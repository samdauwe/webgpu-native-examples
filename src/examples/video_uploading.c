#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example -
 * Video_360°._Timelapse._Bled_Lake_in_Slovenia..webm.720p.vp9.webm
 *
 * This example shows how to upload video frames to WebGPU using FFmpeg pipes.
 * It supports both 2D video playback and 360-degree panoramic videos with
 * mouse interaction.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/videoUploading
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* fullscreen_textured_quad_wgsl;
static const char* sample_external_texture_wgsl;
static const char* sample_external_texture_as_panorama_wgsl;

/* -------------------------------------------------------------------------- *
 * Video Uploading Example
 * -------------------------------------------------------------------------- */

#define VIDEO_COUNT (4)
#define MAX_PATH_LENGTH (256)

/* Video information */
typedef struct {
  char path[MAX_PATH_LENGTH];
  char name[64];
  bool is_360;
} video_info_t;

/* Video decode state */
typedef struct {
  FILE* ffmpeg_pipe;
  pthread_t decode_thread;
  bool thread_running;
  bool has_frame;
  uint8_t* frame_buffer;
  uint8_t* display_buffer;
  pthread_mutex_t buffer_mutex;
  int width;
  int height;
  bool looping;
  bool eof_reached;
} video_decode_t;

/* State struct */
static struct {
  /* Videos */
  video_info_t videos[VIDEO_COUNT];
  int current_video_index;
  video_decode_t video_decode;

  /* Vertex buffer */
  wgpu_buffer_t vertices;

  /* Textures */
  struct {
    WGPUSampler sampler;
    WGPUTexture texture;
    WGPUTextureView view;
  } video_texture;

  /* Pipelines */
  WGPURenderPipeline pipeline_2d;
  WGPURenderPipeline pipeline_360;

  /* Bind groups */
  WGPUBindGroup bind_group_2d;
  WGPUBindGroup bind_group_360;

  /* Uniform buffers */
  wgpu_buffer_t uniform_buffer;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Camera for 360 view */
  struct {
    float y_rotation;
    float x_rotation;
    bool dragging;
    float start_x;
    float start_y;
    float start_y_rotation;
    float start_x_rotation;
  } camera;

  /* GUI */
  bool enable_gui;
  uint64_t last_frame_time;

  /* Fallback */
  bool using_fallback;
  bool ffmpeg_available;

  WGPUBool initialized;
} state = {
  .videos =
    {
      {
        .path   = "assets/videos/5214261-hd_1920_1080_25fps.mp4",
        .name   = "Giraffe (2D)",
        .is_360 = false,
      },
      {
        .path   = "assets/videos/big_buck_bunny_trailer.mp4",
        .name   = "Big Buck Bunny (2D)",
        .is_360 = false,
      },
      {
        .path   = "assets/videos/Video_360°._Timelapse._Bled_Lake_in_Slovenia..webm.720p.vp9.webm",
        .name   = "Lake Bled (360°, drag to aim)",
        .is_360 = true,
      },
      {
        .path   = "assets/videos/underwater_diving_360degrees.mp4",
        .name   = "Underwater Diving (360°, drag to aim)",
        .is_360 = true,
      },
    },
  .current_video_index = 0,
  .color_attachment =
    {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
  .render_pass_descriptor =
    {
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.color_attachment,
    },
  .camera =
    {
      .y_rotation = 0.0f,
      .x_rotation = 0.0f,
      .dragging   = false,
    },
  .enable_gui        = true,
  .using_fallback    = false,
  .ffmpeg_available  = false,
};

/* -------------------------------------------------------------------------- *
 * FFmpeg Helper Functions
 * -------------------------------------------------------------------------- */

static bool check_ffmpeg_available(void)
{
  int ret = system("which ffmpeg > /dev/null 2>&1");
  if (ret != 0) {
    printf("FFmpeg not found in PATH\n");
    return false;
  }

  ret = system("which ffprobe > /dev/null 2>&1");
  if (ret != 0) {
    printf("ffprobe not found in PATH\n");
    return false;
  }

  return true;
}

static bool get_video_dimensions(const char* video_path, int* width,
                                 int* height)
{
  char cmd[512];
  snprintf(cmd, sizeof(cmd),
           "ffprobe -v error -select_streams v:0 -show_entries "
           "stream=width,height -of csv=s=x:p=0 \"%s\"",
           video_path);

  FILE* pipe = popen(cmd, "r");
  if (!pipe) {
    printf("Failed to run ffprobe\n");
    return false;
  }

  if (fscanf(pipe, "%dx%d", width, height) != 2) {
    printf("Failed to parse video dimensions\n");
    pclose(pipe);
    return false;
  }

  pclose(pipe);
  return true;
}

/* Video decode thread */
static void* video_decode_thread(void* arg)
{
  video_decode_t* decode = (video_decode_t*)arg;

  while (decode->thread_running) {
    if (decode->eof_reached && decode->looping) {
      /* Restart the video by closing and reopening */
      if (decode->ffmpeg_pipe) {
        pclose(decode->ffmpeg_pipe);
        decode->ffmpeg_pipe = NULL;
      }

      /* Reopen FFmpeg pipe */
      const char* video_path = state.videos[state.current_video_index].path;
      char cmd[512];
      snprintf(cmd, sizeof(cmd),
               "ffmpeg -re -i \"%s\" -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
               video_path);

      decode->ffmpeg_pipe = popen(cmd, "r");
      if (!decode->ffmpeg_pipe) {
        printf("Failed to restart FFmpeg pipe\n");
        break;
      }
      decode->eof_reached = false;
    }

    /* Read one frame from FFmpeg */
    size_t frame_size = decode->width * decode->height * 3;
    size_t count
      = fread(decode->frame_buffer, 1, frame_size, decode->ffmpeg_pipe);

    if (count != frame_size) {
      decode->eof_reached = true;
      struct timespec ts  = {.tv_sec = 0, .tv_nsec = 16000000}; /* 16ms */
      nanosleep(&ts, NULL);
      continue;
    }

    /* Copy to display buffer with RGB to RGBA conversion */
    pthread_mutex_lock(&decode->buffer_mutex);
    for (int y = 0; y < decode->height; ++y) {
      for (int x = 0; x < decode->width; ++x) {
        int src_idx  = (y * decode->width + x) * 3;
        int dest_idx = (y * decode->width + x) * 4;
        decode->display_buffer[dest_idx + 0]
          = decode->frame_buffer[src_idx + 0];
        decode->display_buffer[dest_idx + 1]
          = decode->frame_buffer[src_idx + 1];
        decode->display_buffer[dest_idx + 2]
          = decode->frame_buffer[src_idx + 2];
        decode->display_buffer[dest_idx + 3] = 255;
      }
    }
    decode->has_frame = true;
    pthread_mutex_unlock(&decode->buffer_mutex);

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 16000000}; /* ~60 FPS */
    nanosleep(&ts, NULL);
  }

  return NULL;
}

static bool start_video_decode(const char* video_path)
{
  video_decode_t* decode = &state.video_decode;

  /* Stop any existing decode */
  if (decode->thread_running) {
    decode->thread_running = false;
    pthread_join(decode->decode_thread, NULL);
  }

  if (decode->ffmpeg_pipe) {
    pclose(decode->ffmpeg_pipe);
    decode->ffmpeg_pipe = NULL;
  }

  /* Get video dimensions */
  if (!get_video_dimensions(video_path, &decode->width, &decode->height)) {
    return false;
  }

  /* Allocate buffers */
  size_t frame_size = decode->width * decode->height * 3;
  size_t rgba_size  = decode->width * decode->height * 4;

  if (decode->frame_buffer) {
    free(decode->frame_buffer);
  }
  if (decode->display_buffer) {
    free(decode->display_buffer);
  }

  decode->frame_buffer   = (uint8_t*)malloc(frame_size);
  decode->display_buffer = (uint8_t*)malloc(rgba_size);

  if (!decode->frame_buffer || !decode->display_buffer) {
    printf("Failed to allocate frame buffers\n");
    return false;
  }

  /* Initialize mutex */
  pthread_mutex_init(&decode->buffer_mutex, NULL);

  /* Start FFmpeg pipe */
  char cmd[512];
  snprintf(cmd, sizeof(cmd),
           "ffmpeg -re -i \"%s\" -f rawvideo -pix_fmt rgb24 - 2>/dev/null",
           video_path);

  decode->ffmpeg_pipe = popen(cmd, "r");
  if (!decode->ffmpeg_pipe) {
    printf("Failed to open FFmpeg pipe\n");
    return false;
  }

  /* Start decode thread */
  decode->thread_running = true;
  decode->has_frame      = false;
  decode->looping        = true;
  decode->eof_reached    = false;

  if (pthread_create(&decode->decode_thread, NULL, video_decode_thread, decode)
      != 0) {
    printf("Failed to create decode thread\n");
    pclose(decode->ffmpeg_pipe);
    decode->ffmpeg_pipe = NULL;
    return false;
  }

  return true;
}

static void stop_video_decode(void)
{
  video_decode_t* decode = &state.video_decode;

  if (decode->thread_running) {
    decode->thread_running = false;
    pthread_join(decode->decode_thread, NULL);
  }

  if (decode->ffmpeg_pipe) {
    pclose(decode->ffmpeg_pipe);
    decode->ffmpeg_pipe = NULL;
  }

  pthread_mutex_destroy(&decode->buffer_mutex);

  if (decode->frame_buffer) {
    free(decode->frame_buffer);
    decode->frame_buffer = NULL;
  }

  if (decode->display_buffer) {
    free(decode->display_buffer);
    decode->display_buffer = NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * WebGPU Resource Initialization
 * -------------------------------------------------------------------------- */

static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Fullscreen quad vertices (position only, UVs generated in shader) */
  static const float quad_vertices[12] = {
    // clang-format off
    -1.0f, -1.0f,
    -1.0f,  3.0f,
     3.0f, -1.0f,
    // clang-format on
  };

  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Video quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(quad_vertices),
                    .count = 3,
                    .initial.data = quad_vertices,
                  });
}

static void init_video_texture(wgpu_context_t* wgpu_context)
{
  /* Cleanup existing texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.video_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.video_texture.texture)

  int width  = state.video_decode.width;
  int height = state.video_decode.height;

  if (width == 0 || height == 0) {
    /* Use fallback size */
    width  = 1920;
    height = 1080;
  }

  /* Create the texture */
  state.video_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Video - Texture"),
      .size =
        (WGPUExtent3D){
          .width              = width,
          .height             = height,
          .depthOrArrayLayers = 1,
        },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    });
  ASSERT(state.video_texture.texture != NULL);

  /* Create the texture view */
  state.video_texture.view = wgpuTextureCreateView(
    state.video_texture.texture, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("Video - Texture view"),
                                   .format    = WGPUTextureFormat_RGBA8Unorm,
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .baseMipLevel    = 0,
                                   .mipLevelCount   = 1,
                                   .baseArrayLayer  = 0,
                                   .arrayLayerCount = 1,
                                 });
  ASSERT(state.video_texture.view != NULL);

  /* Create sampler if not already created */
  if (!state.video_texture.sampler) {
    state.video_texture.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label = STRVIEW("Video - Texture sampler"),
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .maxAnisotropy = 1,
                            });
    ASSERT(state.video_texture.sampler != NULL);
  }

  /* Upload fallback texture if needed */
  if (state.using_fallback) {
    wgpu_texture_t fallback_tex = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      });

    if (fallback_tex.desc.pixels.ptr) {
      wgpu_image_to_texure(wgpu_context, state.video_texture.texture,
                           (void*)fallback_tex.desc.pixels.ptr,
                           (WGPUExtent3D){
                             .width              = width,
                             .height             = height,
                             .depthOrArrayLayers = 1,
                           },
                           4u);
      wgpu_destroy_texture(&fallback_tex);
    }
  }
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Buffer large enough for either pipeline */
  const uint64_t buffer_size
    = (16 + 2 + 2) * sizeof(float); /* mat4 + vec2 + padding */

  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Video - Uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = buffer_size,
                  });
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* 2D Video Pipeline */
  {
    WGPUShaderModule vs_module = wgpu_create_shader_module(
      wgpu_context->device, fullscreen_textured_quad_wgsl);
    WGPUShaderModule fs_module = wgpu_create_shader_module(
      wgpu_context->device, sample_external_texture_wgsl);

    /* Vertex state */
    WGPU_VERTEX_BUFFER_LAYOUT(
      video, 2 * sizeof(float),
      /* Attribute location 0: Position */
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2, 0))

    WGPUVertexState vertex_state = {
      .module      = vs_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &video_vertex_buffer_layout,
    };

    /* Fragment state */
    WGPUBlendState blend_state        = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUFragmentState fragment_state = {
      .module      = fs_module,
      .entryPoint  = STRVIEW("main"),
      .targetCount = 1,
      .targets     = &color_target,
    };

    /* Multisample state */
    WGPUMultisampleState multisample_state = {
      .count                  = 1,
      .mask                   = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    };

    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    /* Create pipeline */
    state.pipeline_2d = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label  = STRVIEW("Video 2D - Render pipeline"),
                              .layout = NULL, /* Auto layout */
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(state.pipeline_2d != NULL);

    wgpuShaderModuleRelease(vs_module);
    wgpuShaderModuleRelease(fs_module);
  }

  /* 360 Video Pipeline */
  {
    WGPUShaderModule module = wgpu_create_shader_module(
      wgpu_context->device, sample_external_texture_as_panorama_wgsl);

    /* Vertex state */
    WGPUVertexState vertex_state = {
      .module      = module,
      .entryPoint  = STRVIEW("vs"),
      .bufferCount = 0, /* No vertex buffer, vertices generated in shader */
      .buffers     = NULL,
    };

    /* Fragment state */
    WGPUBlendState blend_state        = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUFragmentState fragment_state = {
      .module      = module,
      .entryPoint  = STRVIEW("main"),
      .targetCount = 1,
      .targets     = &color_target,
    };

    /* Multisample state */
    WGPUMultisampleState multisample_state = {
      .count                  = 1,
      .mask                   = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    };

    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    /* Create pipeline */
    state.pipeline_360 = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label  = STRVIEW("Video 360 - Render pipeline"),
                              .layout = NULL, /* Auto layout */
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(state.pipeline_360 != NULL);

    wgpuShaderModuleRelease(module);
  }
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* 2D Bind Group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] =
        (WGPUBindGroupEntry){
          .binding = 0,
          .sampler = state.video_texture.sampler,
        },
      [1] =
        (WGPUBindGroupEntry){
          .binding     = 1,
          .textureView = state.video_texture.view,
        },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Video 2D - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline_2d, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    state.bind_group_2d
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.bind_group_2d != NULL);
  }

  /* 360 Bind Group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] =
        (WGPUBindGroupEntry){
          .binding = 0,
          .sampler = state.video_texture.sampler,
        },
      [1] =
        (WGPUBindGroupEntry){
          .binding     = 1,
          .textureView = state.video_texture.view,
        },
      [2] =
        (WGPUBindGroupEntry){
          .binding = 2,
          .buffer  = state.uniform_buffer.buffer,
          .size    = state.uniform_buffer.size,
        },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Video 360 - Bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline_360, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    state.bind_group_360
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.bind_group_360 != NULL);
  }
}

static void switch_video(wgpu_context_t* wgpu_context, int index)
{
  if (index < 0 || index >= VIDEO_COUNT) {
    return;
  }

  state.current_video_index = index;

  /* Stop current video */
  stop_video_decode();

  /* Start new video if FFmpeg is available */
  if (state.ffmpeg_available) {
    const char* video_path = state.videos[index].path;
    if (start_video_decode(video_path)) {
      state.using_fallback = false;
      /* Recreate texture with new dimensions */
      init_video_texture(wgpu_context);
      /* Recreate bind groups */
      WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group_2d)
      WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group_360)
      init_bind_groups(wgpu_context);
    }
    else {
      printf("Failed to start video decode, using fallback\n");
      state.using_fallback = true;
    }
  }

  /* Reset camera for 360 videos */
  if (state.videos[index].is_360) {
    state.camera.y_rotation = 0.0f;
    state.camera.x_rotation = 0.0f;
  }
}

/* -------------------------------------------------------------------------- *
 * Event Handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  imgui_overlay_handle_input(wgpu_context, event);

  if (event->type == INPUT_EVENT_TYPE_MOUSE_DOWN) {
    if (event->mouse_button == BUTTON_LEFT
        && !imgui_overlay_want_capture_mouse()) {
      state.camera.dragging         = true;
      state.camera.start_x          = event->mouse_x;
      state.camera.start_y          = event->mouse_y;
      state.camera.start_y_rotation = state.camera.y_rotation;
      state.camera.start_x_rotation = state.camera.x_rotation;
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    if (event->mouse_button == BUTTON_LEFT) {
      state.camera.dragging = false;
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (state.camera.dragging && state.videos[state.current_video_index].is_360
        && !imgui_overlay_want_capture_mouse()) {
      float delta_x = event->mouse_x - state.camera.start_x;
      float delta_y = event->mouse_y - state.camera.start_y;

      state.camera.y_rotation = state.camera.start_y_rotation + delta_x * 0.01f;
      state.camera.x_rotation
        = CLAMP(state.camera.start_x_rotation + delta_y * -0.01f, -PI * 0.4f,
                PI * 0.4f);
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_RESIZED) {
    UNUSED_VAR(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Render Functions
 * -------------------------------------------------------------------------- */

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  bool is_360 = state.videos[state.current_video_index].is_360;

  if (is_360) {
    /* 360 video: Update camera matrices */
    const float time     = stm_sec(stm_now());
    const float rotation = time * 0.1f + state.camera.y_rotation;
    const float aspect
      = (float)wgpu_context->width / (float)wgpu_context->height;

    /* Projection matrix */
    mat4 projection;
    glm_perspective((75.0f * PI) / 180.0f, aspect, 0.5f, 100.0f, projection);

    /* Camera matrix */
    mat4 camera;
    glm_mat4_identity(camera);
    glm_rotate_y(camera, rotation, camera);
    glm_rotate_x(camera, state.camera.x_rotation, camera);

    /* View matrix (inverse of camera) */
    mat4 view;
    glm_mat4_inv(camera, view);

    /* View-direction-projection matrix */
    mat4 view_dir_proj;
    glm_mat4_mul(projection, view, view_dir_proj);

    /* Inverse */
    mat4 view_dir_proj_inv;
    glm_mat4_inv(view_dir_proj, view_dir_proj_inv);

    /* Upload uniforms: mat4x4 + vec2 (target size) */
    struct {
      mat4 view_dir_proj_inv;
      float target_width;
      float target_height;
      float padding[2];
    } uniforms;

    glm_mat4_copy(view_dir_proj_inv, uniforms.view_dir_proj_inv);
    uniforms.target_width  = (float)wgpu_context->width;
    uniforms.target_height = (float)wgpu_context->height;

    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                         &uniforms, sizeof(uniforms));
  }
  /* 2D video: No uniforms needed */
}

static void update_video_texture(wgpu_context_t* wgpu_context)
{
  if (state.using_fallback) {
    return;
  }

  video_decode_t* decode = &state.video_decode;

  pthread_mutex_lock(&decode->buffer_mutex);
  if (decode->has_frame) {
    wgpu_image_to_texure(wgpu_context, state.video_texture.texture,
                         decode->display_buffer,
                         (WGPUExtent3D){
                           .width              = decode->width,
                           .height             = decode->height,
                           .depthOrArrayLayers = 1,
                         },
                         4u);
    decode->has_frame = false;
  }
  pthread_mutex_unlock(&decode->buffer_mutex);
}

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  if (!state.enable_gui) {
    return;
  }

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 150.0f}, ImGuiCond_FirstUseEver);

  if (igBegin("Video Uploading", NULL, ImGuiWindowFlags_None)) {
    /* Video selection */
    const char* current_video = state.videos[state.current_video_index].name;
    if (igBeginCombo("Video", current_video, ImGuiComboFlags_None)) {
      for (int i = 0; i < VIDEO_COUNT; ++i) {
        bool is_selected = (i == state.current_video_index);
        if (igSelectable(state.videos[i].name, is_selected,
                         ImGuiSelectableFlags_None, (ImVec2){0, 0})) {
          switch_video(wgpu_context, i);
        }
        if (is_selected) {
          igSetItemDefaultFocus();
        }
      }
      igEndCombo();
    }

    igSeparator();

    /* Status */
    if (state.ffmpeg_available) {
      if (state.using_fallback) {
        igTextColored((ImVec4){1.0f, 0.5f, 0.0f, 1.0f},
                      "Status: Using fallback texture");
      }
      else {
        igTextColored((ImVec4){0.0f, 1.0f, 0.0f, 1.0f},
                      "Status: Playing video");
        igText("Resolution: %dx%d", state.video_decode.width,
               state.video_decode.height);
      }
    }
    else {
      igTextColored((ImVec4){1.0f, 0.0f, 0.0f, 1.0f}, "FFmpeg not available");
      igTextWrapped("Please install FFmpeg and ffprobe to play videos.");
    }

    /* 360 camera info */
    if (state.videos[state.current_video_index].is_360) {
      igSeparator();
      igText("360° Video: Drag to look around");
      igText("Y Rotation: %.2f", state.camera.y_rotation);
      igText("X Rotation: %.2f", state.camera.x_rotation);
    }
  }
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Lifecycle Functions
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();

    /* Check if FFmpeg is available */
    state.ffmpeg_available = check_ffmpeg_available();

    if (!state.ffmpeg_available) {
      printf("Warning: FFmpeg not available, using fallback texture\n");
      state.using_fallback      = true;
      state.video_decode.width  = 1920;
      state.video_decode.height = 1080;
    }
    else {
      /* Start playing first video */
      const char* video_path = state.videos[0].path;
      if (!start_video_decode(video_path)) {
        printf("Failed to start video decode, using fallback\n");
        state.using_fallback      = true;
        state.video_decode.width  = 1920;
        state.video_decode.height = 1080;
      }
    }

    init_vertex_buffer(wgpu_context);
    init_video_texture(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups(wgpu_context);
    imgui_overlay_init(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_SUCCESS;
  }

  /* Update video texture */
  update_video_texture(wgpu_context);

  /* Update uniforms */
  update_uniforms(wgpu_context);

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render */
  WGPUTextureView backbuffer_view = wgpu_context->swapchain_view;
  if (!backbuffer_view) {
    return EXIT_FAILURE;
  }

  /* Update render pass color attachment */
  state.color_attachment.view = backbuffer_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  ASSERT(cmd_encoder != NULL);

  /* Render pass */
  WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
    cmd_encoder, &state.render_pass_descriptor);
  ASSERT(rpass_enc != NULL);

  /* Select pipeline and bind group based on video type */
  bool is_360 = state.videos[state.current_video_index].is_360;
  if (is_360) {
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline_360);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group_360, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  }
  else {
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline_2d);
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group_2d, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)

  /* Submit command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  ASSERT(cmd_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer)

  /* Draw GUI overlay on top */
  render_gui(wgpu_context);
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void cleanup(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  stop_video_decode();

  imgui_overlay_shutdown();

  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.video_texture.sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.video_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.video_texture.texture)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline_2d)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline_360)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group_2d)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group_360)
}

/* -------------------------------------------------------------------------- *
 * Main Entry Point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Video Uploading",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = cleanup,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

/* Fullscreen textured quad vertex shader */
static const char* fullscreen_textured_quad_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4f, @location(0) uv : vec2f,
  };

  @vertex fn main(@location(0) position : vec2f)->VertexOutput {
    var output : VertexOutput;
    output.position = vec4f(position, 0.0, 1.0);
    output.uv       = position * vec2f(0.5, -0.5) + vec2f(0.5);
    return output;
  });

/*2D video fragment shader */
static const char* sample_external_texture_wgsl
  = CODE(@group(0) @binding(0) var mySampler : sampler;
         @group(0) @binding(1) var myTexture : texture_2d<f32>;

         @fragment fn main(@location(0) fragUV : vec2f)->@location(0) vec4f {
           return textureSample(myTexture, mySampler, fragUV);
         });

/* 360 panoramic video shader */
static const char* sample_external_texture_as_panorama_wgsl = CODE(
  struct Uniforms {
    viewDirectionProjectionInverse: mat4x4f,
    targetSize: vec2f,
  };

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
  };

  @vertex
  fn vs(@builtin(vertex_index) vertexIndex: u32) -> VSOutput {
    let pos = array(
      vec2f(-1, -1),
      vec2f(-1,  3),
      vec2f( 3, -1),
    );

    let xy = pos[vertexIndex];
    return VSOutput(
        vec4f(xy, 0.0, 1.0),
        xy * vec2f(0.5, -0.5) + vec2f(0.5)
    );
  }

  @group(0) @binding(0) var panoramaSampler: sampler;
  @group(0) @binding(1) var panoramaTexture: texture_2d<f32>;
  @group(0) @binding(2) var<uniform> uniforms: Uniforms;

  const PI = radians(180.0);
  @fragment
  fn main(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let pos = position.xy / uniforms.targetSize * 2.0 - 1.0;
    let t = uniforms.viewDirectionProjectionInverse * vec4f(pos, 0, 1);
    let dir = normalize(t.xyz / t.w);

    let longitude = atan2(dir.z, dir.x);
    let latitude = asin(dir.y / length(dir));

    let uv = vec2f(
      longitude / (2.0 * PI) + 0.5,
      latitude / PI + 0.5,
    );

    return textureSample(panoramaTexture, panoramaSampler, uv);
  }
);
