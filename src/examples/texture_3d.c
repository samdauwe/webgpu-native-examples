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

#include "core/camera.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - 3D Textures
 *
 * Demonstrates 3D texture loading and generation using Perlin noise. A 3D
 * noise texture is procedurally generated on the CPU using fractal Perlin
 * noise and uploaded to the GPU as a 3D texture. The depth slice through the
 * volume is animated over time, showing a smoothly changing cross-section of
 * the noise field rendered on a lit quad.
 *
 * Features:
 * - Procedural 3D Perlin noise generation (fractal noise with 6 octaves)
 * - 3D texture creation and upload (R8Unorm, 128x128x128)
 * - Animated depth slice through the 3D volume
 * - Phong lighting (diffuse + specular)
 * - GUI button to regenerate the noise texture
 * - Camera interaction (LookAt orbit) via mouse
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/texture3d
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* texture_3d_vertex_shader_wgsl;
static const char* texture_3d_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Perlin noise implementation
 * Translation of Ken Perlin's JAVA reference
 * (http://mrl.nyu.edu/~perlin/noise/)
 * -------------------------------------------------------------------------- */

#define NOISE_TEX_WIDTH 128
#define NOISE_TEX_HEIGHT 128
#define NOISE_TEX_DEPTH 128
#define NOISE_TEX_SIZE (NOISE_TEX_WIDTH * NOISE_TEX_HEIGHT * NOISE_TEX_DEPTH)

typedef struct perlin_noise_t {
  uint32_t permutations[512];
} perlin_noise_t;

static float perlin_fade(float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float perlin_lerp(float t, float a, float b)
{
  return a + t * (b - a);
}

static float perlin_grad(int hash, float x, float y, float z)
{
  int h   = hash & 15;
  float u = h < 8 ? x : y;
  float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
  return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

static void perlin_noise_init(perlin_noise_t* pn, bool random_seed)
{
  /* Generate identity permutation */
  uint8_t plookup[256];
  for (int i = 0; i < 256; ++i) {
    plookup[i] = (uint8_t)i;
  }

  /* Fisher-Yates shuffle */
  if (random_seed) {
    srand((unsigned int)time(NULL));
  }
  else {
    srand(0);
  }
  for (int i = 255; i > 0; --i) {
    int j       = rand() % (i + 1);
    uint8_t tmp = plookup[i];
    plookup[i]  = plookup[j];
    plookup[j]  = tmp;
  }

  /* Duplicate the permutation table */
  for (int i = 0; i < 256; ++i) {
    pn->permutations[i]       = plookup[i];
    pn->permutations[256 + i] = plookup[i];
  }
}

static float perlin_noise_sample(const perlin_noise_t* pn, float x, float y,
                                 float z)
{
  /* Find unit cube containing point */
  int32_t X = (int32_t)floorf(x) & 255;
  int32_t Y = (int32_t)floorf(y) & 255;
  int32_t Z = (int32_t)floorf(z) & 255;

  /* Relative position in cube */
  x -= floorf(x);
  y -= floorf(y);
  z -= floorf(z);

  /* Fade curves */
  float u = perlin_fade(x);
  float v = perlin_fade(y);
  float w = perlin_fade(z);

  /* Hash coordinates of the 8 cube corners */
  uint32_t A  = pn->permutations[X] + Y;
  uint32_t AA = pn->permutations[A] + Z;
  uint32_t AB = pn->permutations[A + 1] + Z;
  uint32_t B  = pn->permutations[X + 1] + Y;
  uint32_t BA = pn->permutations[B] + Z;
  uint32_t BB = pn->permutations[B + 1] + Z;

  /* Blend results from 8 corners */
  float res = perlin_lerp(
    w,
    perlin_lerp(
      v,
      perlin_lerp(u, perlin_grad(pn->permutations[AA], x, y, z),
                  perlin_grad(pn->permutations[BA], x - 1, y, z)),
      perlin_lerp(u, perlin_grad(pn->permutations[AB], x, y - 1, z),
                  perlin_grad(pn->permutations[BB], x - 1, y - 1, z))),
    perlin_lerp(
      v,
      perlin_lerp(u, perlin_grad(pn->permutations[AA + 1], x, y, z - 1),
                  perlin_grad(pn->permutations[BA + 1], x - 1, y, z - 1)),
      perlin_lerp(u, perlin_grad(pn->permutations[AB + 1], x, y - 1, z - 1),
                  perlin_grad(pn->permutations[BB + 1], x - 1, y - 1, z - 1))));

  return res;
}

/* -------------------------------------------------------------------------- *
 * Fractal noise generator (multiple octaves of Perlin noise)
 * -------------------------------------------------------------------------- */

static float fractal_noise(const perlin_noise_t* pn, float x, float y, float z,
                           uint32_t octaves, float persistence)
{
  float sum       = 0.0f;
  float frequency = 1.0f;
  float amplitude = 1.0f;
  float max_val   = 0.0f;

  for (uint32_t i = 0; i < octaves; ++i) {
    sum += perlin_noise_sample(pn, x * frequency, y * frequency, z * frequency)
           * amplitude;
    max_val += amplitude;
    amplitude *= persistence;
    frequency *= 2.0f;
  }

  sum = sum / max_val;
  return (sum + 1.0f) / 2.0f; /* Normalize to [0, 1] */
}

/* -------------------------------------------------------------------------- *
 * Vertex data
 * -------------------------------------------------------------------------- */

typedef struct vertex_t {
  float pos[3];
  float uv[2];
  float normal[3];
} vertex_t;

/* -------------------------------------------------------------------------- *
 * Texture 3D example
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;
  bool view_updated;

  /* Vertex / Index buffers */
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 model_view;
    vec4 view_pos;
    float depth;
    float _padding[3]; /* align to 16 bytes */
  } ubo;
  wgpu_buffer_t uniform_buffer;

  /* 3D noise texture */
  wgpu_texture_t texture;

  /* Bind group */
  struct {
    WGPUBindGroup handle;
  } bind_group;
  WGPUBindGroupLayout bind_group_layout;

  /* Pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* Timing */
  uint64_t last_frame_time;
  float frame_timer;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  WGPUBool initialized;
} state = {
  .ubo = {
    .depth = 0.0f,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
};

/* -------------------------------------------------------------------------- *
 * Geometry setup
 * -------------------------------------------------------------------------- */

static void init_geometry(wgpu_context_t* wgpu_context)
{
  /*
   * Quad in XY plane, normals +Z. UVs adapted for WebGPU (V flipped).
   * Vulkan UVs: (1,1), (0,1), (0,0), (1,0) with Y-down clip.
   * WebGPU UVs: (1,0), (0,0), (0,1), (1,1) with Y-up clip.
   */
  // clang-format off
  static const vertex_t vertices[4] = {
    { .pos = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f} },
  };
  // clang-format on

  static const uint32_t indices[6] = {0, 1, 2, 2, 3, 0};

  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Texture 3D - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Texture 3D - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });
}

/* -------------------------------------------------------------------------- *
 * Camera setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -2.5f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 15.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
}

/* -------------------------------------------------------------------------- *
 * 3D Noise texture generation
 * -------------------------------------------------------------------------- */

/**
 * @brief Generate Perlin noise data and upload to the 3D texture.
 */
static void generate_noise_texture(wgpu_context_t* wgpu_context)
{
  const uint32_t w = NOISE_TEX_WIDTH;
  const uint32_t h = NOISE_TEX_HEIGHT;
  const uint32_t d = NOISE_TEX_DEPTH;

  /* Allocate noise data — single channel R8 */
  uint8_t* data = (uint8_t*)malloc(NOISE_TEX_SIZE);
  if (!data) {
    printf("Error: Failed to allocate noise data\n");
    return;
  }
  memset(data, 0, NOISE_TEX_SIZE);

  /* Initialize Perlin noise with random seed */
  perlin_noise_t pn;
  perlin_noise_init(&pn, true);

  const float noise_scale = (float)(rand() % 10) + 4.0f;
  const uint32_t octaves  = 6;
  const float persistence = 0.5f;

  /* Generate 3D noise */
  for (uint32_t z = 0; z < d; ++z) {
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        float nx = (float)x / (float)w;
        float ny = (float)y / (float)h;
        float nz = (float)z / (float)d;
        float n  = fractal_noise(&pn, nx * noise_scale, ny * noise_scale,
                                 nz * noise_scale, octaves, persistence);
        n        = n - floorf(n);
        data[x + y * w + z * w * h] = (uint8_t)(floorf(n * 255.0f));
      }
    }
  }

  /* Upload noise data to the 3D texture */
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.texture.handle,
                          .mipLevel = 0,
                          .origin   = (WGPUOrigin3D){0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        data, NOISE_TEX_SIZE,
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = w * 1, /* 1 byte per texel (R8) */
                          .rowsPerImage = h,
                        },
                        &(WGPUExtent3D){w, h, d});

  free(data);
}

/**
 * @brief Create the GPU texture, sampler and view for the 3D noise texture.
 */
static void init_noise_texture(wgpu_context_t* wgpu_context)
{
  /* Create 3D texture using the common helper */
  state.texture = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent       = {NOISE_TEX_WIDTH, NOISE_TEX_HEIGHT, NOISE_TEX_DEPTH},
      .format       = WGPUTextureFormat_R8Unorm,
      .dimension    = WGPUTextureDimension_3D,
      .address_mode = WGPUAddressMode_ClampToEdge,
    });

  /* Generate and upload noise data */
  generate_noise_texture(wgpu_context);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Texture 3D - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.ubo),
                  });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  camera_t* camera = &state.camera;

  glm_mat4_copy(camera->matrices.perspective, state.ubo.projection);
  glm_mat4_copy(camera->matrices.view, state.ubo.model_view);
  glm_vec4_copy((vec4){-camera->position[0], -camera->position[1],
                       -camera->position[2], 0.0f},
                state.ubo.view_pos);

  /* Animate depth slice through the 3D texture */
  state.ubo.depth += state.frame_timer * 0.15f;
  if (state.ubo.depth > 1.0f) {
    state.ubo.depth -= 1.0f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &state.ubo, sizeof(state.ubo));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout and bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      /* Uniform buffer — vertex + fragment */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(state.ubo),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      /* 3D texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_3D,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Texture 3D - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.bind_group_layout != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group.handle)

  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer.size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.texture.sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.texture.view,
    },
  };

  state.bind_group.handle = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Texture 3D - Bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.bind_group.handle != NULL);
}

/* -------------------------------------------------------------------------- *
 * Pipeline
 * -------------------------------------------------------------------------- */

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Texture 3D - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, texture_3d_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, texture_3d_fragment_shader_wgsl);

  /* Blend state — no blending */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout: pos (float32x3), uv (float32x2), normal (float32x3)
   */
  WGPU_VERTEX_BUFFER_LAYOUT(
    texture_3d, sizeof(vertex_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)),
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Texture 3D - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &texture_3d_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = frag_shader_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_None,
      .frontFace = WGPUFrontFace_CCW,
    },
    .depthStencil = &depth_stencil_state,
    .multisample  = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Texture", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igButton("Generate new texture", (ImVec2){0, 0})) {
      generate_noise_texture(wgpu_context);
    }
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (imgui_overlay_want_capture_mouse()) {
    return;
  }

  camera_on_input_event(&state.camera, input_event);
  state.view_updated = true;

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    camera_update_aspect_ratio(&state.camera,
                               (float)input_event->window_width
                                 / (float)input_event->window_height);
    state.view_updated = true;
  }
}

/* -------------------------------------------------------------------------- *
 * Lifecycle
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();

    init_camera(wgpu_context);
    init_geometry(wgpu_context);
    init_noise_texture(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_bind_group_layout(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
    imgui_overlay_init(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate frame delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;
  state.frame_timer     = delta_time;

  /* Update uniforms (camera + animated depth) */
  update_uniform_buffers(wgpu_context);

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* Begin render pass */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record draw commands */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.index_buffer.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group.handle, 0,
                                    0);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, 6, 1, 0, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  wgpu_destroy_buffer(&state.vertex_buffer);
  wgpu_destroy_buffer(&state.index_buffer);
  wgpu_destroy_buffer(&state.uniform_buffer);
  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group.handle)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "3D Textures",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* texture_3d_vertex_shader_wgsl = CODE(
  struct Uniforms {
    projection : mat4x4f,
    modelView  : mat4x4f,
    viewPos    : vec4f,
    depth      : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : Uniforms;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) uv       : vec2f,
    @location(2) normal   : vec3f,
  };

  struct VertexOutput {
    @builtin(position) Position  : vec4f,
    @location(0)       fragUV    : vec3f,
    @location(1)       fragNormal: vec3f,
    @location(2)       viewVec   : vec3f,
    @location(3)       lightVec  : vec3f,
  };

  @vertex
  fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;

    /* Pass 2D UV + animated depth as 3D texture coordinate */
    output.fragUV = vec3f(input.uv, ubo.depth);

    let worldPos = ubo.modelView * vec4f(input.position, 1.0);
    output.Position = ubo.projection * worldPos;

    /* Transform normal by upper-left 3x3 of model-view matrix */
    let normalMat = mat3x3f(
      ubo.modelView[0].xyz,
      ubo.modelView[1].xyz,
      ubo.modelView[2].xyz
    );
    output.fragNormal = normalMat * input.normal;

    /* Light at origin */
    let lightPos = vec3f(0.0, 0.0, 0.0);
    output.lightVec = lightPos - worldPos.xyz;
    output.viewVec  = ubo.viewPos.xyz - worldPos.xyz;

    return output;
  }
);

static const char* texture_3d_fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var texSampler : sampler;
  @group(0) @binding(2) var texColor   : texture_3d<f32>;

  @fragment
  fn fs_main(
    @location(0) fragUV     : vec3f,
    @location(1) fragNormal : vec3f,
    @location(2) viewVec    : vec3f,
    @location(3) lightVec   : vec3f,
  ) -> @location(0) vec4f {
    /* Sample 3D texture — R8Unorm so only .r channel has data */
    let color = textureSample(texColor, texSampler, fragUV);

    /* Phong lighting */
    let N = normalize(fragNormal);
    let L = normalize(lightVec);
    let V = normalize(viewVec);
    let R = reflect(-L, N);

    let diffuse  = max(dot(N, L), 0.0) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 16.0) * color.r;

    return vec4f(diffuse * color.r + specular, 1.0);
  }
);
// clang-format on
