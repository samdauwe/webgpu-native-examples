#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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
#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Textured Quad
 *
 * This example shows how to upload a 2D texture to the GPU and display it on a
 * quad with Phong lighting.
 *
 * Features:
 * - Asynchronous texture loading via sokol_fetch + stb_image
 * - Phong lighting (diffuse + specular) in the fragment shader
 * - LOD bias control via GUI slider
 * - Camera interaction (LookAt orbit) via mouse
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/texture/texture.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* textured_quad_vertex_shader_wgsl;
static const char* textured_quad_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Vertex data
 * -------------------------------------------------------------------------- */

typedef struct vertex_t {
  float pos[3];
  float uv[2];
  float normal[3];
} vertex_t;

/* -------------------------------------------------------------------------- *
 * Textured Quad example
 * -------------------------------------------------------------------------- */

/* State struct */
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
    float lod_bias;
    float _padding[3]; /* align to 16 bytes */
  } ubo;
  wgpu_buffer_t uniform_buffer;

  /* Texture */
  wgpu_texture_t texture;
  uint8_t file_buffer[512 * 512 * 4];

  /* Bind group */
  struct {
    WGPUBindGroup handle;
    bool is_dirty;
  } bind_group;
  WGPUBindGroupLayout bind_group_layout;

  /* Pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* GUI settings */
  struct {
    float lod_bias;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  WGPUBool initialized;
} state = {
  .settings = {
    .lod_bias = 0.0f,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1f, 0.1f, 0.1f, 1.0f},
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

/**
 * @brief Initialize the quad vertex and index buffers.
 *
 * The quad lies in the XY plane at Z=0 with normals pointing +Z.
 * WebGPU uses a Y-up, right-handed coordinate system with clip-space Z in
 * [0,1]. The UV origin is at top-left (U right, V down).
 *
 * Vulkan vertex data (Y-down clip, UV origin top-left):
 *   { 1, 1, 0, uv(1,1)}, {-1, 1, 0, uv(0,1)},
 *   {-1,-1, 0, uv(0,0)}, { 1,-1, 0, uv(1,0)}
 *
 * In WebGPU with Y-up clip space the visual appearance is the same if we keep
 * the vertex positions and flip the V coordinate so the texture is not
 * upside-down. We also flip the winding to keep front-face CCW.
 */
static void init_geometry(wgpu_context_t* wgpu_context)
{
  /* Quad vertices — positions in XY plane, UVs with V flipped for WebGPU */
  // clang-format off
  static const vertex_t vertices[4] = {
    { .pos = { 1.0f,  1.0f, 0.0f}, .uv = {1.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = {-1.0f,  1.0f, 0.0f}, .uv = {0.0f, 0.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = {-1.0f, -1.0f, 0.0f}, .uv = {0.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f} },
    { .pos = { 1.0f, -1.0f, 0.0f}, .uv = {1.0f, 1.0f}, .normal = {0.0f, 0.0f, 1.0f} },
  };
  // clang-format on

  /* Two triangles — CCW winding for WebGPU front-face */
  static const uint32_t indices[6] = {0, 1, 2, 2, 3, 0};

  /* Create vertex buffer */
  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Textured Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  /* Create index buffer */
  state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Textured Quad - Index buffer",
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
 * Texture loading (asynchronous via sokol_fetch)
 * -------------------------------------------------------------------------- */

/**
 * @brief Callback invoked by sokol_fetch when texture data is available.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  uint8_t* pixels            = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);

  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc              = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_width,
        .height             = (uint32_t)img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_RenderAttachment,
      .pixels = {
        .ptr  = pixels,
        .size = (size_t)(img_width * img_height * 4),
      },
      .generate_mipmaps = true,
    };
    texture->desc.is_dirty = true;
  }
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Create a small 2D placeholder texture (solid grey) while the real one
   * loads asynchronously */
  static const uint8_t placeholder[4] = {128, 128, 128, 255};
  state.texture = wgpu_create_texture(
    wgpu_context, &(wgpu_texture_desc_t){
                    .extent = (WGPUExtent3D){
                      .width              = 1,
                      .height             = 1,
                      .depthOrArrayLayers = 1,
                    },
                    .format          = WGPUTextureFormat_RGBA8Unorm,
                    .mip_level_count = 1,
                    .usage = WGPUTextureUsage_TextureBinding
                             | WGPUTextureUsage_CopyDst,
                    .pixels = {
                      .ptr  = placeholder,
                      .size = sizeof(placeholder),
                    },
                  });

  /* Kick off async load */
  wgpu_texture_t* texture = &state.texture;
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/metalplate01_rgba.png",
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Textured Quad - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.ubo),
                  });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  camera_t* camera = &state.camera;

  /* Copy camera matrices */
  glm_mat4_copy(camera->matrices.perspective, state.ubo.projection);
  glm_mat4_copy(camera->matrices.view, state.ubo.model_view);

  /* View position from camera */
  glm_vec4_copy((vec4){-camera->position[0], -camera->position[1],
                       -camera->position[2], 0.0f},
                state.ubo.view_pos);

  /* LOD bias from GUI */
  state.ubo.lod_bias = state.settings.lod_bias;

  /* Upload */
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
      /* Uniform buffer */
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
      /* Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Textured Quad - Bind group layout"),
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
                            .label      = STRVIEW("Textured Quad - Bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.bind_group.handle != NULL);
  state.bind_group.is_dirty = false;
}

/* -------------------------------------------------------------------------- *
 * Pipeline
 * -------------------------------------------------------------------------- */

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Textured Quad - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, textured_quad_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, textured_quad_fragment_shader_wgsl);

  /* Blend state */
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
    textured_quad, sizeof(vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    /* Attribute location 1: UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)),
    /* Attribute location 2: Normal */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3,
                       offsetof(vertex_t, normal)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Textured Quad - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &textured_quad_vertex_buffer_layout,
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
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Texture", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("LOD Bias", &state.settings.lod_bias, 0.0f,
                               10.0f, "%.1f");
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

  /* Skip camera input when ImGui captures the mouse */
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
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });

    init_camera(wgpu_context);
    init_geometry(wgpu_context);
    init_texture(wgpu_context);
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

  sfetch_dowork();

  /* Recreate texture when pixel data has been loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    /* Recreate bind group with the new texture view / sampler */
    init_bind_group(wgpu_context);
  }

  /* Update uniforms (camera + LOD bias) */
  update_uniform_buffers(wgpu_context);

  /* ImGui delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

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
  sfetch_shutdown();

  wgpu_destroy_texture(&state.texture);
  wgpu_destroy_buffer(&state.vertex_buffer);
  wgpu_destroy_buffer(&state.index_buffer);
  wgpu_destroy_buffer(&state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group.handle)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Textured Quad",
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
static const char* textured_quad_vertex_shader_wgsl = CODE(
  struct Uniforms {
    projection : mat4x4f,
    modelView  : mat4x4f,
    viewPos    : vec4f,
    lodBias    : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : Uniforms;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) uv       : vec2f,
    @location(2) normal   : vec3f,
  };

  struct VertexOutput {
    @builtin(position) Position  : vec4f,
    @location(0)       fragUV    : vec2f,
    @location(1)       fragNormal: vec3f,
    @location(2)       viewVec   : vec3f,
    @location(3)       lightVec  : vec3f,
  };

  @vertex
  fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;

    output.fragUV = input.uv;

    let worldPos = ubo.modelView * vec4f(input.position, 1.0);
    output.Position = ubo.projection * worldPos;

    /* Transform normal by the upper-left 3x3 of the model-view matrix.
     * For a rigid-body transform (no non-uniform scale) this is correct. */
    let normalMat = mat3x3f(
      ubo.modelView[0].xyz,
      ubo.modelView[1].xyz,
      ubo.modelView[2].xyz
    );
    output.fragNormal = normalMat * input.normal;

    /* Light at origin — transform into model space */
    let lightPos = vec3f(0.0, 0.0, 0.0);
    output.lightVec = lightPos - worldPos.xyz;
    output.viewVec  = ubo.viewPos.xyz - worldPos.xyz;

    return output;
  }
);

static const char* textured_quad_fragment_shader_wgsl = CODE(
  struct Uniforms {
    projection : mat4x4f,
    modelView  : mat4x4f,
    viewPos    : vec4f,
    lodBias    : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : Uniforms;
  @group(0) @binding(1) var texSampler : sampler;
  @group(0) @binding(2) var texColor   : texture_2d<f32>;

  @fragment
  fn fs_main(
    @location(0) fragUV     : vec2f,
    @location(1) fragNormal : vec3f,
    @location(2) viewVec    : vec3f,
    @location(3) lightVec   : vec3f,
  ) -> @location(0) vec4f {
    /* Sample texture with LOD bias */
    let color = textureSampleBias(texColor, texSampler, fragUV, ubo.lodBias);

    /* Phong lighting */
    let N = normalize(fragNormal);
    let L = normalize(lightVec);
    let V = normalize(viewVec);
    let R = reflect(-L, N);

    let diffuse  = max(dot(N, L), 0.0) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 16.0) * color.a;

    return vec4f(diffuse * color.rgb + specular, 1.0);
  }
);
// clang-format on
