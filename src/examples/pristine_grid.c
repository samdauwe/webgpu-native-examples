#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pristine Grid
 *
 * A simple WebGPU implementation of the "Pristine Grid" technique described in
 * this wonderful little blog post:
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 *
 * Ref:
 * https://github.com/toji/pristine-grid-webgpu
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* grid_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Pristine Grid example
 * -------------------------------------------------------------------------- */

/* Vertex layout */
typedef struct {
  vec3 position;
  vec2 uv;
} vertex_t;

/* Camera uniforms */
typedef struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  vec3 camera_position;
  float time;
} camera_uniforms_t;

/* Grid uniforms */
typedef struct {
  vec4 line_color;
  vec4 base_color;
  vec2 line_width;
  vec2 padding;
} grid_uniforms_t;

/* Orbit camera */
typedef struct {
  vec2 orbit;
  vec3 distance;
  vec3 target;
  mat4 view_mat;
  mat4 camera_mat;
  vec3 position;
  bool dirty;
} orbit_camera_t;

/* Example state */
static struct {
  // Camera
  orbit_camera_t camera;
  camera_uniforms_t camera_uniforms;
  mat4 projection_matrix;

  // Grid options
  struct {
    vec4 clear_color;
    vec4 line_color;
    vec4 base_color;
    float line_width_x;
    float line_width_y;
  } grid_options;
  grid_uniforms_t grid_uniforms;

  // Buffers
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  wgpu_buffer_t frame_uniform_buffer;
  wgpu_buffer_t uniform_buffer;

  // Pipeline
  WGPUBindGroupLayout frame_bind_group_layout;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup frame_bind_group;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  // Textures
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } msaa_color, depth;

  // Render pass
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  // Settings
  uint32_t sample_count;

  // Mouse state
  struct {
    vec2 last_pos;
    bool dragging;
  } mouse;

  // Timing
  uint64_t last_time;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.2, 1.0},
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
  .sample_count = 4,
  .grid_options = {
    .clear_color  = {0.0f, 0.0f, 0.2f, 1.0f},
    .line_color   = {1.0f, 1.0f, 1.0f, 1.0f},
    .base_color   = {0.0f, 0.0f, 0.0f, 1.0f},
    .line_width_x = 0.05f,
    .line_width_y = 0.05f,
  },
};

/* -------------------------------------------------------------------------- *
 * Orbit Camera functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Transform vec3 by 4x4 matrix.
 * @param v - the vector
 * @param m - The matrix.
 * @param dst - vec3 to store result.
 * @returns the transformed vector dst
 */
static vec3* glm_vec3_transform_mat4(vec3 v, mat4 m, vec3* dst)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];
  const float w = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3];
  (*dst)[0]     = (m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0]) / w;
  (*dst)[1]     = (m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1]) / w;
  (*dst)[2]     = (m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2]) / w;
  return dst;
}

static void orbit_camera_init(orbit_camera_t* cam)
{
  glm_vec2_zero(cam->orbit);
  glm_vec3_copy((vec3){0.0f, 0.0f, 1.0f}, cam->distance);
  glm_vec3_zero(cam->target);
  glm_mat4_identity(cam->view_mat);
  glm_mat4_identity(cam->camera_mat);
  glm_vec3_zero(cam->position);
  cam->dirty = true;
}

static void orbit_camera_update_matrices(orbit_camera_t* cam)
{
  if (cam->dirty) {
    glm_mat4_identity(cam->camera_mat);
    glm_translate(cam->camera_mat, cam->target);
    glm_rotate_y(cam->camera_mat, -cam->orbit[1], cam->camera_mat);
    glm_rotate_x(cam->camera_mat, -cam->orbit[0], cam->camera_mat);
    glm_translate(cam->camera_mat, cam->distance);
    glm_mat4_inv(cam->camera_mat, cam->view_mat);
    cam->dirty = false;
  }
}

static void orbit_camera_get_view_matrix(orbit_camera_t* cam, mat4 view_matrix,
                                         vec3 camera_position)
{
  orbit_camera_update_matrices(cam);
  glm_mat4_copy(cam->view_mat, view_matrix);

  glm_vec3_zero(cam->position);
  glm_vec3_transform_mat4(cam->position, cam->camera_mat, &cam->position);
  glm_vec3_copy(cam->position, camera_position);
}

static void orbit_camera_orbit(orbit_camera_t* cam, float x_delta,
                               float y_delta)
{
  cam->orbit[0] -= y_delta;
  cam->orbit[1] = fmaxf(-PI / 2.0f, fminf(PI / 2.0f, cam->orbit[1] - x_delta));
  cam->dirty    = true;
}

/* -------------------------------------------------------------------------- *
 * Math helper functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Generates a perspective projection matrix with the given bounds.
 * The near/far clip planes correspond to a normalized device coordinate Z range
 * of [0, 1], which matches WebGPU's clip volume.
 */
static mat4* glm_mat4_perspective_zo(mat4* out, float fovy, float aspect,
                                     float near, const float* far)
{
  const float f = 1.0f / tanf(fovy / 2.0f);
  (*out)[0][0]  = f / aspect;
  (*out)[0][1]  = 0.0f;
  (*out)[0][2]  = 0.0f;
  (*out)[0][3]  = 0.0f;
  (*out)[1][0]  = 0.0f;
  (*out)[1][1]  = f;
  (*out)[1][2]  = 0.0f;
  (*out)[1][3]  = 0.0f;
  (*out)[2][0]  = 0.0f;
  (*out)[2][1]  = 0.0f;
  (*out)[2][3]  = -1.0f;
  (*out)[3][0]  = 0.0f;
  (*out)[3][1]  = 0.0f;
  (*out)[3][3]  = 0.0f;
  if (far != NULL && *far != INFINITY) {
    const float nf = 1.0f / (near - *far);
    (*out)[2][2]   = *far * nf;
    (*out)[3][2]   = *far * near * nf;
  }
  else {
    (*out)[2][2] = -1.0f;
    (*out)[3][2] = -near;
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

static void update_projection_matrix(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;
  const float fov   = PI / 2.0f; // 90 degrees
  const float z_far = 128.0f;
  glm_mat4_perspective_zo(&state.projection_matrix, fov, aspect_ratio, 0.01f,
                          &z_far);
}

static void update_camera_uniforms(wgpu_context_t* wgpu_context)
{
  if (state.camera.dirty) {
    orbit_camera_get_view_matrix(&state.camera,
                                 state.camera_uniforms.view_matrix,
                                 state.camera_uniforms.camera_position);
  }
  glm_mat4_copy(state.projection_matrix,
                state.camera_uniforms.projection_matrix);
  state.camera_uniforms.time = (float)stm_sec(stm_now());

  wgpuQueueWriteBuffer(wgpu_context->queue, state.frame_uniform_buffer.buffer,
                       0, &state.camera_uniforms, sizeof(camera_uniforms_t));
}

static void update_grid_uniforms(wgpu_context_t* wgpu_context)
{
  // Update clear color
  state.color_attachment.clearValue = (WGPUColor){
    .r = state.grid_options.clear_color[0],
    .g = state.grid_options.clear_color[1],
    .b = state.grid_options.clear_color[2],
    .a = state.grid_options.clear_color[3],
  };

  // Update grid uniforms
  glm_vec4_copy(state.grid_options.line_color, state.grid_uniforms.line_color);
  glm_vec4_copy(state.grid_options.base_color, state.grid_uniforms.base_color);
  state.grid_uniforms.line_width[0] = state.grid_options.line_width_x;
  state.grid_uniforms.line_width[1] = state.grid_options.line_width_y;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &state.grid_uniforms, sizeof(grid_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Initialization functions
 * -------------------------------------------------------------------------- */

static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  static const vertex_t vertex_array[4] = {
    {.position = {-20.0f, -0.5f, -20.0f}, .uv = {0.0f, 0.0f}},
    {.position = {20.0f, -0.5f, -20.0f}, .uv = {200.0f, 0.0f}},
    {.position = {-20.0f, -0.5f, 20.0f}, .uv = {0.0f, 200.0f}},
    {.position = {20.0f, -0.5f, 20.0f}, .uv = {200.0f, 200.0f}},
  };

  state.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Pristine Grid - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_array),
                    .initial.data = vertex_array,
                  });

  static const uint16_t index_array[6] = {0, 1, 2, 1, 3, 2};

  state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Pristine Grid - Index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(index_array),
                    .initial.data = index_array,
                  });
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.frame_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Pristine Grid - Frame uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(camera_uniforms_t),
                  });

  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Pristine Grid - Grid uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(grid_uniforms_t),
                  });
}

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {0};

  // Frame bind group layout (camera uniforms)
  bgl_entries[0] = (WGPUBindGroupLayoutEntry){
    .binding    = 0,
    .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
    .buffer = (WGPUBufferBindingLayout){
      .type             = WGPUBufferBindingType_Uniform,
      .minBindingSize   = sizeof(camera_uniforms_t),
    },
  };

  state.frame_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Frame - Bind group layout"),
                            .entryCount = 1,
                            .entries    = &bgl_entries[0],
                          });

  // Grid bind group layout (grid uniforms)
  bgl_entries[1] = (WGPUBindGroupLayoutEntry){
    .binding    = 0,
    .visibility = WGPUShaderStage_Fragment,
    .buffer = (WGPUBufferBindingLayout){
      .type             = WGPUBufferBindingType_Uniform,
      .minBindingSize   = sizeof(grid_uniforms_t),
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Grid - Bind group layout"),
                            .entryCount = 1,
                            .entries    = &bgl_entries[1],
                          });
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    state.frame_bind_group_layout,
    state.bind_group_layout,
  };

  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Pipeline layout"),
                            .bindGroupLayoutCount = 2,
                            .bindGroupLayouts     = bind_group_layouts,
                          });
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  // Frame bind group
  state.frame_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Frame - Bind group"),
      .layout     = state.frame_bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.frame_uniform_buffer.buffer,
        .size    = state.frame_uniform_buffer.size,
      },
    });

  // Grid bind group
  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Grid - Bind group"),
      .layout     = state.bind_group_layout,
      .entryCount = 1,
      .entries    = &(WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffer.buffer,
        .size    = state.uniform_buffer.size,
      },
    });
}

static void init_render_targets(wgpu_context_t* wgpu_context)
{
  // MSAA color texture
  if (state.sample_count > 1) {
    WGPUTextureDescriptor tex_desc = {
      .label         = STRVIEW("MSAA color texture"),
      .size          = {wgpu_context->width, wgpu_context->height, 1},
      .mipLevelCount = 1,
      .sampleCount   = state.sample_count,
      .dimension     = WGPUTextureDimension_2D,
      .format        = wgpu_context->render_format,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    state.msaa_color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

    state.msaa_color.view = wgpuTextureCreateView(
      state.msaa_color.texture, &(WGPUTextureViewDescriptor){
                                  .label  = STRVIEW("MSAA color texture view"),
                                  .format = wgpu_context->render_format,
                                  .dimension      = WGPUTextureViewDimension_2D,
                                  .baseMipLevel   = 0,
                                  .mipLevelCount  = 1,
                                  .baseArrayLayer = 0,
                                  .arrayLayerCount = 1,
                                });
  }

  // Depth texture
  {
    WGPUTextureDescriptor tex_desc = {
      .label         = STRVIEW("Depth texture"),
      .size          = {wgpu_context->width, wgpu_context->height, 1},
      .mipLevelCount = 1,
      .sampleCount   = state.sample_count,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    state.depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);

    state.depth.view = wgpuTextureCreateView(
      state.depth.texture, &(WGPUTextureViewDescriptor){
                             .label     = STRVIEW("Depth texture view"),
                             .format    = WGPUTextureFormat_Depth24PlusStencil8,
                             .dimension = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });
  }

  // Set up render pass
  state.depth_stencil_attachment.view = state.depth.view;

  if (state.sample_count > 1) {
    state.color_attachment.view = state.msaa_color.view;
  }
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  // Vertex state
  WGPUVertexAttribute attributes[2] = {
    [0] = {.format         = WGPUVertexFormat_Float32x3,
           .offset         = offsetof(vertex_t, position),
           .shaderLocation = 0},
    [1] = {.format         = WGPUVertexFormat_Float32x2,
           .offset         = offsetof(vertex_t, uv),
           .shaderLocation = 1},
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = attributes,
  };

  // Shader modules
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, grid_shader_wgsl);

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = {
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state = {
    .format               = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled    = true,
    .depthCompare         = WGPUCompareFunction_Less,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  // Create pipeline
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label        = STRVIEW("Pristine grid - Render pipeline"),
      .layout       = state.pipeline_layout,
      .primitive    = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .vertex       = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment     = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &color_target_state,
       },
      .depthStencil = &depth_stencil_state,
      .multisample  = {.count = state.sample_count, .mask = ~0u},
    });

  wgpuShaderModuleRelease(shader_module);
}

/* -------------------------------------------------------------------------- *
 * Render function
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  if (igBegin("Settings", NULL, ImGuiWindowFlags_None)) {
    bool changed = false;
    changed |= igColorEdit4("clearColor", state.grid_options.clear_color,
                            ImGuiColorEditFlags_None);
    changed |= igColorEdit4("lineColor", state.grid_options.line_color,
                            ImGuiColorEditFlags_None);
    changed |= igColorEdit4("baseColor", state.grid_options.base_color,
                            ImGuiColorEditFlags_None);
    changed |= igSliderFloat("lineWidthX", &state.grid_options.line_width_x,
                             0.0f, 1.0f, "%.3f", 0);
    changed |= igSliderFloat("lineWidthY", &state.grid_options.line_width_y,
                             0.0f, 1.0f, "%.3f", 0);
    if (changed) {
      update_grid_uniforms(wgpu_context);
    }
    igEnd();
  }
}

static int frame(wgpu_context_t* wgpu_context)
{
  // Update camera uniforms
  update_camera_uniforms(wgpu_context);

  // Prepare frame
  state.color_attachment.view = state.sample_count > 1 ?
                                  state.msaa_color.view :
                                  wgpu_context->swapchain_view;
  if (state.sample_count > 1) {
    state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  }

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
    cmd_encoder, &state.render_pass_descriptor);

  wgpuRenderPassEncoderSetPipeline(render_pass, state.pipeline);
  wgpuRenderPassEncoderSetViewport(render_pass, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(render_pass, 0, 0, wgpu_context->width,
                                      wgpu_context->height);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0, state.frame_bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, state.bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, state.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(render_pass, state.index_buffer.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);

  wgpuRenderPassEncoderEnd(render_pass);
  wgpuRenderPassEncoderRelease(render_pass);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuCommandEncoderRelease(cmd_encoder);

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);
  wgpuCommandBufferRelease(cmd_buffer);

  // Render GUI overlay (creates its own render pass)
  imgui_overlay_new_frame(wgpu_context, stm_sec(stm_laptime(&state.last_time)));
  render_gui(wgpu_context);
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Input event callback
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  /* Pass input events to ImGui */
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Check if ImGui wants to capture input */
  ImGuiIO* io            = igGetIO();
  bool imgui_wants_input = io->WantCaptureMouse || io->WantCaptureKeyboard;

  /* Handle resize events always */
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    if (state.msaa_color.texture) {
      wgpuTextureViewRelease(state.msaa_color.view);
      wgpuTextureRelease(state.msaa_color.texture);
      state.msaa_color.texture = NULL;
      state.msaa_color.view    = NULL;
    }
    if (state.depth.texture) {
      wgpuTextureViewRelease(state.depth.view);
      wgpuTextureRelease(state.depth.texture);
      state.depth.texture = NULL;
      state.depth.view    = NULL;
    }

    init_render_targets(wgpu_context);
    update_projection_matrix(wgpu_context);
    return;
  }

  /* Skip camera interaction if ImGui wants input */
  if (imgui_wants_input) {
    return;
  }

  /* Handle camera orbit with mouse */
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.mouse.dragging    = true;
      state.mouse.last_pos[0] = input_event->mouse_x;
      state.mouse.last_pos[1] = input_event->mouse_y;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.mouse.dragging = false;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (state.mouse.dragging) {
      float dx = input_event->mouse_x - state.mouse.last_pos[0];
      float dy = input_event->mouse_y - state.mouse.last_pos[1];
      orbit_camera_orbit(&state.camera, dx * 0.01f, dy * 0.01f);
      state.mouse.last_pos[0] = input_event->mouse_x;
      state.mouse.last_pos[1] = input_event->mouse_y;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Init & shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  // Initialize camera
  orbit_camera_init(&state.camera);

  // Setup
  init_vertex_and_index_buffers(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layout(wgpu_context);
  init_bind_groups(wgpu_context);
  init_render_targets(wgpu_context);
  init_pipeline(wgpu_context);

  // Initialize projection
  update_projection_matrix(wgpu_context);

  // Initialize uniforms
  update_camera_uniforms(wgpu_context);
  update_grid_uniforms(wgpu_context);

  // Initialize GUI
  imgui_overlay_init(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  wgpu_destroy_buffer(&state.vertex_buffer);
  wgpu_destroy_buffer(&state.index_buffer);
  wgpu_destroy_buffer(&state.frame_uniform_buffer);
  wgpu_destroy_buffer(&state.uniform_buffer);

  wgpuBindGroupRelease(state.frame_bind_group);
  wgpuBindGroupRelease(state.bind_group);
  wgpuBindGroupLayoutRelease(state.frame_bind_group_layout);
  wgpuBindGroupLayoutRelease(state.bind_group_layout);
  wgpuPipelineLayoutRelease(state.pipeline_layout);
  wgpuRenderPipelineRelease(state.pipeline);

  wgpuTextureViewRelease(state.msaa_color.view);
  wgpuTextureRelease(state.msaa_color.texture);
  wgpuTextureViewRelease(state.depth.view);
  wgpuTextureRelease(state.depth.texture);
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  stm_setup();

  wgpu_start(&(wgpu_desc_t){
    .title           = "Pristine Grid",
    .init_cb         = init,
    .frame_cb        = frame,
    .shutdown_cb     = shutdown,
    .input_event_cb  = input_event_cb,
    .width           = 1280,
    .height          = 720,
    .sample_count    = 4,
    .no_depth_buffer = false,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* grid_shader_wgsl = CODE(
  fn PristineGrid(uv: vec2f, lineWidth: vec2f) -> f32 {
    let uvDDXY = vec4f(dpdx(uv), dpdy(uv));
    let uvDeriv = vec2f(length(uvDDXY.xz), length(uvDDXY.yw));
    let invertLine: vec2<bool> = lineWidth > vec2f(0.5);
    let targetWidth: vec2f = select(lineWidth, 1 - lineWidth, invertLine);
    let drawWidth: vec2f = clamp(targetWidth, uvDeriv, vec2f(0.5));
    let lineAA: vec2f = uvDeriv * 1.5;
    var gridUV: vec2f = abs(fract(uv) * 2.0 - 1.0);
    gridUV = select(1 - gridUV, gridUV, invertLine);
    var grid2: vec2f = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = mix(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2 = select(grid2, 1.0 - grid2, invertLine);
    return mix(grid2.x, 1.0, grid2.y);
  }

  struct VertexIn {
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
  }

  struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
  }

  struct Camera {
    projection: mat4x4f,
    view: mat4x4f,
    position: vec3f,
    time: f32,
  }
  @group(0) @binding(0) var<uniform> camera: Camera;

  struct GridArgs {
    lineColor: vec4f,
    baseColor: vec4f,
    lineWidth: vec2f,
  }
  @group(1) @binding(0) var<uniform> gridArgs: GridArgs;

  @vertex
  fn vertexMain(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = camera.projection * camera.view * in.pos;
    out.uv = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
    var grid = PristineGrid(in.uv, gridArgs.lineWidth);
    return mix(gridArgs.baseColor, gridArgs.lineColor, grid * gridArgs.lineColor.a);
  }
);
// clang-format on
