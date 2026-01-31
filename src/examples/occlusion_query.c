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
 * WebGPU Example - Occlusion Query
 *
 * This example demonstrates using Occlusion Queries.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/occlusionQuery
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* solid_color_lit_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Math functions
 * -------------------------------------------------------------------------- */

static float lerp(float a, float b, float t)
{
  return a + (b - a) * t;
}

static void lerp_v(vec3 a, vec3 b, float t, vec3* dst)
{
  (*dst)[0] = lerp(a[0], b[0], t);
  (*dst)[1] = lerp(a[1], b[1], t);
  (*dst)[2] = lerp(a[2], b[2], t);
}

static float ping_pong_sine(float t)
{
  return sin(t * PI2) * 0.5f + 0.5f;
}

/**
 * @brief Sets a matrix from a vector translation.
 * This is equivalent to (but much faster than):
 *
 *     mat4.identity(dest);
 *     mat4.translate(dest, dest, vec);
 *
 * @param {ReadonlyVec3} v Translation vector
 * @param {mat4} dst mat4 receiving operation result
 */
static void glm_mat4_translation(vec3 v, mat4* dst)
{
  glm_mat4_identity(*dst);
  (*dst)[3][0] = v[0];
  (*dst)[3][1] = v[1];
  (*dst)[3][2] = v[2];
}

/* -------------------------------------------------------------------------- *
 * Occlusion Query example
 * -------------------------------------------------------------------------- */

typedef enum cube_id_t {
  CUBE_ID_RED,
  CUBE_ID_YELLOW,
  CUBE_ID_GREEN,
  CUBE_ID_ORANGE,
  CUBE_ID_BLUE,
  CUBE_ID_PURPLE,
  CUBE_ID_COUNT,
} cube_id_t;

typedef struct cube_uniform_values_t {
  mat4 world_view_projection;
  mat4 world_inverse_transpose;
  vec4 color_value;
} cube_uniform_values_t;

/* State struct */
static struct {
  struct {
    vec3 position;
    vec4 color;
  } cube_positions[CUBE_ID_COUNT];
  struct {
    cube_id_t id;
    vec3 position;
    wgpu_buffer_t uniform_buffer;
    WGPUBindGroup uniform_buffer_bind_group;
    cube_uniform_values_t uniform_values;
    bool is_visible;
  } cubes[CUBE_ID_COUNT];
  struct {
    WGPUQuerySet set;
    WGPUBuffer resolve_buffer;
    WGPUBuffer result_buffer;
    size_t result_buffer_size;
  } occlusion_query;
  struct {
    wgpu_buffer_t vertices;
    wgpu_buffer_t indices;
  } buffers;
  WGPUTextureFormat depth_format;
  wgpu_texture_t depth_texture;
  WGPURenderPipeline render_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    float time;
    float then;
    vec3 lerp_a;
    vec3 lerp_b;
    mat4 projection;
    mat4 m;
    vec3 translation;
    mat4 view;
    mat4 view_projection;
  } render_state;
  struct {
    bool animate;
  } settings;
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .cube_positions = {
    // clang-format off
    [CUBE_ID_RED]    = { .position = {-1,  0,  0}, .color = { 1,   0,   0,   1} },
    [CUBE_ID_YELLOW] = { .position = { 1,  0,  0}, .color = { 1,   1,   0,   1} },
    [CUBE_ID_GREEN]  = { .position = { 0, -1,  0}, .color = { 0,   0.5, 0,   1} },
    [CUBE_ID_ORANGE] = { .position = { 0,  1,  0}, .color = { 1,   0.6, 0,   1} },
    [CUBE_ID_BLUE]   = { .position = { 0,  0, -1}, .color = { 0,   0,   1,   1} },
    [CUBE_ID_PURPLE] = { .position = { 0,  0,  1}, .color = { 0.5, 0,   0.5, 1} },
    // clang-format on
  },
  .depth_format = WGPUTextureFormat_Depth24Plus,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.5, 0.5, 0.5, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .render_state = {
    .time            = 0.0f,
    .then            = 0.0f,
    .lerp_a          = {0.0f, 0.0f, 5.0f},
    .lerp_b          = {0.0f, 0.0f, 40.0f},
    .projection      = GLM_MAT4_ZERO_INIT,
    .m               = GLM_MAT4_IDENTITY_INIT,
    .translation     = GLM_VEC3_ZERO_INIT,
    .view            = GLM_MAT4_ZERO_INIT,
    .view_projection = GLM_MAT4_ZERO_INIT,
  },
  .settings = {
    .animate = true,
  },
};

static void init_occlusion_query_set(wgpu_context_t* wgpu_context)
{
  state.occlusion_query.set = wgpuDeviceCreateQuerySet(
    wgpu_context->device, &(WGPUQuerySetDescriptor){
                            .label = STRVIEW("Occlusion - Query set"),
                            .type  = WGPUQueryType_Occlusion,
                            .count = CUBE_ID_COUNT,
                          });
}

/* Initialize buffers for storing the occlusion query result */
static void init_occlusion_query_set_buffers(wgpu_context_t* wgpu_context)
{
  state.occlusion_query.resolve_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Query set - Resolve buffer"),
      /* Query results are 64bit unsigned integers.*/
      .size  = CUBE_ID_COUNT * sizeof(uint64_t),
      .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
    });

  state.occlusion_query.result_buffer_size = CUBE_ID_COUNT * sizeof(uint64_t);
  state.occlusion_query.result_buffer      = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
           .label = STRVIEW("Query set - Result buffer"),
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
           .size  = state.occlusion_query.result_buffer_size,
    });
}

static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Cube vertices */
  {
    typedef struct {
      vec3 position;
      vec3 normal;
    } vertex_t;
    static const vertex_t vertex_data[24] = {
      // clang-format off
      // position                   normal
      { .position = { 1,  1, -1}, .normal = { 1,  0,  0} },
      { .position = { 1,  1,  1}, .normal = { 1,  0,  0} },
      { .position = { 1, -1,  1}, .normal = { 1,  0,  0} },
      { .position = { 1, -1, -1}, .normal = { 1,  0,  0} },
      { .position = {-1,  1,  1}, .normal = {-1,  0,  0} },
      { .position = {-1,  1, -1}, .normal = {-1,  0,  0} },
      { .position = {-1, -1, -1}, .normal = {-1,  0,  0} },
      { .position = {-1, -1,  1}, .normal = {-1,  0,  0} },
      { .position = {-1,  1,  1}, .normal = { 0,  1,  0} },
      { .position = { 1,  1,  1}, .normal = { 0,  1,  0} },
      { .position = { 1,  1, -1}, .normal = { 0,  1,  0} },
      { .position = {-1,  1, -1}, .normal = { 0,  1,  0} },
      { .position = {-1, -1, -1}, .normal = { 0, -1,  0} },
      { .position = { 1, -1, -1}, .normal = { 0, -1,  0} },
      { .position = { 1, -1,  1}, .normal = { 0, -1,  0} },
      { .position = {-1, -1,  1}, .normal = { 0, -1,  0} },
      { .position = { 1,  1,  1}, .normal = { 0,  0,  1} },
      { .position = {-1,  1,  1}, .normal = { 0,  0,  1} },
      { .position = {-1, -1,  1}, .normal = { 0,  0,  1} },
      { .position = { 1, -1,  1}, .normal = { 0,  0,  1} },
      { .position = {-1,  1, -1}, .normal = { 0,  0, -1} },
      { .position = { 1,  1, -1}, .normal = { 0,  0, -1} },
      { .position = { 1, -1, -1}, .normal = { 0,  0, -1} },
      { .position = {-1, -1, -1}, .normal = { 0,  0, -1} },
      // clang-format on
    };
    state.buffers.vertices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Cube - Vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(vertex_data),
                      .count = (uint32_t)ARRAY_SIZE(vertex_data),
                      .initial.data = vertex_data,
                    });
  }

  /* Cube indices */
  {
    static const uint16_t indices[36] = {
      // clang-format off
       0,  1,  2,  0,  2,  3, /* +x face */
       4,  5,  6,  4,  6,  7, /* -x face */
       8,  9, 10,  8, 10, 11, /* +y face */
      12, 13, 14, 12, 14, 15, /* -y face */
      16, 17, 18, 16, 18, 19, /* +z face */
      20, 21, 22, 20, 22, 23, /* -z face */
      // clang-format on
    };
    state.buffers.indices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Cube - Index buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                      .size  = sizeof(indices),
                      .count = (uint32_t)ARRAY_SIZE(indices),
                      .initial.data = indices,
                    });
  }
}

static void init_cubes(wgpu_context_t* wgpu_context)
{
  const uint32_t uniform_buffer_size = (2 * 16 + 3 + 1 + 4) * 4;
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    state.cubes[i].id  = i;
    vec3 cube_position = {state.cube_positions[i].position[0] * 10.0f,
                          state.cube_positions[i].position[1] * 10.0f,
                          state.cube_positions[i].position[2] * 10.0f};
    glm_vec3_copy(cube_position, state.cubes[i].position);
    state.cubes[i].uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Cube - Uniform buffer",
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = uniform_buffer_size,
                                         });
    glm_vec4_copy(state.cube_positions[i].color,
                  state.cubes[i].uniform_values.color_value);
    state.cubes[i].uniform_buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = STRVIEW("Uniform buffer - Bind group"),
        .layout     = wgpuRenderPipelineGetBindGroupLayout(state.render_pipeline, 0),
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  =state.cubes[i].uniform_buffer.buffer,
          .size    = state.cubes[i].uniform_buffer.size,
        },
      }
    );
    ASSERT(state.cubes[i].uniform_buffer_bind_group != NULL);
  }
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule solid_color_lit_shader_module = wgpu_create_shader_module(
    wgpu_context->device, solid_color_lit_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = state.depth_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    cube, 6 * 4 /* 3x2 floats, 4 bytes each */,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, 12))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Cube - Render pipeline"),
    .vertex = {
      .module      = solid_color_lit_shader_module,
      .entryPoint  = STRVIEW("vs"),
      .bufferCount = 1,
      .buffers     = &cube_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .module      = solid_color_lit_shader_module,
      .entryPoint  = STRVIEW("fs"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(solid_color_lit_shader_module);
}

static void update_view_projection_matrix(wgpu_context_t* wgpu_context)
{
  const float now         = stm_sec(stm_now());
  const float delta_time  = now - state.render_state.then;
  state.render_state.then = now;

  if (state.settings.animate) {
    state.render_state.time += delta_time;
  }

  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_perspective((30.0f * PI) / 180.0f, aspect_ratio, 0.5f, 100.0f,
                  state.render_state.projection);

  glm_mat4_identity(state.render_state.m);
  glm_rotate_x(state.render_state.m, state.render_state.time,
               state.render_state.m);
  glm_rotate_y(state.render_state.m, state.render_state.time * 0.7f,
               state.render_state.m);
  lerp_v(state.render_state.lerp_a, state.render_state.lerp_b,
         ping_pong_sine(state.render_state.time * 0.2f),
         &state.render_state.translation);
  glm_translate(state.render_state.m, state.render_state.translation);
  glm_mat4_inv(state.render_state.m, state.render_state.view);
  glm_mat4_mul(state.render_state.projection, state.render_state.view,
               state.render_state.view_projection);
}

static void update_cubes_unform_buffer(wgpu_context_t* wgpu_context)
{
  /* Update view-projection matrix */
  update_view_projection_matrix(wgpu_context);

  /* Update uniform buffer of each cube */
  mat4 world = GLM_MAT4_ZERO_INIT;
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    glm_mat4_translation(state.cubes[i].position, &world);
    glm_mat4_inv(world, world);
    glm_mat4_transpose_to(
      world, state.cubes[i].uniform_values.world_inverse_transpose);
    glm_mat4_mul(state.render_state.view_projection, world,
                 state.cubes[i].uniform_values.world_view_projection);

    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.cubes[i].uniform_buffer.buffer, 0,
      &state.cubes[i].uniform_values, sizeof(cube_uniform_values_t));
  }
}

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Create the texture  */
  wgpu_destroy_texture(&state.depth_texture);
  state.depth_texture.desc.extent = (WGPUExtent3D){
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };
  state.depth_texture.handle = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth - Texture"),
                            .size          = state.depth_texture.desc.extent,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = state.depth_format,
                            .usage         = WGPUTextureUsage_RenderAttachment,
                          });
  ASSERT(state.depth_texture.handle != NULL);

  /* Create the texture view */
  state.depth_texture.view = wgpuTextureCreateView(
    state.depth_texture.handle, &(WGPUTextureViewDescriptor){
                                  .label     = STRVIEW("Depth - Texture view"),
                                  .dimension = WGPUTextureViewDimension_2D,
                                  .format    = state.depth_format,
                                  .baseMipLevel    = 0,
                                  .mipLevelCount   = 1,
                                  .baseArrayLayer  = 0,
                                  .arrayLayerCount = 1,
                                  .aspect          = WGPUTextureAspect_All,
                                });
  ASSERT(state.depth_texture.view != NULL);
}

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  /* Set initial window size to prevent resizing */
  igSetNextWindowSize((ImVec2){220.0f, 110.0f}, ImGuiCond_Always);

  /* Build GUI - similar to TypeScript version */
  igBegin("Occlusion Query", NULL,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);

  /* Animate checkbox - matches TypeScript: gui.add(settings, 'animate') */
  igCheckbox("animate", &state.settings.animate);

  /* Display visible cubes with colored squares */
  igText("Visible:");

  /* Cube colors and labels matching the TypeScript version */
  const char* cube_labels[CUBE_ID_COUNT] = {
    "Red",    /* CUBE_ID_RED */
    "Yellow", /* CUBE_ID_YELLOW */
    "Green",  /* CUBE_ID_GREEN */
    "Orange", /* CUBE_ID_ORANGE */
    "Blue",   /* CUBE_ID_BLUE */
    "Purple", /* CUBE_ID_PURPLE */
  };

  /* Display colored squares horizontally for all visible cubes */
  bool first_visible = true;
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    if (state.cubes[i].is_visible) {
      /* Get cube color */
      vec4 color;
      glm_vec4_copy(state.cube_positions[i].color, color);

      /* Place on same line as previous square (except first) */
      if (!first_visible) {
        igSameLine(0.0f, 3.0f);
      }
      first_visible = false;

      /* Draw colored square button (non-interactive) */
      ImVec4 col = {color[0], color[1], color[2], color[3]};
      igPushIDInt((int)i);
      igColorButton(cube_labels[i], col,
                    ImGuiColorEditFlags_NoTooltip
                      | ImGuiColorEditFlags_NoPicker,
                    (ImVec2){20.0f, 20.0f});
      igPopID();
    }
  }

  igEnd();
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_pipeline(wgpu_context);
    init_cubes(wgpu_context);
    init_occlusion_query_set(wgpu_context);
    init_occlusion_query_set_buffers(wgpu_context);
    init_vertex_and_index_buffers(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void read_buffer_map_cb(WGPUMapAsyncStatus status,
                               WGPUStringView message, void* userdata1,
                               void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);

  if (status == WGPUMapAsyncStatus_Success) {
    uint64_t const* mapping = (uint64_t*)wgpuBufferGetConstMappedRange(
      state.occlusion_query.result_buffer, 0,
      state.occlusion_query.result_buffer_size);
    ASSERT(mapping)
    for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
      /* A non-zero value means the cube was visible */
      state.cubes[i].is_visible = mapping[i] > 0;
    }
    wgpuBufferUnmap(state.occlusion_query.result_buffer);
  }
}

static void get_occlusion_query_results(void)
{
  if (wgpuBufferGetMapState(state.occlusion_query.result_buffer)
      == WGPUBufferMapState_Unmapped) {
    wgpuBufferMapAsync(state.occlusion_query.result_buffer, WGPUMapMode_Read, 0,
                       state.occlusion_query.result_buffer_size,
                       (WGPUBufferMapCallbackInfo){
                         .mode     = WGPUCallbackMode_AllowProcessEvents,
                         .callback = read_buffer_map_cb,
                       });
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

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

  /* Render GUI controls */
  render_gui(wgpu_context);

  /* Update unform data */
  update_cubes_unform_buffer(wgpu_context);

  /* Update depth texture */
  if (!state.depth_texture.handle
      || state.depth_texture.desc.extent.width != (uint32_t)wgpu_context->width
      || state.depth_texture.desc.extent.height
           != (uint32_t)wgpu_context->height) {
    init_depth_texture(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Set color and depth-stencil attachments */
  state.color_attachment.view                    = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view            = state.depth_texture.view;
  state.render_pass_descriptor.occlusionQuerySet = state.occlusion_query.set;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.buffers.vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.buffers.indices.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    wgpuRenderPassEncoderSetBindGroup(
      rpass_enc, 0, state.cubes[i].uniform_buffer_bind_group, 0, 0);
    wgpuRenderPassEncoderBeginOcclusionQuery(rpass_enc, i);
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.buffers.indices.count, 1,
                                     0, 0, 0);
    wgpuRenderPassEncoderEndOcclusionQuery(rpass_enc);
  }
  wgpuRenderPassEncoderEnd(rpass_enc);

  /* Resolve query set */
  wgpuCommandEncoderResolveQuerySet(cmd_enc, state.occlusion_query.set, 0,
                                    CUBE_ID_COUNT,
                                    state.occlusion_query.resolve_buffer, 0);
  if (wgpuBufferGetMapState(state.occlusion_query.result_buffer)
      == WGPUBufferMapState_Unmapped) {
    wgpuCommandEncoderCopyBufferToBuffer(
      cmd_enc, state.occlusion_query.resolve_buffer, 0,
      state.occlusion_query.result_buffer, 0,
      state.occlusion_query.result_buffer_size);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  /* Map and read results buffer */
  get_occlusion_query_results();

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    wgpu_destroy_buffer(&state.cubes[i].uniform_buffer);
    WGPU_RELEASE_RESOURCE(BindGroup, state.cubes[i].uniform_buffer_bind_group)
  }

  WGPU_RELEASE_RESOURCE(QuerySet, state.occlusion_query.set)
  WGPU_RELEASE_RESOURCE(Buffer, state.occlusion_query.resolve_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.occlusion_query.result_buffer);

  wgpu_destroy_buffer(&state.buffers.vertices);
  wgpu_destroy_buffer(&state.buffers.indices);

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline);

  wgpu_destroy_texture(&state.depth_texture);
}

/**
 * @brief Input event callback for ImGui interaction
 */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  imgui_overlay_handle_input(wgpu_context, event);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Occlusion Query",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* solid_color_lit_shader_wgsl = CODE(
  struct Uniforms {
    worldViewProjectionMatrix: mat4x4f,
    worldMatrix: mat4x4f,
    color: vec4f,
  };

  struct Vertex {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
  };

  struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
  };

  @group(0) @binding(0) var<uniform> uni: Uniforms;

  @vertex fn vs(vin: Vertex) -> VSOut {
    var vOut: VSOut;
    vOut.position = uni.worldViewProjectionMatrix * vin.position;
    vOut.normal = (uni.worldMatrix * vec4f(vin.normal, 0)).xyz;
    return vOut;
  }

  @fragment fn fs(vin: VSOut) -> @location(0) vec4f {
    let lightDirection = normalize(vec3f(4, 10, 6));
    let light = dot(normalize(vin.normal), lightDirection) * 0.5 + 0.5;
    return vec4f(uni.color.rgb * light, uni.color.a);
  }
);
// clang-format on
