#include "example_base.h"

#include "../webgpu/imgui_overlay.h"

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

static struct {
  vec3 position;
  vec4 color;
} cube_positions[CUBE_ID_COUNT] = {
  // clang-format off
  [CUBE_ID_RED]    = { .position = {-1,  0,  0}, .color = { 1,   0,   0,   1} },
  [CUBE_ID_YELLOW] = { .position = { 1,  0,  0}, .color = { 1,   1,   0,   1} },
  [CUBE_ID_GREEN]  = { .position = { 0, -1,  0}, .color = { 0,   0.5, 0,   1} },
  [CUBE_ID_ORANGE] = { .position = { 0,  1,  0}, .color = { 1,   0.6, 0,   1} },
  [CUBE_ID_BLUE]   = { .position = { 0,  0, -1}, .color = { 0,   0,   1,   1} },
  [CUBE_ID_PURPLE] = { .position = { 0,  0,  1}, .color = { 0.5, 0,   0.5, 1} },
  // clang-format on
};

typedef struct cube_uniform_values_t {
  mat4 world_view_projection;
  mat4 world_inverse_transpose;
  vec4 color_value;
} cube_uniform_values_t;

static struct {
  cube_id_t id;
  vec3 position;
  wgpu_buffer_t uniform_buffer;
  WGPUBindGroup uniform_buffer_bind_group;
  cube_uniform_values_t uniform_values;
} cubes[CUBE_ID_COUNT] = {0};

static struct {
  WGPUQuerySet set;
  WGPUBuffer resolve_buffer;
  WGPUBuffer result_buffer;
  size_t result_buffer_size;
} occlusion_query = {0};

static struct {
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
} buffers = {0};

static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

static WGPURenderPipeline render_pipeline = NULL;

static struct {
  float time;
  float then;
  vec3 lerp_a;
  vec3 lerp_b;
  mat4 projection;
  mat4 m;
  vec3 translation;
  mat4 view;
  mat4 view_projection;
} render_state = {
  .time            = 0.0f,
  .then            = 0.0f,
  .lerp_a          = {0.0f, 0.0f, 5.0f},
  .lerp_b          = {0.0f, 0.0f, 40.0f},
  .projection      = GLM_MAT4_ZERO_INIT,
  .m               = GLM_MAT4_IDENTITY_INIT,
  .translation     = GLM_VEC3_ZERO_INIT,
  .view            = GLM_MAT4_ZERO_INIT,
  .view_projection = GLM_MAT4_ZERO_INIT,
};

static struct {
  bool animate;
} settings = {
  .animate = true,
};

static WGPUTextureFormat depth_format = WGPUTextureFormat_Depth24Plus;
static texture_t depth_texture        = {0};

/* Other variables */
static const char* example_title = "Occlusion Query";
static bool prepared             = false;

static void create_occlusion_query_set(wgpu_context_t* wgpu_context)
{
  occlusion_query.set = wgpuDeviceCreateQuerySet(
    wgpu_context->device, &(WGPUQuerySetDescriptor){
                            .label = "Occlusion query set",
                            .type  = WGPUQueryType_Occlusion,
                            .count = CUBE_ID_COUNT,
                          });
}

// Create buffers for storing the occlusion query result
static void create_occlusion_query_set_buffers(wgpu_context_t* wgpu_context)
{
  occlusion_query.resolve_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = "Query set - Resolve buffer",
      /* Query results are 64bit unsigned integers.*/
      .size  = CUBE_ID_COUNT * sizeof(size_t),
      .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
    });

  occlusion_query.result_buffer_size = CUBE_ID_COUNT * sizeof(uint64_t);
  occlusion_query.result_buffer      = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
           .label = "Query set - Result buffer",
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
           .size  = occlusion_query.result_buffer_size,
    });
}

// Prepare vertex and index buffers for an indexed triangle
static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
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
    buffers.vertices = wgpu_create_buffer(
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
    buffers.indices = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Cube - Index buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                      .size  = sizeof(indices),
                      .count = (uint32_t)ARRAY_SIZE(indices),
                      .initial.data = indices,
                    });
  }
}

static void prepare_cubes(wgpu_context_t* wgpu_context)
{
  const uint32_t uniform_buffer_size = (2 * 16 + 3 + 1 + 4) * 4;
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    cubes[i].id        = i;
    vec3 cube_position = {cube_positions[i].position[0] * 10.0f,
                          cube_positions[i].position[1] * 10.0f,
                          cube_positions[i].position[2] * 10.0f};
    glm_vec3_copy(cube_position, cubes[i].position);
    cubes[i].uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Cube - Uniform buffer",
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = uniform_buffer_size,
                                         });
    glm_vec4_copy(cube_positions[i].color, cubes[i].uniform_values.color_value);
    cubes[i].uniform_buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = "Uniform buffer - Bind group",
        .layout     = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = cubes[i].uniform_buffer.buffer,
          .size    = cubes[i].uniform_buffer.size,
        },
      }
    );
    ASSERT(cubes[i].uniform_buffer_bind_group != NULL);
  }
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = depth_format,
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

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      /* Vertex shader WGSL */
                      .label            = "Cube - Vertex shader WGSL",
                      .wgsl_code.source = solid_color_lit_shader_wgsl,
                      .entry            = "vs",
                    },
                    .buffer_count = 1,
                    .buffers = &cube_vertex_buffer_layout,
                  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      /* Fragment shader WGSL */
                      .label            = "Cube - Fragment shader WGSL",
                      .wgsl_code.source = solid_color_lit_shader_wgsl,
                      .entry            = "fs",
                    },
                    .target_count = 1,
                    .targets = &color_target_state,
                  });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Cube - Render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(render_pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_render_pass(void)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
    .view       = NULL, /* Assigned later */
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor) {
      .r = 0.5f,
      .g = 0.5f,
      .b = 0.5f,
      .a = 1.0f,
    },
  };

  /* Depth-stencil attachment */
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view            = NULL, /* Assigned later */
    .depthClearValue = 1.0f,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
  };

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
    .occlusionQuerySet      = occlusion_query.set,
  };
}

static void update_view_projection_matrix(wgpu_example_context_t* context)
{
  const float now
    = context->frame.timestamp_millis / 1000.0f; /* Convert to seconds */
  const float delta_time = now - render_state.then;
  render_state.then      = now;

  if (settings.animate) {
    render_state.time += delta_time;
  }

  wgpu_context_t* wgpu_context = context->wgpu_context;
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_perspective((30.0f * PI) / 180.0f, aspect_ratio, 0.5f, 100.0f,
                  render_state.projection);

  glm_mat4_identity(render_state.m);
  glm_rotate_x(render_state.m, render_state.time, render_state.m);
  glm_rotate_y(render_state.m, render_state.time * 0.7f, render_state.m);
  lerp_v(render_state.lerp_a, render_state.lerp_b,
         ping_pong_sine(render_state.time * 0.2f), &render_state.translation);
  glm_translate(render_state.m, render_state.translation);
  glm_mat4_inv(render_state.m, render_state.view);
  glm_mat4_mul(render_state.projection, render_state.view,
               render_state.view_projection);
}

static void update_cubes_inform_buffer(wgpu_example_context_t* context)
{
  if (!settings.animate) {
    return;
  }

  /* Update view-projection matrix */
  update_view_projection_matrix(context);

  /* Update uniform buffer of each cube */
  mat4 world = GLM_MAT4_ZERO_INIT;
  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    glm_mat4_translation(cubes[i].position, &world);
    glm_mat4_inv(world, world);
    glm_mat4_transpose_to(world,
                          cubes[i].uniform_values.world_inverse_transpose);
    glm_mat4_mul(render_state.view_projection, world,
                 cubes[i].uniform_values.world_view_projection);

    wgpu_queue_write_buffer(
      context->wgpu_context, cubes[i].uniform_buffer.buffer, 0,
      &cubes[i].uniform_values, sizeof(cube_uniform_values_t));
  }
}

static void create_depth_texture(wgpu_context_t* wgpu_context)
{
  if (depth_texture.texture
      && depth_texture.size.width == (uint32_t)wgpu_context->surface.width
      && depth_texture.size.height == (uint32_t)wgpu_context->surface.height) {
    return;
  }

  /* Create the texture  */
  wgpu_destroy_texture(&depth_texture);
  depth_texture.size = (WGPUExtent3D){
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  depth_texture.texture = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = "Depth texture",
                            .size          = depth_texture.size,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = depth_format,
                            .usage         = WGPUTextureUsage_RenderAttachment,
                          });
  ASSERT(depth_texture.texture != NULL);

  /* Create the texture view */
  depth_texture.view = wgpuTextureCreateView(
    depth_texture.texture, &(WGPUTextureViewDescriptor){
                             .label           = "Depth texture view",
                             .dimension       = WGPUTextureViewDimension_2D,
                             .format          = depth_format,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                             .aspect          = WGPUTextureAspect_All,
                           });
  ASSERT(depth_texture.view != NULL);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_pipeline(context->wgpu_context);
    prepare_cubes(context->wgpu_context);
    create_occlusion_query_set(context->wgpu_context);
    create_occlusion_query_set_buffers(context->wgpu_context);
    prepare_vertex_and_index_buffers(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Animate",
                           &settings.animate);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set color and depth-stencil attachments */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;
  create_depth_texture(wgpu_context);
  render_pass.depth_stencil_attachment.view = depth_texture.view;

  /* Create command encoder and render pass encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Draw cubes */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 0, buffers.vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, buffers.indices.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);

  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      cubes[i].uniform_buffer_bind_group, 0, 0);
    wgpuRenderPassEncoderBeginOcclusionQuery(wgpu_context->rpass_enc, i);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     buffers.indices.count, 1, 0, 0, 0);
    wgpuRenderPassEncoderEndOcclusionQuery(wgpu_context->rpass_enc);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Resolve query set */
  wgpuCommandEncoderResolveQuerySet(wgpu_context->cmd_enc, occlusion_query.set,
                                    0, CUBE_ID_COUNT,
                                    occlusion_query.resolve_buffer, 0);
  if (wgpuBufferGetMapState(occlusion_query.result_buffer)
      == WGPUBufferMapState_Unmapped) {
    wgpuCommandEncoderCopyBufferToBuffer(
      wgpu_context->cmd_enc, occlusion_query.resolve_buffer, 0,
      occlusion_query.result_buffer, 0, occlusion_query.result_buffer_size);
  }

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static void read_buffer_map_cb(WGPUBufferMapAsyncStatus status, void* user_data)
{
  UNUSED_VAR(user_data);

  if (status == WGPUBufferMapAsyncStatus_Success) {
    uint64_t const* mapping = (uint64_t*)wgpuBufferGetConstMappedRange(
      occlusion_query.result_buffer, 0, occlusion_query.result_buffer_size);
    ASSERT(mapping)
    for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
      printf("%lu ", mapping[i]);
    }
    printf("\n");
    wgpuBufferUnmap(occlusion_query.result_buffer);
  }
}

static void get_occlusion_query_results(void)
{
  if (wgpuBufferGetMapState(occlusion_query.result_buffer)
      == WGPUBufferMapState_Unmapped) {
    wgpuBufferMapAsync(occlusion_query.result_buffer, WGPUMapMode_Read, 0,
                       occlusion_query.result_buffer_size, read_buffer_map_cb,
                       NULL);
  }
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit command buffers to queue */
  submit_command_buffers(context);

  /* Map and read results buffer */
  get_occlusion_query_results();

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  update_cubes_inform_buffer(context);
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  for (uint32_t i = 0; i < CUBE_ID_COUNT; ++i) {
    wgpu_destroy_buffer(&cubes[i].uniform_buffer);
    WGPU_RELEASE_RESOURCE(BindGroup, cubes[i].uniform_buffer_bind_group)
  }

  WGPU_RELEASE_RESOURCE(QuerySet, occlusion_query.set)
  WGPU_RELEASE_RESOURCE(Buffer, occlusion_query.resolve_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, occlusion_query.result_buffer);

  wgpu_destroy_buffer(&buffers.vertices);
  wgpu_destroy_buffer(&buffers.indices);

  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline);

  wgpu_destroy_texture(&depth_texture);
}

void example_occlusion_query(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
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
