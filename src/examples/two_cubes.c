#include "common_shaders.h"
#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Two Cubes
 *
 * This example shows some of the alignment requirements involved when updating
 * and binding multiple slices of a uniform buffer. It renders two rotating
 * cubes which have transform matrices at different offsets in a uniform buffer.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/twoCubes
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_CUBES (2ull)

/* Cube struct */
typedef struct cube_t {
  WGPUBindGroup uniform_buffer_bind_group;
  struct {
    mat4 model;
    mat4 model_view_projection;
    mat4 tmp;
  } view_mtx;
} cube_t;

/* State struct */
static struct {
  cube_mesh_t cube_mesh;
  cube_t cubes[NUMBER_OF_CUBES];
  wgpu_buffer_t vertices;
  struct {
    WGPUBuffer buffer;
    uint64_t offset;
    uint64_t size;
    uint64_t size_with_offset;
  } uniform_buffer;
  struct {
    mat4 projection;
    mat4 view;
  } view_matrices;
  WGPURenderPipeline pipeline;
  WGPURenderBundle render_bundle;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  struct {
    uint64_t number_of_cubes;
    bool render_bundles;
  } settings;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1, 0.2, 0.3, 1.0},
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
  .render_pass_dscriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .number_of_cubes = NUMBER_OF_CUBES,
    .render_bundles  = true,
  }
};

/* Prepare the cube geometry */
static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

/* Create a vertex buffer from the cube data. */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

static void update_transformation_matrix(void)
{
  const float now     = stm_sec(stm_now());
  const float sin_now = sin(now), cos_now = cos(now);

  cube_t* cube = NULL;
  for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {
    cube = &state.cubes[i];
    glm_mat4_copy(cube->view_mtx.model, cube->view_mtx.tmp);
    if (i % 2 == 0) {
      glm_rotate(cube->view_mtx.tmp, 1.0f, (vec3){sin_now, cos_now, 0.0f});
    }
    else if (i % 2 == 1) {
      glm_rotate(cube->view_mtx.tmp, 1.0f, (vec3){cos_now, sin_now, 0.0f});
    }
    glm_mat4_mul(state.view_matrices.view, cube->view_mtx.tmp,
                 cube->view_mtx.model_view_projection);
    glm_mat4_mul(state.view_matrices.projection,
                 cube->view_mtx.model_view_projection,
                 cube->view_mtx.model_view_projection);
  }
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  update_transformation_matrix();

  for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer,
                         i * state.uniform_buffer.offset,
                         &state.cubes[i].view_mtx.model_view_projection,
                         sizeof(mat4));
  }
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);

  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -7.0f});

  const float start_x = -2.0f, increment_x = 4.0f;
  cube_t* cube = NULL;
  float x      = 0.0f;
  for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {
    cube = &state.cubes[i];
    x    = start_x + i * increment_x;

    /* Model matrices */
    glm_mat4_identity(cube->view_mtx.model);
    glm_translate(cube->view_mtx.model, (vec3){x, 0.0f, 0.0f});

    /* Model view matrices */
    glm_mat4_identity(cube->view_mtx.model_view_projection);

    /* Temporary matrices */
    glm_mat4_identity(cube->view_mtx.tmp);
  }
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  init_view_matrices(wgpu_context);

  /* Unform buffer */
  state.uniform_buffer.size = sizeof(mat4); /* 4x4 matrix */
  state.uniform_buffer.offset
    = 256; /* uniformBindGroup offset must be 256-byte aligned */
  state.uniform_buffer.size_with_offset
    = ((state.settings.number_of_cubes - 1) * state.uniform_buffer.offset)
      + state.uniform_buffer.size;

  state.uniform_buffer.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Cube - Uniform buffer"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = state.uniform_buffer.size_with_offset,
    });
  ASSERT(state.uniform_buffer.buffer != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {
    state.cubes[i].uniform_buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = STRVIEW("Uniform buffer - Bind group"),
        .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 0),
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = state.uniform_buffer.buffer,
          .offset  = i * state.uniform_buffer.offset,
          .size    = state.uniform_buffer.size,
        },
      }
    );
    ASSERT(state.cubes[i].uniform_buffer_bind_group != NULL);
  }
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, basic_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, vertex_position_color_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(two_cubes, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: Color */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.color_offset))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Two cubes - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &two_cubes_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
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

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    wgpu##Type##SetPipeline(rpass_enc, state.pipeline);                        \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,        \
                                WGPU_WHOLE_SIZE);                              \
    for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {            \
      wgpu##Type##SetBindGroup(                                                \
        rpass_enc, 0, state.cubes[i].uniform_buffer_bind_group, 0, 0);         \
      wgpu##Type##Draw(rpass_enc, state.cube_mesh.vertex_count, 1, 0, 0);      \
    }                                                                          \
  }

static void init_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->render_format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device,
      &(WGPURenderBundleEncoderDescriptor){
        .label              = STRVIEW("Two cubes - Render bundle encoder"),
        .colorFormatCount   = (uint32_t)ARRAY_SIZE(color_formats),
        .colorFormats       = color_formats,
        .depthStencilFormat = wgpu_context->depth_stencil_format,
        .sampleCount        = 1,
      });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  state.render_bundle
    = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);
  ASSERT(state.render_bundle != NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_cube_mesh();
    init_vertex_buffer(wgpu_context);
    init_pipeline(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_bind_groups(wgpu_context);
    init_render_bundle_encoder(wgpu_context);
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

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  if (state.settings.render_bundles) {
    wgpuRenderPassEncoderExecuteBundles(rpass_enc, 1, &state.render_bundle);
  }
  else {
    RECORD_RENDER_PASS(RenderPassEncoder, rpass_enc)
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer)
  for (uint64_t i = 0; i < state.settings.number_of_cubes; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.cubes[i].uniform_buffer_bind_group)
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Two Cubes",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}
