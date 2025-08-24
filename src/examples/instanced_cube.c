#include "common_shaders.h"
#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Instanced Cube
 *
 * This example shows the use of instancing.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/sample/instancedCube
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* instanced_vertex_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Instanced Cube example
 * -------------------------------------------------------------------------- */

#define MAX_NUM_INSTANCES (16u)

static const uint32_t x_count             = 4;
static const uint32_t y_count             = 4;
static const uint32_t num_instances       = x_count * y_count;
static const uint32_t matrix_float_count  = 16; // 4x4 matrix
static const uint32_t matrix_size         = 4 * matrix_float_count;
static const uint32_t uniform_buffer_size = num_instances * matrix_size;

static struct {
  cube_mesh_t cube_mesh;
  uint32_t cube_vertex_count;
  wgpu_buffer_t vertices;
  struct {
    wgpu_buffer_t buffer;
    WGPUBindGroup bind_group;
  } uniform_buffer;
  struct {
    mat4 projection;
    mat4 view;
    mat4 model[MAX_NUM_INSTANCES];
    float model_view_projection[16 * MAX_NUM_INSTANCES];
    mat4 tmp;
  } view_matrices;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  int8_t initialized;
} state = {
  .cube_vertex_count = 36,
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
                    .label = "Cube data - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

static void update_transformation_matrix(void)
{
  const float now = nano_time() * powf(10, -9);

  uint32_t m = 0, i = 0;
  for (uint32_t x = 0; x < x_count; ++x) {
    for (uint32_t y = 0; y < y_count; ++y) {
      memcpy(state.view_matrices.tmp, state.view_matrices.model[i],
             sizeof(mat4));
      glm_rotate(state.view_matrices.tmp, 1.0f,
                 (vec3){
                   sin(((float)x + 0.5f) * now), /* x */
                   cos(((float)y + 0.5f) * now), /* y */
                   0.0f                          /* z */
                 });

      glm_mat4_mul(state.view_matrices.view, state.view_matrices.tmp,
                   state.view_matrices.tmp);
      glm_mat4_mul(state.view_matrices.projection, state.view_matrices.tmp,
                   state.view_matrices.tmp);

      memcpy(&state.view_matrices.model_view_projection[m],
             state.view_matrices.tmp, sizeof(state.view_matrices.tmp));

      ++i;
      m += matrix_float_count;
    }
  }
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  update_transformation_matrix();

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer.buffer,
                       0, &state.view_matrices.model_view_projection,
                       state.uniform_buffer.buffer.size);
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  /* Calculate aspect ratio */
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);

  /* View matrix */
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -12.0f});

  /* Temporary matrix */
  glm_mat4_identity(state.view_matrices.tmp);
}

static void init_model_matrices(void)
{
  const float step = 4.0f;

  /* Initialize the matrix data for every instance. */
  uint32_t m = 0;
  for (uint32_t x = 0; x < x_count; x++) {
    for (uint32_t y = 0; y < y_count; y++) {
      glm_mat4_identity(state.view_matrices.model[m]);
      glm_translate(state.view_matrices.model[m],
                    (vec3){
                      step * (x - x_count / 2.0f + 0.5f), /* x */
                      step * (y - y_count / 2.0f + 0.5f), /* y */
                      0.0f                                /* z */
                    });
      ++m;
    }
  }
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Prepare camera view matrices */
  init_view_matrices(wgpu_context);

  // Uniform buffer: allocate a buffer large enough to hold transforms for every
  // instance.
  state.uniform_buffer.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Camera view matrices - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                  });
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  /* Uniform bind group */
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Cube - Bind group"),
    .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pipeline, 0),
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer.buffer.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer.buffer.size,
    },
  };
  state.uniform_buffer.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.uniform_buffer.bind_group != NULL);

  /* Model matrices */
  init_model_matrices();
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, instanced_vertex_shader_wgsl);
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
  WGPU_VERTEX_BUFFER_LAYOUT(instanced_cube, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: Color */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.color_offset))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Instanced cube - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &instanced_cube_vertex_buffer_layout,
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

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_cube_mesh();
    init_vertex_buffer(wgpu_context);
    init_pipeline(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_bind_group(wgpu_context);
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

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);

  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.uniform_buffer.bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(rpass_enc, state.cube_vertex_count, num_instances,
                            0, 0);
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
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_buffer.bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Instanced Cube",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* instanced_vertex_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix : array<mat4x4f, 16>,
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) fragUV : vec2f,
    @location(1) fragPosition: vec4f,
  }

  @vertex
  fn main(
    @builtin(instance_index) instanceIdx : u32,
    @location(0) position : vec4f,
    @location(1) uv : vec2f
  ) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix[instanceIdx] * position;
    output.fragUV = uv;
    output.fragPosition = 0.5 * (position + vec4(1.0));
    return output;
  }
);
// clang-format on
