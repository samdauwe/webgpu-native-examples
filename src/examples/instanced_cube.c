#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "meshes.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Instanced Cube
 *
 * This example shows the use of instancing.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/pages/samples/instancedCube.ts
 * -------------------------------------------------------------------------- */

#define MAX_NUM_INSTANCES 16

static const uint32_t x_count             = 4;
static const uint32_t y_count             = 4;
static const uint32_t num_instances       = x_count * y_count;
static const uint32_t matrix_float_count  = 16; // 4x4 matrix
static const uint32_t matrix_size         = 4 * matrix_float_count;
static const uint32_t uniform_buffer_size = num_instances * matrix_size;

// Cube mesh
static cube_mesh_t cube_mesh = {0};

// Vertex buffer
static struct vertices_t {
  WGPUBuffer buffer;
  uint32_t size;
} vertices = {0};

// Uniform buffer object
static struct uniform_buffer_t {
  WGPUBuffer buffer;
  uint64_t size;
  WGPUBindGroup bind_group;
} uniform_buffer = {0};

static struct view_matrices_t {
  mat4 projection;
  mat4 view;
  mat4 model[MAX_NUM_INSTANCES];
  float model_view_projection[16 * MAX_NUM_INSTANCES];
  mat4 tmp;
} view_matrices = {0};

// Pipeline
static WGPURenderPipeline pipeline;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// Other variables
static const char* example_title = "Instanced Cube";
static bool prepared             = false;

// Prepare the cube geometry
static void prepare_cube_mesh()
{
  cube_mesh_init(&cube_mesh);
}

// Prepare vertex buffer
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices.size                    = sizeof(cube_mesh.vertex_array);
  WGPUBufferDescriptor buffer_desc = {
    .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
    .size             = vertices.size,
    .mappedAtCreation = true,
  };
  vertices.buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
  ASSERT(vertices.buffer)
  void* mapping = wgpuBufferGetMappedRange(vertices.buffer, 0, vertices.size);
  ASSERT(mapping)
  memcpy(mapping, cube_mesh.vertex_array, vertices.size);
  wgpuBufferUnmap(vertices.buffer);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .attachment = NULL, // attachment is acquired in render loop.
      .loadOp = WGPULoadOp_Clear,
      .storeOp = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now = context->frame.timestamp_millis / 1000.0f;

  uint32_t m = 0, i = 0;
  for (uint32_t x = 0; x < x_count; x++) {
    for (uint32_t y = 0; y < y_count; y++) {
      memcpy(view_matrices.tmp, view_matrices.model[i], sizeof(mat4));
      glm_rotate(view_matrices.tmp, 1.0f,
                 (vec3){sin(((float)x + 0.5f) * now),
                        cos(((float)y + 0.5f) * now), 0.0f});

      glm_mat4_mul(view_matrices.view, view_matrices.tmp, view_matrices.tmp);
      glm_mat4_mul(view_matrices.projection, view_matrices.tmp,
                   view_matrices.tmp);

      memcpy(&view_matrices.model_view_projection[m], view_matrices.tmp,
             sizeof(view_matrices.tmp));

      ++i;
      m += matrix_float_count;
    }
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  update_transformation_matrix(context);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer.buffer, 0,
                          &view_matrices.model_view_projection,
                          uniform_buffer_size);
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  // Projection matrix
  glm_mat4_identity(view_matrices.projection);
  glm_perspective((2 * PI) / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);

  // View matrix
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -12.0f});

  // Temporary matrix
  glm_mat4_identity(view_matrices.tmp);
}

static void prepare_model_matrices()
{
  const float step = 4.0f;

  uint32_t m = 0;
  for (uint32_t x = 0; x < x_count; x++) {
    for (uint32_t y = 0; y < y_count; y++) {
      glm_mat4_identity(view_matrices.model[m]);
      glm_translate(view_matrices.model[m],
                    (vec3){step * (x - x_count / 2.0f + 0.5f),
                           step * (y - y_count / 2.0f + 0.5f), 0.0f});
      ++m;
    }
  }
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  // Camera
  prepare_view_matrices(wgpu_context);

  // Unform buffer
  uniform_buffer.size   = uniform_buffer_size;
  uniform_buffer.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = uniform_buffer.size,
    });
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Uniform bind group
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer.buffer,
      .offset  = 0,
      .size    = uniform_buffer.size,
    },
  };
  uniform_buffer.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(uniform_buffer.bind_group != NULL)

  // Model matrices
  prepare_model_matrices();
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  // Construct the different states making up the pipeline

  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state_desc
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face = WGPUFrontFace_CCW,
        .cull_mode  = WGPUCullMode_Back,
      });

  // Color blend state
  WGPUColorStateDescriptor color_state_desc
    = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
      .format       = wgpu_context->swap_chain.format,
      .enable_blend = true,
    });

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc
    = wgpu_create_depth_stencil_state_descriptor(
      &(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = true,
      });

  // Vertex input binding (=> Input assembly)
  WGPU_VERTSTATE(
    instanced_cube, cube_mesh.vertex_size,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    // Attribute location 1: Color
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, cube_mesh.color_offset))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/instanced_cube/shader.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/instanced_cube/shader.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state_desc,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 1,
      .colorStates            = &color_state_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_instanced_cube,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_cube_mesh();
    prepare_vertex_buffer(context->wgpu_context);
    prepare_pipeline(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  rp_color_att_descriptors[0].attachment
    = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, 0);

  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_buffer.bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 36, num_instances, 0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_buffer.bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_instanced_cube(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title  = example_title,
     .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy
  });
  // clang-format on
}
