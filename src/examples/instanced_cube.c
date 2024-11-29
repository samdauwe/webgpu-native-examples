#include "common_shaders.h"
#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

#define MAX_NUM_INSTANCES 16u

static const uint32_t x_count             = 4;
static const uint32_t y_count             = 4;
static const uint32_t num_instances       = x_count * y_count;
static const uint32_t matrix_float_count  = 16; // 4x4 matrix
static const uint32_t matrix_size         = 4 * matrix_float_count;
static const uint32_t uniform_buffer_size = num_instances * matrix_size;

// Cube mesh
static cube_mesh_t cube_mesh      = {0};
static uint32_t cube_vertex_count = 36;

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// Uniform buffer object
static struct {
  wgpu_buffer_t buffer;
  WGPUBindGroup bind_group;
} uniform_buffer = {0};

static struct {
  mat4 projection;
  mat4 view;
  mat4 model[MAX_NUM_INSTANCES];
  float model_view_projection[16 * MAX_NUM_INSTANCES];
  mat4 tmp;
} view_matrices = {0};

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Instanced Cube";
static bool prepared             = false;

// Prepare the cube geometry
static void prepare_cube_mesh(void)
{
  cube_mesh_init(&cube_mesh);
}

// Create a vertex buffer from the cube data.
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube data - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(cube_mesh.vertex_array),
                    .initial.data = cube_mesh.vertex_array,
                  });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* View is acquired in render loop. */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Depth-stencil attachment
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now = context->frame.timestamp_millis / 1000.0f;

  uint32_t m = 0, i = 0;
  for (uint32_t x = 0; x < x_count; ++x) {
    for (uint32_t y = 0; y < y_count; ++y) {
      memcpy(view_matrices.tmp, view_matrices.model[i], sizeof(mat4));
      glm_rotate(view_matrices.tmp, 1.0f,
                 (vec3){
                   sin(((float)x + 0.5f) * now), /* x */
                   cos(((float)y + 0.5f) * now), /* y */
                   0.0f                          /* z */
                 });

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

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer.buffer.buffer,
                          0, &view_matrices.model_view_projection,
                          uniform_buffer.buffer.size);
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  /* Calculate aspect ratio */
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);

  /* View matrix */
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -12.0f});

  /* Temporary matrix */
  glm_mat4_identity(view_matrices.tmp);
}

static void prepare_model_matrices(void)
{
  const float step = 4.0f;

  /* Initialize the matrix data for every instance. */
  uint32_t m = 0;
  for (uint32_t x = 0; x < x_count; x++) {
    for (uint32_t y = 0; y < y_count; y++) {
      glm_mat4_identity(view_matrices.model[m]);
      glm_translate(view_matrices.model[m],
                    (vec3){
                      step * (x - x_count / 2.0f + 0.5f), /* x */
                      step * (y - y_count / 2.0f + 0.5f), /* y */
                      0.0f                                /* z */
                    });
      ++m;
    }
  }
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Prepare camera view matrices */
  prepare_view_matrices(wgpu_context);

  // Uniform buffer: allocate a buffer large enough to hold transforms for every
  // instance.
  uniform_buffer.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Camera view matrices - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = uniform_buffer_size,
                  });
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Uniform bind group */
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Cube - Bind group",
    .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer.buffer.buffer,
      .offset  = 0,
      .size    = uniform_buffer.buffer.size,
    },
  };
  uniform_buffer.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(uniform_buffer.bind_group != NULL);

  /* Model matrices */
  prepare_model_matrices();
}

static void prepare_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    // Backface culling since the cube is solid piece of geometry.
    // Faces pointing away from the camera will be occluded by faces pointing
    // toward the camera.
    .cullMode = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    instanced_cube, cube_mesh.vertex_size,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    // Attribute location 1: Color
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, cube_mesh.color_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader WGSL */
              .label            = "Instanced - Vertex shader WGSL",
              .wgsl_code.source = instanced_vertex_shader_wgsl,
            },
            .buffer_count = 1,
            .buffers      = &instanced_cube_vertex_buffer_layout,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader WGSL */
              .label            = "Vertex position color - Fragment shader WGSL",
              .wgsl_code.source = vertex_position_color_fragment_shader_wgsl,
            },
            .target_count = 1,
            .targets      = &color_target_state,
          });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Instanced cube - Render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
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
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_buffer.bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, cube_vertex_count,
                            num_instances, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
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

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
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
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_instanced_cube(int argc, char* argv[])
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
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* instanced_vertex_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix : array<mat4x4<f32>, 16>,
  }

  @binding(0) @group(0) var<uniform> uniforms : Uniforms;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
    @location(1) fragPosition: vec4<f32>,
  }

  @vertex
  fn main(
    @builtin(instance_index) instanceIdx : u32,
    @location(0) position : vec4<f32>,
    @location(1) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.Position = uniforms.modelViewProjectionMatrix[instanceIdx] * position;
    output.fragUV = uv;
    output.fragPosition = 0.5 * (position + vec4(1.0));
    return output;
  }
);
// clang-format on
