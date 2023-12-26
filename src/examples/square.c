#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Square
 *
 * This example shows how to render a static colored square in WebGPU with only
 * using vertex buffers.
 *
 * Ref:
 * https://github.com/cx20/webgpu-test/tree/master/examples/webgpu_wgsl/square
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Square example
 * -------------------------------------------------------------------------- */

// Vertex buffers
static struct {
  wgpu_buffer_t positions; /* Positions */
  wgpu_buffer_t colors;    /* Colors */
} square = {0};

// Pipeline
static WGPURenderPipeline pipeline;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Square";
static bool prepared             = false;

static void prepare_vertex_buffers(wgpu_context_t* wgpu_context)
{
  // Square data
  //             1.0 y
  //              ^  -1.0
  //              | / z
  //              |/       x
  // -1.0 -----------------> +1.0
  //            / |
  //      +1.0 /  |
  //           -1.0
  //
  //        [0]------[1]
  //         |        |
  //         |        |
  //         |        |
  //        [2]------[3]
  //
  // clang-format off
  static const float positions[12] = {
    -0.5f,  0.5f, 0.0f, // v0
     0.5f,  0.5f, 0.0f, // v1
    -0.5f, -0.5f, 0.0f, // v2
     0.5f, -0.5f, 0.0f, // v3
  };
  // clang-format on
  square.positions = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(positions),
                    .count = 4,
                    .initial.data = positions,
                  });

  static const float colors[16] = {
    1.0f, 0.0f, 0.0f, 1.0f, /* v0 */
    0.0f, 1.0f, 0.0f, 1.0f, /* v1 */
    0.0f, 0.0f, 1.0f, 1.0f, /* v2 */
    1.0f, 1.0f, 0.0f, 1.0f  /* v3 */
  };
  square.colors = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Colored square vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(colors),
                    .count = 4,
                    .initial.data = colors,
                  });
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology         = WGPUPrimitiveTopology_TriangleStrip,
    .stripIndexFormat = WGPUIndexFormat_Uint32,
    .frontFace        = WGPUFrontFace_CCW,
    .cullMode         = WGPUCullMode_Front,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex buffer layout
  WGPUVertexBufferLayout vertex_buffer_layouts[2] = {0};
  {
    WGPUVertexAttribute attribute = {
      // Shader location 0 : vertex position
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      // Shader location 1 : vertex color
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x4,
    };
    vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 4 * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Square vertex shader wgsl",
                      .wgsl_code.source = vertex_shader_wgsl,
                    },
                    .buffer_count = (uint32_t) ARRAY_SIZE(vertex_buffer_layouts),
                    .buffers      = vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Square fragment shader wgsl",
                      .wgsl_code.source = fragment_shader_wgsl,
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
  pipeline = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
                                            &(WGPURenderPipelineDescriptor){
                                              .label = "Square render pipeline",
                                              .primitive   = primitive_state,
                                              .vertex      = vertex_state,
                                              .fragment    = &fragment_state,
                                              .multisample = multisample_state,
                                            });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffers(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Bind vertex buffers (contain position & colors)
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 0, square.positions.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 1, square.colors.buffer, 0, WGPU_WHOLE_SIZE);

  // Draw quad
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, square.positions.count, 1,
                            0, 0);

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Buffer, square.positions.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, square.colors.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_square(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title  = example_title,
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
static const char* vertex_shader_wgsl = CODE(
  struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color : vec4<f32>
  };

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>
  };

  @vertex
  fn main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.fragColor = input.color;
    output.Position = vec4<f32>(input.position, 1.0);
    return output;
  };
);

static const char* fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(0) fragColor : vec4<f32>
  };

  struct FragmentOutput {
    @location(0) outColor : vec4<f32>
  };

  @fragment
  fn main(input : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.outColor = input.fragColor;
    return output;
  };
);
// clang-format on
