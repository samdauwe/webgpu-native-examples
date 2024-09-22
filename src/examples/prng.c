#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pseudorandom Number Generation
 *
 * A WebGPU example demonstrating pseudorandom number generation on the GPU. A
 * 32-bit PCG hash is used which is fast enough to be useful for real-time,
 * while also being high-quality enough for almost any graphics use-case.
 *
 * A pseudorandom number generator (PRNG), also known as a deterministic random
 * bit generator (DRBG), is an algorithm for generating a sequence of numbers
 * whose properties approximate the properties of sequences of random numbers.
 *
 * Ref:
 * https://en.wikipedia.org/wiki/Pseudorandom_number_generator
 * https://github.com/wwwtyro/webgpu-prng-example
 * https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* prng_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Pseudorandom Number Generation example
 * -------------------------------------------------------------------------- */

// Vertex layout used in this example
typedef struct {
  vec2 position;
} vertex_t;

// Vertex buffer and attributes
static wgpu_buffer_t vertices = {0};

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_fs = {0};

// Uniform block fragment shader
static struct {
  uint32_t offset;
} ubo_fs = {0};

static WGPUPipelineLayout pipeline_layout = NULL;
static WGPURenderPipeline pipeline        = NULL;

static WGPUBindGroupLayout uniform_bind_group_layout = NULL;
static WGPUBindGroup uniform_bind_group              = NULL;

static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Other variables
static const char* example_title = "Pseudorandom Number Generation";
static bool prepared             = false;

static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Vertices */
  static const vertex_t vertex_buffer[6] = {
    {
      .position = {-1.0f, -1.0f},
    },
    {
      .position = {1.0f, -1.0f},
    },
    {
      .position = {1.0f, 1.0f},
    },
    {
      .position = {-1.0f, -1.0f},
    },
    {
      .position = {1.0f, 1.0f},
    },
    {
      .position = {-1.0f, 1.0f},
    },
  };

  /* Create vertex buffer */
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout
  uniform_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor) {
      .label      = "Bind group layout",
      .entryCount = 1,
      .entries = &(WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Fragment shader)
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(ubo_fs),
        },
        .sampler = {0},
      }
    }
  );
  ASSERT(uniform_bind_group_layout != NULL);

  // Pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &uniform_bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Bind Group
  uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor) {
     .label      = "Bind group",
     .layout     = uniform_bind_group_layout,
     .entryCount = 1,
     .entries    = &(WGPUBindGroupEntry) {
       // Binding 0: Uniform buffer (Fragment shader)
       .binding = 0,
       .buffer  = uniform_buffer_fs.buffer,
       .offset  = 0,
       .size    = uniform_buffer_fs.size,
     },
   }
  );
  ASSERT(uniform_bind_group != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.125f,
        .g = 0.125f,
        .b = 0.250f,
        .a = 1.000f,
      },
  };

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = NULL,
  };
}

void update_uniform_buffers(wgpu_example_context_t* context)
{
  ubo_fs.offset = (uint32_t)roundf(random_float() * 4294967295.f);

  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_fs.buffer, 0,
                          &ubo_fs, uniform_buffer_fs.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Create a uniform buffer */
  uniform_buffer_fs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(ubo_fs), /* One u32, 4 bytes each */
    });

  /* Upload the uniform buffer to the GPU */
  update_uniform_buffers(context);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  /* Construct the different states making up the pipeline */

  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(quad, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                                               offsetof(vertex_t, position)))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Vertex shader WGSL
      .label            = "PRNG - Vertex shader WGSL",
      .wgsl_code.source = prng_shader_wgsl,
      .entry            = "vs_main",
    },
    .buffer_count = 1,
    .buffers      = &quad_vertex_buffer_layout,
  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      // Fragment shader WGSL
      .label            = "PRNG - Fragment shader WGSL",
      .wgsl_code.source = prng_shader_wgsl,
      .entry            = "fs_main",
    },
    .target_count = 1,
    .targets      = &color_target_state,
  });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  pipeline = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
                                            &(WGPURenderPipelineDescriptor){
                                              .label = "PRNG - Render pipeline",
                                              .layout       = pipeline_layout,
                                              .primitive    = primitive_state,
                                              .vertex       = vertex_state,
                                              .fragment     = &fragment_state,
                                              .depthStencil = NULL,
                                              .multisample  = multisample_state,
                                            });
  ASSERT(pipeline != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
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
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    uniform_bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, vertices.count, 1, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay */
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

  /* Submit command buffers to the queue */
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

  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_fs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, uniform_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_prng(int argc, char* argv[])
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
static const char* prng_shader_wgsl = CODE(
  struct Uniforms {
    offset: u32
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  var<private> state: u32;

  // From https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/\n"
  fn pcg_hash(input: u32) -> u32 {
    state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
  }

  @vertex
  fn vs_main(@location(0) position : vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
  }

  @fragment
  fn fs_main(
    @builtin(position) position: vec4<f32>,
  ) -> @location(0) vec4<f32> {
    var seed = u32(512.0 * position.y + position.x) + uniforms.offset;
    var pcg = pcg_hash(seed);
    var v = f32(pcg) * (1.0 / 4294967295.0);
    return vec4<f32>(v, v, v, 1.0);
  }
);
// clang-format on
