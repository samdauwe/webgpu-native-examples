#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - MSAA Line
 *
 * The parts of this example enabling MSAA are:
 * *    The render pipeline is created with a sample_count > 1.
 * *    A new texture with a sample_count > 1 is created and set as the
 *      color_attachment instead of the swapchain.
 * *    The swapchain is now specified as a resolve_target.
 *
 * The parts of this example enabling LineList are:
 * *   Set the primitive_topology to PrimitiveTopology::LineList.
 * *   Vertices and Indices describe the two points that make up a line.
 *
 * Ref:
 * https://github.com/gfx-rs/wgpu-rs/tree/master/examples/msaa-line
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * MSAA Line example
 * -------------------------------------------------------------------------- */

#define NUMBER_OF_LINES 50u
static const uint32_t sample_count = 4u;

typedef struct {
  vec2 position;
  vec4 color;
} vertex_t;

// Vertex buffer
static wgpu_buffer_t vertices = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Render bundle
static WGPURenderBundle render_bundle = NULL;

// Multi-sampled texture
static WGPUTexture multisampled_texture         = NULL;
static WGPUTextureView multisampled_framebuffer = NULL;

// Other variables
static const char* example_title = "MSAA Line";
static bool prepared             = false;

static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  const uint32_t vertex_count = NUMBER_OF_LINES * 2;
  vertex_t vertex_data[vertex_count];
  float percent = 0.0f, sin_value = 0.0f, cos_value = 0.0f;
  for (uint32_t i = 0; i < NUMBER_OF_LINES; ++i) {
    percent                = (float)i / (float)NUMBER_OF_LINES;
    sin_value              = sinf(percent * PI2);
    cos_value              = cosf(percent * PI2);
    vertex_data[i * 2 + 0] = (vertex_t){
      .position = {0.f, 0.f},
      .color    = {1.f, -sin_value, cos_value, 1.f},
    };
    vertex_data[i * 2 + 1] = (vertex_t){
      .position = {1.f * cos_value, 1.f * sin_value},
      .color    = {sin_value, -cos_value, 1.f, 1.f},
    };
  }

  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_data),
                    .count = (uint32_t)ARRAY_SIZE(vertex_data),
                    .initial.data = vertex_data,
                  });
}

static void create_multisampled_framebuffer(wgpu_context_t* wgpu_context)
{
  /* Create the multi-sampled texture */
  WGPUTextureDescriptor multisampled_frame_desc = {
    .label         = "Multi-sampled texture",
    .size          = (WGPUExtent3D){
      .width               = wgpu_context->surface.width,
      .height              = wgpu_context->surface.height,
      .depthOrArrayLayers  = 1,
     },
    .mipLevelCount = 1,
    .sampleCount   = sample_count,
    .dimension     = WGPUTextureDimension_2D,
    .format        = wgpu_context->swap_chain.format,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  multisampled_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
  ASSERT(multisampled_texture != NULL);

  /* Create the multi-sampled texture view */
  multisampled_framebuffer = wgpuTextureCreateView(
    multisampled_texture, &(WGPUTextureViewDescriptor){
                            .label           = "Multi-sampled texture view",
                            .format          = wgpu_context->swap_chain.format,
                            .dimension       = WGPUTextureViewDimension_2D,
                            .baseMipLevel    = 0,
                            .mipLevelCount   = 1,
                            .baseArrayLayer  = 0,
                            .arrayLayerCount = 1,
                          });
  ASSERT(multisampled_framebuffer != NULL);
}

static void setup_render_bundle(wgpu_context_t* wgpu_context)
{
  WGPURenderBundleEncoderDescriptor rbe_desc = {
    .colorFormatCount = 1,
    .colorFormats     = &wgpu_context->swap_chain.format,
    .sampleCount      = sample_count,
  };
  WGPURenderBundleEncoder encoder
    = wgpuDeviceCreateRenderBundleEncoder(wgpu_context->device, &rbe_desc);
  wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);
  wgpuRenderBundleEncoderSetVertexBuffer(encoder, 0, vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
  wgpuRenderBundleEncoderDraw(encoder, vertices.count, 1, 0, 0);
  render_bundle
    = wgpuRenderBundleEncoderFinish(encoder, &(WGPURenderBundleDescriptor){
                                               .label = "main",
                                             });
  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, encoder);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view          = NULL, /* Assigned later */
      .resolveTarget = NULL,
      .depthSlice    = ~0,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = rp_color_att_descriptors,
  };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {0};
  pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                   &pipeline_layout_desc);
  ASSERT(pipeline_layout != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_LineList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(msaa_line, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                                               offsetof(vertex_t, position)),
                            // Attribute location 1: Color
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                               offsetof(vertex_t, color)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Vertex shader WGSL",
                      .wgsl_code.source = vertex_shader_wgsl,
                      .entry            = "main",
                    },
                    .buffer_count = 1,
                    .buffers      = &msaa_line_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Fragment shader WGSL",
                      .wgsl_code.source = fragment_shader_wgsl,
                      .entry            = "main",
                    },
                    .target_count = 1,
                    .targets      = &color_target_state,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = sample_count,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "MSAA line - Render pipeline",
                            .layout      = pipeline_layout,
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
    prepare_vertex_buffer(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    create_multisampled_framebuffer(context->wgpu_context);
    setup_render_bundle(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  if (sample_count == 1) {
    rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;
    rp_color_att_descriptors[0].resolveTarget = NULL;
  }
  else {
    rp_color_att_descriptors[0].view = multisampled_framebuffer;
    rp_color_att_descriptors[0].resolveTarget
      = wgpu_context->swap_chain.frame_buffer;
  }

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

  /* Execute render bundles */
  wgpuRenderPassEncoderExecuteBundles(wgpu_context->rpass_enc, 1,
                                      &render_bundle);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

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

  /* Submit command buffer to queue */
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
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Texture, multisampled_texture)
  WGPU_RELEASE_RESOURCE(TextureView, multisampled_framebuffer)
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_msaa_line(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .vsync = true,
    },
    .example_window_config = (window_config_t){
      .width  = 800,
      .height = 600,
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
  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) f_Color : vec4<f32>,
  };

  @vertex
  fn main(
    @location(0) a_Pos : vec2<f32>,
    @location(1) a_Color : vec4<f32>
  ) -> Output {
    var output : Output;
    output.position = vec4(a_Pos, 0.0, 1.0);
    output.f_Color = a_Color;
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn main(@location(0) v_Color : vec4<f32>) -> @location(0) vec4<f32> {
    return v_Color;
  }
);
// clang-format on
