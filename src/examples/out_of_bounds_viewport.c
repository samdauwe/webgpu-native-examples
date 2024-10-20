#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Out-of-bounds Viewport
 *
 * WebGPU doesn't let you set the viewport’s values to be out-of-bounds.
 * Therefore, the viewport’s values need to be clamped to the screen-size, which
 * means the viewport values can’t be defined in a way that makes the viewport
 * go off the screen. This example shows how to render a viewport out-of-bounds.
 *
 * Ref:
 * https://babylonjs.medium.com/how-to-simulate-out-of-bounds-viewports-when-using-webgpu-or-babylonnative-2280637c0660
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Out-of-bounds Viewport example
 * -------------------------------------------------------------------------- */

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

// Uniform block data - inputs of the shader
static struct {
  float x;
  float y;
  float width;
  float height;
} viewport_params = {
  .x      = 0.0f,
  .y      = 0.0f,
  .width  = 1.0f,
  .height = 1.0f,
};

// Texture and sampler
static texture_t texture = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

// Pipeline
static WGPURenderPipeline pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// The bind group layout
static WGPUBindGroupLayout bind_group_layout = NULL;

// The bind group
static WGPUBindGroup bind_group = NULL;

// Other variables
static const char* example_title = "Out-of-bounds Viewport";
static bool prepared             = false;

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(viewport_params),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader texture view */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Fragment shader texture sampler */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type  = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
  };
  bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout != NULL);

  // Create the pipeline layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Pipeline layout",
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding = 2,
      .sampler = texture.sampler,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Bind group",
    .layout     = bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_group != NULL);
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/Di-3d.png";
  texture          = wgpu_create_texture_from_file(wgpu_context, file, NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
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

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &viewport_params, uniform_buffer_vs.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Uniform buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size         = sizeof(viewport_params),
      .initial.data = &viewport_params,
    });
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Vertex shader WGSL",
                      .wgsl_code.source = vertex_shader_wgsl,
                      .entry            = "vertex_main",
                    },
                    .buffer_count = 0,
                    .buffers      = NULL,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Fragment shader WGSL",
                      .wgsl_code.source = fragment_shader_wgsl,
                      .entry            = "fragment_main",
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
                                              .label       = "Render pipeline",
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
    prepare_texture(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Viewport")) {
    if (imgui_overlay_slider_float(context->imgui_overlay, "x",
                                   &viewport_params.x, -0.5f, 0.5f, "%.2f")) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "y",
                                   &viewport_params.y, -0.5f, 0.5f, "%.2f")) {
      update_uniform_buffers(context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "width",
                                   &viewport_params.width, 0.f, 2.f, "%.2f")) {
      viewport_params.width += 0.000001f;
      update_uniform_buffers(context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "height",
                                   &viewport_params.height, 0.f, 2.f, "%.2f")) {
      viewport_params.height += 0.000001f;
      update_uniform_buffers(context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  /* Set the bind group */
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  /* Set viewport */
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  /* Draw indexed quad */
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, 1, 0, 0);

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
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Update the uniform buffers
  update_uniform_buffers(context);

  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit command buffer to queue
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
  wgpu_destroy_texture(&texture);
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_out_of_bounds_viewport(int argc, char* argv[])
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
static const char* vertex_shader_wgsl = CODE(
  struct Viewport {
    x : f32,
    y : f32,
    w : f32,
    h : f32,
  };

  @group(0) @binding(0)
  var<uniform> viewport : Viewport;

  struct VertexOutput{
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
  }

  @vertex
  fn vertex_main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> VertexOutput {

    var pos = array<vec2<f32>, 6>(
      vec2( 1.0,  1.0),  vec2( 1.0, -1.0), vec2(-1.0, -1.0),
      vec2( 1.0,  1.0),  vec2(-1.0, -1.0), vec2(-1.0,  1.0)
    );

    const uv = array(
      vec2( 1.0,  0.0),  vec2( 1.0,  1.0), vec2( 0.0,  1.0),
      vec2( 1.0,  0.0),  vec2( 0.0,  1.0), vec2( 0.0,  0.0)
    );

    var position : vec4<f32> = vec4(pos[VertexIndex], 0.0, 1.0);
    position.x = position.x * viewport.w
                  + (viewport.x + viewport.w - 1.0 + viewport.x) * position.w;
    position.y = position.y * viewport.h
                  + (viewport.y + viewport.h - 1.0 + viewport.y) * position.w;

    var output : VertexOutput;
    output.Position = position;
    output.fragUV =  uv[VertexIndex];

    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @group(0) @binding(1) var myTexture : texture_2d<f32>;
  @group(0) @binding(2) var mySampler : sampler;

  @fragment
  fn fragment_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    // PIXELATE
    var dx:f32 = 8.0 / 640.0;
    var dy:f32 = 8.0 / 640.0;
    var uv:vec2<f32> = vec2(dx*(floor(fragUV.x/dx)), dy*(floor(fragUV.y/dy)));

    var color:vec3<f32> = (textureSample(myTexture, mySampler, uv)).rgb;

    return  vec4<f32>(color, 1.0);
  }
);
// clang-format on
