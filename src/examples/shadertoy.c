#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Shadertoy
 *
 * Minimal "shadertoy launcher" using WebGPU, demonstrating how to load an
 * example Shadertoy shader 'Seascape'.
 *
 * Ref:
 * https://www.shadertoy.com/view/Ms2SD1
 * https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
 * -------------------------------------------------------------------------- */

// Uniform buffer block object
static wgpu_buffer_t uniform_buffer_vs = {0};

// Uniform block data - inputs of the ShaderToy shader
static struct {
  vec2 iResolution;  // viewport resolution (in pixels)
  float iTime;       // shader playback time (in seconds)
  float iTimeDelta;  // render time (in seconds)
  int iFrame;        // shader playback frame
  vec4 iMouse;       // mouse pixel coords. xy: current (if MLB down), zw: click
  vec4 iDate;        // (year, month, day, time in seconds)
  float iSampleRate; // sound sample rate (i.e., 44100)
} shader_inputs_ubo = {0};

// Used for mouse pixel coordinates calculation
static struct {
  vec2 initial_mouse_position;
  vec2 prev_mouse_position;
  vec2 mouse_drag_distance;
  bool dragging;
} mouse_state = {
  .initial_mouse_position = GLM_VEC2_ZERO_INIT,
  .prev_mouse_position    = GLM_VEC2_ZERO_INIT,
  .mouse_drag_distance    = GLM_VEC2_ZERO_INIT,
  .dragging               = false,
};

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
static const char* example_title = "Shadertoy";
static bool prepared             = false;

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Create the bind group layout
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Bind group layout",
    .entryCount = 1,
    .entries = &(WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Fragment shader)
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(shader_inputs_ubo),
      },
      .sampler = {0},
    }
  };
  bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
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
  // Bind Group
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Bind group",
    .layout     = bind_group_layout,
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_group != NULL);
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
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // iResolution: viewport resolution (in pixels)
  shader_inputs_ubo.iResolution[0]
    = (float)context->wgpu_context->surface.width;
  shader_inputs_ubo.iResolution[1]
    = (float)context->wgpu_context->surface.height;

  // iTime: Time since the shader started (in seconds)
  shader_inputs_ubo.iTime = context->run_time;

  // iTimeDelta: time between each frame (duration since the previous frame)
  shader_inputs_ubo.iTimeDelta = context->frame_timer;

  // iFrame: shader playback frame
  shader_inputs_ubo.iFrame = (int)context->frame.index;

  // iMouse: mouse pixel coords. xy: current (if MLB down), zw: click
  context->mouse_position[1]
    = context->wgpu_context->surface.height - context->mouse_position[1];
  if (!mouse_state.dragging && context->mouse_buttons.left) {
    glm_vec2_copy(context->mouse_position, mouse_state.prev_mouse_position);
    mouse_state.dragging = true;
  }
  else if (mouse_state.dragging && context->mouse_buttons.left) {
    glm_vec2_sub(context->mouse_position, mouse_state.prev_mouse_position,
                 mouse_state.mouse_drag_distance);
    glm_vec2_add(shader_inputs_ubo.iMouse, mouse_state.mouse_drag_distance,
                 shader_inputs_ubo.iMouse);
    glm_vec2_copy(context->mouse_position, mouse_state.prev_mouse_position);
  }
  else if (mouse_state.dragging && !context->mouse_buttons.left) {
    mouse_state.dragging = false;
  }

  // iDate: year, month, day, time in seconds
  struct date_t current_date;
  get_local_time(&current_date);
  shader_inputs_ubo.iDate[0] = current_date.year,
  shader_inputs_ubo.iDate[1] = current_date.month,
  shader_inputs_ubo.iDate[2] = current_date.day,
  shader_inputs_ubo.iDate[3] = current_date.day_sec;

  // iSampleRate: iSampleRate
  shader_inputs_ubo.iSampleRate = 44100.0f;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &shader_inputs_ubo, uniform_buffer_vs.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create the uniform bind group (note 'rotDeg' is copied here, not bound in
  // any way)
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size         = sizeof(shader_inputs_ubo),
      .initial.data = &shader_inputs_ubo,
    });

  // Update uniform buffer data and uniform buffer
  update_uniform_buffers(context);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
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
                      // Vertex shader SPIR-V
                      .label = "main_vertex_shader",
                      .file  = "shaders/shadertoy/main.vert.spv",
                    },
                    .buffer_count = 0,
                    .buffers      = NULL,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .label = "main_fragment_shader",
                      .file  = "shaders/shadertoy/main.frag.spv",
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
                            .label       = "shadertoy_render_pipeline",
                            .layout      = pipeline_layout,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
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

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0, bind_group, 0,
                                    0);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Draw quad
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

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

  // Update the uniform buffers
  update_uniform_buffers(context);

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
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
}

void example_shadertoy(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title = example_title,
     .vsync = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
