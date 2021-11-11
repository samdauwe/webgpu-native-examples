#include "example_base.h"
#include "examples.h"

#include <string.h>

// Vertex layout used in this example
typedef struct vertex_t {
  vec3 position;
} vertex_t;

// Vertex buffer
static struct vertices_t {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

// Index buffer
static struct indices_t {
  WGPUBuffer buffer;
  uint32_t count;
} indices = {0};

// Uniform buffer block object
static struct uniform_buffer_vs_t {
  WGPUBuffer buffer;
  uint32_t count;
} uniform_buffer_vs = {0};

// Uniform block data - inputs of the ShaderToy shader
static struct shader_inputs_ubo_t {
  vec2 iResolution;  // viewport resolution (in pixels)
  float iTime;       // shader playback time (in seconds)
  float iTimeDelta;  // render time (in seconds)
  int iFrame;        // shader playback frame
  vec4 iMouse;       // mouse pixel coords. xy: current (if MLB down), zw: click
  vec4 iDate;        // (year, month, day, time in seconds)
  float iSampleRate; // sound sample rate (i.e., 44100)
} shader_inputs_ubo = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout;

// Pipeline
static WGPURenderPipeline pipeline;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// The bind group layout
static WGPUBindGroupLayout bind_group_layout;

// The bind group
static WGPUBindGroup bind_group;

// Other variables
static const char* example_title = "Shadertoy";
static bool prepared             = false;

static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  // Setup vertices (x, y, z)
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
  static const vertex_t vertex_buffer[4] = {
    {
      .position = {-1.0f, 1.0f, 0.0f}, // Vertex 0
    },
    {
      .position = {1.0f, 1.0f, 0.0f}, // Vertex 1
    },
    {
      .position = {-1.0f, -1.0f, 0.0f}, // Vertex 2
    },
    {
      .position = {1.0f, -1.0f, 0.0f}, // Vertex 3
    },
  };
  vertices.count              = (uint32_t)ARRAY_SIZE(vertex_buffer);
  uint32_t vertex_buffer_size = vertices.count * sizeof(vertex_t);

  // Setup indices
  static const uint16_t index_buffer[6] = {
    0, 1, 2, // Triangle 0
    2, 1, 3  // Triangle 1
  };
  indices.count              = (uint32_t)ARRAY_SIZE(index_buffer);
  uint32_t index_buffer_size = indices.count * sizeof(uint32_t);

  // Create vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertex_buffer, vertex_buffer_size, WGPUBufferUsage_Vertex);

  // Create index buffer
  indices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, index_buffer, index_buffer_size, WGPUBufferUsage_Index);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Create the bind group layout
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = 1,
    .entries = &(WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform buffer (Vertex shader)
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type = WGPUBufferBindingType_Uniform,
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
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  // Bind Group
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = bind_group_layout,
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = sizeof(shader_inputs_ubo),
    },
  };

  bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(bind_group != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount = 1,
    .colorAttachments     = rp_color_att_descriptors,
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
  if (context->mouse_buttons.left) {
    shader_inputs_ubo.iMouse[0] = context->mouse_position[0];
    shader_inputs_ubo.iMouse[1]
      = context->wgpu_context->surface.height - context->mouse_position[1];
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
                          &shader_inputs_ubo, uniform_buffer_vs.count);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create the uniform bind group (note 'rotDeg' is copied here, not bound in
  // any way)
  uniform_buffer_vs.buffer = wgpu_create_buffer_from_data(
    context->wgpu_context, &shader_inputs_ubo, sizeof(shader_inputs_ubo),
    WGPUBufferUsage_Uniform);
  uniform_buffer_vs.count = sizeof(shader_inputs_ubo);

  update_uniform_buffers(context);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Front,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(shadertoy, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, position)))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader SPIR-V
                      .file = "shaders/shadertoy/main.vert.spv",
                    },
                    .buffer_count = 1,
                    .buffers = &shadertoy_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader SPIR-V
                      .file = "shaders/shadertoy/main.frag.spv",
                    },
                    .target_count = 1,
                    .targets = &color_target_state_desc,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "shadertoy_render_pipeline",
                            .layout      = pipeline_layout,
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_and_index_buffers(context->wgpu_context);
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
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass_desc);

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

  // Bind vertex buffer (contains position)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, 0);

  // Bind index buffer
  wgpuRenderPassEncoderSetIndexBuffer(wgpu_context->rpass_enc, indices.buffer,
                                      WGPUIndexFormat_Uint16, 0, 0);

  // Draw indexed quad
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc, indices.count, 1, 0,
                                   0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
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

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, indices.buffer)
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
     .title  = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
