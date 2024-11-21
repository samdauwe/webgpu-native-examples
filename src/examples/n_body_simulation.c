#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - N-Body Simulation
 *
 * A simple N-body simulation implemented using WebGPU.
 *
 * Ref:
 * https://github.com/jrprice/NBody-WebGPU
 * https://en.wikipedia.org/wiki/N-body_simulation
 * -------------------------------------------------------------------------- */

#define NUM_BODIES 8192u
#define WORKGROUP_SIZE 64u
#define INITIAL_EYE_POSITION {0.0f, 0.0f, -1.5f}

// Simulation parameters
static const uint32_t num_bodies = NUM_BODIES;

// Shader parameters.
static const uint32_t workgroup_size = WORKGROUP_SIZE;

// Render parameters
static vec3 eye_position = INITIAL_EYE_POSITION;

// Uniform buffer block object
static struct {
  wgpu_buffer_t render_params;
} uniform_buffers = {0};

static struct {
  mat4 view_projection_matrix;
  mat4 projection_matrix;
  bool changed;
} render_params = {
  .view_projection_matrix = GLM_MAT4_ZERO_INIT,
  .projection_matrix      = GLM_MAT4_ZERO_INIT,
  .changed                = true,
};

// Storage buffer block objects
static struct {
  struct {
    wgpu_buffer_t buffer;
    float positions[NUM_BODIES * 4];
  } positions_in;
  wgpu_buffer_t positions_out;
  wgpu_buffer_t velocities;
} storage_buffers = {0};

// Bind group layouts
static struct {
  WGPUBindGroupLayout compute;
  WGPUBindGroupLayout render;
} bind_group_layouts = {0};

// Bind groups
static struct {
  WGPUBindGroup compute[2];
  WGPUBindGroup render;
} bind_groups = {0};

// Pipeline layouts
static struct {
  WGPUPipelineLayout compute;
  WGPUPipelineLayout render;
} pipeline_layouts = {0};

// Pipelines
static struct {
  WGPUComputePipeline compute;
  WGPURenderPipeline render;
} pipelines = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// FPS counter
static struct {
  float fps_update_interval;
  float num_frames_since_fps_update;
  float last_fps_update_time;
  float fps;
  bool last_fps_update_time_valid;
} fps_counter = {
  .fps_update_interval         = 500.0f,
  .num_frames_since_fps_update = 0.0f,
  .last_fps_update_time_valid  = false,
};

static uint32_t frame_idx = 0;

// Other variables
static const char* example_title = "N-Body Simulation";
static bool prepared             = false;

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Generate the view projection matrix
  glm_mat4_identity(render_params.projection_matrix);
  glm_mat4_identity(render_params.view_projection_matrix);
  const float aspect
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  const float far = 50.0f;
  perspective_zo(&render_params.projection_matrix, 1.0f, aspect, 0.1f, &far);
  glm_translate(render_params.view_projection_matrix, eye_position);
  glm_mat4_mul(render_params.projection_matrix,
               render_params.view_projection_matrix,
               render_params.view_projection_matrix);

  // Write the render parameters to the uniform buffer
  wgpu_queue_write_buffer(wgpu_context, uniform_buffers.render_params.buffer, 0,
                          render_params.view_projection_matrix,
                          uniform_buffers.render_params.size);

  render_params.changed = false;
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block
  uniform_buffers.render_params = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Vertex shader - Uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(mat4), // sizeof(mat4x4<f32>)
    });

  update_uniform_buffers(context);
}

/* Generate initial positions on the surface of a sphere */
static void init_bodies(wgpu_context_t* wgpu_context)
{
  const float radius = 0.6f;
  float* positions   = storage_buffers.positions_in.positions;
  ASSERT(positions != NULL);
  float longitude = 0.0f, latitude = 0.0f;
  for (uint32_t i = 0; i < num_bodies; ++i) {
    longitude            = 2.0f * PI * random_float();
    latitude             = acos((2.0f * random_float() - 1.0f));
    positions[i * 4 + 0] = radius * sin(latitude) * cos(longitude);
    positions[i * 4 + 1] = radius * sin(latitude) * sin(longitude);
    positions[i * 4 + 2] = radius * cos(latitude);
    positions[i * 4 + 3] = 1.0f;
  }

  /* Write the render parameters to the uniform buffer */
  wgpu_queue_write_buffer(wgpu_context,
                          storage_buffers.positions_in.buffer.buffer, 0,
                          storage_buffers.positions_in.positions,
                          storage_buffers.positions_in.buffer.size);
}

/* Create buffers for body positions and velocities. */
static void prepare_storage_buffers(wgpu_example_context_t* context)
{
  storage_buffers.positions_in.buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Positions in - Storage buffers",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
               | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size = num_bodies * 4 * 4,
    });

  storage_buffers.positions_out = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Positions out - Storage buffers",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
               | WGPUBufferUsage_Vertex,
      .size = num_bodies * 4 * 4,
    });

  storage_buffers.velocities = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Velocities - Storage buffers",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
      .size  = num_bodies * 4 * 4,
    });

  /* Generate initial positions on the surface of a sphere */
  init_bodies(context->wgpu_context);
}

static void setup_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = storage_buffers.positions_in.buffer.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = storage_buffers.positions_out.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = storage_buffers.velocities.size,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Compute - Bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  bind_group_layouts.compute
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layouts.compute != NULL);

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = "Compute - Pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &bind_group_layouts.compute,
  };
  pipeline_layouts.compute = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(pipeline_layouts.compute != NULL);
}

static void setup_render_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_buffers.render_params.size,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Render - Pipeline layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  bind_group_layouts.render
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layouts.render != NULL);

  WGPUPipelineLayoutDescriptor render_pipeline_layout_desc = {
    .label                = "Render - Pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &bind_group_layouts.render,
  };
  pipeline_layouts.render = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &render_pipeline_layout_desc);
  ASSERT(pipeline_layouts.render != NULL);
}

/* Create the bind group for the compute shader. */
static void setup_compute_bind_group(wgpu_context_t* wgpu_context)
{
  {
    WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Input Positions */
      .binding = 0,
      .buffer  = storage_buffers.positions_in.buffer.buffer,
      .offset  = 0,
      .size    = storage_buffers.positions_in.buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Binding 1 : Output Positions */
      .binding = 1,
      .buffer  = storage_buffers.positions_out.buffer,
      .offset  = 0,
      .size    = storage_buffers.positions_out.size,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2 : Velocities */
      .binding = 2,
      .buffer  = storage_buffers.velocities.buffer,
      .offset  = 0,
      .size    = storage_buffers.velocities.size,
    },
  };

    bind_groups.compute[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Compute shader - Bind group 0",
                              .layout     = bind_group_layouts.compute,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.compute[0] != NULL);
  }

  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : Output Positions */
        .binding = 0,
        .buffer  = storage_buffers.positions_out.buffer,
        .offset  = 0,
        .size    = storage_buffers.positions_out.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1 : Input Positions */
        .binding = 1,
        .buffer  = storage_buffers.positions_in.buffer.buffer,
        .offset  = 0,
        .size    = storage_buffers.positions_in.buffer.size,
      },
    [2] = (WGPUBindGroupEntry) {
      /* Binding 2 : Velocities */
      .binding = 2,
      .buffer  = storage_buffers.velocities.buffer,
      .offset  = 0,
      .size    = storage_buffers.velocities.size,
    },
  };

    bind_groups.compute[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Compute shader - Bind group 1",
                              .layout     = bind_group_layouts.compute,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.compute[1] != NULL);
  }
}

static void setup_render_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Render params uniform buffer */
      .binding = 0,
      .buffer  = uniform_buffers.render_params.buffer,
      .offset  = 0,
      .size    = uniform_buffers.render_params.size,
    },
  };

  bind_groups.render = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "Render - Bind group",
                            .layout     = bind_group_layouts.render,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(bind_groups.render != NULL);
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
        .b = 0.1f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

/* Create the compute pipeline */
static void prepare_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  wgpu_shader_t compute_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    /* Compute shader WGSL */
                    .label = "Compute shader WGSL",
                    .file  = "shaders/n_body_simulation/n_body_simulation.wgsl",
                    .entry = "cs_main",
                  });

  pipelines.compute = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "N-Body simulation - Compute pipeline",
      .layout  = pipeline_layouts.compute,
      .compute = compute_shader.programmable_stage_descriptor,
    });
  ASSERT(pipelines.compute != NULL);

  /* Partial cleanup */
  wgpu_shader_release(&compute_shader);
}

// Create the graphics pipeline
static void prepare_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state = {
    .color.operation = WGPUBlendOperation_Add,
    .color.srcFactor = WGPUBlendFactor_One,
    .color.dstFactor = WGPUBlendFactor_One,
    .alpha.operation = WGPUBlendOperation_Add,
    .alpha.srcFactor = WGPUBlendFactor_One,
    .alpha.dstFactor = WGPUBlendFactor_One,
  };
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    position, 4 * sizeof(float),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0))
  position_vertex_buffer_layout.stepMode = WGPUVertexStepMode_Instance;

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Vertex shader WGSL */
                  .label = "N-Body simulation - Vertex shader WGSL",
                  .file  = "shaders/n_body_simulation/n_body_simulation.wgsl",
                  .entry = "vs_main",
                },
                .buffer_count = 1,
                .buffers = &position_vertex_buffer_layout,
              });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Fragment shader WGSL */
                  .label = "N-Body simulation - Fragment shader WGSL",
                  .file  = "shaders/n_body_simulation/n_body_simulation.wgsl",
                  .entry = "fs_main",
                },
                .target_count = 1,
                .targets = &color_target_state,
              });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  /* Create rendering pipeline using the specified states */
  pipelines.render = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = "N-Body simulation - Render pipeline",
                            .layout    = pipeline_layouts.render,
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .depthStencil = NULL,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipelines.render != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_uniform_buffers(context);
    prepare_storage_buffers(context);
    setup_compute_pipeline_layout(context->wgpu_context);
    setup_render_pipeline_layout(context->wgpu_context);
    prepare_compute_pipeline(context->wgpu_context);
    prepare_render_pipeline(context->wgpu_context);
    setup_compute_bind_group(context->wgpu_context);
    setup_render_bind_group(context->wgpu_context);
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

static WGPUCommandBuffer build_command_buffer(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context          = context->wgpu_context;
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass */
  if (!context->paused) {
    // Set up the compute shader dispatch
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      pipelines.compute);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       bind_groups.compute[frame_idx], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      wgpu_context->cpass_enc, ceil(num_bodies / (float)workgroup_size), 1, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
    frame_idx = (frame_idx + 1) % 2;
  }

  /* Render pass */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipelines.render);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.render, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0,
      frame_idx == 0 ? storage_buffers.positions_in.buffer.buffer :
                       storage_buffers.positions_out.buffer,
      0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, num_bodies, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
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
  wgpu_context->submit_info.command_buffers[0] = build_command_buffer(context);

  /* Submit command buffer to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static void update_fps_counter(wgpu_example_context_t* context)
{
  if (fps_counter.last_fps_update_time_valid) {
    const float now                 = context->frame.timestamp_millis;
    const float time_since_last_log = now - fps_counter.last_fps_update_time;
    if (time_since_last_log >= fps_counter.fps_update_interval) {
      fps_counter.fps = fps_counter.num_frames_since_fps_update
                        / (time_since_last_log / 1000.0f);
      fps_counter.last_fps_update_time        = now;
      fps_counter.num_frames_since_fps_update = 0.0f;
    }
  }
  else {
    fps_counter.last_fps_update_time       = context->frame.timestamp_millis;
    fps_counter.last_fps_update_time_valid = true;
  }
  ++fps_counter.num_frames_since_fps_update;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  update_fps_counter(context);
  bool result = example_draw(context);
  if (render_params.changed) {
    update_uniform_buffers(context);
  }
  return result;
}

static void example_on_key_pressed(keycode_t key)
{
  if (key == KEY_UP || key == KEY_DOWN) {
    static const float z_inc = 0.025f;
    if (key == KEY_UP) {
      eye_position[2] += z_inc;
    }
    else if (key == KEY_DOWN) {
      eye_position[2] -= z_inc;
    }
    // Update render parameters based on key presses
    render_params.changed = true;
  }
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.render_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, storage_buffers.positions_in.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, storage_buffers.positions_out.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, storage_buffers.velocities.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.compute)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.render)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.compute[0])
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.compute[1])
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.render)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.compute)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.render)
  WGPU_RELEASE_RESOURCE(ComputePipeline, pipelines.compute)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.render)
}

void example_n_body_simulation(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func     = &example_initialize,
    .example_render_func         = &example_render,
    .example_destroy_func        = &example_destroy,
    .example_on_key_pressed_func = &example_on_key_pressed,
  });
  // clang-format on
}
