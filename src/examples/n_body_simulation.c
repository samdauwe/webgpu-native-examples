#include "webgpu/wgpu_common.h"

#include "core/camera.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdbool.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - N-Body Simulation
 *
 * A simple N-body simulation implemented using WebGPU.
 *
 * Ref:
 * https://github.com/jrprice/NBody-WebGPU
 * https://en.wikipedia.org/wiki/N-body_simulation
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* n_body_simulation_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * N-Body Simulation example
 * -------------------------------------------------------------------------- */

#define NUM_BODIES (8192u)
#define WORKGROUP_SIZE (64u)
#define INITIAL_EYE_POSITION {0.0f, 0.0f, -1.5f}

/* State struct */
static struct {
  const uint32_t num_bodies;
  const uint32_t workgroup_size;
  vec3 eye_position;
  const float z_inc;
  struct {
    wgpu_buffer_t render_params;
  } uniform_buffers;
  struct {
    mat4 view_projection_matrix;
    mat4 projection_matrix;
    bool changed;
  } render_params;
  struct {
    struct {
      wgpu_buffer_t buffer;
      float positions[NUM_BODIES * 4];
    } positions_in;
    wgpu_buffer_t positions_out;
    wgpu_buffer_t velocities;
  } storage_buffers;
  struct {
    WGPUBindGroupLayout compute;
    WGPUBindGroupLayout render;
  } bind_group_layouts;
  struct {
    WGPUBindGroup compute[2];
    WGPUBindGroup render;
  } bind_groups;
  struct {
    WGPUPipelineLayout compute;
    WGPUPipelineLayout render;
  } pipeline_layouts;
  struct {
    WGPUComputePipeline compute;
    WGPURenderPipeline render;
  } pipelines;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    float fps_update_interval;
    float num_frames_since_fps_update;
    float last_fps_update_time;
    float fps;
    bool last_fps_update_time_valid;
  } fps_counter;
  uint64_t frame_idx;
  float prev_time_millis;
  bool paused;
  bool initialized;
} state = {
  .num_bodies     = NUM_BODIES,
  .workgroup_size = WORKGROUP_SIZE,
  .eye_position   = INITIAL_EYE_POSITION,
  .z_inc          = 0.025f,
  .render_params = {
     .view_projection_matrix = GLM_MAT4_ZERO_INIT,
     .projection_matrix      = GLM_MAT4_ZERO_INIT,
     .changed                = true,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .fps_counter = {
    .fps_update_interval         = 500.0f,
    .num_frames_since_fps_update = 0.0f,
    .last_fps_update_time_valid  = false,
  },
};

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Generate the view projection matrix */
  glm_mat4_identity(state.render_params.projection_matrix);
  glm_mat4_identity(state.render_params.view_projection_matrix);
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  const float far    = 50.0f;
  perspective_zo(&state.render_params.projection_matrix, 1.0f, aspect, 0.1f,
                 &far);
  glm_translate(state.render_params.view_projection_matrix, state.eye_position);
  glm_mat4_mul(state.render_params.projection_matrix,
               state.render_params.view_projection_matrix,
               state.render_params.view_projection_matrix);

  /* Write the render parameters to the uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.uniform_buffers.render_params.buffer, 0,
                       state.render_params.view_projection_matrix,
                       state.uniform_buffers.render_params.size);

  state.render_params.changed = false;
}

/* Initialize uniform buffer containing shader uniforms */
static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Vertex shader uniform buffer block */
  state.uniform_buffers.render_params = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex shader - Uniform buffer block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(mat4), /* sizeof(mat4x4<f32>) */
                  });

  update_uniform_buffers(wgpu_context);
}

/* Generate initial positions on the surface of a sphere */
static void init_bodies(wgpu_context_t* wgpu_context)
{
  const float radius = 0.6f;
  float* positions   = state.storage_buffers.positions_in.positions;
  ASSERT(positions != NULL);
  float longitude = 0.0f, latitude = 0.0f;
  for (uint32_t i = 0; i < state.num_bodies; ++i) {
    longitude            = 2.0f * PI * random_float();
    latitude             = acos((2.0f * random_float() - 1.0f));
    positions[i * 4 + 0] = radius * sin(latitude) * cos(longitude);
    positions[i * 4 + 1] = radius * sin(latitude) * sin(longitude);
    positions[i * 4 + 2] = radius * cos(latitude);
    positions[i * 4 + 3] = 1.0f;
  }

  /* Write the render parameters to the uniform buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.storage_buffers.positions_in.buffer.buffer, 0,
                       state.storage_buffers.positions_in.positions,
                       state.storage_buffers.positions_in.buffer.size);
}

/* Create buffers for body positions and velocities. */
static void init_storage_buffers(wgpu_context_t* wgpu_context)
{
  state.storage_buffers.positions_in.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Positions in - Storage buffers",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                             | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size = state.num_bodies * 4 * 4,
                  });

  state.storage_buffers.positions_out = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Positions out - Storage buffers",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage
                             | WGPUBufferUsage_Vertex,
                    .size = state.num_bodies * 4 * 4,
                  });

  state.storage_buffers.velocities = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Velocities - Storage buffers",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                    .size  = state.num_bodies * 4 * 4,
                  });

  /* Generate initial positions on the surface of a sphere */
  init_bodies(wgpu_context);
}

static void init_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = state.storage_buffers.positions_in.buffer.size,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = state.storage_buffers.positions_out.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = state.storage_buffers.velocities.size,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Compute - Bind group layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.bind_group_layouts.compute
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.bind_group_layouts.compute != NULL);

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = STRVIEW("Compute - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.bind_group_layouts.compute,
  };
  state.pipeline_layouts.compute = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(state.pipeline_layouts.compute != NULL);
}

static void init_render_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = state.uniform_buffers.render_params.size,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Render - Pipeline layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.bind_group_layouts.render
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.bind_group_layouts.render != NULL);

  WGPUPipelineLayoutDescriptor render_pipeline_layout_desc = {
    .label                = STRVIEW("Render - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.bind_group_layouts.render,
  };
  state.pipeline_layouts.render = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &render_pipeline_layout_desc);
  ASSERT(state.pipeline_layouts.render != NULL);
}

static void init_compute_bind_group(wgpu_context_t* wgpu_context)
{
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0 : Input Positions */
        .binding = 0,
        .buffer  = state.storage_buffers.positions_in.buffer.buffer,
        .offset  = 0,
        .size    = state.storage_buffers.positions_in.buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1 : Output Positions */
        .binding = 1,
        .buffer  = state.storage_buffers.positions_out.buffer,
        .offset  = 0,
        .size    = state.storage_buffers.positions_out.size,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2 : Velocities */
        .binding = 2,
        .buffer  = state.storage_buffers.velocities.buffer,
        .offset  = 0,
        .size    = state.storage_buffers.velocities.size,
      },
    };

    state.bind_groups.compute[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label = STRVIEW("Compute shader - Bind group 0"),
                              .layout     = state.bind_group_layouts.compute,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.compute[0] != NULL);
  }

  {
    WGPUBindGroupEntry bg_entries[3] = {
        [0] = (WGPUBindGroupEntry) {
          /* Binding 0 : Output Positions */
          .binding = 0,
          .buffer  = state.storage_buffers.positions_out.buffer,
          .offset  = 0,
          .size    = state.storage_buffers.positions_out.size,
        },
        [1] = (WGPUBindGroupEntry) {
          /* Binding 1 : Input Positions */
          .binding = 1,
          .buffer  = state.storage_buffers.positions_in.buffer.buffer,
          .offset  = 0,
          .size    = state.storage_buffers.positions_in.buffer.size,
        },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2 : Velocities */
        .binding = 2,
        .buffer  = state.storage_buffers.velocities.buffer,
        .offset  = 0,
        .size    = state.storage_buffers.velocities.size,
      },
    };

    state.bind_groups.compute[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label = STRVIEW("Compute shader - Bind group 1"),
                              .layout     = state.bind_group_layouts.compute,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.compute[1] != NULL);
  }
}

static void init_render_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Render params uniform buffer */
      .binding = 0,
      .buffer  = state.uniform_buffers.render_params.buffer,
      .offset  = 0,
      .size    = state.uniform_buffers.render_params.size,
    },
  };

  state.bind_groups.render = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Render - Bind group"),
                            .layout     = state.bind_group_layouts.render,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.bind_groups.render != NULL);
}

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, n_body_simulation_shader_wgsl);

  /* Create pipeline */
  state.pipelines.compute = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("N-Body simulation - Compute pipeline"),
      .layout  = state.pipeline_layouts.compute,
      .compute = {
        .module     = shader_module,
        .entryPoint = STRVIEW("cs_main"),
      },
    });
  ASSERT(state.pipelines.compute != NULL);

  /* Partial cleanup */
  wgpuShaderModuleRelease(shader_module);
}

static void init_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, n_body_simulation_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = {
    .color.operation = WGPUBlendOperation_Add,
    .color.srcFactor = WGPUBlendFactor_One,
    .color.dstFactor = WGPUBlendFactor_One,
    .alpha.operation = WGPUBlendOperation_Add,
    .alpha.srcFactor = WGPUBlendFactor_One,
    .alpha.dstFactor = WGPUBlendFactor_One,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    position, 4 * sizeof(float),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0))
  position_vertex_buffer_layout.stepMode = WGPUVertexStepMode_Instance;

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("N-Body simulation - render pipeline"),
    .layout = state.pipeline_layouts.render,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &position_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .module      = shader_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_None,
      .frontFace = WGPUFrontFace_CW
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff
    },
  };

  state.pipelines.render
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipelines.render != NULL);

  wgpuShaderModuleRelease(shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_uniform_buffers(wgpu_context);
    init_storage_buffers(wgpu_context);
    init_compute_pipeline_layout(wgpu_context);
    init_render_pipeline_layout(wgpu_context);
    init_compute_pipeline(wgpu_context);
    init_render_pipeline(wgpu_context);
    init_compute_bind_group(wgpu_context);
    init_render_bind_group(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void update_fps_counter(void)
{
  const float now               = stm_ms(stm_now());
  const float frame_time_millis = now - state.prev_time_millis;
  state.prev_time_millis        = now;
  if (state.fps_counter.last_fps_update_time_valid) {
    const float now = frame_time_millis;
    const float time_since_last_log
      = now - state.fps_counter.last_fps_update_time;
    if (time_since_last_log >= state.fps_counter.fps_update_interval) {
      state.fps_counter.fps = state.fps_counter.num_frames_since_fps_update
                              / (time_since_last_log / 1000.0f);
      state.fps_counter.last_fps_update_time        = now;
      state.fps_counter.num_frames_since_fps_update = 0.0f;
    }
  }
  else {
    state.fps_counter.last_fps_update_time       = frame_time_millis;
    state.fps_counter.last_fps_update_time_valid = true;
  }
  ++state.fps_counter.num_frames_since_fps_update;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Updated FPS counter and uniform buffers */
  update_fps_counter();
  if (state.render_params.changed) {
    update_uniform_buffers(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  if (!state.paused) {
    // Set up the compute shader dispatch
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.pipelines.compute);
    wgpuComputePassEncoderSetBindGroup(
      cpass_enc, 0, state.bind_groups.compute[state.frame_idx], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      cpass_enc, ceil(state.num_bodies / (float)state.workgroup_size), 1, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
    state.frame_idx = (state.frame_idx + 1) % 2;
  }

  /* Render pass */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.render);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.render, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0,
      state.frame_idx == 0 ? state.storage_buffers.positions_in.buffer.buffer :
                             state.storage_buffers.positions_out.buffer,
      0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, state.num_bodies, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  /* Get command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  if (input_event->type == INPUT_EVENT_TYPE_KEY_UP
      || input_event->type == INPUT_EVENT_TYPE_KEY_DOWN) {
    if (input_event->type == INPUT_EVENT_TYPE_KEY_UP) {
      state.eye_position[2] += state.z_inc;
    }
    else if (input_event->type == INPUT_EVENT_TYPE_KEY_DOWN) {
      state.eye_position[2] -= state.z_inc;
    }
    /* Update render parameters based on key presses */
    state.render_params.changed = true;
  }
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.render_params.buffer)
  WGPU_RELEASE_RESOURCE(Buffer,
                        state.storage_buffers.positions_in.buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.storage_buffers.positions_out.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.storage_buffers.velocities.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.compute)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.render)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.compute[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.compute[1])
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.render)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.compute)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.render)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.pipelines.compute)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.render)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "N-Body Simulation",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* n_body_simulation_shader_wgsl = CODE(
  // Simulation parameters.
  const kNumBodies = 8192;
  const kWorkgroupSize = 64;
  const kDelta = 0.000025;
  const kSoftening = 0.2;

  struct Float4Buffer {
    data : array<vec4<f32>>
  };

  @group(0) @binding(0)
  var<storage, read> positionsIn : Float4Buffer;

  @group(0) @binding(1)
  var<storage, read_write> positionsOut : Float4Buffer;

  @group(0) @binding(2)
  var<storage, read_write> velocities : Float4Buffer;

  fn computeForce(ipos : vec4<f32>,
                  jpos : vec4<f32>,
                  ) -> vec4<f32> {
    var d      = vec4<f32>((jpos - ipos).xyz, 0.0);
    var distSq = d.x*d.x + d.y*d.y + d.z*d.z + kSoftening*kSoftening;
    var dist   = inverseSqrt(distSq);
    var coeff  = jpos.w * (dist*dist*dist);
    return coeff * d;
  }

  @compute @workgroup_size(kWorkgroupSize)
  fn cs_main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    ) {
    var idx = gid.x;
    var pos = positionsIn.data[idx];

    // Compute force.
    var force = vec4<f32>(0.0);
    for (var i = 0; i < kNumBodies; i = i + 1) {
      force = force + computeForce(pos, positionsIn.data[i]);
    }

    // Update velocity.
    var velocity = velocities.data[idx];
    velocity = velocity + force * kDelta;
    velocities.data[idx] = velocity;

    // Update position.
    positionsOut.data[idx] = pos + velocity * kDelta;
  }

  struct RenderParams {
    viewProjectionMatrix : mat4x4<f32>
  };

  @group(0) @binding(0)
  var<uniform> renderParams : RenderParams;

  struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) positionInQuad : vec2<f32>,
    @location(1) @interpolate(flat) color : vec3<f32>
  };

  @vertex
  fn vs_main(
    @builtin(instance_index) idx : u32,
    @builtin(vertex_index) vertex : u32,
    @location(0) position : vec4<f32>,
    ) -> VertexOut {

    const kPointRadius = 0.005;
    var vertexOffsets = array<vec2<f32>, 6>(
      vec2<f32>(1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(1.0, -1.0),
    );
    var offset = vertexOffsets[vertex];

    var out : VertexOut;
    out.position = renderParams.viewProjectionMatrix *
      vec4<f32>(position.xy + offset * kPointRadius, position.zw);
    out.positionInQuad = offset;
    if (idx % 2u == 0u) {
      out.color = vec3<f32>(0.4, 0.4, 1.0);
    } else {
      out.color = vec3<f32>(1.0, 0.4, 0.4);
    }
    return out;
  }

  @fragment
  fn fs_main(
    @builtin(position) position : vec4<f32>,
    @location(0) positionInQuad : vec2<f32>,
    @location(1) @interpolate(flat) color : vec3<f32>,
    ) -> @location(0) vec4<f32> {
    // Calculate the normalized distance from this fragment to the quad center.
    var distFromCenter = length(positionInQuad);

    // Discard fragments that are outside the circle.
    if (distFromCenter > 1.0) {
      discard;
    }

    var intensity = 1.0 - distFromCenter;
    return vec4<f32>(intensity*color, 1.0);
  }
);
// clang-format on
