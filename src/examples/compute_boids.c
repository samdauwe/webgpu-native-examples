#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Boids
 *
 * Flocking boids example with gpu compute update pass.
 * Adapted from:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/computeBoids
 *
 * A GPU compute particle simulation that mimics the flocking behavior of birds.
 * A compute shader updates two ping-pong buffers which store particle data. The
 * data is used to draw instanced particles.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/computeBoids
 * https://github.com/gfx-rs/wgpu-rs/tree/master/examples/boids
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* sprite_vertex_shader_wgsl;
static const char* sprite_fragment_shader_wgsl;
static const char* update_sprites_compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Boids example
 * -------------------------------------------------------------------------- */

/* Number of boid particles to simulate */
#define NUM_PARTICLES (1500u)

/* Number of single-particle calculations (invocations) in each gpu work group
 */
#define PARTICLES_PER_GROUP (64u)

/* Number for simumation parameters */
#define SIM_PARAMS_COUNT (7u)

/* State struct */
static struct {
  WGPUBuffer sim_param_buffer; /* Simulation Parameter Buffer */
  WGPUBuffer particle_buffers[2];
  WGPUBuffer sprite_vertex_buffer;
  struct {
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
  } compute;
  WGPUBindGroup particle_bind_groups[2];
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  uint32_t work_group_count;
  uint64_t frame_index;
  struct {
    float delta_t;        /* deltaT */
    float rule1_distance; /* rule1Distance */
    float rule2_distance; /* rule2Distance */
    float rule3_distance; /* rule3Distance */
    float rule1_scale;    /* rule1Scale */
    float rule2_scale;    /* rule2Scale */
    float rule3_scale;    /* rule3Scale */
  } sim_param_data;
  struct {
    const char* label;
    float* param_ref;
  } sim_params_mappings[SIM_PARAMS_COUNT];
  bool initialized;
  /* Timing for ImGui */
  uint64_t last_frame_time;
  bool sim_params_changed;
  /* Timestamp query support for performance measurement */
  bool has_timestamp_query;
  WGPUQuerySet query_set;
  WGPUBuffer resolve_buffer;
  /* Pool of spare buffers for reading back timestamps */
  WGPUBuffer spare_result_buffers[8];
  uint32_t spare_buffer_count;
  /* Performance statistics */
  double compute_pass_duration_sum;
  double render_pass_duration_sum;
  uint32_t timer_samples;
  /* Averaged performance stats for display */
  uint32_t avg_compute_microseconds;
  uint32_t avg_render_microseconds;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
  },
  .sim_param_data =  {
    .delta_t        = 0.04f,  /* deltaT */
    .rule1_distance = 0.10f,  /* rule1Distance */
    .rule2_distance = 0.025f, /* rule2Distance */
    .rule3_distance = 0.025f, /* rule3Distance */
    .rule1_scale    = 0.02f,  /* rule1Scale */
    .rule2_scale    = 0.05f,  /* rule2Scale */
    .rule3_scale    = 0.005f, /* rule3Scale */
  },
  .sim_params_mappings = {
    // clang-format off
    /* deltaT */
    { .label = "deltaT",        .param_ref = &state.sim_param_data.delta_t },
    /* rule1Distance */
    { .label = "rule1Distance", .param_ref = &state.sim_param_data.rule1_distance },
    /* rule2Distance */
    { .label = "rule2Distance", .param_ref = &state.sim_param_data.rule2_distance },
    /* rule3Distance */
    { .label = "rule3Distance", .param_ref = &state.sim_param_data.rule3_distance },
    /* rule1Scale */
    { .label = "rule1Scale",    .param_ref = &state.sim_param_data.rule1_scale },
    /* rule2Scale */
    { .label = "rule2Scale",    .param_ref = &state.sim_param_data.rule2_scale },
    /* rule3Scale */
    { .label = "rule3Scale",    .param_ref = &state.sim_param_data.rule3_scale },
    // clang-format on
  },
};

/* Intialize vertex buffers */
static void init_vertices(wgpu_context_t* wgpu_context)
{
  /* Buffer for the three 2d triangle vertices of each instance */
  // clang-format off
  const float vertex_buffer_data[6] = {
    -0.01f, -0.02f, 0.01f, /* */
    -0.02f,  0.00f, 0.02f, /* */
  };
  // clang-format on
  const uint32_t vertex_buffer_size
    = (uint32_t)(ARRAY_SIZE(vertex_buffer_data) * sizeof(float));
  state.sprite_vertex_buffer
    = wgpu_create_buffer_from_data(wgpu_context, vertex_buffer_data,
                                   vertex_buffer_size, WGPUBufferUsage_Vertex);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.sim_param_data),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = NUM_PARTICLES * 16,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = NUM_PARTICLES * 16,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Compute - Bind group layout"),
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  state.compute.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.compute.bind_group_layout != NULL);

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = STRVIEW("Compute - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.compute.bind_group_layout,
  };
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(state.compute.pipeline_layout != NULL)

  /* Render pipeline layout (with empty bind group layout) */
  WGPUPipelineLayoutDescriptor render_pipeline_layout_desc = {0};
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &render_pipeline_layout_desc);
  ASSERT(state.graphics.pipeline_layout != NULL);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Buffer for simulation parameters uniform */
  state.sim_param_buffer = wgpu_create_buffer_from_data(
    wgpu_context, &state.sim_param_data, sizeof(state.sim_param_data),
    WGPUBufferUsage_Uniform);

  /* Buffer for all particles data of type [(posx,posy,velx,vely),...] */
  float particle_data[NUM_PARTICLES * 4];
  memset(particle_data, 0.f, sizeof(particle_data));
  srand((unsigned int)time(NULL)); // randomize seed
  for (uint32_t i = 0; i < NUM_PARTICLES; i += 4) {
    const size_t chunk       = i * 4;
    particle_data[chunk + 0] = 2 * (random_float() - 0.5f);        /* posx */
    particle_data[chunk + 1] = 2 * (random_float() - 0.5f);        /* posy */
    particle_data[chunk + 2] = 2 * (random_float() - 0.5f) * 0.1f; /* velx */
    particle_data[chunk + 3] = 2 * (random_float() - 0.5f) * 0.1f; /* vely */
  }

  // Creates two buffers of particle data each of size NUM_PARTICLES the two
  // buffers alternate as dst and src for each frame
  for (uint32_t i = 0; i < 2; ++i) {
    state.particle_buffers[i] = wgpu_create_buffer_from_data(
      wgpu_context, &particle_data, sizeof(particle_data),
      WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage);
  }

  // Create two bind groups, one for each buffer as the src where the alternate
  // buffer is used as the dst
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.sim_param_buffer,
        .offset  = 0,
        .size    = sizeof(state.sim_param_data),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.particle_buffers[i],
        .offset  = 0,
        .size    = sizeof(particle_data),
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.particle_buffers[(i + 1) % 2],
        .offset  = 0,
        .size    = sizeof(particle_data), /* Bind to opposite buffer */
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Particle compute - Bind group layout"),
      .layout     = state.compute.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.particle_bind_groups[i]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.particle_bind_groups[i] != NULL);
  }

  /* Calculates number of work groups from PARTICLES_PER_GROUP constant */
  state.work_group_count
    = (uint32_t)ceilf((float)NUM_PARTICLES / (float)PARTICLES_PER_GROUP);
}

static void update_sim_params(wgpu_context_t* wgpu_context)
{
  wgpuQueueWriteBuffer(wgpu_context->queue, state.sim_param_buffer, 0,
                       &state.sim_param_data, sizeof(state.sim_param_data));
}
/* Callback for processing timestamp query results */
static void on_buffer_mapped(WGPUMapAsyncStatus status, WGPUStringView message,
                             void* userdata1, void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata2);

  if (status != WGPUMapAsyncStatus_Success) {
    return;
  }

  WGPUBuffer result_buffer = (WGPUBuffer)userdata1;
  const int64_t* times     = (const int64_t*)wgpuBufferGetConstMappedRange(
    result_buffer, 0, 4 * sizeof(int64_t));
  if (!times) {
    return;
  }

  /* Calculate pass durations in nanoseconds */
  int64_t compute_pass_duration = times[1] - times[0];
  int64_t render_pass_duration  = times[3] - times[2];

  /* In some cases timestamps may wrap around and produce negative values */
  /* These can safely be ignored */
  if (compute_pass_duration > 0 && render_pass_duration > 0) {
    state.compute_pass_duration_sum += (double)compute_pass_duration;
    state.render_pass_duration_sum += (double)render_pass_duration;
    state.timer_samples++;
  }

  wgpuBufferUnmap(result_buffer);

  /* Update display periodically (every 100 samples) */
  const uint32_t kNumTimerSamplesPerUpdate = 100;
  if (state.timer_samples >= kNumTimerSamplesPerUpdate) {
    /* Convert from nanoseconds to microseconds and calculate average */
    state.avg_compute_microseconds
      = (uint32_t)((state.compute_pass_duration_sum / state.timer_samples)
                   / 1000.0);
    state.avg_render_microseconds
      = (uint32_t)((state.render_pass_duration_sum / state.timer_samples)
                   / 1000.0);

    /* Reset accumulators */
    state.compute_pass_duration_sum = 0.0;
    state.render_pass_duration_sum  = 0.0;
    state.timer_samples             = 0;
  }

  /* Return buffer to spare pool */
  if (state.spare_buffer_count < 8) {
    state.spare_result_buffers[state.spare_buffer_count++] = result_buffer;
  }
  else {
    /* Pool is full, release the buffer */
    wgpuBufferRelease(result_buffer);
  }
}
/* Create the compute & graphics pipelines */
static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline */
  {
    /* Compute shader */
    WGPUShaderModule comp_shader_module = wgpu_create_shader_module(
      wgpu_context->device, update_sprites_compute_shader_wgsl);

    /* Create compute pipeline */
    state.compute.pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Compute boids - Compute pipeline"),
        .layout  = state.compute.pipeline_layout,
        .compute = {
          .module     = comp_shader_module,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(state.compute.pipeline != NULL);

    /* Partial cleanup */
    wgpuShaderModuleRelease(comp_shader_module);
  }

  /* Graphics pipeline */
  {
    WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
      wgpu_context->device, sprite_vertex_shader_wgsl);
    WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
      wgpu_context->device, sprite_fragment_shader_wgsl);

    /* Vertex state */
    WGPUVertexAttribute vert_buff_attrs_0[2] = {
      [0] = (WGPUVertexAttribute) {
        /* Attribute location 0: instance position */
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x2,
      },
      [1] = (WGPUVertexAttribute) {
        /* Attribute location 1: instance velocity */
        .shaderLocation = 1,
        .offset         = 2 * 4,
        .format         = WGPUVertexFormat_Float32x2,
      },
    };
    WGPUVertexAttribute vert_buff_attrs_1 = {
      /* Attribute location 2: vertex positions */
      .shaderLocation = 2,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    WGPUVertexBufferLayout vert_buf[2] = {
      [0] = (WGPUVertexBufferLayout) {
        /* Instanced particles buffer */
        .arrayStride    = 4 * 4,
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = (uint32_t)ARRAY_SIZE(vert_buff_attrs_0),
        .attributes     = vert_buff_attrs_0,
      },
      [1] = (WGPUVertexBufferLayout) {
        /* vertex buffer */
        .arrayStride    = 2 * 4,
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &vert_buff_attrs_1,
      },
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Compute boids - Render pipeline"),
      .layout = state.graphics.pipeline_layout,
      .vertex = {
        .module      = vert_shader_module,
        .entryPoint  = STRVIEW("vert_main"),
        .bufferCount = 2,
        .buffers     = vert_buf,
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("frag_main"),
        .module      = frag_shader_module,
        .targetCount = 1,
        .targets = &(WGPUColorTargetState) {
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Back,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    state.graphics.pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(state.graphics.pipeline != NULL);

    wgpuShaderModuleRelease(vert_shader_module);
    wgpuShaderModuleRelease(frag_shader_module);
  }
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    /* Initialize sokol_time */
    stm_setup();

    /* Check for timestamp query support */
    state.has_timestamp_query
      = wgpuAdapterHasFeature(wgpu_context->adapter,
                              WGPUFeatureName_TimestampQuery)
        && wgpuDeviceHasFeature(wgpu_context->device,
                                WGPUFeatureName_TimestampQuery);

    if (state.has_timestamp_query) {
      /* Create query set for timestamps (4 queries: compute begin/end,
       * render begin/end) */
      state.query_set = wgpuDeviceCreateQuerySet(
        wgpu_context->device, &(WGPUQuerySetDescriptor){
                                .label = STRVIEW("Timestamp query set"),
                                .type  = WGPUQueryType_Timestamp,
                                .count = 4,
                              });

      /* Create resolve buffer for timestamp results */
      state.resolve_buffer = wgpuDeviceCreateBuffer(
        wgpu_context->device,
        &(WGPUBufferDescriptor){
          .label = STRVIEW("Timestamp resolve buffer"),
          .size  = 4 * sizeof(int64_t),
          .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
        });
    }

    init_vertices(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipelines(wgpu_context);

    /* Initialize ImGui overlay */
    imgui_overlay_init(wgpu_context);

    state.initialized = 1;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  /* Set initial window size with content-aware padding */
  igSetNextWindowSize((ImVec2){320.0f, 0.0f}, ImGuiCond_FirstUseEver);

  /* Build GUI - similar to TypeScript version's dat.gui */
  /* Use AlwaysAutoResize flag to adapt to content size dynamically */
  igBegin("Boid Simulation Parameters", NULL,
          ImGuiWindowFlags_AlwaysAutoResize);

  state.sim_params_changed = false;

  if (imgui_overlay_slider_float("deltaT", &state.sim_param_data.delta_t, 0.0f,
                                 0.1f, "%.4f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float("rule1Distance",
                                 &state.sim_param_data.rule1_distance, 0.0f,
                                 0.5f, "%.3f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float("rule2Distance",
                                 &state.sim_param_data.rule2_distance, 0.0f,
                                 0.1f, "%.3f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float("rule3Distance",
                                 &state.sim_param_data.rule3_distance, 0.0f,
                                 0.1f, "%.3f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float(
        "rule1Scale", &state.sim_param_data.rule1_scale, 0.0f, 0.1f, "%.4f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float(
        "rule2Scale", &state.sim_param_data.rule2_scale, 0.0f, 0.2f, "%.4f")) {
    state.sim_params_changed = true;
  }
  if (imgui_overlay_slider_float(
        "rule3Scale", &state.sim_param_data.rule3_scale, 0.0f, 0.1f, "%.4f")) {
    state.sim_params_changed = true;
  }

  /* Separator for performance statistics section */
  igSeparator();
  igSpacing();
  igText("Performance Statistics");
  igSeparator();

  /* Display performance stats if timestamp queries are supported */
  if (state.has_timestamp_query) {
    igText("avg compute pass duration: %u µs", state.avg_compute_microseconds);
    igText("avg render pass duration:  %u µs", state.avg_render_microseconds);
    igText("spare readback buffers:    %u", state.spare_buffer_count);
  }
  else {
    igTextDisabled("Timestamp queries not supported");
  }

  igEnd();

  /* Update simulation parameters if changed */
  if (state.sim_params_changed) {
    update_sim_params(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI controls */
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    WGPUComputePassDescriptor compute_pass_desc = {
      .label = STRVIEW("Compute pass"),
    };

    /* Add timestamp writes if supported */
    WGPUPassTimestampWrites timestamp_writes = {0};
    if (state.has_timestamp_query) {
      timestamp_writes.querySet                  = state.query_set;
      timestamp_writes.beginningOfPassWriteIndex = 0;
      timestamp_writes.endOfPassWriteIndex       = 1;
      compute_pass_desc.timestampWrites          = &timestamp_writes;
    }

    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, &compute_pass_desc);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(
      cpass_enc, 0, state.particle_bind_groups[state.frame_index % 2], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass_enc, state.work_group_count,
                                             1, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* Render pass */
  {
    /* Add timestamp writes if supported */
    WGPUPassTimestampWrites timestamp_writes = {0};
    if (state.has_timestamp_query) {
      timestamp_writes.querySet                    = state.query_set;
      timestamp_writes.beginningOfPassWriteIndex   = 2;
      timestamp_writes.endOfPassWriteIndex         = 3;
      state.render_pass_descriptor.timestampWrites = &timestamp_writes;
    }

    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    /* Render dst particles */
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0, state.particle_buffers[(state.frame_index + 1) % 2], 0,
      WGPU_WHOLE_SIZE);
    /* The three instance-local vertices */
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 1, state.sprite_vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rpass_enc, 3, NUM_PARTICLES, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  /* Resolve timestamp queries if supported */
  WGPUBuffer result_buffer = NULL;
  if (state.has_timestamp_query) {
    /* Get a spare buffer from the pool or create a new one */
    if (state.spare_buffer_count > 0) {
      result_buffer = state.spare_result_buffers[--state.spare_buffer_count];
    }
    else {
      result_buffer = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                  .label = STRVIEW("Timestamp result buffer"),
                  .size  = 4 * sizeof(int64_t),
                  .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
                });
    }

    /* Resolve query results into resolve buffer, then copy to result buffer */
    wgpuCommandEncoderResolveQuerySet(cmd_enc, state.query_set, 0, 4,
                                      state.resolve_buffer, 0);
    wgpuCommandEncoderCopyBufferToBuffer(cmd_enc, state.resolve_buffer, 0,
                                         result_buffer, 0, 4 * sizeof(int64_t));
  }

  /* Get command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Map the result buffer for reading back timestamps */
  if (state.has_timestamp_query && result_buffer) {
    wgpuBufferMapAsync(result_buffer, WGPUMapMode_Read, 0, 4 * sizeof(int64_t),
                       (WGPUBufferMapCallbackInfo){
                         .mode      = WGPUCallbackMode_AllowProcessEvents,
                         .callback  = on_buffer_mapped,
                         .userdata1 = result_buffer,
                       });
  }

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  /* Update frame count */
  ++state.frame_index;

  return EXIT_SUCCESS;
}
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown ImGui overlay */
  imgui_overlay_shutdown();

  /* Clean up timestamp query resources */
  if (state.has_timestamp_query) {
    WGPU_RELEASE_RESOURCE(QuerySet, state.query_set)
    WGPU_RELEASE_RESOURCE(Buffer, state.resolve_buffer)
    for (uint32_t i = 0; i < state.spare_buffer_count; ++i) {
      WGPU_RELEASE_RESOURCE(Buffer, state.spare_result_buffers[i])
    }
  }

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bind_groups[1])
  WGPU_RELEASE_RESOURCE(Buffer, state.sim_param_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.particle_buffers[0])
  WGPU_RELEASE_RESOURCE(Buffer, state.particle_buffers[1])
  WGPU_RELEASE_RESOURCE(Buffer, state.sprite_vertex_buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)
}

/**
 * @brief Input event callback for ImGui interaction
 */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  imgui_overlay_handle_input(wgpu_context, event);
}

int main(void)
{
  /* Request timestamp query feature for performance measurement */
  static const WGPUFeatureName required_features[1]
    = {WGPUFeatureName_TimestampQuery};

  wgpu_start(&(wgpu_desc_t){
    .title                  = "Compute Boids",
    .init_cb                = init,
    .frame_cb               = frame,
    .input_event_cb         = input_event_cb,
    .shutdown_cb            = shutdown,
    .required_features      = required_features,
    .required_feature_count = 1,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* sprite_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(4) color : vec4<f32>,
  }

  @vertex
  fn vert_main(
    @location(0) a_particlePos : vec2<f32>,
    @location(1) a_particleVel : vec2<f32>,
    @location(2) a_pos : vec2<f32>
  ) -> VertexOutput {
    let angle = -atan2(a_particleVel.x, a_particleVel.y);
    let pos = vec2(
      (a_pos.x * cos(angle)) - (a_pos.y * sin(angle)),
      (a_pos.x * sin(angle)) + (a_pos.y * cos(angle))
    );

    var output : VertexOutput;
    output.position = vec4(pos + a_particlePos, 0.0, 1.0);
    output.color = vec4(
      1.0 - sin(angle + 1.0) - a_particleVel.y,
      pos.x * 100.0 - a_particleVel.y + 0.1,
      a_particleVel.x + cos(angle + 0.5),
      1.0);
    return output;
  }
);

static const char* sprite_fragment_shader_wgsl = CODE(
  @fragment
  fn frag_main(@location(4) color : vec4<f32>) -> @location(0) vec4<f32> {
    return color;
  }
);

static const char* update_sprites_compute_shader_wgsl = CODE(
  struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
  }
  struct SimParams {
    deltaT : f32,
    rule1Distance : f32,
    rule2Distance : f32,
    rule3Distance : f32,
    rule1Scale : f32,
    rule2Scale : f32,
    rule3Scale : f32,
  }
  struct Particles {
    particles : array<Particle>,
  }
  @binding(0) @group(0) var<uniform> params : SimParams;
  @binding(1) @group(0) var<storage, read> particlesA : Particles;
  @binding(2) @group(0) var<storage, read_write> particlesB : Particles;

  // https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var index = GlobalInvocationID.x;

    var vPos = particlesA.particles[index].pos;
    var vVel = particlesA.particles[index].vel;
    var cMass = vec2(0.0);
    var cVel = vec2(0.0);
    var colVel = vec2(0.0);
    var cMassCount = 0u;
    var cVelCount = 0u;
    var pos : vec2<f32>;
    var vel : vec2<f32>;

    for (var i = 0u; i < arrayLength(&particlesA.particles); i++) {
      if (i == index) {
        continue;
      }

      pos = particlesA.particles[i].pos.xy;
      vel = particlesA.particles[i].vel.xy;
      if (distance(pos, vPos) < params.rule1Distance) {
        cMass += pos;
        cMassCount++;
      }
      if (distance(pos, vPos) < params.rule2Distance) {
        colVel -= pos - vPos;
      }
      if (distance(pos, vPos) < params.rule3Distance) {
        cVel += vel;
        cVelCount++;
      }
    }
    if (cMassCount > 0) {
      cMass = (cMass / vec2(f32(cMassCount))) - vPos;
    }
    if (cVelCount > 0) {
      cVel /= f32(cVelCount);
    }
    vVel += (cMass * params.rule1Scale) + (colVel * params.rule2Scale) + (cVel * params.rule3Scale);

    // clamp velocity for a more pleasing simulation
    vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);
    // kinematic update
    vPos = vPos + (vVel * params.deltaT);
    // Wrap around boundary
    if (vPos.x < -1.0) {
      vPos.x = 1.0;
    }
    if (vPos.x > 1.0) {
      vPos.x = -1.0;
    }
    if (vPos.y < -1.0) {
      vPos.y = 1.0;
    }
    if (vPos.y > 1.0) {
      vPos.y = -1.0;
    }
    // Write back
    particlesB.particles[index].pos = vPos;
    particlesB.particles[index].vel = vVel;
  }
);
// clang-format on
