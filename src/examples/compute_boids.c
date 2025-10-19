#include "webgpu/wgpu_common.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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
  UNUSED_FUNCTION(update_sim_params);

  if (wgpu_context) {
    init_vertices(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipelines(wgpu_context);
    state.initialized = 1;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
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

  /* Get command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Update frame count */
  ++state.frame_index;

  return EXIT_SUCCESS;
}
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

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

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Compute Boids",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
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
