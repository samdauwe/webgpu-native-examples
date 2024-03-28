#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

// Number of boid particles to simulate
static const uint32_t NUM_PARTICLES = 1500u;

// Number of single-particle calculations (invocations) in each gpu work group
static const uint32_t PARTICLES_PER_GROUP = 64u;

// Sim parameters
static struct sim_params_t {
  float delta_t;        /* deltaT */
  float rule1_distance; /* rule1Distance */
  float rule2_distance; /* rule2Distance */
  float rule3_distance; /* rule3Distance */
  float rule1_scale;    /* rule1Scale */
  float rule2_scale;    /* rule2Scale */
  float rule3_scale;    /* rule3Scale */
} sim_param_data = {
  .delta_t        = 0.04f,  /* deltaT */
  .rule1_distance = 0.10f,  /* rule1Distance */
  .rule2_distance = 0.025f, /* rule2Distance */
  .rule3_distance = 0.025f, /* rule3Distance */
  .rule1_scale    = 0.02f,  /* rule1Scale */
  .rule2_scale    = 0.05f,  /* rule2Scale */
  .rule3_scale    = 0.005f, /* rule3Scale */
};

// Used to configure Sim parameters in GUI
static const uint8_t sim_params_count = 7;
static struct {
  const char* label;
  float* param_ref;
} sim_params_mappings[7] = {
  // clang-format off
  /* deltaT */
  { .label = "deltaT",        .param_ref = &sim_param_data.delta_t },
  /* rule1Distance */
  { .label = "rule1Distance", .param_ref = &sim_param_data.rule1_distance },
  /* rule2Distance */
  { .label = "rule2Distance", .param_ref = &sim_param_data.rule2_distance },
  /* rule3Distance */
  { .label = "rule3Distance", .param_ref = &sim_param_data.rule3_distance },
  /* rule1Scale */
  { .label = "rule1Scale",    .param_ref = &sim_param_data.rule1_scale },
  /* rule2Scale */
  { .label = "rule2Scale",    .param_ref = &sim_param_data.rule2_scale },
  /* rule3Scale */
  { .label = "rule3Scale",    .param_ref = &sim_param_data.rule3_scale },
  // clang-format on
};

// WebGPU buffers
static WGPUBuffer sim_param_buffer     = NULL; /* Simulation Parameter Buffer */
static WGPUBuffer particle_buffers[2]  = {0};
static WGPUBuffer sprite_vertex_buffer = NULL;

// The pipeline layouts
static WGPUPipelineLayout compute_pipeline_layout = NULL;
static WGPUPipelineLayout render_pipeline_layout  = NULL;

// Pipelines
static WGPUComputePipeline compute_pipeline = NULL;
static WGPURenderPipeline render_pipeline   = NULL;

// Bind groups and layouts
static WGPUBindGroup particle_bind_groups[2]         = {0};
static WGPUBindGroupLayout compute_bind_group_layout = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Compute Boids";
static bool prepared             = false;
static uint32_t work_group_count = 0;

// Prepare vertex buffers
static void prepare_vertices(wgpu_context_t* wgpu_context)
{
  // Buffer for the three 2d triangle vertices of each instance
  // clang-format off
  const float vertex_buffer_data[6] = {
    -0.01f, -0.02f, 0.01f, //
    -0.02f,  0.00f, 0.02f, //
  };
  // clang-format on
  const uint32_t vertex_buffer_size
    = (uint32_t)(ARRAY_SIZE(vertex_buffer_data) * sizeof(float));
  sprite_vertex_buffer
    = wgpu_create_buffer_from_data(wgpu_context, vertex_buffer_data,
                                   vertex_buffer_size, WGPUBufferUsage_Vertex);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(sim_param_data),
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
    .label      = "Compute bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  compute_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(compute_bind_group_layout != NULL);

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = "Compute pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &compute_bind_group_layout,
  };
  compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(compute_pipeline_layout != NULL)

  /* Render pipeline layout (with empty bind group layout) */
  WGPUPipelineLayoutDescriptor render_pipeline_layout_desc = {0};
  render_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &render_pipeline_layout_desc);
  ASSERT(render_pipeline_layout != NULL);
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
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
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

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Buffer for simulation parameters uniform
  sim_param_buffer = wgpu_create_buffer_from_data(
    context->wgpu_context, &sim_param_data, sizeof(sim_param_data),
    WGPUBufferUsage_Uniform);

  // Buffer for all particles data of type [(posx,posy,velx,vely),...]
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
    particle_buffers[i] = wgpu_create_buffer_from_data(
      context->wgpu_context, &particle_data, sizeof(particle_data),
      WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage);
  }

  // Create two bind groups, one for each buffer as the src where the alternate
  // buffer is used as the dst
  for (uint32_t i = 0; i < 2; ++i) {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = sim_param_buffer,
        .offset  = 0,
        .size    = sizeof(sim_param_data),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = particle_buffers[i],
        .offset  = 0,
        .size    = sizeof(particle_data),
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = particle_buffers[(i + 1) % 2],
        .offset  = 0,
        .size    = sizeof(particle_data), /* bind to opposite buffer */
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Particle compute bind group layout",
      .layout     = compute_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    particle_bind_groups[i]
      = wgpuDeviceCreateBindGroup(context->wgpu_context->device, &bg_desc);
    ASSERT(particle_bind_groups[i] != NULL);
  }

  // Calculates number of work groups from PARTICLES_PER_GROUP constant
  work_group_count
    = (uint32_t)ceilf((float)NUM_PARTICLES / (float)PARTICLES_PER_GROUP);
}

static void update_sim_params(wgpu_context_t* wgpu_context)
{
  wgpu_queue_write_buffer(wgpu_context, sim_param_buffer, 0, &sim_param_data,
                          sizeof(sim_param_data));
}

// Create the compute & graphics pipelines
static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
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

  // Vertex state
  WGPUVertexAttribute vert_buff_attrs_0[2] = {
    [0] = (WGPUVertexAttribute) {
      // Attribute location 0: instance position
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    },
    [1] = (WGPUVertexAttribute) {
      // Attribute location 1: instance velocity
      .shaderLocation = 1,
      .offset         = 2 * 4,
      .format         = WGPUVertexFormat_Float32x2,
    },
  };
  WGPUVertexAttribute vert_buff_attrs_1 = {
    // Attribute location 2: vertex positions
    .shaderLocation = 2,
    .offset         = 0,
    .format         = WGPUVertexFormat_Float32x2,
  };
  WGPUVertexBufferLayout vert_buf[2] = {
    [0] = (WGPUVertexBufferLayout) {
      // Instanced particles buffer
      .arrayStride    = 4 * 4,
      .stepMode       = WGPUVertexStepMode_Instance,
      .attributeCount = (uint32_t)ARRAY_SIZE(vert_buff_attrs_0),
      .attributes     = vert_buff_attrs_0,
    },
    [1] = (WGPUVertexBufferLayout) {
      // vertex buffer
      .arrayStride    = 2 * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vert_buff_attrs_1,
    },
  };

  // Compute shader
  wgpu_shader_t boids_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Compute shader WGSL
                    .label            = "Update sprites compute shader",
                    .wgsl_code.source = update_sprites_compute_shader_wgsl,
                    .entry            = "main",
                  });

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader WGSL
              .label            = "Sprite vertex shader WGSL",
              .wgsl_code.source = sprite_vertex_shader_wgsl,
              .entry            = "vert_main",
            },
            .buffer_count = 2,
            .buffers      = vert_buf,
          });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader WGSL
              .label            = "Sprite fragment shader WGSL",
              .wgsl_code.source = sprite_fragment_shader_wgsl,
              .entry            = "frag_main",
            },
            .target_count = 1,
            .targets = &color_target_state,
          });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Compute pipeline
  compute_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Compute boids - compute pipeline",
      .layout  = compute_pipeline_layout,
      .compute = boids_comp_shader.programmable_stage_descriptor,
    });
  ASSERT(compute_pipeline != NULL);

  // Create rendering pipeline using the specified states
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Compute boids - render pipeline",
                            .layout      = render_pipeline_layout,
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(render_pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  wgpu_shader_release(&boids_comp_shader);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertices(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_uniform_buffers(context);
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
    for (uint8_t i = 0; i < sim_params_count; ++i) {
      if (imgui_overlay_input_float(
            context->imgui_overlay, sim_params_mappings[i].label,
            sim_params_mappings[i].param_ref, 0.01, "%.3f")) {
        update_sim_params(context->wgpu_context);
      }
    }
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
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      wgpu_context->cpass_enc, 0,
      particle_bind_groups[context->frame.index % 2], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
                                             work_group_count, 1, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Render pass */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);
    // render dst particles
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0,
      particle_buffers[(context->frame.index + 1) % 2], 0, WGPU_WHOLE_SIZE);
    // the three instance-local vertices
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 1, sprite_vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, NUM_PARTICLES, 0, 0);
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

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, render_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, particle_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, particle_bind_groups[1])
  WGPU_RELEASE_RESOURCE(Buffer, sim_param_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, particle_buffers[0])
  WGPU_RELEASE_RESOURCE(Buffer, particle_buffers[1])
  WGPU_RELEASE_RESOURCE(Buffer, sprite_vertex_buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute_pipeline)
}

void example_compute_boids(int argc, char* argv[])
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
