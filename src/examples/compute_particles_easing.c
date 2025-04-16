#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particle Easing
 *
 * Particle system using compute shaders. Particle data is stored in a shader
 * storage buffer, particle movement is implemented using easing functions.
 *
 * Ref:
 * https://redcamel.github.io/webgpu/14_compute
 * https://github.com/redcamel/webgpu/tree/master/14_compute
 * -------------------------------------------------------------------------- */

#define PARTICLE_NUM 60000u
#define PROPERTY_NUM 40u

static struct {
  float time;
  float min_life;
  float max_life;
} sim_param_data = {
  .time     = 0.0f,     /* startTime time */
  .min_life = 2000.0f,  /* Min lifeRange  */
  .max_life = 10000.0f, /* Max lifeRange  */
};

// Vertex buffer and attributes
static struct wgpu_buffer_t vertices = {0};

// Particles data
static float initial_particle_data[PARTICLE_NUM * PROPERTY_NUM] = {0};

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout uniforms_bind_group_layout;
  WGPUBindGroup uniforms_bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
} graphics = {0};

// Resources for the compute part of the example
static struct {
  wgpu_buffer_t sim_param_buffer;
  wgpu_buffer_t particle_buffer;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup particle_bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
} compute = {0};

// Texture and sampler
static texture_t particle_texture = {0};

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1] = {0};
static WGPURenderPassDescriptor render_pass_desc                 = {0};

// Other variables
static const char* example_title = "Compute Shader Particle Easing";
static bool prepared             = false;

static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  const float t_scale                     = 0.005f;
  static const float vertex_buffer[6 * 6] = {
    // clang-format off
    -t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 0.0f, //
     t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 1.0f, //
    -t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 0.0f, //
    //
    -t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 0.0f, //
     t_scale, -t_scale, 0.0f, 1.0f, 0.0f, 1.0f, //
     t_scale,  t_scale, 0.0f, 1.0f, 1.0f, 1.0f, //
    // clang-format on
  };

  // Create vertex buffer
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particle - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });
}

static void prepare_particle_buffer(wgpu_context_t* wgpu_context)
{
  // Particle data
  const float current_time = platform_get_time();
  for (uint32_t i = 0; i < (uint32_t)PARTICLE_NUM; ++i) {
    const float life = random_float() * 8000.0f + 2000.0f;
    const float age  = random_float() * life;
    initial_particle_data[PROPERTY_NUM * i + 0]
      = current_time - age;                             // start time
    initial_particle_data[PROPERTY_NUM * i + 1] = life; // life
    // position
    initial_particle_data[PROPERTY_NUM * i + 4] = random_float() * 2 - 1; // x
    initial_particle_data[PROPERTY_NUM * i + 5] = random_float() * 2 - 1; // y
    initial_particle_data[PROPERTY_NUM * i + 6] = random_float() * 2 - 1; // z
    initial_particle_data[PROPERTY_NUM * i + 7] = 0.0f;
    // scale
    initial_particle_data[PROPERTY_NUM * i + 8]  = 0.0f; // scaleX
    initial_particle_data[PROPERTY_NUM * i + 9]  = 0.0f; // scaleY
    initial_particle_data[PROPERTY_NUM * i + 10] = 0.0f; // scaleZ
    initial_particle_data[PROPERTY_NUM * i + 11] = 0.0f;
    // x
    initial_particle_data[PROPERTY_NUM * i + 12] = 0.0f; // startValue
    initial_particle_data[PROPERTY_NUM * i + 13]
      = random_float() * 2.0f - 1.0f; // endValue
    initial_particle_data[PROPERTY_NUM * i + 14]
      = (int)(random_float() * 27.0f); // ease
    // y
    initial_particle_data[PROPERTY_NUM * i + 16] = 0.0f; // startValue
    initial_particle_data[PROPERTY_NUM * i + 17]
      = random_float() * 2.0f - 1.0f; // endValue
    initial_particle_data[PROPERTY_NUM * i + 18]
      = (int)(random_float() * 27.0f); // ease
    // z
    initial_particle_data[PROPERTY_NUM * i + 20] = 0.0f; // startValue
    initial_particle_data[PROPERTY_NUM * i + 21]
      = random_float() * 2.0f - 1.0f; // endValue
    initial_particle_data[PROPERTY_NUM * i + 22]
      = (int)(random_float() * 27.0f); // ease
    // scaleX
    const float t_scale                          = random_float() * 12.0f;
    initial_particle_data[PROPERTY_NUM * i + 24] = 0.0f;    // startValue
    initial_particle_data[PROPERTY_NUM * i + 25] = t_scale; // endValue
    initial_particle_data[PROPERTY_NUM * i + 26] = 0.0f;    // ease
    // scaleY
    initial_particle_data[PROPERTY_NUM * i + 28] = 0.0f;    // startValue
    initial_particle_data[PROPERTY_NUM * i + 29] = t_scale; // endValue
    initial_particle_data[PROPERTY_NUM * i + 30] = 0.0f;    // ease
    // scaleZ
    initial_particle_data[PROPERTY_NUM * i + 32] = 0.0f;    // startValue
    initial_particle_data[PROPERTY_NUM * i + 33] = t_scale; // endValue
    initial_particle_data[PROPERTY_NUM * i + 34] = 0.0f;    // ease
    // alpha
    initial_particle_data[PROPERTY_NUM * i + 36] = random_float(); // startValue
    initial_particle_data[PROPERTY_NUM * i + 37] = 0.0f;           // endValue
    initial_particle_data[PROPERTY_NUM * i + 38]
      = (int)(random_float() * 27.0f);                   // ease
    initial_particle_data[PROPERTY_NUM * i + 39] = 0.0f; // value
  }

  // Create vertex buffer
  compute.particle_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute particle - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = sizeof(initial_particle_data),
                    .initial.data = initial_particle_data,
                  });
}

static void prepare_compute(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0 : SimParams
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(sim_param_data),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1 : ParticlesA
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = sizeof(initial_particle_data),
      },
      .sampler = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Compute - Bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  compute.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(compute.bind_group_layout != NULL)

  WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
    .label                = "Compute - Pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &compute.bind_group_layout,
  };
  compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(compute.pipeline_layout != NULL)

  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : SimParams */
      .binding = 0,
      .buffer  = compute.sim_param_buffer.buffer,
      .size    =  compute.sim_param_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
     /* Binding 1 : Particles A */
      .binding = 1,
      .buffer  = compute.particle_buffer.buffer,
      .offset  = 0,
      .size    = compute.particle_buffer.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Compute pipeline - Bind group",
    .layout     = compute.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  compute.particle_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);

  /* Compute shader */
  wgpu_shader_t particle_comp_shader = wgpu_shader_create(
    wgpu_context,
    &(wgpu_shader_desc_t){
      /* Compute shader SPIR-V */
      .label = "Particle - Compute shader SPIR-V",
      .file  = "shaders/compute_particles_easing/particle.comp.spv",
    });

  /* Create pipeline */
  compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Particle - Compute pipeline",
      .layout  = compute.pipeline_layout,
      .compute = particle_comp_shader.programmable_stage_descriptor,
    });
  ASSERT(compute.pipeline != NULL);

  /* Partial clean-up */
  wgpu_shader_release(&particle_comp_shader);
}

static void prepare_particle_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/particle.png";
  particle_texture = wgpu_create_texture_from_file(wgpu_context, file, NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Uniforms bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : sampler */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : texture */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Uniforms bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  graphics.uniforms_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(graphics.uniforms_bind_group_layout != NULL)

  /* Create the pipeline layout */
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = "Render pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &graphics.uniforms_bind_group_layout,
  };
  graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &pipeline_layout_desc);
  ASSERT(graphics.pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
       /* Binding 0 : sampler */
      .binding = 0,
      .sampler = particle_texture.sampler,
    },
    [1] = (WGPUBindGroupEntry) {
       /* Binding 1 : texture view */
      .binding     = 1,
      .textureView = particle_texture.view,
    }
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Bind group",
    .layout     = graphics.uniforms_bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };

  graphics.uniforms_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(graphics.uniforms_bind_group != NULL)
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  sim_param_data.time += (1.0 / 60.f) * 1000.0f;

  /* Map uniform buffer and update it */
  wgpu_queue_write_buffer(wgpu_context, compute.sim_param_buffer.buffer, 0,
                          &sim_param_data, compute.sim_param_buffer.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Compute shader uniform buffer block */
  compute.sim_param_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Compute shader uniform buffer block",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(sim_param_data),
    });

  /* Update uniform buffer */
  update_uniform_buffers(context);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
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

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void prepare_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);
  {
    blend_state.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_state.color.dstFactor = WGPUBlendFactor_One;
    blend_state.color.operation = WGPUBlendOperation_Add;
  }
  {
    blend_state.alpha.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_state.alpha.dstFactor = WGPUBlendFactor_One;
    blend_state.alpha.operation = WGPUBlendOperation_Add;
  }
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = false,
    });

  /* Vertex buffer layout */
  WGPUVertexBufferLayout buffers[2] = {0};
  /* Instanced particles buffer */
  buffers[0].arrayStride              = PROPERTY_NUM * 4;
  buffers[0].stepMode                 = WGPUVertexStepMode_Instance;
  buffers[0].attributeCount           = 3;
  WGPUVertexAttribute attributes_0[3] = {0};
  {
    /* position */
    attributes_0[0] = (WGPUVertexAttribute){
      .shaderLocation = 0,
      .offset         = 4 * 4,
      .format         = WGPUVertexFormat_Float32x3,
    };
    /* scale */
    attributes_0[1] = (WGPUVertexAttribute){
      .shaderLocation = 1,
      .offset         = 8 * 4,
      .format         = WGPUVertexFormat_Float32x3,
    };
    /* alpha */
    attributes_0[2] = (WGPUVertexAttribute){
      .shaderLocation = 2,
      .offset         = 39 * 4,
      .format         = WGPUVertexFormat_Float32,
    };
  }
  buffers[0].attributes = attributes_0;
  /* vertex buffer */
  buffers[1].arrayStride              = 6 * 4;
  buffers[1].stepMode                 = WGPUVertexStepMode_Vertex;
  buffers[1].attributeCount           = 2;
  WGPUVertexAttribute attributes_1[2] = {0};
  {
    /* position*/
    attributes_1[0] = (WGPUVertexAttribute){
      .shaderLocation = 3,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x4,
    };
    /* scale*/
    attributes_1[1] = (WGPUVertexAttribute){
      .shaderLocation = 4,
      .offset         = 4 * 4,
      .format         = WGPUVertexFormat_Float32x2,
    };
  }
  buffers[1].attributes = attributes_1;

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Vertex shader SPIR-V */
                  .label = "Particle vertex shader SPIR-V",
                  .file  = "shaders/compute_particles_easing/particle.vert.spv",
                },
                .buffer_count = (uint32_t) ARRAY_SIZE(buffers),
                .buffers      = buffers,
              });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  /* Fragment shader SPIR-V */
                  .label = "Particle fragment shader SPIR-V",
                  .file  = "shaders/compute_particles_easing/particle.frag.spv",
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
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Particle render pipeline",
                            .layout       = graphics.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_particle_buffer(context->wgpu_context);
    prepare_compute(context->wgpu_context);
    prepare_particle_texture(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_graphics_pipeline(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Set target frame buffer */
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass: Compute particle movement */
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    /* Dispatch the compute job */
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute.particle_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
                                             PARTICLE_NUM, 1, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Render pass: Draw the particle system using the update vertex buffer */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      graphics.uniforms_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         compute.particle_buffer.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         vertices.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, PARTICLE_NUM, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL)
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

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  update_uniform_buffers(context);

  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  /* Textures */
  wgpu_destroy_texture(&particle_texture);

  /* Vertices */
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.uniforms_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.uniforms_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, compute.sim_param_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, compute.particle_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.particle_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline)
}

void example_compute_particles_easing(int argc, char* argv[])
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
