#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particle System
 *
 * Attraction based 2D GPU particle system using compute shaders. Particle data
 * is stored in a shader storage buffer.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/computeparticles/computeparticles.cpp
 * https://github.com/gpuweb/gpuweb/issues/332
 * -------------------------------------------------------------------------- */

#define PARTICLE_COUNT (256u * 1024u)

static float timer           = 0.0f;
static float animStart       = 20.0f;
static bool attach_to_cursor = false;

static struct {
  texture_t particle;
  texture_t gradient;
} textures = {0};

// Resources for the graphics part of the example
static struct {
  WGPUBindGroupLayout bind_group_layout; // Particle system rendering shader
                                         // binding layout
  WGPUBindGroup bind_group; // Particle system rendering shader bindings
  WGPUPipelineLayout pipeline_layout; // Layout of the graphics pipeline
  WGPURenderPipeline pipeline;        // Particle rendering pipeline
} graphics = {0};

// Resources for the compute part of the example
static struct {
  wgpu_buffer_t storage_buffer; // (Shader) storage buffer object containing the
                                // particles
  wgpu_buffer_t uniform_buffer; // Uniform buffer object containing particle
                                // system parameters
  WGPUBindGroupLayout bind_group_layout; // Compute shader binding layout
  WGPUBindGroup bind_group;              // Compute shader bindings
  WGPUPipelineLayout pipeline_layout;    // Layout of the compute pipeline
  WGPUComputePipeline pipeline; // Compute pipeline for updating particle
                                // positions
  struct compute_ubo_t {        // Compute shader uniform block object
    float delta_t;              // Frame delta time
    float dest_x;               // x position of the attractor
    float dest_y;               // y position of the attractor
    int32_t particle_count;
  } ubo;
} compute = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// SSBO particle declaration
typedef struct particle_t {
  vec2 pos;          // Particle position
  vec2 vel;          // Particle velocity
  vec4 gradient_pos; // Texture coordinates for the gradient ramp map
} particle_t;

// Other variables
static const char* example_title = "Compute Shader Particle System";
static bool prepared             = false;

static void load_assets(wgpu_context_t* wgpu_context)
{
  textures.particle = wgpu_create_texture_from_file(
    wgpu_context, "textures/particle01_rgba.ktx", NULL);
  textures.gradient = wgpu_create_texture_from_file(
    wgpu_context, "textures/particle_gradient_rgba.ktx", NULL);
}

/* Setup and fill the compute shader storage buffers containing the particles */
static void prepare_storage_buffers(wgpu_context_t* wgpu_context)
{
  /* Initial particle positions */
  static particle_t particle_buffer[PARTICLE_COUNT] = {0};
  for (uint32_t i = 0; i < (uint32_t)PARTICLE_COUNT; ++i) {
    particle_buffer[i] = (particle_t){
      .pos = {
        random_float_min_max(-1.0f, 1.0f), /* x */
        random_float_min_max(-1.0f, 1.0f)  /* y */
       },
      .vel = GLM_VEC2_ZERO_INIT,
      .gradient_pos = GLM_VEC4_ZERO_INIT,
    };
    particle_buffer[i].gradient_pos[0] = particle_buffer[i].pos[0] / 2.0f;
  }

  /* Staging */
  /* SSBO won't be changed on the host after upload */
  compute.storage_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute shader - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = PARTICLE_COUNT * sizeof(particle_t),
                    .initial.data = particle_buffer,
                  });
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  /* Update uniform buffer data */
  compute.ubo.delta_t = context->frame_timer * 2.5f;
  if (!attach_to_cursor) {
    compute.ubo.dest_x = sin(glm_rad(timer * 360.0f)) * 0.75f;
    compute.ubo.dest_y = 0.0f;
  }
  else {
    const float width  = (float)wgpu_context->surface.width;
    const float height = (float)wgpu_context->surface.height;

    const float normalized_mx
      = (context->mouse_position[0] - (width / 2.0f)) / (width / 2.0f);
    const float normalized_my
      = ((height / 2.0f) - context->mouse_position[1]) / (height / 2.0f);
    compute.ubo.dest_x = normalized_mx;
    compute.ubo.dest_y = normalized_my;
  }

  /* Map uniform buffer and update it */
  wgpu_queue_write_buffer(wgpu_context, compute.uniform_buffer.buffer, 0,
                          &compute.ubo, compute.uniform_buffer.size);
}

/* Prepare and initialize uniform buffer containing shader uniforms */
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  /* Initialize the uniform buffer block */
  compute.ubo.particle_count = PARTICLE_COUNT;

  /* Compute shader uniform buffer block */
  compute.uniform_buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Compute shader - Uniform buffer",
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(compute.ubo),
    });

  update_uniform_buffers(context);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.025f,
        .g = 0.025f,
        .b = 0.025f,
        .a = 1.0f,
      },
  };

  /* Depth attachment */
  wgpu_setup_deph_stencil(wgpu_context, NULL);

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Particle color map texture */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1 : Particle color map sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2 : Particle gradient ramp texture */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      /* Binding 3 : Particle gradient ramp sampler */
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Graphics - Bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  graphics.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(graphics.bind_group_layout != NULL)

  // Create the pipeline layout
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = "Graphics - Pipeline layout",
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &graphics.bind_group_layout,
  };
  graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &pipeline_layout_desc);
  ASSERT(graphics.pipeline_layout != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_PointList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  blend_state.color.srcFactor             = WGPUBlendFactor_One;
  blend_state.color.dstFactor             = WGPUBlendFactor_One;
  blend_state.alpha.srcFactor             = WGPUBlendFactor_SrcAlpha;
  blend_state.alpha.dstFactor             = WGPUBlendFactor_DstAlpha;
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = false,
    });

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    particle, sizeof(particle_t),
    // Attribute descriptions
    // Describes memory layout and shader positions
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                       offsetof(particle_t, pos)),
    // Attribute location 1: Gradient position
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                       offsetof(particle_t, gradient_pos)))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .label = "Particle - Vertex shader SPIR-V",
                  .file  = "shaders/compute_particles/particle.vert.spv",
                },
                .buffer_count = 1,
                .buffers      = &particle_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .label = "Particle - Fragment shader SPIR-V",
                  .file  = "shaders/compute_particles/particle.frag.spv",
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
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Particle - Render pipeline",
                            .layout       = graphics.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(graphics.pipeline != NULL)

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Particle color map texture */
      .binding     = 0,
      .textureView = textures.particle.view,
    },
    [1] = (WGPUBindGroupEntry) {
       /* Binding 1 : Particle color map sampler */
      .binding = 1,
      .sampler = textures.particle.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
       /* Binding 2 : Particle gradient ramp texture */
      .binding     = 2,
      .textureView = textures.gradient.view,
    },
    [3] = (WGPUBindGroupEntry) {
      /* Binding 3 : Particle gradient ramp sampler */
      .binding = 3,
      .sampler = textures.gradient.sampler,
    }
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Particle color and gradient - Bind group",
    .layout     = graphics.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };

  graphics.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(graphics.bind_group != NULL)
}

static void prepare_graphics(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  prepare_storage_buffers(wgpu_context);
  prepare_uniform_buffers(context);
  setup_pipeline_layout(wgpu_context);
  prepare_pipelines(wgpu_context);
  setup_bind_groups(wgpu_context);
  setup_render_pass(wgpu_context);
}

static void prepare_compute(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0 : Particle position storage buffer
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = PARTICLE_COUNT * sizeof(particle_t),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1 : Uniform buffer
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(compute.ubo),
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
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &compute.bind_group_layout,
  };
  compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &compute_pipeline_layout_desc);
  ASSERT(compute.pipeline_layout != NULL)

  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Particle position storage buffer
      .binding = 0,
      .buffer  = compute.storage_buffer.buffer,
      .size    = compute.storage_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
     // Binding 1 : Uniform buffer
      .binding = 1,
      .buffer  = compute.uniform_buffer.buffer,
      .offset  = 0,
      .size    = compute.uniform_buffer.size,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = compute.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  compute.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);

  /* Compute shader */
  wgpu_shader_t particle_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Compute shader SPIR-V
                    .label = "Particle - Compute shader SPIR-V",
                    .file  = "shaders/compute_particles/particle.comp.spv",
                  });

  /* Create pipeline */
  compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Particle - Compute pipeline",
      .layout  = compute.pipeline_layout,
      .compute = particle_comp_shader.programmable_stage_descriptor,
    });

  /* Partial clean-up */
  wgpu_shader_release(&particle_comp_shader);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    load_assets(context->wgpu_context);
    prepare_graphics(context);
    prepare_compute(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Attach Attractor to Cursor",
                           &attach_to_cursor);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
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
                                       compute.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
                                             PARTICLE_COUNT / 256, 1, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  /* Render pass: Draw the particle system using the update vertex buffer */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      graphics.bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         compute.storage_buffer.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, PARTICLE_COUNT, 1, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  int result = example_draw(context);

  if (!attach_to_cursor) {
    if (animStart > 0.0f) {
      animStart -= context->frame_timer * 5.0f;
    }
    else if (animStart <= 0.0f) {
      timer += context->frame_timer * 0.04f;
      if (timer > 1.f) {
        timer = 0.f;
      }
    }
  }

  update_uniform_buffers(context);

  return result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  // Textures
  wgpu_destroy_texture(&textures.particle);
  wgpu_destroy_texture(&textures.gradient);

  // Graphics pipeline
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)

  // Compute pipeline
  WGPU_RELEASE_RESOURCE(Buffer, compute.storage_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline)
}

void example_compute_particles(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title  = example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
