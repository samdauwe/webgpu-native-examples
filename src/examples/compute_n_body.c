#include "example_base.h"
#include "examples.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - N-body simulation
 *
 * N-body simulation based particle system with multiple attractors and
 * particle-to-particle interaction using two passes separating particle
 * movement calculation and final integration. Shared compute shader memory is
 * used to speed up compute calculations.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/computenbody/computenbody.cpp
 * -------------------------------------------------------------------------- */

#define PARTICLES_PER_ATTRACTOR 4 * 1024
#define NUMBER_OF_ATTRACTORS 6

static uint32_t num_particles = 0;

static struct {
  texture_t particle;
  texture_t gradient;
} textures;

// Resources for the graphics part of the example
static struct {
  WGPUBuffer uniform_buffer;             // Contains scene matrices
  WGPUBindGroupLayout bind_group_layout; // Particle system rendering shader
                                         // binding layout
  WGPUBindGroup bind_group; // Particle system rendering shader bindings
  WGPUPipelineLayout pipeline_layout; // Layout of the graphics pipeline
  WGPURenderPipeline pipeline;        // Particle rendering pipeline
  struct {
    mat4 projection;
    mat4 view;
    vec2 screen_dim;
  } ubo;
} graphics;

// Resources for the compute part of the example
static struct {
  WGPUBuffer storage_buffer; // (Shader) storage buffer object containing the
                             // particles
  WGPUBuffer uniform_buffer; // Uniform buffer object containing particle
                             // system parameters
  WGPUBindGroupLayout bind_group_layout;  // Compute shader binding layout
  WGPUBindGroup bind_group;               // Compute shader bindings
  WGPUPipelineLayout pipeline_layout;     // Layout of the compute pipeline
  WGPUComputePipeline pipeline_calculate; // Compute pipeline for N-Body
                                          // velocity calculation (1st pass)
  WGPUComputePipeline
    pipeline_integrate;  // Compute pipeline for euler integration (2nd pass)
  struct compute_ubo_t { // Compute shader uniform block object
    float delta_t;       // Frame delta time
    int32_t particle_count;
  } ubo;
} compute;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// SSBO particle declaration
typedef struct particle_t {
  vec4 pos; // xyz = position, w = mass
  vec4 vel; // xyz = velocity, w = gradient texture position
} particle_t;

// Other variables
static const char* example_title = "Compute shader N-body system";
static bool prepared             = false;

static float rand_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 0.1f, 512.0f);
  camera_set_rotation(context->camera, (vec3){-26.0f, 75.0f, 0.0f});
  camera_set_translation(context->camera, (vec3){0.0f, 0.0f, -14.0f});
  camera_set_movement_speed(context->camera, 2.5f);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  textures.particle = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/particle01_rgba.ktx");
  textures.gradient = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/particle_gradient_rgba.ktx");
}

// Setup and fill the compute shader storage buffers containing the particles
static void prepare_storage_buffers(wgpu_context_t* wgpu_context)
{
  static vec3 attractors[NUMBER_OF_ATTRACTORS] = {
    {5.0f, 0.0f, 0.0f},  //
    {-5.0f, 0.0f, 0.0f}, //
    {0.0f, 0.0f, 5.0f},  //
    {0.0f, 0.0f, -5.0f}, //
    {0.0f, 4.0f, 0.0f},  //
    {0.0f, -8.0f, 0.0f}, //
  };

  num_particles = NUMBER_OF_ATTRACTORS * PARTICLES_PER_ATTRACTOR;

  // Initial particle positions
  static particle_t
    particle_buffer[NUMBER_OF_ATTRACTORS * PARTICLES_PER_ATTRACTOR]
    = {0};

  for (uint32_t i = 0; i < (uint32_t)NUMBER_OF_ATTRACTORS; ++i) {
    for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; ++j) {
      particle_t* particle = &particle_buffer[i * PARTICLES_PER_ATTRACTOR + j];

      // First particle in group as heavy center of gravity
      if (j == 0) {
        vec3 scaled;
        glm_vec3_scale(attractors[i], 1.5f, scaled);
        memcpy(particle->pos, scaled, sizeof(vec3));
        particle->pos[3] = 90000.0f;
        glm_vec4_zero(particle->vel);
      }
      else {
        // Position
        vec3 position
          = {rand_float_min_max(0.0f, 1.0f), rand_float_min_max(0.0f, 1.0f),
             rand_float_min_max(0.0f, 1.0f)};
        glm_vec3_scale(position, 0.75f, position);
        glm_vec3_add(attractors[i], position, position);
        vec3 diff;
        glm_vec3_sub(position, attractors[i], diff);
        glm_normalize(diff);
        float len
          = sqrtf(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        position[1] *= 2.0f - (len * len);

        // Velocity
        vec3 angular = {0.5f, 1.5f, 0.5f};
        glm_vec3_scale(angular, ((i % 2) == 0) ? 1.0f : -1.0f, angular);
        vec3 velocity
          = {rand_float_min_max(0.0f, 1.0f), rand_float_min_max(0.0f, 1.0f),
             rand_float_min_max(0.0f, 1.0f) * 0.025f};
        glm_vec3_sub(position, attractors[i], diff);
        glm_cross(diff, angular, diff);
        glm_vec3_add(diff, velocity, velocity);

        float mass = (rand_float_min_max(0.0f, 1.0f) * 0.5f + 0.5f) * 75.0f;
        memcpy(particle->pos, position, sizeof(vec3));
        particle->pos[3] = mass;
        memcpy(particle->vel, velocity, sizeof(vec3));
        particle->vel[3] = 0.0f;
      }

      // Color gradient offset
      particle->vel[3] = (float)i * 1.0f / (int32_t)NUMBER_OF_ATTRACTORS;
    }
  }

  compute.ubo.particle_count = num_particles;

  uint64_t storage_buffer_size
    = (uint64_t)ARRAY_SIZE(particle_buffer) * sizeof(particle_t);

  // Staging
  // SSBO won't be changed on the host after upload
  compute.storage_buffer = wgpu_create_buffer_from_data(
    wgpu_context, &particle_buffer, storage_buffer_size,
    WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage);
}

static void update_compute_uniform_buffers(wgpu_example_context_t* context)
{
  compute.ubo.delta_t = context->paused ? 0.0f : context->frame_timer * 0.05f;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, compute.uniform_buffer, 0,
                          &compute.ubo, sizeof(compute.ubo));
}

static void update_graphics_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Pass matrices to the shaders
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective, graphics.ubo.projection);
  glm_mat4_copy(camera->matrices.view, graphics.ubo.view);
  graphics.ubo.screen_dim[0] = (float)wgpu_context->surface.width;
  graphics.ubo.screen_dim[1] = (float)wgpu_context->surface.height;

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, graphics.uniform_buffer, 0,
                          &graphics.ubo, sizeof(graphics.ubo));
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Compute shader uniform buffer block
  {
    WGPUBufferDescriptor buffer_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(compute.ubo),
      .mappedAtCreation = false,
    };
    compute.uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(compute.uniform_buffer)
  }

  // Vertex shader uniform buffer block
  {
    WGPUBufferDescriptor buffer_desc = {
      .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size             = sizeof(graphics.ubo),
      .mappedAtCreation = false,
    };
    graphics.uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(graphics.uniform_buffer)
  }

  update_compute_uniform_buffers(context);
  update_graphics_uniform_buffers(context);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[5] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0 : Particle color map texture
      .binding = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1 : Particle color map sampler
      .binding = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2 : Particle gradient ramp texture
      .binding = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      // Binding 3 : Particle gradient ramp sampler
      .binding = 3,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      // Binding 4 : Uniform Buffer Object
      .binding = 4,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type=WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(graphics.ubo),
      },
      .texture = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  graphics.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(graphics.bind_group_layout != NULL)

  // Create the pipeline layout
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &graphics.bind_group_layout,
  };
  graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &pipeline_layout_desc);
  ASSERT(graphics.pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[5] = {
    [0] = (WGPUBindGroupEntry) {
      // Binding 0 : Particle color map texture
      .binding = 0,
      .textureView = textures.particle.view,
    },
    [1] = (WGPUBindGroupEntry) {
       // Binding 1 : Particle color map sampler
      .binding = 1,
      .sampler = textures.particle.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
       // Binding 2 : Particle gradient ramp texture
      .binding = 2,
      .textureView = textures.gradient.view,
    },
    [3] = (WGPUBindGroupEntry) {
      // Binding 3 : Particle gradient ramp sampler
      .binding = 3,
      .sampler = textures.gradient.sampler,
    },
    [4] = (WGPUBindGroupEntry) {
      // Binding 4 : Uniform Buffer Object
      .binding = 4,
      .buffer = graphics.uniform_buffer,
      .size = sizeof(graphics.ubo),
    },
  };

  WGPUBindGroupDescriptor bg_desc = {
    .layout     = graphics.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };

  graphics.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(graphics.bind_group != NULL)
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state_desc
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face = WGPUFrontFace_CCW,
        .cull_mode  = WGPUCullMode_None,
      });

  // Color blend state: additive blending
  WGPUColorStateDescriptor color_state_desc
    = wgpu_create_color_state_descriptor(&(create_color_state_desc_t){
      .format       = wgpu_context->swap_chain.format,
      .enable_blend = true,
    });
  color_state_desc.colorBlend.srcFactor = WGPUBlendFactor_One;
  color_state_desc.colorBlend.dstFactor = WGPUBlendFactor_One;
  color_state_desc.alphaBlend.srcFactor = WGPUBlendFactor_SrcAlpha;
  color_state_desc.alphaBlend.dstFactor = WGPUBlendFactor_DstAlpha;

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc
    = wgpu_create_depth_stencil_state_descriptor(
      &(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = false,
      });

  // Vertex input binding (=> Input assembly)
  WGPU_VERTSTATE(particle, sizeof(particle_t),
                 // Attribute descriptions
                 // Describes memory layout and shader positions
                 // Attribute location 0: Position
                 WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                    offsetof(particle_t, pos)),
                 // Attribute location 1: Velocity (used for gradient lookup)
                 WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                    offsetof(particle_t, vel)))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/compute_n_body/particle.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/compute_n_body/particle.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  graphics.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .layout = graphics.pipeline_layout,
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state_desc,
      .primitiveTopology      = WGPUPrimitiveTopology_PointList,
      .colorStateCount        = 1,
      .colorStates            = &color_state_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_particle,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .view = NULL,
      .attachment = NULL,
      .loadOp = WGPULoadOp_Clear,
      .storeOp = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  wgpu_setup_deph_stencil(wgpu_context);

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
  };
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
      .binding = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Storage,
        .minBindingSize = num_particles * sizeof(particle_t),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1 : Uniform buffer
      .binding = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(compute.ubo),
      },
      .sampler = {0},
    }
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
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
      .buffer = compute.storage_buffer,
      .size = num_particles * sizeof(particle_t),
    },
    [1] = (WGPUBindGroupEntry) {
     // Binding 1 : Uniform buffer
      .binding = 1,
      .buffer = compute.uniform_buffer,
      .offset = 0,
      .size = sizeof(compute.ubo),
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .layout     = compute.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  compute.bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);

  /* 1st pass */

  /* Compute shader */
  wgpu_shader_t particle_calculate_comp_shader = wgpu_shader_create(
    wgpu_context,
    &(wgpu_shader_desc_t){
      // Compute shader SPIR-V
      .file = "shaders/compute_n_body/particle_calculate.comp.spv",
    });

#if 0
  // Set shader parameters via specialization constants
  struct specialization_data {
    uint32_t shared_data_size;
    float gravity;
    float power;
    float soften;
  } specialization_data = {
    .shared_data_size = 1024,
    .gravity          = 0.002f,
    .power            = 0.75f,
    .soften           = 0.05f,
  };
#endif

  /* Create pipeline */
  compute.pipeline_calculate = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .layout = compute.pipeline_layout,
      .computeStage
      = particle_calculate_comp_shader.programmable_stage_descriptor,
    });

  /* Partial clean-up */
  wgpu_shader_release(&particle_calculate_comp_shader);

  /* 2nd pass */

  /* Compute shader */
  wgpu_shader_t particle_integrate_comp_shader = wgpu_shader_create(
    wgpu_context,
    &(wgpu_shader_desc_t){
      // Compute shader SPIR-V
      .file = "shaders/compute_n_body/particle_integrate.comp.spv",
    });

  /* Create pipeline */
  compute.pipeline_integrate = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .layout = compute.pipeline_layout,
      .computeStage
      = particle_integrate_comp_shader.programmable_stage_descriptor,
    });

  /* Partial clean-up */
  wgpu_shader_release(&particle_integrate_comp_shader);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_graphics(context);
    prepare_compute(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // First compute pass: Compute particle movement
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    // Dispatch the compute job
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline_calculate);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc, num_particles / 256,
                                   1, 1);
    wgpuComputePassEncoderEndPass(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  // Second compute pass: Integrate particle
  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    // Dispatch the compute job
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute.pipeline_integrate);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute.bind_group, 0, NULL);
    wgpuComputePassEncoderDispatch(wgpu_context->cpass_enc, num_particles / 256,
                                   1, 1);
    wgpuComputePassEncoderEndPass(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  // Render pass: Draw the particle system using the update vertex buffer
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      graphics.bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         compute.storage_buffer, 0, 0);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, num_particles, 1, 0, 0);
    wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, NULL);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL)
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
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
  int result = example_draw(context);
  update_compute_uniform_buffers(context);

  return result;
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_graphics_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  // Textures
  wgpu_destroy_texture(&textures.particle);
  wgpu_destroy_texture(&textures.gradient);

  // Graphics pipeline
  WGPU_RELEASE_RESOURCE(Buffer, graphics.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, graphics.pipeline)

  // Compute pipeline
  WGPU_RELEASE_RESOURCE(Buffer, compute.storage_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, compute.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline_calculate)
  WGPU_RELEASE_RESOURCE(ComputePipeline, compute.pipeline_integrate)
}

void example_compute_n_body(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title  = example_title,
     .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}
