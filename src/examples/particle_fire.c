#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/gltf_model.h"
#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - CPU Based Particle System
 *
 * Implements a simple CPU based particle system. Particle data is stored in
 * host memory, updated on the CPU per-frame and synchronized with the device
 * before it's rendered using pre-multiplied alpha.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/particlefire/particlefire.cpp
 * -------------------------------------------------------------------------- */

#define PARTICLE_COUNT 512
#define PARTICLE_SIZE 10.0f

#define FLAME_RADIUS 8.0f

#define PARTICLE_TYPE_FLAME 0
#define PARTICLE_TYPE_SMOKE 1

typedef struct particle_t {
  vec4 pos;
  vec4 color;
  float alpha;
  float size;
  float rotation;
  uint32_t type;
  // Attributes not used in shader
  vec4 vel;
  float rotation_speed;
} particle_t;

static struct {
  struct {
    texture_t smoke;
    texture_t fire;
    // Use a custom sampler to change sampler attributes required for rotating
    // the uvs in the shader for alpha blended textures
    WGPUSampler sampler;
  } particles;
  struct {
    texture_t color_map;
    texture_t normal_map;
  } floor;
} textures;

struct gltf_model_t* environment = NULL;

static vec3 emitter_pos = {0.0f, -FLAME_RADIUS + 2.0f, 0.0f};
static vec3 min_vel     = {-3.0f, 0.5f, -3.0f};
static vec3 max_vel     = {3.0f, 7.0f, 3.0f};

static struct {
  WGPUBuffer buffer;
  // Size of the particle buffer in bytes
  size_t size;
} particles;

static struct {
  WGPUBuffer fire;
  WGPUBuffer environment;
} uniform_buffers;

static struct ubo_vs_t {
  mat4 projection;
  mat4 model_view;
  vec2 viewport_dim;
  float point_size;
} ubo_vs = {
  .point_size = PARTICLE_SIZE,
};

static struct ubo_env_t {
  mat4 projection;
  mat4 model_view;
  mat4 normal;
  vec4 light_pos;
} ubo_env = {
  .light_pos = {0.0f, 0.0f, 0.0f, 0.0f},
};

static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

static struct {
  WGPURenderPipeline particles;
  WGPURenderPipeline environment;
} pipelines;

static struct {
  WGPUPipelineLayout particles;
  WGPUPipelineLayout environment;
} pipeline_layouts;

static struct {
  WGPUBindGroupLayout particles;
  WGPUBindGroupLayout environment;
} bind_group_layouts;

static struct {
  WGPUBindGroup particles;
  WGPUBindGroup environment;
} bind_groups;

static particle_t particle_buffer[PARTICLE_COUNT] = {0};
static uint64_t particle_buffer_size              = (uint64_t)PARTICLE_COUNT;

static const char* example_title = "CPU Based Particle System";
static bool prepared             = false;

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){0.0f, 2.0f, -15.0f});
  camera_set_rotation(context->camera, (vec3){-15.0f + 180.0f, 45.0f, 0.0f});
  camera_set_perspective(context->camera, 60.0f,
                         context->window_size.aspect_ratio, 1.0f, 256.0f);
}

static float rand_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

static float rnd(float range)
{
  return rand_float_min_max(0.0f, range);
}

static void init_particle(particle_t* particle, vec3 emitter_pos)
{
  glm_vec4_copy(
    (vec4){0.0f, min_vel[1] + rnd(max_vel[1] - min_vel[1]), 0.0f, 0.0f},
    particle->vel);
  particle->alpha = rnd(0.75f);
  particle->size  = 1.0f + rnd(0.5f);
  glm_vec4_copy((vec4){1.0f, 0.0f, 0.0f, 0.0f}, particle->color);
  particle->type           = PARTICLE_TYPE_FLAME;
  particle->rotation       = rnd(2.0f * (float)PI);
  particle->rotation_speed = rnd(2.0f) - rnd(2.0f);

  // Get random sphere point
  float theta = rnd(2.0f * (float)PI);
  float phi   = rnd((float)PI) - (float)PI / 2.0f;
  float r     = rnd(FLAME_RADIUS);

  particle->pos[0] = r * cos(theta) * cos(phi);
  particle->pos[1] = r * sin(phi);
  particle->pos[2] = r * sin(theta) * cos(phi);

  glm_vec4_add(particle->pos,
               (vec4){emitter_pos[0], emitter_pos[1], emitter_pos[2], 0.0f},
               particle->pos);
}

static void transition_particle(particle_t* particle)
{
  switch (particle->type) {
    case PARTICLE_TYPE_FLAME:
      // Flame particles have a chance of turning into smoke
      if (rnd(1.0f) < 0.05f) {
        particle->alpha = 0.0f;
        glm_vec4_copy((vec4){0.25f + rnd(0.25f), 0.0f, 0.0f, 0.0f},
                      particle->color);
        particle->pos[0] *= 0.5f;
        particle->pos[2] *= 0.5f;
        glm_vec4_copy((vec4){rnd(1.0f) - rnd(1.0f),
                             (min_vel[1] * 2) + rnd(max_vel[1] - min_vel[1]),
                             rnd(1.0f) - rnd(1.0f), 0.0f},
                      particle->vel);
        particle->size           = 1.0f + rnd(0.5f);
        particle->rotation_speed = rnd(1.0f) - rnd(1.0f);
        particle->type           = PARTICLE_TYPE_SMOKE;
      }
      else {
        init_particle(particle, emitter_pos);
      }
      break;
    case PARTICLE_TYPE_SMOKE:
      // Respawn at end of life
      init_particle(particle, emitter_pos);
      break;
  }
}

static void prepare_particles(wgpu_context_t* wgpu_context)
{
  for (uint64_t i = 0; i < particle_buffer_size; ++i) {
    particle_t* particle = &particle_buffer[i];
    init_particle(particle, emitter_pos);
    particle->alpha = 1.0f - (fabs(particle->pos[1]) / (FLAME_RADIUS * 2.0f));
  }

  particles.size = particle_buffer_size * sizeof(particle_t);

  particles.buffer = wgpu_create_buffer_from_data(
    wgpu_context, particle_buffer, particles.size, WGPUBufferUsage_Vertex);
}

static void update_particles(wgpu_example_context_t* context)
{
  float particle_timer = context->frame_timer * 0.45f;
  for (uint64_t i = 0; i < particle_buffer_size; ++i) {
    particle_t* particle = &particle_buffer[i];
    switch (particle->type) {
      case PARTICLE_TYPE_FLAME: {
        particle->pos[1] -= particle->vel[1] * particle_timer * 3.5f;
        particle->alpha += particle_timer * 2.5f;
        particle->size -= particle_timer * 0.5f;
      } break;
      case PARTICLE_TYPE_SMOKE: {
        vec4 vel = GLM_VEC4_ZERO_INIT;
        glm_vec4_copy(particle->vel, vel);
        glm_vec4_scale(vel, context->frame_timer * 1.0f, particle->pos);
        glm_vec4_sub(particle->pos, vel, particle->pos);
        particle->alpha += particle_timer * 1.25f;
        particle->size += particle_timer * 0.125f;
        particle->color[0] -= particle_timer * 0.05f;
      } break;
    }
    particle->rotation += particle_timer * particle->rotation_speed;
    // Transition particle state
    if (particle->alpha > 2.0f) {
      transition_particle(particle);
    }
  }
  wgpu_queue_write_buffer(context->wgpu_context, particles.buffer, 0,
                          particle_buffer, particles.size);
}

static void load_assets(wgpu_context_t* wgpu_context)
{
  // Particles
  textures.particles.smoke = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/particle_smoke.ktx");
  textures.particles.fire = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/particle_fire.ktx");

  // Floor
  textures.floor.color_map = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/fireplace_colormap_rgba.ktx");
  textures.floor.normal_map = wgpu_texture_load_from_ktx_file(
    wgpu_context, "textures/fireplace_normalmap_rgba.ktx");

  // Create a custom sampler to be used with the particle textures
  // Create sampler
  WGPUSamplerDescriptor sampler_desc = {0};
  sampler_desc.minFilter             = WGPUFilterMode_Linear;
  sampler_desc.magFilter             = WGPUFilterMode_Linear;
  sampler_desc.mipmapFilter          = WGPUFilterMode_Linear;
  // Different address mode
  sampler_desc.addressModeU = WGPUAddressMode_ClampToEdge;
  sampler_desc.addressModeV = sampler_desc.addressModeU;
  sampler_desc.addressModeW = sampler_desc.addressModeU;
  sampler_desc.lodMinClamp  = 0.0f;
  // Both particle textures have the same number of mip maps
  sampler_desc.lodMaxClamp = (float)textures.particles.fire.mip_level_count;

  {
    // Enable anisotropic filtering
    sampler_desc.maxAnisotropy = 8.0f;
  }

  textures.particles.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);

  const uint32_t gltf_loading_flags
    = WGPU_GLTF_FileLoadingFlags_PreTransformVertices
      | WGPU_GLTF_FileLoadingFlags_PreMultiplyVertexColors
      | WGPU_GLTF_FileLoadingFlags_FlipY
      | WGPU_GLTF_FileLoadingFlags_DontLoadImages;
  environment
    = wgpu_gltf_model_load_from_file(&(wgpu_gltf_model_load_options_t){
      .wgpu_context       = wgpu_context,
      .filename           = "models/fireplace.gltf",
      .file_loading_flags = gltf_loading_flags,
    });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Particles bind group and pipeline layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_vs),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader smoke texture view
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader smoke texture sampler
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Fragment shader fire texture view
        .binding = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        // Binding 4: Fragment shader fire texture sampler
        .binding = 4,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    bind_group_layouts.particles = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.particles != NULL)

    // Create the pipeline layout
    pipeline_layouts.particles = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts = &bind_group_layouts.particles,
                            });
    ASSERT(pipeline_layouts.particles != NULL)
  }

  // Environment bind group and pipeline layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = sizeof(ubo_env),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Fragment shader image view
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Fragment shader image sampler
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        // Binding 3: Fragment shader image view
        .binding = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        // Binding 4: Fragment shader image sampler
        .binding = 4,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    bind_group_layouts.environment = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layouts.environment != NULL)

    // Create the pipeline layout
    pipeline_layouts.environment = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &bind_group_layouts.environment,
      });
    ASSERT(pipeline_layouts.environment != NULL)
  }
}

static void setup_bind_group(wgpu_context_t* wgpu_context)
{
  // Particles bind Group
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.fire,
        .offset = 0,
        .size = sizeof(ubo_vs),
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader smoke texture view
        .binding = 1,
        .textureView = textures.particles.smoke.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader smoke texture sampler
        .binding = 2,
        .sampler = textures.particles.sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader fire texture view
        .binding = 3,
        .textureView = textures.particles.fire.view,
      },
      [4] = (WGPUBindGroupEntry) {
        // Binding 4: Fragment shader fire texture sampler
        .binding = 4,
        .sampler = textures.particles.sampler,
      }
    };

    bind_groups.particles = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layouts.particles,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.particles != NULL)
  }

  // Environment bind Group
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0 : Vertex shader uniform buffer
        .binding = 0,
        .buffer = uniform_buffers.environment,
        .offset = 0,
        .size = sizeof(ubo_env),
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: Fragment shader color map texture view
        .binding = 1,
        .textureView = textures.floor.color_map.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: Fragment shader color map texture sampler
        .binding = 2,
        .sampler = textures.floor.color_map.sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        // Binding 3: Fragment shader normal map texture view
        .binding = 3,
        .textureView = textures.floor.normal_map.view,
      },
      [4] = (WGPUBindGroupEntry) {
        // Binding 4: Fragment shader normal map texture sampler
        .binding = 4,
        .sampler = textures.floor.normal_map.sampler,
      }
    };

    bind_groups.environment = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layouts.environment,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.environment != NULL)
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .view       = NULL,
      .attachment = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.5f,
        .g = 0.5f,
        .b = 0.5f,
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

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Particle rendering pipeline
  {
    // Premultiplied alpha

    // Primitive state
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_PointList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state_desc
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = false,
      });

    // Vertex input state
    WGPU_VERTEX_BUFFER_LAYOUT(
      particle, sizeof(particle_t),
      // Attribute location 0: Position
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                         offsetof(particle_t, pos)),
      // Attribute location 1: Color
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                         offsetof(particle_t, color)),
      // Attribute location 2: Alpha
      WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32,
                         offsetof(particle_t, alpha)),
      // Attribute location 3: Size
      WGPU_VERTATTR_DESC(3, WGPUVertexFormat_Float32,
                         offsetof(particle_t, size)),
      // Attribute location 4: Rotation
      WGPU_VERTATTR_DESC(4, WGPUVertexFormat_Float32,
                         offsetof(particle_t, rotation)),
      // Attribute location 5: Particle type
      WGPU_VERTATTR_DESC(5, WGPUVertexFormat_Sint32,
                         offsetof(particle_t, type)))

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                  wgpu_context, &(wgpu_vertex_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/particle_fire/particle.vert.spv",
                  },
                  .buffer_count = 1,
                  .buffers = &particle_vertex_buffer_layout,
                });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                  wgpu_context, &(wgpu_fragment_state_t){
                  .shader_desc = (wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/particle_fire/particle.frag.spv",
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
    pipelines.particles = wgpuDeviceCreateRenderPipeline2(
      wgpu_context->device, &(WGPURenderPipelineDescriptor2){
                              .label        = "particle_render_pipeline",
                              .layout       = pipeline_layouts.particles,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }

  // Environment rendering pipeline (normal mapped)
  {
    // Primitive state
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state_desc
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = true,
      });

    // Vertex buffer layout
    WGPU_GLTF_VERTEX_BUFFER_LAYOUT(
      cube,
      // Location 0: Position
      WGPU_GLTF_VERTATTR_DESC(0, WGPU_GLTF_VertexComponent_Position),
      // Location 1: Texture coordinates
      WGPU_GLTF_VERTATTR_DESC(1, WGPU_GLTF_VertexComponent_UV),
      // Location 2: Vertex normal
      WGPU_GLTF_VERTATTR_DESC(2, WGPU_GLTF_VertexComponent_Normal),
      // Location 3: Vertex tangent
      WGPU_GLTF_VERTATTR_DESC(3, WGPU_GLTF_VertexComponent_Tangent));

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Vertex shader SPIR-V
                .file = "shaders/particle_fire/normalmap.vert.spv",
              },
              .buffer_count = 1,
              .buffers = &cube_vertex_buffer_layout,
            });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                // Fragment shader SPIR-V
                .file = "shaders/particle_fire/normalmap.frag.spv",
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
    pipelines.environment = wgpuDeviceCreateRenderPipeline2(
      wgpu_context->device, &(WGPURenderPipelineDescriptor2){
                              .label        = "environment_render_pipeline",
                              .layout       = pipeline_layouts.environment,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });
  }
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Particle system fire
  glm_mat4_copy(context->camera->matrices.perspective, ubo_vs.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_vs.model_view);
  glm_vec2_copy((vec2){(float)context->wgpu_context->surface.width,
                       (float)context->wgpu_context->surface.height},
                ubo_vs.viewport_dim);
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.fire, 0,
                          &ubo_vs, sizeof(ubo_vs));

  // Environment
  glm_mat4_copy(context->camera->matrices.perspective, ubo_env.projection);
  glm_mat4_copy(context->camera->matrices.view, ubo_env.model_view);
  mat4 inv = GLM_MAT4_ZERO_INIT;
  glm_mat4_inv(ubo_env.model_view, inv);
  glm_mat4_transpose_to(inv, ubo_env.normal);
  glm_rotate(ubo_env.model_view, glm_rad(-90), (vec3){1.0f, 0.0f, 0.0f});
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.environment, 0,
                          &ubo_env, sizeof(ubo_env));
}

static void update_uniform_buffer_light(wgpu_example_context_t* context)
{
  // Environment
  ubo_env.light_pos[0] = sin(context->timer * 2.0f * (float)PI) * 1.5f;
  ubo_env.light_pos[1] = 0.0f;
  ubo_env.light_pos[2] = cos(context->timer * 2.0f * (float)PI) * 1.5f;
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffers.environment, 0,
                          &ubo_env, sizeof(ubo_env));
}

// Prepare and initialize uniform buffer containing shader uniforms
static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Vertex shader uniform buffer block
  WGPUBufferDescriptor ubo_vs_desc = {
    .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size             = sizeof(ubo_vs),
    .mappedAtCreation = false,
  };
  uniform_buffers.fire
    = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_vs_desc);

  // Vertex shader uniform buffer block
  WGPUBufferDescriptor ubo_env_desc = {
    .usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size             = sizeof(ubo_env),
    .mappedAtCreation = false,
  };
  uniform_buffers.environment
    = wgpuDeviceCreateBuffer(context->wgpu_context->device, &ubo_env_desc);

  update_uniform_buffers(context);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    load_assets(context->wgpu_context);
    prepare_particles(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_bind_group(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
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

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Environment
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                   pipelines.environment);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.environment, 0, 0);
  wgpu_gltf_model_draw(environment, (wgpu_gltf_model_render_options_t){0});

  // Particle system (no index buffer)
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                   pipelines.particles);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    bind_groups.particles, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       particles.buffer, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, PARTICLE_COUNT, 1, 0, 0);

  // End render pass
  wgpuRenderPassEncoderEndPass(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
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
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffer_light(context);
    update_particles(context);
  }
  if (context->camera->updated) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  wgpu_destroy_texture(&textures.particles.smoke);
  wgpu_destroy_texture(&textures.particles.fire);
  WGPU_RELEASE_RESOURCE(Sampler, textures.particles.sampler)

  wgpu_destroy_texture(&textures.floor.color_map);
  wgpu_destroy_texture(&textures.floor.normal_map);

  wgpu_gltf_model_destroy(environment);

  WGPU_RELEASE_RESOURCE(Buffer, particles.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.fire)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.environment)

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.particles)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.environment)

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.particles)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.environment)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.particles)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layouts.environment)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.particles)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.environment)
}

void example_texture_particle_fire(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
