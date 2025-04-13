#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particles WebGPU Logo
 *
 * This example demonstrates rendering of particles simulated with compute
 * shaders.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/particles
 * -------------------------------------------------------------------------- */

const uint32_t num_particles               = 50000;
const uint32_t particle_position_offset    = 0;
const uint32_t particle_color_offset       = 4 * 4;
const uint32_t particle_instance_byte_size = 3 * 4 + /* position */
                                             1 * 4 + /* lifetime */
                                             4 * 4 + /* color    */
                                             3 * 4 + /* velocity */
                                             1 * 4 + /* padding  */
                                             0;

/* Particles buffer */
static struct wgpu_buffer_t particles_buffer = {0};

/* Quad vertex buffer */
static struct wgpu_buffer_t quad_vertices = {0};

/* Uniform buffer block object */
static struct uniform_buffer_vs_t {
  struct {
    mat4 model_view_projection_matrix;
    vec3 right;
    float padding1;
    vec3 up;
    float padding2;
  } data;
  wgpu_buffer_t buffer;
} uniform_buffer_vs = {0};

static WGPUBindGroup uniform_bind_group;
static texture_t depth_texture;
static texture_t texture;
static WGPURenderPipeline render_pipeline;

static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;
static WGPURenderPassDepthStencilAttachment render_pass_depth_stencil_att_desc;

// Probability map generation
static WGPUComputePipeline probability_map_import_level_pipeline;
static WGPUComputePipeline probability_map_export_level_pipeline;

static struct probability_ubo_buffer_t {
  int32_t data[4];
  WGPUBuffer buffer;
  uint32_t size;
} probability_ubo_buffer, buffer_a, buffer_b = {0};

/* Simulation compute pipeline */
static struct simulation_params_t {
  bool simulate;
  float delta_time;
} simulation_params = {
  .simulate   = true,
  .delta_time = 0.04f,
};

static struct simulation_ubo_buffer_t {
  struct {
    float delta_time;
    vec3 padding;
    struct {
      float x;
      float y;
      float z;
      float w;
    } seed;
  } data;
  WGPUBuffer buffer;
  uint32_t size;
} simulation_ubo_buffer = {0};

struct {
  mat4 projection;
  mat4 view;
  mat4 model_view_projection;
} view_matrices;

static WGPUComputePipeline compute_pipeline;
static WGPUBindGroup compute_bind_group;

// Other variables
static const char* example_title = "Compute Shader Particles WebGPU Logo";
static bool prepared             = false;

static void prepare_particles_buffer(wgpu_context_t* wgpu_context)
{
  particles_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particles - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size = num_particles * particle_instance_byte_size,
                  });
}

static void prepare_render_pipelines(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  blend_state.color.srcFactor             = WGPUBlendFactor_SrcAlpha;
  blend_state.color.dstFactor             = WGPUBlendFactor_One;
  blend_state.color.operation             = WGPUBlendOperation_Add;
  blend_state.alpha.srcFactor             = WGPUBlendFactor_Zero;
  blend_state.alpha.dstFactor             = WGPUBlendFactor_One;
  blend_state.alpha.operation             = WGPUBlendOperation_Add;
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = false,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPUVertexBufferLayout vertex_buffer_layouts[2] = {0};
  {
    WGPUVertexAttribute attributes[2] = {
      [0] = (WGPUVertexAttribute) {
        /* Shader location 0 : position */
        .shaderLocation = 0,
        .offset         = particle_position_offset,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute) {
        /* Shader location 1 : color */
        .shaderLocation = 1,
        .offset         = particle_color_offset,
        .format         = WGPUVertexFormat_Float32x4,
      },
    };
    vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      /* instanced particles  */
      .arrayStride    = particle_instance_byte_size,
      .stepMode       = WGPUVertexStepMode_Instance,
      .attributeCount = (uint32_t)ARRAY_SIZE(attributes),
      .attributes     = attributes,
    };
  }
  {
    WGPUVertexAttribute attributes[1] = {
      [0] = (WGPUVertexAttribute) {
        /* Shader location 2 : vertex positions */
        .shaderLocation = 2,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x2,
      },
    };
    vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      /* quad vertex  */
      .arrayStride    = 2 * 4, /* vec2<f32> */
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = (uint32_t)ARRAY_SIZE(attributes),
      .attributes     = attributes,
    };
  }

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Vertex shader WGSL */
              .label = "Particle - Vertex shader",
              .file  = "shaders/compute_particles_webgpu_logo/particle.wgsl",
              .entry = "vs_main",
            },
            .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts),
            .buffers      = vertex_buffer_layouts,
          });

  /* Fragment  */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              /* Fragment shader WGSL */
              .label = "Particle - Fragment shader",
              .file  = "shaders/compute_particles_webgpu_logo/particle.wgsl",
              .entry = "fs_main",
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
  render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Particle - Render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(render_pipeline);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPUTextureDescriptor texture_desc = {
    .label         = "Depth - Texture",
    .size          = (WGPUExtent3D) {
      .width              = wgpu_context->surface.width,
      .height             = wgpu_context->surface.height,
      .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24Plus,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Depth - Texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = WGPUTextureFormat_Depth24Plus,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  depth_texture.view
    = wgpuTextureCreateView(depth_texture.texture, &texture_view_dec);
}

static void prepare_uniform_buffer(wgpu_context_t* wgpu_context)
{
  uniform_buffer_vs.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex shader - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(uniform_buffer_vs.data),
                  });
}

static void prepare_uniform_bind_group(wgpu_context_t* wgpu_context)
{
  // Uniform bind group
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer.buffer,
      .offset  = 0,
      .size    =  uniform_buffer_vs.buffer.size,
    },
  };
  uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = "Vertex shader - Uniform bind group",
      .layout     = wgpuRenderPipelineGetBindGroupLayout(render_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(uniform_bind_group != NULL)
}

static void setup_render_pass(void)
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
  render_pass_depth_stencil_att_desc = (WGPURenderPassDepthStencilAttachment){
    .view            = depth_texture.view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };

  /* Render pass descriptor */
  render_pass_desc = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = &render_pass_depth_stencil_att_desc,
  };
}

static void prepare_quad_vertex_buffer(wgpu_context_t* wgpu_context)
{
  static const float vertex_buffer[3 * 4] = {
    -1.0f, -1.0f, +1.0f, //
    -1.0f, -1.0f, +1.0f, //
    -1.0f, +1.0f, +1.0f, //
    -1.0f, +1.0f, +1.0f, //
  };

  /* Create vertex buffer */
  quad_vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_buffer),
                    .count = (uint32_t)ARRAY_SIZE(vertex_buffer),
                    .initial.data = vertex_buffer,
                  });
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  const char* file = "textures/webgpu.png";
  struct wgpu_texture_load_options_t texture_load_options = {
    .label            = "WebGPU logo - Texture",
    .generate_mipmaps = true,
    .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding
             | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment,
    .format       = WGPUTextureFormat_RGBA8Unorm,
    .address_mode = WGPUAddressMode_Repeat,
  };
  texture
    = wgpu_create_texture_from_file(wgpu_context, file, &texture_load_options);
}

/**
 * @brief Probability map generation.
 * The 0'th mip level of texture holds the color data and spawn-probability in
 * the alpha channel. The mip levels 1..N are generated to hold spawn
 * probabilities up to the top 1x1 mip level.
 * @param wgpu_context the WebGPU context
 */
static void generate_probability_map(wgpu_context_t* wgpu_context)
{
  /* Compute shaders */
  wgpu_shader_t probability_map_import_lvl_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Compute shader WGSL
                    .label = "Probability map import level - Compute shader",
                    .file  = "shaders/compute_particles_webgpu_logo/"
                             "probabilityMap.wgsl",
                    .entry = "import_level",
                  });
  wgpu_shader_t probability_map_export_lvl_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Compute shader WGSL
                    .label = "Probability map export level - Compute shader",
                    .file  = "shaders/compute_particles_webgpu_logo/"
                             "probabilityMap.wgsl",
                    .entry = "export_level",
                  });

  /* Create compute pipelines */
  probability_map_import_level_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label = "Probability map import level - Compute pipeline",
      .compute
      = probability_map_import_lvl_comp_shader.programmable_stage_descriptor,
    });
  probability_map_export_level_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label = "Probability map export level - Compute pipeline",
      .compute
      = probability_map_export_lvl_comp_shader.programmable_stage_descriptor,
    });

  /* ProbabilityMap UBO Buffer */
  {
    probability_ubo_buffer.size = 1 * 4 + /* stride  */
                                  3 * 4 + /* padding */
                                  0;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label            = "ProbabilityMap - UBO Buffer",
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = probability_ubo_buffer.size,
      .mappedAtCreation = false,
    };
    probability_ubo_buffer.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(probability_ubo_buffer.buffer)

    probability_ubo_buffer.data[0] = texture.size.width;
    wgpu_queue_write_buffer(wgpu_context, probability_ubo_buffer.buffer, 0,
                            probability_ubo_buffer.data,
                            probability_ubo_buffer.size);
  }

  /* Buffer A */
  {
    buffer_a.size = texture.size.width * texture.size.height * 4;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label            = "Storage buffer A",
      .usage            = WGPUBufferUsage_Storage,
      .size             = buffer_a.size,
      .mappedAtCreation = false,
    };
    buffer_a.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(buffer_a.buffer)
  }

  /* Buffer B */
  {
    buffer_b.size = texture.size.width * texture.size.height * 4;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label            = "Storage buffer B",
      .usage            = WGPUBufferUsage_Storage,
      .size             = buffer_b.size,
      .mappedAtCreation = false,
    };
    buffer_b.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(buffer_b.buffer)
  }

  /* Probability map generation */
  {
    wgpu_context->cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPUTextureView* texture_views = (WGPUTextureView*)calloc(
      texture.mip_level_count, sizeof(WGPUTextureView));
    WGPUBindGroup* probability_map_bind_groups
      = (WGPUBindGroup*)calloc(texture.mip_level_count, sizeof(WGPUBindGroup));
    for (uint32_t level = 0; level < texture.mip_level_count; ++level) {
      /* Pipeline layouts */
      const uint32_t level_width  = texture.size.width >> level;
      const uint32_t level_height = texture.size.height >> level;
      const WGPUBindGroupLayout pipeline_layout
        = level == 0 ? wgpuComputePipelineGetBindGroupLayout(
                         probability_map_import_level_pipeline, 0) :
                       wgpuComputePipelineGetBindGroupLayout(
                         probability_map_export_level_pipeline, 0);
      /* Texture view */
      texture_views[level] = wgpuTextureCreateView(
        texture.texture, &(WGPUTextureViewDescriptor){
                           .label           = "Probability map - Texture view",
                           .format          = WGPUTextureFormat_RGBA8Unorm,
                           .aspect          = WGPUTextureAspect_All,
                           .baseMipLevel    = level,
                           .mipLevelCount   = 1,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .baseArrayLayer  = 0,
                           .arrayLayerCount = 1,
                         });
      /* Compute pipeline bind group */
      WGPUBindGroupEntry bg_entries[4] = {
        [0] = (WGPUBindGroupEntry) {
          // Binding 0 : ubo
          .binding = 0,
          .buffer  = probability_ubo_buffer.buffer,
          .size    = probability_ubo_buffer.size,
        },
        [1] = (WGPUBindGroupEntry) {
         // Binding 1 : buf_in
          .binding = 1,
          .buffer  = level & 1 ? buffer_a.buffer : buffer_b.buffer,
          .offset  = 0,
          .size    = level & 1 ? buffer_a.size : buffer_b.size,
        },
        [2] = (WGPUBindGroupEntry) {
          // Binding 2 : buf_out
           .binding = 2,
           .buffer  = level & 1 ? buffer_b.buffer : buffer_a.buffer,
           .offset  = 0,
           .size    = level & 1 ? buffer_b.size : buffer_a.size,
        },
        [3] = (WGPUBindGroupEntry) {
          // Binding 3 : tex_in / tex_out
           .binding     = 3,
           .textureView = texture_views[level],
        },
      };
      WGPUBindGroupDescriptor bg_desc = {
        .label      = "Probability map - Bind group layout",
        .layout     = pipeline_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      probability_map_bind_groups[level]
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(probability_map_bind_groups[level])
      /* Compute pass */
      if (level == 0) {
        wgpu_context->cpass_enc
          = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
        wgpuComputePassEncoderSetPipeline(
          wgpu_context->cpass_enc, probability_map_import_level_pipeline);
        wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                           probability_map_bind_groups[level],
                                           0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
          wgpu_context->cpass_enc, ceil(level_width / 64.f), level_height, 1);
        wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
        WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
      }
      else {
        wgpu_context->cpass_enc
          = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
        wgpuComputePassEncoderSetPipeline(
          wgpu_context->cpass_enc, probability_map_export_level_pipeline);
        wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                           probability_map_bind_groups[level],
                                           0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
          wgpu_context->cpass_enc, ceil(level_width / 64.f), level_height, 1);
        wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
        WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
      }
    }

    /* Get command buffer */
    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(wgpu_context->cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

    /* Sumbit commmand buffer and cleanup */
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

    /* Cleanup */
    for (uint32_t i = 0; i < texture.mip_level_count; ++i) {
      WGPU_RELEASE_RESOURCE(TextureView, texture_views[i]);
    }
    free(texture_views);
    for (uint32_t i = 0; i < texture.mip_level_count; ++i) {
      WGPU_RELEASE_RESOURCE(BindGroup, probability_map_bind_groups[i]);
    }
    free(probability_map_bind_groups);
  }

  /* Partial clean-up */
  wgpu_shader_release(&probability_map_import_lvl_comp_shader);
  wgpu_shader_release(&probability_map_export_lvl_comp_shader);
}

static void prepare_simulation_uniform_buffer(wgpu_context_t* wgpu_context)
{
  simulation_ubo_buffer.size = sizeof(simulation_ubo_buffer.data);

  WGPUBufferDescriptor uniform_buffer_desc = {
    .label            = "Simulation UBO buffer",
    .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    .size             = simulation_ubo_buffer.size,
    .mappedAtCreation = false,
  };
  simulation_ubo_buffer.buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
  ASSERT(simulation_ubo_buffer.buffer)
}

static void prepare_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  wgpu_shader_t particle_comp_shader = wgpu_shader_create(
    wgpu_context,
    &(wgpu_shader_desc_t){
      // Compute shader WGSL
      .label = "Particle - Compute shader WGSL",
      .file  = "shaders/compute_particles_webgpu_logo/particle.wgsl",
      .entry = "simulate",
    });

  /* Create pipeline */
  compute_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = "Particle - Compute pipeline",
      .compute = particle_comp_shader.programmable_stage_descriptor,
    });
  ASSERT(compute_pipeline)

  /* Partial clean-up */
  wgpu_shader_release(&particle_comp_shader);
}

static void prepare_compute_bind_group(wgpu_context_t* wgpu_context)
{
  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Simulation UBO buffer */
      .binding = 0,
      .buffer  = simulation_ubo_buffer.buffer,
      .size    = simulation_ubo_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
     /* Binding 1 : Particles buffer */
      .binding = 1,
      .buffer  = particles_buffer.buffer,
      .offset  = 0,
      .size    = particles_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
     /* Binding 2 : Texture view */
      .binding     = 2,
      .textureView = texture.view,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "Compute pipeline - Bind group",
    .layout     = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0),
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  compute_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(compute_bind_group)
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  /* Projection matrix */
  glm_mat4_identity(view_matrices.projection);
  glm_mat4_identity(view_matrices.view);
  glm_mat4_identity(view_matrices.model_view_projection);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);
}

static void update_simulation_ubo_data(wgpu_context_t* wgpu_context)
{
  simulation_ubo_buffer.data.delta_time
    = simulation_params.simulate ? simulation_params.delta_time : 0.0f;
  glm_vec3_zero(simulation_ubo_buffer.data.padding);
  simulation_ubo_buffer.data.seed.x = random_float() * 100.0f;
  simulation_ubo_buffer.data.seed.y = random_float() * 100.0f;
  simulation_ubo_buffer.data.seed.z = 1.0f + random_float();
  simulation_ubo_buffer.data.seed.w = 1.0f + random_float();

  wgpu_queue_write_buffer(wgpu_context, simulation_ubo_buffer.buffer, 0,
                          &simulation_ubo_buffer.data,
                          simulation_ubo_buffer.size);
}

static void update_transformation_matrix(void)
{
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -3.0f});
  glm_rotate(view_matrices.view, PI * -0.2f, (vec3){1.0f, 0.0f, 0.0f});
  glm_mat4_mul(view_matrices.projection, view_matrices.view,
               view_matrices.model_view_projection);
}

static void update_uniform_buffer_vs_data(void)
{
  mat4* view = &view_matrices.view;

  glm_mat4_copy(view_matrices.model_view_projection,
                uniform_buffer_vs.data.model_view_projection_matrix);
  glm_vec3_copy((vec3){(*view)[0][0], (*view)[1][0], (*view)[2][0]},
                uniform_buffer_vs.data.right);
  uniform_buffer_vs.data.padding1 = 0.f;
  glm_vec3_copy((vec3){(*view)[0][1], (*view)[1][1], (*view)[2][1]},
                uniform_buffer_vs.data.up);
  uniform_buffer_vs.data.padding2 = 0.f;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update the model-view-projection matrix */
  update_transformation_matrix();

  /* Update uniform buffer data */
  update_uniform_buffer_vs_data();

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(wgpu_context, uniform_buffer_vs.buffer.buffer, 0,
                          &uniform_buffer_vs.data,
                          uniform_buffer_vs.buffer.size);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_particles_buffer(context->wgpu_context);
    prepare_render_pipelines(context->wgpu_context);
    prepare_depth_texture(context->wgpu_context);
    prepare_uniform_buffer(context->wgpu_context);
    prepare_uniform_bind_group(context->wgpu_context);
    setup_render_pass();
    prepare_quad_vertex_buffer(context->wgpu_context);
    prepare_texture(context->wgpu_context);
    generate_probability_map(context->wgpu_context);
    prepare_simulation_uniform_buffer(context->wgpu_context);
    prepare_compute_pipeline(context->wgpu_context);
    prepare_compute_bind_group(context->wgpu_context);
    prepare_view_matrices(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Simulate",
                           &simulation_params.simulate);
    imgui_overlay_input_float(context->imgui_overlay, "Delta Time",
                              &simulation_params.delta_time, 0.01, "%.2f");
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    wgpu_context->cpass_enc
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                      compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                       compute_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
                                             ceil(num_particles / 64.f), 1, 1);
    wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
  }

  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      uniform_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0, particles_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 1, quad_vertices.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 6, num_particles, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
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

  update_simulation_ubo_data(context->wgpu_context);
  update_uniform_buffers(context->wgpu_context);

  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(Buffer, particles_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, quad_vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer.buffer)

  WGPU_RELEASE_RESOURCE(BindGroup, uniform_bind_group)
  wgpu_destroy_texture(&depth_texture);
  wgpu_destroy_texture(&texture);
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipeline)

  WGPU_RELEASE_RESOURCE(ComputePipeline, probability_map_import_level_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, probability_map_export_level_pipeline)

  WGPU_RELEASE_RESOURCE(Buffer, probability_ubo_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, buffer_a.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, buffer_b.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, simulation_ubo_buffer.buffer)

  WGPU_RELEASE_RESOURCE(ComputePipeline, compute_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, compute_bind_group)
}

void example_compute_particles_webgpu_logo(int argc, char* argv[])
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
