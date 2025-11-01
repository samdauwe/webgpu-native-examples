#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

#include "stdbool.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader Particles WebGPU Logo
 *
 * This example demonstrates rendering of particles simulated with compute
 * shaders.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/particles
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* particle_shader_wgsl;
static const char* probability_map_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Shader Particles WebGPU Logo
 * -------------------------------------------------------------------------- */

#define PARTICLE_NUM (50000u)

/* State struct */
static struct {
  const uint32_t particle_position_offset;
  const uint32_t particle_color_offset;
  const uint32_t particle_instance_byte_size;
  wgpu_buffer_t particles_buffer;
  wgpu_buffer_t quad_vertices;
  struct uniform_buffer_vs_t {
    struct {
      mat4 model_view_projection_matrix;
      vec3 right;
      float padding1;
      vec3 up;
      float padding2;
    } data;
    wgpu_buffer_t buffer;
  } uniform_buffer_vs;
  struct simulation_ubo_buffer_t {
    struct {
      float delta_time;
      float brightness_factor;
      vec2 padding;
      struct {
        float x;
        float y;
        float z;
        float w;
      } seed;
    } data;
    WGPUBuffer buffer;
    uint32_t size;
  } simulation_ubo_buffer;
  struct {
    mat4 projection;
    mat4 view;
    mat4 model_view_projection;
  } view_matrices;
  wgpu_texture_t depth_texture;
  wgpu_texture_t texture;
  uint8_t file_buffer[512 * 512 * 4];
  WGPUBindGroup uniform_bind_group;
  WGPUBindGroup compute_bind_group;
  WGPURenderPipeline render_pipeline;
  WGPUComputePipeline compute_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct simulation_params_t {
    bool simulate;
    float delta_time;
    float brightness_factor;
  } simulation_params;
  bool initialized;
} state = {
  .particle_position_offset    = 0,
  .particle_color_offset       = 4 * 4,
  .particle_instance_byte_size = 3 * 4 + /* position */
                                 1 * 4 + /* lifetime */
                                 4 * 4 + /* color    */
                                 3 * 4 + /* velocity */
                                 1 * 4 + /* padding  */
                                 0,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .simulation_params = {
    .simulate          = true,
    .delta_time        = 0.04f,
    .brightness_factor = 1.0f,
  },
};

static void init_particles_buffer(wgpu_context_t* wgpu_context)
{
  state.particles_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particles - Storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size = PARTICLE_NUM * state.particle_instance_byte_size,
                  });
}

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule particle_shader_module
    = wgpu_create_shader_module(wgpu_context->device, particle_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state  = wgpu_create_blend_state(true);
  blend_state.color.srcFactor = WGPUBlendFactor_SrcAlpha;
  blend_state.color.dstFactor = WGPUBlendFactor_One;
  blend_state.color.operation = WGPUBlendOperation_Add;
  blend_state.alpha.srcFactor = WGPUBlendFactor_Zero;
  blend_state.alpha.dstFactor = WGPUBlendFactor_One;
  blend_state.alpha.operation = WGPUBlendOperation_Add;

  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
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

  /* instanced particles buffer */
  WGPUVertexAttribute attributes_0[2] = {
      [0] = (WGPUVertexAttribute) {
        /* Shader location 0 : position */
        .shaderLocation = 0,
        .offset         = state.particle_position_offset,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute) {
        /* Shader location 1 : color */
        .shaderLocation = 1,
        .offset         = state.particle_color_offset,
        .format         = WGPUVertexFormat_Float32x4,
      },
    };
  vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
    /* instanced particles  */
    .arrayStride    = state.particle_instance_byte_size,
    .stepMode       = WGPUVertexStepMode_Instance,
    .attributeCount = (uint32_t)ARRAY_SIZE(attributes_0),
    .attributes     = attributes_0,
  };

  /* Quad vertex buffer */
  WGPUVertexAttribute attributes_1[1] = {
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
    .attributeCount = (uint32_t)ARRAY_SIZE(attributes_1),
    .attributes     = attributes_1,
  };

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Particle - Render pipeline"),
    .vertex = {
      .module      = particle_shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = (uint32_t) ARRAY_SIZE(vertex_buffer_layouts),
      .buffers     = vertex_buffer_layouts,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("fs_main"),
      .module      = particle_shader_module,
      .targetCount = 1,
      .targets     = &color_target_state,
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(particle_shader_module);
}

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture);

  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth - Texture"),
    .size          = (WGPUExtent3D) {
      .width              = wgpu_context->width,
      .height             = wgpu_context->height,
      .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24Plus,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Depth - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.handle, &texture_view_dec);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer_vs.buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertex shader - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.uniform_buffer_vs.data),
                  });
}

static void init_uniform_bind_group(wgpu_context_t* wgpu_context)
{
  // Uniform bind group
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.buffer.size,
    },
  };
  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Vertex shader - Uniform bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.render_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(state.uniform_bind_group != NULL)
}

static void init_quad_vertex_buffer(wgpu_context_t* wgpu_context)
{
  static const float vertex_data[6 * 2] = {
    -1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, +1.0f,
  };

  /* Create vertex buffer */
  state.quad_vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_data),
                    .count = (uint32_t)ARRAY_SIZE(vertex_data),
                    .initial.data = vertex_data,
                  });
}

/**
 * @brief Determines if the given value is a power of two.
 *
 * @param {int} n - Number to evaluate.
 * @returns {bool} - True if the number is a power of two.
 */
static bool is_power_of_2(int n)
{
  return (n & (n - 1)) == 0;
}

static int32_t get_mip_level_count(int width, int height)
{
  return log2(fmax(width, height)) + 1;
}

/**
 * @brief The fetch-callback is called by sokol_fetch.h when the data is loaded,
 * or when an error has occurred.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    ASSERT(is_power_of_2(img_width)) /* image must be a power of 2 */
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D) {
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 4,
      },
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .mip_level_count = get_mip_level_count(img_width, img_height),
      .usage           = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding
                         | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  state.texture           = wgpu_create_texture(wgpu_context, NULL);
  wgpu_texture_t* texture = &state.texture;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/webgpu.png",
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

/**
 * @brief Probability map generation.
 * The 0'th mip level of texture holds the color data and spawn-probability in
 * the alpha channel. The mip levels 1..N are generated to hold spawn
 * probabilities up to the top 1x1 mip level.
 * @param wgpu_context the WebGPU context
 * @param texture the source texture
 */
static void generate_probability_map(wgpu_context_t* wgpu_context,
                                     wgpu_texture_t* texture)
{
  /* Compute shaders */
  WGPUShaderModule probability_map_shader_module
    = wgpu_create_shader_module(wgpu_context->device, probability_map_wgsl);

  /* Create compute pipelines */
  WGPUComputePipeline probability_map_import_level_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Probability map import level - Compute pipeline"),
      .compute = {
        .module     = probability_map_shader_module,
        .entryPoint = STRVIEW("import_level"),
      },
    });
  ASSERT(probability_map_import_level_pipeline != NULL);

  WGPUComputePipeline probability_map_export_level_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Probability map export level - Compute pipeline"),
      .compute = {
        .module     = probability_map_shader_module,
        .entryPoint = STRVIEW("export_level"),
      },
    });
  ASSERT(probability_map_export_level_pipeline != NULL);

  /* ProbabilityMap UBO Buffer */
  struct {
    int32_t data[4];
    WGPUBuffer buffer;
    uint32_t size;
  } probability_ubo_buffer = {0}, buffer_a = {0}, buffer_b = {0};

  /* ProbabilityMap UBO Buffer */
  {
    probability_ubo_buffer.size = 1 * 4 + /* stride  */
                                  3 * 4 + /* padding */
                                  0;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label = STRVIEW("ProbabilityMap - UBO Buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = probability_ubo_buffer.size,
    };
    probability_ubo_buffer.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(probability_ubo_buffer.buffer)

    probability_ubo_buffer.data[0] = texture->desc.extent.width;
    wgpuQueueWriteBuffer(wgpu_context->queue, probability_ubo_buffer.buffer, 0,
                         probability_ubo_buffer.data,
                         probability_ubo_buffer.size);
  }

  /* Buffer A */
  {
    buffer_a.size
      = texture->desc.extent.width * texture->desc.extent.height * 4;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label = STRVIEW("Storage buffer A"),
      .usage = WGPUBufferUsage_Storage,
      .size  = buffer_a.size,
    };
    buffer_a.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(buffer_a.buffer)
  }

  /* Buffer B */
  {
    buffer_b.size
      = texture->desc.extent.width * texture->desc.extent.height * 4;
    WGPUBufferDescriptor uniform_buffer_desc = {
      .label = STRVIEW("Storage buffer B"),
      .usage = WGPUBufferUsage_Storage,
      .size  = buffer_b.size,
    };
    buffer_b.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
    ASSERT(buffer_b.buffer)
  }

  /* Probability map generation */
  {
    WGPUCommandEncoder cmd_enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPUTextureView* texture_views = (WGPUTextureView*)calloc(
      texture->desc.mip_level_count, sizeof(WGPUTextureView));
    WGPUBindGroup* probability_map_bind_groups = (WGPUBindGroup*)calloc(
      texture->desc.mip_level_count, sizeof(WGPUBindGroup));
    for (uint32_t level = 0; level < texture->desc.mip_level_count; ++level) {
      /* Pipeline layouts */
      const uint32_t level_width  = texture->desc.extent.width >> level;
      const uint32_t level_height = texture->desc.extent.height >> level;
      const WGPUBindGroupLayout pipeline_layout
        = level == 0 ? wgpuComputePipelineGetBindGroupLayout(
                         probability_map_import_level_pipeline, 0) :
                       wgpuComputePipelineGetBindGroupLayout(
                         probability_map_export_level_pipeline, 0);
      /* Texture view */
      texture_views[level] = wgpuTextureCreateView(
        texture->handle, &(WGPUTextureViewDescriptor){
                           .label  = STRVIEW("Probability map - Texture view"),
                           .format = WGPUTextureFormat_RGBA8Unorm,
                           .aspect = WGPUTextureAspect_All,
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
        .label      = STRVIEW("Probability map - Bind group layout"),
        .layout     = pipeline_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      probability_map_bind_groups[level]
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(probability_map_bind_groups[level])
      /* Compute pass */
      if (level == 0) {
        WGPUComputePassEncoder cpass_enc
          = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
        wgpuComputePassEncoderSetPipeline(
          cpass_enc, probability_map_import_level_pipeline);
        wgpuComputePassEncoderSetBindGroup(
          cpass_enc, 0, probability_map_bind_groups[level], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
          cpass_enc, ceil(level_width / 64.f), level_height, 1);
        wgpuComputePassEncoderEnd(cpass_enc);
        WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
      }
      else {
        WGPUComputePassEncoder cpass_enc
          = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
        wgpuComputePassEncoderSetPipeline(
          cpass_enc, probability_map_export_level_pipeline);
        wgpuComputePassEncoderSetBindGroup(
          cpass_enc, 0, probability_map_bind_groups[level], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
          cpass_enc, ceil(level_width / 64.f), level_height, 1);
        wgpuComputePassEncoderEnd(cpass_enc);
        WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
      }
    }

    /* Get command buffer */
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
    ASSERT(command_buffer != NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc)

    /* Sumbit commmand buffer and cleanup */
    wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

    /* Cleanup */
    for (uint32_t i = 0; i < texture->desc.mip_level_count; ++i) {
      WGPU_RELEASE_RESOURCE(TextureView, texture_views[i]);
    }
    free(texture_views);
    for (uint32_t i = 0; i < texture->desc.mip_level_count; ++i) {
      WGPU_RELEASE_RESOURCE(BindGroup, probability_map_bind_groups[i]);
    }
    free(probability_map_bind_groups);
  }

  /* Partial clean-up */
  wgpuShaderModuleRelease(probability_map_shader_module);
  WGPU_RELEASE_RESOURCE(ComputePipeline, probability_map_import_level_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, probability_map_export_level_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, probability_ubo_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, buffer_a.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, buffer_b.buffer)
}

static void init_simulation_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.simulation_ubo_buffer.size = sizeof(state.simulation_ubo_buffer.data);

  WGPUBufferDescriptor uniform_buffer_desc = {
    .label            = STRVIEW("Simulation UBO buffer"),
    .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    .size             = state.simulation_ubo_buffer.size,
    .mappedAtCreation = false,
  };
  state.simulation_ubo_buffer.buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &uniform_buffer_desc);
  ASSERT(state.simulation_ubo_buffer.buffer)
}

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  WGPUShaderModule particle_shader_module
    = wgpu_create_shader_module(wgpu_context->device, particle_shader_wgsl);

  /* Create pipeline */
  state.compute_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Particle - Compute pipeline"),
      .compute = {
        .module     = particle_shader_module,
        .entryPoint = STRVIEW("simulate"),
      },
    });
  ASSERT(state.compute_pipeline != NULL);

  /* Partial cleanup */
  wgpuShaderModuleRelease(particle_shader_module);
}

static void init_compute_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_group)

  /* Compute pipeline bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Binding 0 : Simulation UBO buffer */
      .binding = 0,
      .buffer  = state.simulation_ubo_buffer.buffer,
      .size    = state.simulation_ubo_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
     /* Binding 1 : Particles buffer */
      .binding = 1,
      .buffer  = state.particles_buffer.buffer,
      .offset  = 0,
      .size    = state.particles_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
     /* Binding 2 : Texture view */
      .binding     = 2,
      .textureView = state.texture.view,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label  = STRVIEW("Compute pipeline - Bind group"),
    .layout = wgpuComputePipelineGetBindGroupLayout(state.compute_pipeline, 0),
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.compute_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.compute_bind_group)
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_mat4_identity(state.view_matrices.projection);
  glm_mat4_identity(state.view_matrices.view);
  glm_mat4_identity(state.view_matrices.model_view_projection);
  glm_perspective((2.0f * PI) / 5.0f, aspect, 1.0f, 100.0f,
                  state.view_matrices.projection);
}

static void update_simulation_ubo_data(wgpu_context_t* wgpu_context)
{
  state.simulation_ubo_buffer.data.delta_time
    = state.simulation_params.simulate ? state.simulation_params.delta_time :
                                         0.0f;
  state.simulation_ubo_buffer.data.brightness_factor
    = state.simulation_params.brightness_factor;
  glm_vec2_zero(state.simulation_ubo_buffer.data.padding);
  state.simulation_ubo_buffer.data.seed.x = random_float() * 100.0f;
  state.simulation_ubo_buffer.data.seed.y = random_float() * 100.0f; // seed.xy
  state.simulation_ubo_buffer.data.seed.z = 1.0f + random_float();
  state.simulation_ubo_buffer.data.seed.w = 1.0f + random_float(); // seed.zw

  wgpuQueueWriteBuffer(wgpu_context->queue, state.simulation_ubo_buffer.buffer,
                       0, &state.simulation_ubo_buffer.data,
                       state.simulation_ubo_buffer.size);
}

static void update_transformation_matrix(void)
{
  glm_mat4_identity(state.view_matrices.view);
  glm_translate(state.view_matrices.view, (vec3){0.0f, 0.0f, -3.0f});
  glm_rotate(state.view_matrices.view, PI * -0.2f, (vec3){1.0f, 0.0f, 0.0f});
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
               state.view_matrices.model_view_projection);
}

static void update_uniform_buffer_vs_data(void)
{
  mat4* view = &state.view_matrices.view;

  glm_mat4_copy(state.view_matrices.model_view_projection,
                state.uniform_buffer_vs.data.model_view_projection_matrix);
  glm_vec3_copy((vec3){(*view)[0][0], (*view)[1][0], (*view)[2][0]},
                state.uniform_buffer_vs.data.right);
  state.uniform_buffer_vs.data.padding1 = 0.f;
  glm_vec3_copy((vec3){(*view)[0][1], (*view)[1][1], (*view)[2][1]},
                state.uniform_buffer_vs.data.up);
  state.uniform_buffer_vs.data.padding2 = 0.f;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update the model-view-projection matrix */
  update_transformation_matrix();

  /* Update uniform buffer data */
  update_uniform_buffer_vs_data();

  // Map uniform buffer and update it
  wgpuQueueWriteBuffer(
    wgpu_context->queue, state.uniform_buffer_vs.buffer.buffer, 0,
    &state.uniform_buffer_vs.data, state.uniform_buffer_vs.buffer.size);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    init_particles_buffer(wgpu_context);
    init_graphics_pipeline(wgpu_context);
    init_depth_texture(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_uniform_bind_group(wgpu_context);
    init_quad_vertex_buffer(wgpu_context);
    init_texture(wgpu_context);
    generate_probability_map(wgpu_context, &state.texture);
    init_simulation_uniform_buffer(wgpu_context);
    init_compute_pipeline(wgpu_context);
    init_compute_bind_group(wgpu_context);
    init_view_matrices(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    generate_probability_map(wgpu_context, &state.texture);
    init_compute_bind_group(wgpu_context);
  }

  /* Update inform buffers */
  update_simulation_ubo_data(wgpu_context);
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0, state.compute_bind_group,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass_enc,
                                             ceil(PARTICLE_NUM / 64.f), 1, 1);
    wgpuComputePassEncoderEnd(cpass_enc);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, cpass_enc)
  }

  /* Graphics pass */
  {
    state.color_attachment.view         = wgpu_context->swapchain_view;
    state.depth_stencil_attachment.view = state.depth_texture.view;

    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.uniform_bind_group, 0,
                                      0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0, state.particles_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 1, state.quad_vertices.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rpass_enc, 6, PARTICLE_NUM, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  WGPU_RELEASE_RESOURCE(Buffer, state.particles_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad_vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer.buffer)

  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  wgpu_destroy_texture(&state.depth_texture);
  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)

  WGPU_RELEASE_RESOURCE(Buffer, state.simulation_ubo_buffer.buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_group)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader Particles WebGPU Logo",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* particle_shader_wgsl = CODE(
////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////
var<private> rand_seed : vec2f;

fn init_rand(invocation_id : u32, seed : vec4f) {
  rand_seed = seed.xz;
  rand_seed = fract(rand_seed * cos(35.456+f32(invocation_id) * seed.yw));
  rand_seed = fract(rand_seed * cos(41.235+f32(invocation_id) * seed.xw));
}

fn rand() -> f32 {
  rand_seed.x = fract(cos(dot(rand_seed, vec2f(23.14077926, 232.61690225))) * 136.8168);
  rand_seed.y = fract(cos(dot(rand_seed, vec2f(54.47856553, 345.84153136))) * 534.7645);
  return rand_seed.y;
}

////////////////////////////////////////////////////////////////////////////////
// Vertex shader
////////////////////////////////////////////////////////////////////////////////
struct RenderParams {
  modelViewProjectionMatrix : mat4x4f,
  right : vec3f,
  up : vec3f
}
@binding(0) @group(0) var<uniform> render_params : RenderParams;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) color : vec4f,
  @location(2) quad_pos : vec2f, // -1..+1
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) quad_pos : vec2f, // -1..+1
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  var quad_pos = mat2x3f(render_params.right, render_params.up) * in.quad_pos;
  var position = in.position + quad_pos * 0.01;
  var out : VertexOutput;
  out.position = render_params.modelViewProjectionMatrix * vec4f(position, 1.0);
  out.color = in.color;
  out.quad_pos = in.quad_pos;
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Fragment shader
////////////////////////////////////////////////////////////////////////////////
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
  var color = in.color;
  // Apply a circular particle alpha mask
  color.a = color.a * max(1.0 - length(in.quad_pos), 0.0);
  return color;
}

////////////////////////////////////////////////////////////////////////////////
// Simulation Compute shader
////////////////////////////////////////////////////////////////////////////////
struct SimulationParams {
  deltaTime : f32,
  brightnessFactor : f32,
  seed : vec4f,
}

struct Particle {
  position : vec3f,
  lifetime : f32,
  color    : vec4f,
  velocity : vec3f,
}

struct Particles {
  particles : array<Particle>,
}

@binding(0) @group(0) var<uniform> sim_params : SimulationParams;
@binding(1) @group(0) var<storage, read_write> data : Particles;
@binding(2) @group(0) var texture : texture_2d<f32>;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) global_invocation_id : vec3u) {
  let idx = global_invocation_id.x;

  init_rand(idx, sim_params.seed);

  var particle = data.particles[idx];

  // Apply gravity
  particle.velocity.z = particle.velocity.z - sim_params.deltaTime * 0.5;

  // Basic velocity integration
  particle.position = particle.position + sim_params.deltaTime * particle.velocity;

  // Age each particle. Fade out before vanishing.
  particle.lifetime = particle.lifetime - sim_params.deltaTime;
  particle.color.a = smoothstep(0.0, 0.5, particle.lifetime);

  // If the lifetime has gone negative, then the particle is dead and should be
  // respawned.
  if (particle.lifetime < 0.0) {
    // Use the probability map to find where the particle should be spawned.
    // Starting with the 1x1 mip level.
    var coord : vec2i;
    for (var level = u32(textureNumLevels(texture) - 1); level > 0; level--) {
      // Load the probability value from the mip-level
      // Generate a random number and using the probabilty values, pick the
      // next texel in the next largest mip level:
      //
      // 0.0    probabilites.r    probabilites.g    probabilites.b   1.0
      //  |              |              |              |              |
      //  |   TOP-LEFT   |  TOP-RIGHT   | BOTTOM-LEFT  | BOTTOM_RIGHT |
      //
      let probabilites = textureLoad(texture, coord, level);
      let value = vec4f(rand());
      let mask = (value >= vec4f(0.0, probabilites.xyz)) & (value < probabilites);
      coord = coord * 2;
      coord.x = coord.x + select(0, 1, any(mask.yw)); // x  y
      coord.y = coord.y + select(0, 1, any(mask.zw)); // z  w
    }
    let uv = vec2f(coord) / vec2f(textureDimensions(texture));
    particle.position = vec3f((uv - 0.5) * 3.0 * vec2f(1.0, -1.0), 0.0);
    particle.color = textureLoad(texture, coord, 0);
    particle.color.r *= sim_params.brightnessFactor;
    particle.color.g *= sim_params.brightnessFactor;
    particle.color.b *= sim_params.brightnessFactor;
    particle.velocity.x = (rand() - 0.5) * 0.1;
    particle.velocity.y = (rand() - 0.5) * 0.1;
    particle.velocity.z = rand() * 0.3;
    particle.lifetime = 0.5 + rand() * 3.0;
  }

  // Store the new particle value
  data.particles[idx] = particle;
}
);

static const char* probability_map_wgsl = CODE(
struct UBO {
  width : u32,
}

@binding(0) @group(0) var<uniform> ubo : UBO;
@binding(1) @group(0) var<storage, read> buf_in : array<f32>;
@binding(2) @group(0) var<storage, read_write> buf_out : array<f32>;
@binding(3) @group(0) var tex_in : texture_2d<f32>;
@binding(3) @group(0) var tex_out : texture_storage_2d<rgba8unorm, write>;

////////////////////////////////////////////////////////////////////////////////
// import_level
//
// Loads the alpha channel from a texel of the source image, and writes it to
// the buf_out.weights.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn import_level(@builtin(global_invocation_id) coord : vec3u) {
  _ = &buf_in; // so the bindGroups are similar.
  if (!all(coord.xy < vec2u(textureDimensions(tex_in)))) {
    return;
  }

  let offset = coord.x + coord.y * ubo.width;
  buf_out[offset] = textureLoad(tex_in, vec2i(coord.xy), 0).w;
}

////////////////////////////////////////////////////////////////////////////////
// export_level
//
// Loads 4 f32 weight values from buf_in.weights, and stores summed value into
// buf_out.weights, along with the calculated 'probabilty' vec4 values into the
// mip level of tex_out. See simulate() in particle.wgsl to understand the
// probability logic.
////////////////////////////////////////////////////////////////////////////////
@compute @workgroup_size(64)
fn export_level(@builtin(global_invocation_id) coord : vec3u) {
  if (!all(coord.xy < vec2u(textureDimensions(tex_out)))) {
    return;
  }

  let dst_offset = coord.x    + coord.y    * ubo.width;
  let src_offset = coord.x*2u + coord.y*2u * ubo.width;

  let a = buf_in[src_offset + 0u];
  let b = buf_in[src_offset + 1u];
  let c = buf_in[src_offset + 0u + ubo.width];
  let d = buf_in[src_offset + 1u + ubo.width];
  let sum = a + b + c + d;

  buf_out[dst_offset] = sum / 4.0;

  let probabilities = vec4f(a, a+b, a+b+c, sum) / max(sum, 0.0001);
  textureStore(tex_out, vec2i(coord.xy), probabilities);
}
);
// clang-format on
