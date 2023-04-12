#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cornell Box
 *
 * A classic Cornell box, using a lightmap generated using software ray-tracing.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/cornell
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Common holds the shared WGSL between the shaders, including the common
 * uniform buffer.
 * -------------------------------------------------------------------------- */

typedef struct {
  /** The common uniform buffer bind group and layout */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
  } uniforms;
  wgpu_context_t* wgpu_context;
  wgpu_buffer_t uniform_buffer;
  uint64_t frame;
} common_t;

/* -------------------------------------------------------------------------- *
 * Scene holds the cornell-box scene information.
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 center;
  vec3 right;
  vec3 up;
  vec3 color;
  float emissive;
} quad_t;

//      ─────────┐
//     ╱  +Y    ╱│
//    ┌────────┐ │
//    │        │+X
//    │   +Z   │ │
//    │        │╱
//    └────────┘
typedef enum {
  Positive_X,
  Positive_Y,
  Positive_Z,
  Negative_X,
  Negative_Y,
  Negative_Z,
} cube_face_t;

typedef struct {
  uint32_t vertex_count;
  uint32_t index_count;
  WGPUBuffer vertices;
  WGPUBuffer indices;
  WGPUVertexBufferLayout vertex_buffer_layout;
  WGPUBuffer quad_buffer;
  struct {
    uint32_t length;
    void* data;
  } quads;
  vec3 light_center;
  float light_width;
  float light_height;
} scene_t;

/* -------------------------------------------------------------------------- *
 * Radiosity computes lightmaps, calculated by software raytracing of light in
 * the scene.
 * -------------------------------------------------------------------------- */

typedef struct {
  // The output lightmap format and dimensions
  WGPUTextureFormat lightmap_format;
  uint32_t lightmap_width;
  uint32_t lightmap_height;
  // The output lightmap.
  texture_t lightmap;
  uint32_t lightmap_depth_or_array_layers;
  // Number of photons emitted per workgroup.
  // This is equal to the workgroup size (one photon per invocation)
  uint32_t photons_per_workgroup;
  // Number of radiosity workgroups dispatched per frame.
  uint32_t workgroups_per_frame;
  uint32_t photons_per_frame;
  // Maximum value that can be added to the 'accumulation' buffer, per photon,
  // across all texels.
  uint32_t photon_energy;
  // The total number of lightmap texels for all quads.
  uint32_t total_lightmap_texels;
  uint32_t accumulation_to_lightmap_workgroup_size_x;
  uint32_t accumulation_to_lightmap_workgroup_size_y;
  wgpu_context_t* wgpu_context;
  common_t* common;
  scene_t* scene;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline radiosity_pipeline;
  WGPUComputePipeline accumulation_to_lightmap_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  wgpu_buffer_t accumulation_buffer;
  wgpu_buffer_t uniform_buffer;
  // The 'accumulation' buffer average value
  float accumulation_mean;
  // The maximum value of 'accumulationAverage' before all values in
  // 'accumulation' are reduced to avoid integer overflows.
  uint32_t accumulation_mean_max;
} radiosity_t;

static void radiosity_init_defaults(radiosity_t* this)
{
  memset(this, 0, sizeof(*this));

  this->lightmap_format = WGPUTextureFormat_RGBA16Float;
  this->lightmap_width  = 256;
  this->lightmap_height = 256;

  this->photons_per_workgroup = 256;
  this->workgroups_per_frame  = 1024;
  this->photons_per_frame
    = this->photons_per_workgroup * this->workgroups_per_frame;

  this->photon_energy = 100000;

  this->accumulation_to_lightmap_workgroup_size_x = 16;
  this->accumulation_to_lightmap_workgroup_size_y = 16;

  this->accumulation_mean     = 0.0f;
  this->accumulation_mean_max = 0x10000000;
}

static void radiosity_create(radiosity_t* this, wgpu_context_t* wgpu_context,
                             common_t* common, scene_t* scene)
{
  radiosity_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->scene        = scene;

  /* Lightmap */
  {
    // Texture
    WGPUTextureDescriptor texture_desc = {
      .label         = "Radiosity.lightmap texture",
      .size          = (WGPUExtent3D) {
        .width              = this->lightmap_width,
        .height             = this->lightmap_height,
        .depthOrArrayLayers = this->scene->quads.length,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = this->lightmap_format,
      .usage
      = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
    };
    this->lightmap.texture
      = wgpuDeviceCreateTexture(this->wgpu_context->device, &texture_desc);
    ASSERT(this->lightmap.texture != NULL);
    this->lightmap_depth_or_array_layers = this->scene->quads.length;

    // Texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Radiosity.lightmap texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    this->lightmap.view
      = wgpuTextureCreateView(this->lightmap.texture, &texture_view_dec);
    ASSERT(this->lightmap.view != NULL);

    // Sampler
    WGPUSamplerDescriptor sampler_desc = {
      .label         = "Radiosity.lightmap texture sampler",
      .addressModeU  = WGPUAddressMode_ClampToEdge,
      .addressModeV  = WGPUAddressMode_ClampToEdge,
      .addressModeW  = WGPUAddressMode_ClampToEdge,
      .minFilter     = WGPUFilterMode_Linear,
      .magFilter     = WGPUFilterMode_Linear,
      .mipmapFilter  = WGPUFilterMode_Nearest,
      .lodMinClamp   = 0.0f,
      .lodMaxClamp   = (float)1,
      .maxAnisotropy = 1,
    };
    this->lightmap.sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  }

  /* Accumulation buffer */
  {
    this->accumulation_buffer = wgpu_create_buffer(
      this->wgpu_context, &(wgpu_buffer_desc_t){
                            .label = "Radiosity.accumulationBuffer",
                            .size = this->lightmap_width * this->lightmap_height
                                    * this->scene->quads.length * 16,
                            .usage = WGPUBufferUsage_Storage,
                          });
    this->total_lightmap_texels = this->lightmap_width * this->lightmap_height
                                  * this->scene->quads.length;
  }

  /* Uniform buffer */
  {
    this->accumulation_buffer = wgpu_create_buffer(
      this->wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Radiosity.uniformBuffer",
        .size  = 8 * 4, /* 8 x f32 */
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: accumulation buffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = this->accumulation_buffer.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: lightmap
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access = WGPUStorageTextureAccess_WriteOnly,
          .format = this->lightmap_format,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: radiosity_uniforms
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->uniform_buffer.size,
        },
        .sampler = {0},
      }
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Radiosity.bindGroupLayout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: accumulation buffer
        .binding = 0,
        .buffer  = this->accumulation_buffer.buffer,
        .offset  = 0,
        .size    = this->accumulation_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: lightmap
        .binding     = 1,
        .textureView = this->lightmap.view,
      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: radiosity_uniforms
        .binding = 2,
        .buffer  = this->uniform_buffer.buffer,
        .offset  = 0,
        .size    = this->uniform_buffer.size,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Radiosity.bindGroup",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Radiosity.accumulatePipelineLayout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Radiosity compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "PhotonsPerWorkgroup",
        .value = this->photons_per_workgroup,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "PhotonEnergy",
        .value = this->photon_energy,
      },
    };

    /* Compute shader */
    wgpu_shader_t radiosity_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label           = "radiosity_comp_shader",
                      .file            = "shaders/cornell_box/radiosity.wgsl",
                      .entry           = "radiosity",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->radiosity_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Radiosity.radiosityPipeline",
        .layout  = this->pipeline_layout,
        .compute = radiosity_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&radiosity_comp_shader);
  }

  /* Accumulation to lightmap compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "AccumulationToLightmapWorkgroupSizeX",
        .value = this->accumulation_to_lightmap_workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "AccumulationToLightmapWorkgroupSizeY",
        .value = this->accumulation_to_lightmap_workgroup_size_y,
      },
    };

    /* Compute shader */
    wgpu_shader_t accumulation_to_lightmap_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label           = "accumulation_to_lightmap_comp_shader",
                      .file            = "shaders/cornell_box/radiosity.wgsl",
                      .entry           = "accumulation_to_lightmap",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->accumulation_to_lightmap_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label  = "Radiosity.accumulationToLightmapPipeline",
        .layout = this->pipeline_layout,
        .compute
        = accumulation_to_lightmap_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&accumulation_to_lightmap_comp_shader);
  }
}

static void radiosity_destroy(radiosity_t* this)
{
  wgpu_destroy_texture(&this->lightmap);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->radiosity_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline,
                        this->accumulation_to_lightmap_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, this->accumulation_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffer.buffer)
}

static void radiosity_run(radiosity_t* this,
                          WGPUComputePassEncoder command_encoder)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  // Calculate the new mean value for the accumulation buffer
  this->accumulation_mean += (this->photons_per_frame * this->photon_energy)
                             / (float)this->total_lightmap_texels;

  // Calculate the 'accumulation' -> 'lightmap' scale factor from
  // 'accumulationMean'
  const float accumulation_to_lightmap_scale = 1.0f / this->accumulation_mean;
  // If 'accumulationMean' is greater than 'kAccumulationMeanMax', then reduce
  // the 'accumulation' buffer values to prevent u32 overflow.
  const float accumulation_buffer_scale
    = this->accumulation_mean > 2 * this->accumulation_mean_max ? 0.5f : 1.0f;
  this->accumulation_mean *= accumulation_buffer_scale;

  // Update the radiosity uniform buffer data.
  const float uniform_data_f32[7] = {
    accumulation_to_lightmap_scale, // accumulation_to_lightmap_scale */
    accumulation_buffer_scale,      // accumulation_buffer_scale */
    this->scene->light_width,       // light_width */
    this->scene->light_height,      // light_height */
    this->scene->light_center[0],   // light_center x */
    this->scene->light_center[1],   // light_center y */
    this->scene->light_center[2],   // light_center z */
  };
  wgpu_queue_write_buffer(wgpu_context, this->uniform_buffer.buffer, 0,
                          &uniform_data_f32, sizeof(uniform_data_f32));

  // Dispatch the radiosity workgroups
  wgpuComputePassEncoderSetBindGroup(
    command_encoder, 0, this->common->uniforms.bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(command_encoder, 1, this->bind_group, 0,
                                     NULL);
  wgpuComputePassEncoderSetPipeline(command_encoder, this->radiosity_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(command_encoder,
                                           this->workgroups_per_frame, 1, 1);
  wgpuComputePassEncoderEnd(command_encoder);

  // Then copy the 'accumulation' data to 'lightmap'
  wgpuComputePassEncoderSetPipeline(command_encoder,
                                    this->accumulation_to_lightmap_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    command_encoder,
    ceil(this->lightmap_width
         / (float)this->accumulation_to_lightmap_workgroup_size_x),
    ceil(this->lightmap_height
         / (float)this->accumulation_to_lightmap_workgroup_size_y),
    this->lightmap_depth_or_array_layers);
}

/* -------------------------------------------------------------------------- *
 * Rasterizer renders the scene using a regular raserization graphics pipeline.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  scene_t* scene;
  texture_t depth_texture;
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
} rasterizer_t;

static void rasterizer_init_defaults(rasterizer_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void rasterizer_create(rasterizer_t* this, wgpu_context_t* wgpu_context,
                              common_t* common, scene_t* scene,
                              radiosity_t* radiosity, texture_t* frame_buffer)
{
  rasterizer_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->scene        = scene;

  /* Depth texture */
  {
    // Create the texture
    WGPUExtent3D texture_extent = {
      .width              = frame_buffer->size.width,
      .height             = frame_buffer->size.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "RasterizerRenderer.depthTexture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    this->depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->depth_texture.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->depth_texture.view
      = wgpuTextureCreateView(this->depth_texture.texture, &texture_view_dec);
    ASSERT(this->depth_texture.view != NULL);
  }

  /* Render pass */
  {
    // Color attachment
    this->render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = frame_buffer->view,
     .loadOp     = WGPULoadOp_Clear,
     .storeOp    = WGPUStoreOp_Store,
     .clearValue = (WGPUColor) {
       .r = 0.1f,
       .g = 0.2f,
       .b = 0.3f,
       .a = 1.0f,
       },
     };

    // Depth-stencil attachment
    this->render_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = this->depth_texture.view,
        .depthClearValue = 1.0f,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
      };

    // Render pass descriptor
    this->render_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = "RasterizerRenderer.renderPassDescriptor",
      .colorAttachmentCount   = 1,
      .colorAttachments       = this->render_pass.color_attachments,
      .depthStencilAttachment = &this->render_pass.depth_stencil_attachment,
    };
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: lightmap
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: sampler
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout) {
          .type  = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = "'RasterizerRenderer.bindGroupLayout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: lightmap
        .binding = 0,
        .textureView  = radiosity->lightmap.view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: sampler
        .binding     = 1,
        .sampler = radiosity->lightmap.sampler,

      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "RasterizerRenderer.bindGroup",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "RasterizerRenderer.pipelineLayout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Rasterizer render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = frame_buffer->format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state_desc
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24Plus,
        .depth_write_enabled = true,
      });
    depth_stencil_state_desc.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Vertex shader WGSL
        .label = "RasterizerRenderer.vertex.module",
        .file  = "shaders/cornell_box/rasterizer.wgsl",
        .entry = "vs_main",
      },
      .buffer_count = 1,
      .buffers      = &this->scene->vertex_buffer_layout,
    });

    // Fragment state
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        // Fragment shader WGSL
        .label = "RasterizerRenderer.vertex.module",
        .file  = "shaders/cornell_box/rasterizer.wgsl",
        .entry = "fs_main",
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
    this->pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "RasterizerRenderer.pipeline",
                              .layout       = this->pipeline_layout,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });
    ASSERT(this->pipeline != NULL);

    // Shader modules are no longer needed once the graphics pipeline has been
    // created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
  }
}

static void rasterizer_destroy(rasterizer_t* this)
{
  wgpu_destroy_texture(&this->depth_texture);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void rasterizer_run(rasterizer_t* this,
                           WGPUCommandEncoder command_encoder)
{
  WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &this->render_pass.descriptor);
  wgpuRenderPassEncoderSetPipeline(pass_encoder, this->pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(pass_encoder, 0, this->scene->vertices,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass_encoder, this->scene->indices,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0,
                                    this->common->uniforms.bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(pass_encoder, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(pass_encoder, this->scene->index_count, 1, 0,
                                   0, 0);
  wgpuRenderPassEncoderEnd(pass_encoder);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass_encoder)
}

/* -------------------------------------------------------------------------- *
 * Raytracer renders the scene using a software ray-tracing compute pipeline.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  texture_t* frame_buffer;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
} raytracer_t;

static void raytracer_init_defaults(raytracer_t* this)
{
  memset(this, 0, sizeof(*this));

  this->workgroup_size_x = 16;
  this->workgroup_size_y = 16;
}

static void raytracer_create(raytracer_t* this, wgpu_context_t* wgpu_context,
                             common_t* common, radiosity_t* radiosity,
                             texture_t* frame_buffer)
{
  raytracer_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->frame_buffer = frame_buffer;

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: lightmap
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: sampler
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout) {
          .type  = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: framebuffer
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access = WGPUStorageTextureAccess_WriteOnly,
          .format = radiosity->lightmap_format,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "'Raytracer.bindGroupLayout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: lightmap
        .binding = 0,
        .textureView  = radiosity->lightmap.view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: sampler
        .binding     = 1,
        .sampler = radiosity->lightmap.sampler,

      },
      [2] = (WGPUBindGroupEntry) {
        // Binding 2: framebuffer
        .binding = 2,
        .textureView = frame_buffer->view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "rendererBindGroup",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "raytracerPipelineLayout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Raytracer compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeX",
        .value = this->workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeY",
        .value = this->workgroup_size_y,
      },
    };

    /* Compute shader */
    wgpu_shader_t raytracer_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label           = "raytracer_comp_shader",
                      .file            = "shaders/cornell_box/raytracer.wgsl",
                      .entry           = "main",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "raytracerPipeline",
        .layout  = this->pipeline_layout,
        .compute = raytracer_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&raytracer_comp_shader);
  }
}

static void raytracer_destroy(raytracer_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void raytracer_run(raytracer_t* this, WGPUCommandEncoder command_encoder)
{
  WGPUComputePassEncoder pass_encoder
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
  wgpuComputePassEncoderSetPipeline(pass_encoder, this->pipeline);
  wgpuComputePassEncoderSetBindGroup(
    pass_encoder, 0, this->common->uniforms.bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 1, this->bind_group, 0,
                                     NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass_encoder,
    ceil(this->frame_buffer->size.width / (float)this->workgroup_size_x),
    ceil(this->frame_buffer->size.height / (float)this->workgroup_size_y), 1);
  wgpuComputePassEncoderEnd(pass_encoder);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, pass_encoder)
}

/* -------------------------------------------------------------------------- *
 * Tonemapper implements a tonemapper to convert a linear-light framebuffer to
 * a gamma-correct, tonemapped framebuffer used for presentation.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
  float width;
  float height;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
} tonemapper_t;

static void tonemapper_init_defaults(tonemapper_t* this)
{
  memset(this, 0, sizeof(*this));

  this->workgroup_size_x = 16;
  this->workgroup_size_y = 16;
}

static void tonemapper_create(tonemapper_t* this, wgpu_context_t* wgpu_context,
                              common_t* common, texture_t* input,
                              texture_t* output)
{
  tonemapper_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->width        = input->size.width;
  this->height       = input->size.height;

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: input
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: output
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access = WGPUStorageTextureAccess_WriteOnly,
          .format = output->format,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Tonemapper.bindGroupLayout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: input
        .binding     = 0,
        .textureView = input->view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: output
        .binding     = 1,
        .textureView = output->view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Tonemapper.bindGroup",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->bind_group_layout, /* Group 0 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Tonemap.pipelineLayout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Tonemap compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeX",
        .value = this->workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeY",
        .value = this->workgroup_size_y,
      },
    };

    /* Compute shader */
    wgpu_shader_t tonemapper_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label           = "tonemapper_comp_shader",
                      .file            = "shaders/cornell_box/tonemapper.wgsl",
                      .entry           = "main",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Tonemap.pipeline",
        .layout  = this->pipeline_layout,
        .compute = tonemapper_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&tonemapper_comp_shader);
  }
}

static void tonemapper_destroy(tonemapper_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void tonemapper_run(tonemapper_t* this,
                           WGPUCommandEncoder command_encoder)
{
  WGPUComputePassEncoder pass_encoder
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 0, this->bind_group, 0,
                                     NULL);
  wgpuComputePassEncoderSetPipeline(pass_encoder, this->pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass_encoder, ceil(this->width / (float)this->workgroup_size_x),
    ceil(this->height / (float)this->workgroup_size_y), 1);
  wgpuComputePassEncoderEnd(pass_encoder);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, pass_encoder)
}

/* -------------------------------------------------------------------------- *
 * Cornell box example.
 * -------------------------------------------------------------------------- */

// Example structs
static struct {
  texture_t framebuffer;
  scene_t scene;
  common_t common;
  radiosity_t radiosity;
  rasterizer_t rasterizer;
  raytracer_t raytracer;
} example;

// GUI
typedef enum {
  RENDERER_RASTERIZER,
  RENDERER_RAYTRACER,
} renderer_t;

static struct {
  renderer_t renderer;
  bool rotate_camera;
} example_parms = {
  .renderer      = RENDERER_RASTERIZER,
  .rotate_camera = true,
};

static const char* renderer_names[2] = {"Rasterizer", "Raytracer"};

// Other variables
static const char* example_title = "Cornell box";
static bool prepared             = false;

static create_frame_buffer(wgpu_context_t* wgpu_context)
{
  // Create the texture
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "framebuffer texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_StorageBinding
             | WGPUTextureUsage_TextureBinding,
  };
  example.framebuffer.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(example.framebuffer.texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "framebuffer texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  example.framebuffer.view
    = wgpuTextureCreateView(example.framebuffer.texture, &texture_view_dec);
  ASSERT(example.framebuffer.view != NULL);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    int32_t current_renderer_index
      = (example_parms.renderer == RENDERER_RASTERIZER) ? 0 : 1;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Renderer",
                                &current_renderer_index, renderer_names, 11)) {
      example_parms.renderer = (current_renderer_index == 0) ?
                                 RENDERER_RASTERIZER :
                                 RENDERER_RAYTRACER;
    }
    imgui_overlay_checkBox(context->imgui_overlay, "Rotate Camera",
                           &example_parms.rotate_camera);
  }
}
