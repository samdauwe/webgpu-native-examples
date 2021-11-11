#include "example_base.h"
#include "examples.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Deferred Rendering
 *
 * This example shows how to do deferred rendering with webgpu.
 * Render geometry info to multiple targets in the gBuffers in the first pass.
 * In this sample we have 3 gBuffers for positions, normals, and albedo.
 * And then do the lighting in a second pass with per fragment data read from
 * gBuffers so it's independent of scene complexity. We also update light
 * position in a compute shader, where further operations like tile/cluster
 * culling could happen.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/deferredRendering
 * -------------------------------------------------------------------------- */

// Constants
#define MAX_NUM_LIGHTS 1024
static const uint32_t max_num_lights   = (uint32_t)MAX_NUM_LIGHTS;
static const uint8_t light_data_stride = 8;
static vec3 light_extent_min           = {-50.f, -30.f, -50.f};
static vec3 light_extent_max           = {50.f, 30.f, 50.f};

static struct view_matrices_t {
  vec3 up_vector;
  vec3 origin;
  mat4 projection_matrix;
  mat4 view_proj_matrix;
} view_matrices = {0};

static stanford_dragon_mesh_t stanford_dragon_mesh = {0};

// Vertex and index buffers
static WGPUBuffer vertex_buffer;
static WGPUBuffer index_buffer;
static uint32_t index_count;

// GBuffer
static struct gbuffer_t {
  WGPUTexture texture_2d_float;
  WGPUTexture texture_albedo;
  WGPUTextureView texture_views[3];
} gbuffer = {0};

// Depth texture
static WGPUTexture depth_texture;
static WGPUTextureView depth_texture_view;

// Uniform buffers
static WGPUBuffer model_uniform_buffer;
static WGPUBuffer camera_uniform_buffer;
static WGPUBuffer surface_size_uniform_buffer;

// Lights
static struct lights_t {
  WGPUBuffer buffer;
  uint64_t buffer_size;
  WGPUBuffer extent_buffer;
  uint64_t extent_buffer_size;
  WGPUBuffer config_uniform_buffer;
  uint64_t config_uniform_buffer_size;
  WGPUBindGroup buffer_bind_group;
  WGPUBindGroupLayout buffer_bind_group_layout;
  WGPUBindGroup buffer_compute_bind_group;
  WGPUBindGroupLayout buffer_compute_bind_group_layout;
} lights = {0};

// Bind groups
static WGPUBindGroup scene_uniform_bind_group;
static WGPUBindGroup surface_size_uniform_bind_group;
static WGPUBindGroup gbuffer_textures_bind_group;

// Bind group layouts
static WGPUBindGroupLayout scene_uniform_bind_group_layout;
static WGPUBindGroupLayout surface_size_uniform_bind_group_layout;
static WGPUBindGroupLayout gbuffer_textures_bind_group_layout;

// Pipelines
static WGPURenderPipeline write_gbuffers_pipeline;
static WGPURenderPipeline gbuffers_debug_view_pipeline;
static WGPURenderPipeline deferred_render_pipeline;
static WGPUComputePipeline light_update_compute_pipeline;

// Pipeline layouts
static WGPUPipelineLayout write_gbuffers_pipeline_layout;
static WGPUPipelineLayout gbuffers_debug_view_pipeline_layout;
static WGPUPipelineLayout deferred_render_pipeline_layout;
static WGPUPipelineLayout light_update_compute_pipeline_layout;

// Render pass descriptor
static struct write_gbuffer_pass_t {
  WGPURenderPassColorAttachment color_attachments[3];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} write_gbuffer_pass = {0};

static struct texture_quad_pass_t {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} texture_quad_pass = {0};

typedef enum render_mode_enum {
  RenderMode_Rendering    = 0,
  RenderMode_GBuffer_View = 1,
} render_mode_enum;

static struct settings_t {
  render_mode_enum current_render_mode;
  int32_t num_lights;
} settings = {
  .current_render_mode = RenderMode_Rendering,
  .num_lights          = 128,
};

// Other variables
static const char* example_title = "Deferred Rendering";
static bool prepared             = false;

// Prepare vertex and index buffers for the Stanford dragon mesh
static void
prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context,
                                 stanford_dragon_mesh_t* dragon_mesh)
{
  // Create the model vertex buffer
  {
    const uint8_t ground_plane_vertex_count = 4;
    // position: vec3, normal: vec3, uv: vec2
    const uint8_t vertex_stride = 8;
    uint64_t vertex_buffer_size
      = (dragon_mesh->positions.count + ground_plane_vertex_count)
        * vertex_stride * sizeof(float);
    WGPUBufferDescriptor buffer_desc = {
      .usage            = WGPUBufferUsage_Vertex,
      .size             = vertex_buffer_size,
      .mappedAtCreation = true,
    };
    vertex_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(vertex_buffer)
    float* mapping
      = (float*)wgpuBufferGetMappedRange(vertex_buffer, 0, vertex_buffer_size);
    ASSERT(mapping)
    for (uint64_t i = 0; i < dragon_mesh->positions.count; ++i) {
      memcpy(&mapping[vertex_stride * i], dragon_mesh->positions.data[i],
             sizeof(vec3));
      memcpy(&mapping[vertex_stride * i + 3], dragon_mesh->normals.data[i],
             sizeof(vec3));
      memcpy(&mapping[vertex_stride * i + 6], dragon_mesh->uvs.data[i],
             sizeof(vec2));
    }
    // Push vertex attributes for an additional ground plane
    static const vec3 ground_plane_positions[4] = {
      {-100.0f, 20.0f, -100.0f}, //
      {100.0f, 20.0f, 100.0f},   //
      {-100.0f, 20.0f, 100.0f},  //
      {100.0f, 20.0f, -100.0f}   //
    };
    static const vec3 ground_plane_normals[4] = {
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}  //
    };
    static const vec2 ground_plane_uvs[4] = {
      {0.0f, 0.0f}, //
      {1.0f, 1.0f}, //
      {0.0f, 1.0f}, //
      {1.0f, 0.0f}  //
    };
    const uint64_t offset = dragon_mesh->positions.count * vertex_stride;
    for (uint64_t i = 0; i < ground_plane_vertex_count; ++i) {
      memcpy(&mapping[offset + vertex_stride * i], ground_plane_positions[i],
             sizeof(vec3));
      memcpy(&mapping[offset + vertex_stride * i + 3], ground_plane_normals[i],
             sizeof(vec3));
      memcpy(&mapping[offset + vertex_stride * i + 6], ground_plane_uvs[i],
             sizeof(vec2));
    }
    wgpuBufferUnmap(vertex_buffer);
  }

  // Create the model index buffer
  {
    const uint8_t ground_plane_index_count = 2;
    index_count = (dragon_mesh->triangles.count + ground_plane_index_count) * 3;
    uint64_t index_buffer_size       = index_count * sizeof(uint16_t);
    WGPUBufferDescriptor buffer_desc = {
      .usage            = WGPUBufferUsage_Index,
      .size             = index_buffer_size,
      .mappedAtCreation = true,
    };
    index_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(index_buffer)
    uint16_t* mapping
      = (uint16_t*)wgpuBufferGetMappedRange(index_buffer, 0, index_buffer_size);
    ASSERT(mapping)
    for (uint64_t i = 0; i < dragon_mesh->triangles.count; ++i) {
      memcpy(&mapping[3 * i], dragon_mesh->triangles.data[i],
             sizeof(uint16_t) * 3);
    }
    // Push indices for an additional ground plane
    static const uint16_t ground_plane_indices[2][3] = {
      {POSITION_COUNT_RES_4, POSITION_COUNT_RES_4 + 2,
       POSITION_COUNT_RES_4 + 1},
      {POSITION_COUNT_RES_4, POSITION_COUNT_RES_4 + 1,
       POSITION_COUNT_RES_4 + 3},
    };
    const uint64_t offset = dragon_mesh->triangles.count * 3;
    for (uint64_t i = 0; i < ground_plane_index_count; ++i) {
      memcpy(&mapping[offset + 3 * i], ground_plane_indices[i],
             sizeof(uint16_t) * 3);
    }
    wgpuBufferUnmap(index_buffer);
  }
}

// GBuffer texture render targets
static void prepare_gbuffer_texture_render_targets(wgpu_context_t* wgpu_context)
{
  {
    WGPUTextureDescriptor texture_desc = {
      .size          = (WGPUExtent3D) {
        .width               = wgpu_context->surface.width,
        .height              = wgpu_context->surface.height,
        .depthOrArrayLayers  = 3,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA32Float,
      .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    gbuffer.texture_2d_float
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  }

  {
    WGPUTextureDescriptor texture_desc = {
      .size          = (WGPUExtent3D) {
        .width               = wgpu_context->surface.width,
        .height              = wgpu_context->surface.height,
        .depthOrArrayLayers  = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    gbuffer.texture_albedo
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  }

  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_RGBA32Float,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };

    gbuffer.texture_views[0]
      = wgpuTextureCreateView(gbuffer.texture_2d_float, &texture_view_dec);

    texture_view_dec.baseArrayLayer = 1;
    gbuffer.texture_views[1]
      = wgpuTextureCreateView(gbuffer.texture_2d_float, &texture_view_dec);

    texture_view_dec.format         = WGPUTextureFormat_BGRA8Unorm;
    texture_view_dec.baseArrayLayer = 0;
    gbuffer.texture_views[2]
      = wgpuTextureCreateView(gbuffer.texture_albedo, &texture_view_dec);
  }
}

static void prepare_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  // Lights buffer bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Storage buffer (Fragment shader) - LightsBuffer
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Storage,
          .minBindingSize = sizeof(float) * light_data_stride * max_num_lights,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Fragment shader) - Config
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uint32_t),
        },
        .storageTexture = {0},
      },
    };
    lights.buffer_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(lights.buffer_bind_group_layout != NULL)
  }

  // Lights buffer compute bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Storage buffer (Compute shader) - LightsBuffer
        .binding = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Storage,
          .minBindingSize = sizeof(float) * light_data_stride * max_num_lights,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Compute shader) - Config
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uint32_t),
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Uniform buffer (Compute shader) - LightExtent
        .binding = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(vec4) * 2,
        },
        .storageTexture = {0},
      },
    };
    lights.buffer_compute_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(lights.buffer_compute_bind_group_layout != NULL)
  }

  // Scene uniform bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) - Uniforms
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = 4 * 16 * 2, // two 4x4 matrix
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Vertex shader) - Camera
        .binding = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = 4 * 16, // 4x4 matrix
        },
        .storageTexture = {0},
      },
    };
    scene_uniform_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(scene_uniform_bind_group_layout != NULL)
  }

  // Surface size uniform bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Fragment shader) - SurfaceConstants
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(vec2),
        },
        .sampler = {0},
      }
    };
    surface_size_uniform_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(surface_size_uniform_bind_group_layout != NULL)
  }

  // GBuffer textures bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Texture view
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Texture view
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Texture view
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      }
    };
    gbuffer_textures_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(gbuffer_textures_bind_group_layout != NULL)
  }
}

static void prepare_render_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  // Write GBuffers pipeline layout
  {
    write_gbuffers_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &scene_uniform_bind_group_layout,
      });
    ASSERT(write_gbuffers_pipeline_layout != NULL)
  }

  // GBuffers debug view pipeline layout
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      gbuffer_textures_bind_group_layout,     // set 0
      surface_size_uniform_bind_group_layout, // set 1
    };
    gbuffers_debug_view_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(gbuffers_debug_view_pipeline_layout != NULL)
  }

  // Deferred render pipeline layout
  {
    WGPUBindGroupLayout bind_group_layouts[3] = {
      gbuffer_textures_bind_group_layout,     // set 0
      lights.buffer_bind_group_layout,        // set 1
      surface_size_uniform_bind_group_layout, // set 2
    };
    deferred_render_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(deferred_render_pipeline_layout != NULL)
  }
}

static void prepare_write_gbuffers_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUColorTargetState color_target_state_descs[3] = {
    // position
    [0] = (WGPUColorTargetState){
      .format = WGPUTextureFormat_RGBA32Float,
      .writeMask = WGPUColorWriteMask_All,
    },
    // normal
    [1] = (WGPUColorTargetState){
      .format = WGPUTextureFormat_RGBA32Float,
      .writeMask = WGPUColorWriteMask_All,
    },
    // albedo
    [2] = (WGPUColorTargetState){
      .format = WGPUTextureFormat_BGRA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    },
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state_desc
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state_desc.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    write_gbuffers, sizeof(float) * 8,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3),
    // Attribute location 2: uv
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2, sizeof(float) * 6))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader SPIR-V
                  .file = "shaders/deferred_rendering/write_gbuffers.vert.spv",
                },
                .buffer_count = 1,
                .buffers = &write_gbuffers_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader SPIR-V
                  .file = "shaders/deferred_rendering/write_gbuffers.frag.spv",
                },
                .target_count = 3,
                .targets = color_target_state_descs,
              });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  write_gbuffers_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "write_gbuffers_render_pipeline",
                            .layout       = write_gbuffers_pipeline_layout,
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

static void prepare_gbuffers_debug_view_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .file = "shaders/deferred_rendering/texture_quad.vert.spv",
            },
            .buffer_count = 0,
            .buffers = NULL,
          });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .file = "shaders/deferred_rendering/gbuffers_debug_view.frag.spv",
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
  gbuffers_debug_view_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = "gbuffers_debug_view_render_pipeline",
                            .layout    = gbuffers_debug_view_pipeline_layout,
                            .primitive = primitive_state_desc,
                            .vertex    = vertex_state_desc,
                            .fragment  = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void prepare_deferred_render_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = WGPUTextureFormat_BGRA8Unorm,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
            wgpu_context, &(wgpu_vertex_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Vertex shader SPIR-V
              .file = "shaders/deferred_rendering/texture_quad.vert.spv",
            },
            .buffer_count = 0,
            .buffers = NULL,
          });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
            wgpu_context, &(wgpu_fragment_state_t){
            .shader_desc = (wgpu_shader_desc_t){
              // Fragment shader SPIR-V
              .file = "shaders/deferred_rendering/deferred_rendering.frag.spv",
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
  deferred_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "deferred_render_pipeline",
                            .layout      = deferred_render_pipeline_layout,
                            .primitive   = primitive_state_desc,
                            .vertex      = vertex_state_desc,
                            .fragment    = &fragment_state_desc,
                            .multisample = multisample_state_desc,
                          });

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24Plus,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  depth_texture = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = WGPUTextureFormat_Depth24Plus,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  depth_texture_view = wgpuTextureCreateView(depth_texture, &texture_view_dec);
}

static void setup_render_passes()
{
  // Write GBuffer pass
  {
    // Color attachments
    write_gbuffer_pass.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = gbuffer.texture_views[0],
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 1.0f,
          .g = 1.0f,
          .b = 1.0f,
          .a = 1.0f,
        },
      };

    write_gbuffer_pass.color_attachments[1] =
      (WGPURenderPassColorAttachment) {
        .view       = gbuffer.texture_views[1],
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 1.0f,
          .a = 1.0f,
        },
      };

    write_gbuffer_pass.color_attachments[2] =
      (WGPURenderPassColorAttachment) {
        .view       = gbuffer.texture_views[2],
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };

    // Render pass depth stencil attachment descriptor
    write_gbuffer_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view           = depth_texture_view,
        .depthLoadOp    = WGPULoadOp_Clear,
        .depthStoreOp   = WGPUStoreOp_Store,
        .clearDepth     = 1.0f,
        .stencilLoadOp  = WGPULoadOp_Clear,
        .stencilStoreOp = WGPUStoreOp_Store,
        .clearStencil   = 0,
      };

    // Render pass descriptor
    write_gbuffer_pass.descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 3,
      .colorAttachments       = write_gbuffer_pass.color_attachments,
      .depthStencilAttachment = &write_gbuffer_pass.depth_stencil_attachment,
    };
  }

  // Texture Quad Pass
  {
    // Color attachment
    texture_quad_pass.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = NULL, // attachment is acquired and set in render loop.
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };

    // Render pass descriptor
    texture_quad_pass.descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = texture_quad_pass.color_attachments,
    };
  }
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Config uniform buffer
  {
    lights.config_uniform_buffer_size = sizeof(uint32_t);
    lights.config_uniform_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size             = lights.config_uniform_buffer_size,
        .mappedAtCreation = true,
      });
    ASSERT(lights.config_uniform_buffer);
    uint32_t* config_data = (uint32_t*)wgpuBufferGetMappedRange(
      lights.config_uniform_buffer, 0, lights.config_uniform_buffer_size);
    ASSERT(config_data);
    config_data[0] = settings.num_lights;
    wgpuBufferUnmap(lights.config_uniform_buffer);
  }

  // Model uniform buffer
  {
    const WGPUBufferDescriptor buffer_desc = {
      .size  = 4 * 16 * 2, // two 4x4 matrix
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    model_uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(model_uniform_buffer)
  }

  // Camera uniform buffer
  {
    const WGPUBufferDescriptor buffer_desc = {
      .size  = 4 * 16, // 4x4 matrix
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    camera_uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(camera_uniform_buffer)
  }

  // Scene uniform bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = model_uniform_buffer,
        .size = 4 * 16 * 2, // two 4x4 matrix
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer = camera_uniform_buffer,
        .size = 4 * 16, // 4x4 matrix
      },
    };
    scene_uniform_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                write_gbuffers_pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(scene_uniform_bind_group != NULL)
  }

  // Surface size uniform buffer
  {
    const WGPUBufferDescriptor buffer_desc = {
      .size  = 4 * 2,
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    surface_size_uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(surface_size_uniform_buffer)
  }

  // Surface size uniform bind group
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = surface_size_uniform_buffer,
        .size = 4 * 2,
      },
    };
    surface_size_uniform_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout = surface_size_uniform_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(surface_size_uniform_bind_group != NULL)
  }

  // GBuffer textures bind group
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .textureView = gbuffer.texture_views[0],
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = gbuffer.texture_views[1],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = gbuffer.texture_views[2],
      },
    };
    gbuffer_textures_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                gbuffers_debug_view_pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(gbuffer_textures_bind_group != NULL)
  }
}

static float rand_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

static void prepare_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Light update compute pipeline layout
  {
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &lights.buffer_compute_bind_group_layout,
    };
    light_update_compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(light_update_compute_pipeline_layout != NULL)
  }
}

static void prepare_light_update_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  wgpu_shader_t light_update_comp_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Compute shader SPIR-V
                    .file = "shaders/deferred_rendering/light_update.comp.spv",
                  });

  /* Create pipeline */
  light_update_compute_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .layout  = light_update_compute_pipeline_layout,
      .compute = light_update_comp_shader.programmable_stage_descriptor,
    });

  /* Partial clean-up */
  wgpu_shader_release(&light_update_comp_shader);
}

static void prepare_lights(wgpu_context_t* wgpu_context)
{
  // Lights buffer
  {
    // Lights data are uploaded in a storage buffer
    // which could be updated/culled/etc. with a compute shader
    vec3 extent = GLM_VEC3_ZERO_INIT;
    glm_vec3_sub(light_extent_max, light_extent_min, extent);
    lights.buffer_size = sizeof(float) * light_data_stride * max_num_lights;
    lights.buffer      = wgpuDeviceCreateBuffer(wgpu_context->device,
                                           &(WGPUBufferDescriptor){
                                             .usage = WGPUBufferUsage_Storage,
                                             .size  = lights.buffer_size,
                                             .mappedAtCreation = true,
                                           });
    ASSERT(lights.buffer);

    // We randomly populate lights randomly in a box range
    // And simply move them along y-axis per frame to show they are dynamic
    // lightings
    float* light_data
      = (float*)wgpuBufferGetMappedRange(lights.buffer, 0, lights.buffer_size);
    ASSERT(light_data);
    vec4 tmp_vec4   = GLM_VEC4_ZERO_INIT;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < max_num_lights; ++i) {
      offset = light_data_stride * i;
      // position
      for (uint8_t j = 0; j < 3; j++) {
        tmp_vec4[j]
          = rand_float_min_max(0.0f, 1.0f) * extent[j] + light_extent_min[j];
      }
      tmp_vec4[3] = 1.0f;
      memcpy(&light_data[offset], tmp_vec4, sizeof(vec4));
      // color
      tmp_vec4[0] = rand_float_min_max(0.0f, 1.0f) * 2.0f;
      tmp_vec4[1] = rand_float_min_max(0.0f, 1.0f) * 2.0f;
      tmp_vec4[2] = rand_float_min_max(0.0f, 1.0f) * 2.0f;
      // radius
      tmp_vec4[3] = 20.0f;
      memcpy(&light_data[offset + 4], tmp_vec4, sizeof(vec4));
    }
    wgpuBufferUnmap(lights.buffer);
  }

  // Lights extent buffer
  {
    lights.extent_buffer_size = 4 * 8;
    lights.extent_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = lights.extent_buffer_size,
      });
    float light_extent_data[8] = {0};
    memcpy(&light_extent_data[0], light_extent_min, sizeof(vec3));
    memcpy(&light_extent_data[4], light_extent_max, sizeof(vec3));
    wgpu_queue_write_buffer(wgpu_context, lights.extent_buffer, 0,
                            &light_extent_data, lights.extent_buffer_size);
  }

  // Lights buffer bind group
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = lights.buffer,
        .size =lights.buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer = lights.config_uniform_buffer,
        .size = lights.config_uniform_buffer_size,
      },
    };
    lights.buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = lights.buffer_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(lights.buffer_bind_group != NULL)
  }

  // Lights buffer compute bind group
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = lights.buffer,
        .size =lights.buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer = lights.config_uniform_buffer,
        .size = lights.config_uniform_buffer_size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer = lights.extent_buffer,
        .size =lights.extent_buffer_size,
      },
    };
    lights.buffer_compute_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout = wgpuComputePipelineGetBindGroupLayout(
                                light_update_compute_pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(lights.buffer_compute_bind_group != NULL)
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  // Scene matrices
  vec3 eye_position = {0.0f, 50.0f, -100.0f};
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, view_matrices.up_vector);
  glm_vec3_copy((vec3){0.0f, 0.0f, 0.0f}, view_matrices.origin);

  glm_mat4_identity(view_matrices.projection_matrix);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                  view_matrices.projection_matrix);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,            //
             view_matrices.origin,    //
             view_matrices.up_vector, //
             view_matrix);

  mat4 view_proj_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_mulN((mat4*[]){&view_matrices.projection_matrix, &view_matrix}, 2,
                view_proj_matrix);

  // Move the model so it's centered.
  mat4 model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_translate(model_matrix, (vec3){0.0f, -5.0f, 0.0f});
  glm_translate(model_matrix, (vec3){0.0f, -40.0f, 0.0f});

  // Write data to buffers
  wgpuQueueWriteBuffer(wgpu_context->queue, camera_uniform_buffer, 0,
                       view_proj_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue, model_uniform_buffer, 0,
                       model_matrix, sizeof(mat4));

  // Normal model data
  mat4 invert_transpose_model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_inv(model_matrix, invert_transpose_model_matrix);
  glm_mat4_transpose(invert_transpose_model_matrix);
  wgpuQueueWriteBuffer(wgpu_context->queue, model_uniform_buffer, 64,
                       invert_transpose_model_matrix, sizeof(mat4));

  // Pass the surface size to shader to help sample from gBuffer textures using
  // coord
  const vec2 surface_size_data
    = {(float)wgpu_context->surface.width, (float)wgpu_context->surface.height};
  wgpuQueueWriteBuffer(wgpu_context->queue, surface_size_uniform_buffer, 0,
                       surface_size_data, sizeof(vec2));
}

/**
 * @brief Rotate a 3D vector around the y-axis
 * @param a The vec3 point to rotate
 * @param b The origin of the rotation
 * @param rad The angle of rotation in radians
 * @param  out The receiving vec3
 * @see https://glmatrix.net/docs/vec3.js.html#line593
 */
static void glm_vec3_rotate_y(vec3 a, vec3 b, float rad, vec3* out)
{
  vec3 p, r;

  // Translate point to the origin
  p[0] = a[0] - b[0];
  p[1] = a[1] - b[1];
  p[2] = a[2] - b[2];

  // perform rotation

  r[0] = p[2] * sin(rad) + p[0] * cos(rad);
  r[1] = p[1];
  r[2] = p[2] * cos(rad) - p[0] * sin(rad);

  // translate to correct position
  (*out)[0] = r[0] + b[0];
  (*out)[1] = r[1] + b[1];
  (*out)[2] = r[2] + b[2];
}

// Rotates the camera around the origin based on time.
static mat4* get_camera_view_proj_matrix(wgpu_example_context_t* context)
{
  vec3 eye_position = {0.0f, 50.0f, -100.0f};

  const float rad = PI * (context->frame.timestamp_millis / 5000.0f);
  glm_vec3_rotate_y(eye_position, view_matrices.origin, rad, &eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,            //
             view_matrices.origin,    //
             view_matrices.up_vector, //
             view_matrix);

  glm_mat4_mulN((mat4*[]){&view_matrices.projection_matrix, &view_matrix}, 2,
                view_matrices.view_proj_matrix);
  return &view_matrices.view_proj_matrix;
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  mat4* camera_view_proj = get_camera_view_proj_matrix(context);
  wgpuQueueWriteBuffer(context->wgpu_context->queue, camera_uniform_buffer, 0,
                       *camera_view_proj, sizeof(mat4));
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    stanford_dragon_mesh_init(&stanford_dragon_mesh);
    prepare_vertex_and_index_buffers(context->wgpu_context,
                                     &stanford_dragon_mesh);
    prepare_gbuffer_texture_render_targets(context->wgpu_context);
    prepare_depth_texture(context->wgpu_context);
    prepare_bind_group_layouts(context->wgpu_context);
    prepare_render_pipeline_layouts(context->wgpu_context);
    prepare_write_gbuffers_pipeline(context->wgpu_context);
    prepare_gbuffers_debug_view_pipeline(context->wgpu_context);
    prepare_deferred_render_pipeline(context->wgpu_context);
    setup_render_passes();
    prepare_uniform_buffers(context->wgpu_context);
    prepare_compute_pipeline_layout(context->wgpu_context);
    prepare_light_update_compute_pipeline(context->wgpu_context);
    prepare_lights(context->wgpu_context);
    prepare_view_matrices(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    static const char* mode[3] = {"rendering", "gBuffers view"};
    int32_t item_index         = (int32_t)settings.current_render_mode;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Mode", &item_index,
                                mode, 2)) {
      settings.current_render_mode = (render_mode_enum)item_index;
    }
    if (imgui_overlay_slider_int(context->imgui_overlay, "Number of Lights",
                                 &settings.num_lights, 1, max_num_lights)) {
      uint32_t num_lights[1] = {(uint32_t)settings.num_lights};
      wgpuQueueWriteBuffer(context->wgpu_context->queue,
                           lights.config_uniform_buffer, 0, num_lights,
                           sizeof(uint32_t));
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
#if 10
  {
    // Write position, normal, albedo etc. data to gBuffers
    WGPURenderPassEncoder gbuffer_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &write_gbuffer_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(gbuffer_pass, write_gbuffers_pipeline);
    wgpuRenderPassEncoderSetBindGroup(gbuffer_pass, 0, scene_uniform_bind_group,
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(gbuffer_pass, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      gbuffer_pass, index_buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(gbuffer_pass, index_count, 1, 0, 0, 0);
    wgpuRenderPassEncoderEndPass(gbuffer_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, gbuffer_pass)
  }
#endif
#if 10
  {
    // Update lights position
    WGPUComputePassEncoder light_pass
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(light_pass,
                                      light_update_compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      light_pass, 0, lights.buffer_compute_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatch(light_pass, max_num_lights, 1, 1);
    wgpuComputePassEncoderEndPass(light_pass);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, light_pass)
  }
#endif
  {
    if (settings.current_render_mode == RenderMode_GBuffer_View) {
      // GBuffers debug view
      // Left: position
      // Middle: normal
      // Right: albedo (use uv to mimic a checkerboard texture)
      texture_quad_pass.color_attachments[0].view
        = wgpu_context->swap_chain.frame_buffer;
      WGPURenderPassEncoder debug_view_pass = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc, &texture_quad_pass.descriptor);
      wgpuRenderPassEncoderSetPipeline(debug_view_pass,
                                       gbuffers_debug_view_pipeline);
      wgpuRenderPassEncoderSetBindGroup(debug_view_pass, 0,
                                        gbuffer_textures_bind_group, 0, 0);
      wgpuRenderPassEncoderSetBindGroup(debug_view_pass, 1,
                                        surface_size_uniform_bind_group, 0, 0);
      wgpuRenderPassEncoderDraw(debug_view_pass, 6, 1, 0, 0);
      wgpuRenderPassEncoderEndPass(debug_view_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, debug_view_pass)
    }
    else {
      // Deferred rendering
      texture_quad_pass.color_attachments[0].view
        = wgpu_context->swap_chain.frame_buffer;
      WGPURenderPassEncoder deferred_rendering_pass
        = wgpuCommandEncoderBeginRenderPass(wgpu_context->cmd_enc,
                                            &texture_quad_pass.descriptor);
      wgpuRenderPassEncoderSetPipeline(deferred_rendering_pass,
                                       deferred_render_pipeline);
      wgpuRenderPassEncoderSetBindGroup(deferred_rendering_pass, 0,
                                        gbuffer_textures_bind_group, 0, 0);
      wgpuRenderPassEncoderSetBindGroup(deferred_rendering_pass, 1,
                                        lights.buffer_bind_group, 0, 0);
      wgpuRenderPassEncoderSetBindGroup(deferred_rendering_pass, 2,
                                        surface_size_uniform_bind_group, 0, 0);
      wgpuRenderPassEncoderDraw(deferred_rendering_pass, 6, 1, 0, 0);
      wgpuRenderPassEncoderEndPass(deferred_rendering_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, deferred_rendering_pass)
    }
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
    update_uniform_buffers(context);
  }
  return draw_result;
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, index_buffer)
  WGPU_RELEASE_RESOURCE(Texture, gbuffer.texture_2d_float)
  WGPU_RELEASE_RESOURCE(Texture, gbuffer.texture_albedo)
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(gbuffer.texture_views); ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, gbuffer.texture_views[i])
  }
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, depth_texture_view)
  WGPU_RELEASE_RESOURCE(Buffer, model_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, camera_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, surface_size_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, lights.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, lights.extent_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, lights.config_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, lights.buffer_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, lights.buffer_compute_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, scene_uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, surface_size_uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, gbuffer_textures_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, lights.buffer_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        lights.buffer_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, scene_uniform_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, surface_size_uniform_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, gbuffer_textures_bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, write_gbuffers_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, gbuffers_debug_view_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, deferred_render_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, light_update_compute_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, write_gbuffers_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, gbuffers_debug_view_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, deferred_render_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, light_update_compute_pipeline_layout)
}

void example_deferred_rendering(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title  = example_title,
      .overlay = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy
  });
  // clang-format on
}
