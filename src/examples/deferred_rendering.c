#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Deferred Rendering
 *
 * This example shows how to do deferred rendering with webgpu. Render geometry
 * info to multiple targets in the gBuffers in the first pass. In this sample we
 * have 2 gBuffers for normals and albedo, along with a depth texture. And then
 * do the lighting in a second pass with per fragment data read from gBuffers so
 * it's independent of scene complexity. World-space positions are reconstructed
 * from the depth texture and camera matrix. We also update light position in a
 * compute shader, where further operations like tile/cluster culling could
 * happen. The debug view shows the depth buffer on the left (flipped and scaled
 * a bit to make it more visible), the normal G buffer in the middle, and the
 * albedo G-buffer on the right side of the screen.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/tree/main/src/sample/deferredRendering
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* fragment_deferred_rendering_wgsl;
static const char* fragment_gbuffers_debug_view_wgsl;
static const char* fragment_write_gbuffers_wgsl;
static const char* light_update_wgsl;
static const char* vertex_texture_quad_wgsl;
static const char* vertex_write_gbuffers_wgsl;

/* -------------------------------------------------------------------------- *
 * Deferred Rendering example
 * -------------------------------------------------------------------------- */

/* Constants */
#define MAX_NUM_LIGHTS (1024u)

/* Render mode enum */
typedef enum render_mode_enum {
  RenderMode_Rendering    = 0,
  RenderMode_GBuffer_View = 1,
} render_mode_enum;

/* State struct */
static struct {
  /* Constants */
  uint32_t max_num_lights;
  uint8_t light_data_stride;
  vec3 light_extent_min;
  vec3 light_extent_max;

  /* Scene matrices */
  struct {
    vec4 eye_position;
    vec3 up_vector;
    vec3 origin;
    mat4 projection_matrix;
    mat4 view_proj_matrix;
  } view_matrices;

  /* Mesh */
  stanford_dragon_mesh_t stanford_dragon_mesh;

  /* Vertex and index buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t index_count;

  /* GBuffer */
  struct {
    WGPUTexture texture_2d_float16;
    WGPUTexture texture_albedo;
    WGPUTexture texture_depth;
    WGPUTextureView texture_views[3];
  } gbuffer;

  /* Uniform buffers */
  wgpu_buffer_t model_uniform_buffer;
  wgpu_buffer_t camera_uniform_buffer;
  uint64_t camera_uniform_buffer_size;

  /* Lights */
  struct {
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
  } lights;

  /* Bind groups */
  WGPUBindGroup scene_uniform_bind_group;
  WGPUBindGroup gbuffer_textures_bind_group;

  /* Bind group layouts */
  WGPUBindGroupLayout scene_uniform_bind_group_layout;
  WGPUBindGroupLayout gbuffer_textures_bind_group_layout;

  /* Pipelines */
  WGPURenderPipeline write_gbuffers_pipeline;
  WGPURenderPipeline gbuffers_debug_view_pipeline;
  WGPURenderPipeline deferred_render_pipeline;
  WGPUComputePipeline light_update_compute_pipeline;

  /* Pipeline layouts */
  WGPUPipelineLayout write_gbuffers_pipeline_layout;
  WGPUPipelineLayout gbuffers_debug_view_pipeline_layout;
  WGPUPipelineLayout deferred_render_pipeline_layout;
  WGPUPipelineLayout light_update_compute_pipeline_layout;

  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_attachments[2];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } write_gbuffer_pass;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } texture_quad_pass;

  /* Settings */
  struct {
    render_mode_enum current_render_mode;
    int32_t num_lights;
  } settings;

  /* Time tracking for animation */
  struct {
    uint64_t start_time;
    float elapsed_ms;
  } time;

  /* Initialization flag */
  WGPUBool initialized;
} state = {
  .max_num_lights   = (uint32_t)MAX_NUM_LIGHTS,
  .light_data_stride = 8,
  .light_extent_min  = {-50.f, -30.f, -50.f},
  .light_extent_max  = {50.f, 30.f, 50.f},
  .view_matrices = {
    .eye_position = {0.0f, 50.0f, -100.0f, 0.0f},
    .up_vector    = {0.0f, 1.0f, 0.0f},
    .origin       = GLM_VEC3_ZERO_INIT,
  },
  .settings = {
    .current_render_mode = RenderMode_Rendering,
    .num_lights          = 128,
  },
  .initialized = false,
};

/* Initialize vertex and index buffers for the Stanford dragon mesh */
static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context,
                                          stanford_dragon_mesh_t* dragon_mesh)
{
  /* Create the model vertex buffer */
  {
    const uint8_t ground_plane_vertex_count = 4;
    /* position: vec3, normal: vec3, uv: vec2 */
    const uint8_t vertex_stride = 8;
    uint64_t vertex_buffer_size
      = (dragon_mesh->positions.count + ground_plane_vertex_count)
        * vertex_stride * sizeof(float);
    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex,
      .size             = vertex_buffer_size,
      .mappedAtCreation = true,
    };
    state.vertex_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(state.vertex_buffer);
    float* mapping = (float*)wgpuBufferGetMappedRange(state.vertex_buffer, 0,
                                                      vertex_buffer_size);
    ASSERT(mapping);
    for (uint64_t i = 0; i < dragon_mesh->positions.count; ++i) {
      memcpy(&mapping[vertex_stride * i], dragon_mesh->positions.data[i],
             sizeof(vec3));
      memcpy(&mapping[vertex_stride * i + 3], dragon_mesh->normals.data[i],
             sizeof(vec3));
      memcpy(&mapping[vertex_stride * i + 6], dragon_mesh->uvs.data[i],
             sizeof(vec2));
    }
    /* Push vertex attributes for an additional ground plane */
    // clang-format off
    static const vec3 ground_plane_positions[4] = {
      {-100.0f, 20.0f, -100.0f}, //
      { 100.0f, 20.0f,  100.0f}, //
      {-100.0f, 20.0f,  100.0f}, //
      { 100.0f, 20.0f, -100.0f}, //
    };
    // clang-format on
    static const vec3 ground_plane_normals[4] = {
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
    };
    static const vec2 ground_plane_uvs[4] = {
      {0.0f, 0.0f}, //
      {1.0f, 1.0f}, //
      {0.0f, 1.0f}, //
      {1.0f, 0.0f}, //
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
    wgpuBufferUnmap(state.vertex_buffer);
  }

  /* Create the model index buffer */
  {
    const uint8_t ground_plane_index_count = 2;
    state.index_count
      = (dragon_mesh->triangles.count + ground_plane_index_count) * 3;
    uint64_t index_buffer_size       = state.index_count * sizeof(uint16_t);
    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Index buffer"),
      .usage            = WGPUBufferUsage_Index,
      .size             = index_buffer_size,
      .mappedAtCreation = true,
    };
    state.index_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(state.index_buffer);
    uint16_t* mapping = (uint16_t*)wgpuBufferGetMappedRange(
      state.index_buffer, 0, index_buffer_size);
    ASSERT(mapping);
    for (uint64_t i = 0; i < dragon_mesh->triangles.count; ++i) {
      memcpy(&mapping[3 * i], dragon_mesh->triangles.data[i],
             sizeof(uint16_t) * 3);
    }
    /* Push indices for an additional ground plane */
    static const uint16_t ground_plane_indices[2][3] = {
      {STANFORD_DRAGON_POSITION_COUNT_RES_4,
       STANFORD_DRAGON_POSITION_COUNT_RES_4 + 2,
       STANFORD_DRAGON_POSITION_COUNT_RES_4 + 1},
      {STANFORD_DRAGON_POSITION_COUNT_RES_4,
       STANFORD_DRAGON_POSITION_COUNT_RES_4 + 1,
       STANFORD_DRAGON_POSITION_COUNT_RES_4 + 3},
    };
    const uint64_t offset = dragon_mesh->triangles.count * 3;
    for (uint64_t i = 0; i < ground_plane_index_count; ++i) {
      memcpy(&mapping[offset + 3 * i], ground_plane_indices[i],
             sizeof(uint16_t) * 3);
    }
    wgpuBufferUnmap(state.index_buffer);
  }
}

/* GBuffer texture render targets  */
static void init_gbuffer_texture_render_targets(wgpu_context_t* wgpu_context)
{
  {
    WGPUTextureDescriptor texture_desc = {
      .label = STRVIEW("GBuffer - Texture view"),
      .size  = (WGPUExtent3D) {
        .width               = wgpu_context->width,
        .height              = wgpu_context->height,
        .depthOrArrayLayers  = 2,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    state.gbuffer.texture_2d_float16
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.gbuffer.texture_2d_float16 != NULL);
  }

  {
    WGPUTextureDescriptor texture_desc = {
      .label = STRVIEW("GBuffer - Albedo texture"),
      .size = (WGPUExtent3D) {
        .width               = wgpu_context->width,
        .height              = wgpu_context->height,
        .depthOrArrayLayers  = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    state.gbuffer.texture_albedo
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.gbuffer.texture_albedo != NULL);
  }

  {
    WGPUTextureDescriptor texture_desc = {
        .label = STRVIEW("GBuffer - Depth texture"),
        .size  = (WGPUExtent3D) {
            .width               = wgpu_context->width,
            .height              = wgpu_context->height,
            .depthOrArrayLayers  = 2,
        },
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .dimension     = WGPUTextureDimension_2D,
        .format        = WGPUTextureFormat_Depth24Plus,
        .usage         = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    state.gbuffer.texture_depth
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.gbuffer.texture_depth != NULL);
  }

  {
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("GBuffer albedo - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Undefined,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };

    texture_view_dec.format        = WGPUTextureFormat_RGBA16Float;
    state.gbuffer.texture_views[0] = wgpuTextureCreateView(
      state.gbuffer.texture_2d_float16, &texture_view_dec);
    ASSERT(state.gbuffer.texture_views[0] != NULL);

    texture_view_dec.format = WGPUTextureFormat_BGRA8Unorm;
    state.gbuffer.texture_views[1]
      = wgpuTextureCreateView(state.gbuffer.texture_albedo, &texture_view_dec);
    ASSERT(state.gbuffer.texture_views[1] != NULL);

    texture_view_dec.format = WGPUTextureFormat_Depth24Plus;
    state.gbuffer.texture_views[2]
      = wgpuTextureCreateView(state.gbuffer.texture_depth, &texture_view_dec);
    ASSERT(state.gbuffer.texture_views[2] != NULL);
  }
}

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  // GBuffer textures bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Position texture view
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Normal texture view
        .binding =   1,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: depth texture view
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Depth,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
        .storageTexture = {0},
      }
    };
    state.gbuffer_textures_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("GBuffer textures - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.gbuffer_textures_bind_group_layout != NULL);
  }

  // Lights buffer bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Storage buffer (Fragment shader) - LightsBuffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = sizeof(float) * state.light_data_stride * state.max_num_lights,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Fragment shader) - Config
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uint32_t),
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Uniform buffer (Fragment shader) - LightExtent
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4) * 2,
        },
        .storageTexture = {0},
      },
    };
    state.lights.buffer_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Lights buffer - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.lights.buffer_bind_group_layout != NULL);
  }

  // Scene uniform bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform buffer (Vertex shader) - Uniforms
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 4 * 16 * 2, // two 4x4 matrix
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Vertex shader) - Camera
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 4 * 16 * 2, // two 4x4 matrix
        },
        .storageTexture = {0},
      },
    };
    state.scene_uniform_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Scene uniform - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.scene_uniform_bind_group_layout != NULL);
  }

  // Lights buffer compute bind group layout
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Storage buffer (Compute shader) - LightsBuffer
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = sizeof(float) * state.light_data_stride * state.max_num_lights,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: Uniform buffer (Compute shader) - Config
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uint32_t),
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        // Binding 2: Uniform buffer (Compute shader) - LightExtent
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = state.camera_uniform_buffer_size,
        },
        .storageTexture = {0},
      },
    };
    state.lights.buffer_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device,
        &(WGPUBindGroupLayoutDescriptor){
          .label      = STRVIEW("Lights buffer compute - Bind group layout"),
          .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
          .entries    = bgl_entries,
        });
    ASSERT(state.lights.buffer_compute_bind_group_layout != NULL);
  }
}

static void init_render_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Write GBuffers pipeline layout */
  {
    state.write_gbuffers_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Write gbuffers - Pipeline layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.scene_uniform_bind_group_layout,
      });
    ASSERT(state.write_gbuffers_pipeline_layout != NULL);
  }

  /* GBuffers debug view pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      state.gbuffer_textures_bind_group_layout, /* set 0 */
    };
    state.gbuffers_debug_view_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label = STRVIEW("GBuffers debug view - Pipeline layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(state.gbuffers_debug_view_pipeline_layout != NULL);
  }

  /* Deferred render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      state.gbuffer_textures_bind_group_layout, /* set 0 */
      state.lights.buffer_bind_group_layout,    /* set 1 */
    };
    state.deferred_render_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Deferred render - Pipeline layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(state.deferred_render_pipeline_layout != NULL);
  }
}

static void init_write_gbuffers_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUColorTargetState color_target_states[2] = {
    /* Normal */
    [0] = (WGPUColorTargetState){
      .format    = WGPUTextureFormat_RGBA16Float,
      .writeMask = WGPUColorWriteMask_All,
    },
    /* Albedo */
    [1] = (WGPUColorTargetState){
      .format    = WGPUTextureFormat_BGRA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    },
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    write_gbuffers, sizeof(float) * 8,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3),
    /* Attribute location 2: uv */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2, sizeof(float) * 6))

  /* Vertex shader */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, vertex_write_gbuffers_wgsl);

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module      = vert_shader_module,
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 1,
    .buffers     = &write_gbuffers_vertex_buffer_layout,
  };

  /* Fragment shader */
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, fragment_write_gbuffers_wgsl);

  /* Fragment state */
  WGPUFragmentState fragment_state = {
    .module      = frag_shader_module,
    .entryPoint  = STRVIEW("main"),
    .targetCount = (uint32_t)ARRAY_SIZE(color_target_states),
    .targets     = color_target_states,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Create rendering pipeline using the specified states */
  state.write_gbuffers_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label        = STRVIEW("Write GBuffers - Render pipeline"),
      .layout       = state.write_gbuffers_pipeline_layout,
      .primitive    = primitive_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .depthStencil = &depth_stencil_state,
      .multisample  = multisample_state,
    });

  /* Shader modules are no longer needed once the graphics pipeline has been
   * created */
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module);
}

static void init_gbuffers_debug_view_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Constants */
  WGPUConstantEntry constant_entries[2] = {
    [0] = (WGPUConstantEntry){
      .key   = STRVIEW("canvasSizeWidth"),
      .value = wgpu_context->width,
    },
    [1] = (WGPUConstantEntry){
      .key   = STRVIEW("canvasSizeHeight"),
      .value = wgpu_context->height,
    },
  };

  /* Vertex shader */
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_texture_quad_wgsl);

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module      = vert_shader_module,
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 0,
    .buffers     = NULL,
  };

  /* Fragment shader */
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, fragment_gbuffers_debug_view_wgsl);

  /* Fragment state */
  WGPUFragmentState fragment_state = {
    .module        = frag_shader_module,
    .entryPoint    = STRVIEW("main"),
    .constantCount = (uint32_t)ARRAY_SIZE(constant_entries),
    .constants     = constant_entries,
    .targetCount   = 1,
    .targets       = &color_target_state,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Create rendering pipeline using the specified states */
  state.gbuffers_debug_view_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label       = STRVIEW("GBuffers debug view - Render pipeline"),
      .layout      = state.gbuffers_debug_view_pipeline_layout,
      .primitive   = primitive_state,
      .vertex      = vertex_state,
      .fragment    = &fragment_state,
      .multisample = multisample_state,
    });

  /* Shader modules are no longer needed once the graphics pipeline has been
   * created */
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module);
}

static void init_deferred_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex shader */
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_texture_quad_wgsl);

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module      = vert_shader_module,
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 0,
    .buffers     = NULL,
  };

  /* Fragment shader */
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, fragment_deferred_rendering_wgsl);

  /* Fragment state */
  WGPUFragmentState fragment_state = {
    .module      = frag_shader_module,
    .entryPoint  = STRVIEW("main"),
    .targetCount = 1,
    .targets     = &color_target_state,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count = 1,
    .mask  = 0xffffffff,
  };

  /* Create rendering pipeline using the specified states */
  state.deferred_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label     = STRVIEW("Deferred - Render pipeline"),
                            .layout    = state.deferred_render_pipeline_layout,
                            .primitive = primitive_state,
                            .vertex    = vertex_state,
                            .fragment  = &fragment_state,
                            .multisample = multisample_state,
                          });

  /* Shader modules are no longer needed once the graphics pipeline has been
   * created */
  WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module);
  WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module);
}

static void init_render_passes(void)
{
  /* Write GBuffer pass */
  {
    /* Color attachments */
    state.write_gbuffer_pass.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = state.gbuffer.texture_views[0],
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 1.0f,
          .a = 1.0f,
        },
      };

    state.write_gbuffer_pass.color_attachments[1] =
      (WGPURenderPassColorAttachment) {
        .view       = state.gbuffer.texture_views[1],
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

    /* Render pass depth stencil attachment descriptor */
    state.write_gbuffer_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = state.gbuffer.texture_views[2],
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilClearValue = 1,
      };

    /* Render pass descriptor */
    state.write_gbuffer_pass.descriptor = (WGPURenderPassDescriptor){
      .label = STRVIEW("Write GBuffer - Render pass"),
      .colorAttachmentCount
      = (uint32_t)ARRAY_SIZE(state.write_gbuffer_pass.color_attachments),
      .colorAttachments = state.write_gbuffer_pass.color_attachments,
      .depthStencilAttachment
      = &state.write_gbuffer_pass.depth_stencil_attachment,
    };
  }

  /* Texture Quad Pass */
  {
    /* Color attachment */
    state.texture_quad_pass.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = NULL, /* view is acquired and set in render loop. */
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
    state.texture_quad_pass.descriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Textured Quad - Render pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = state.texture_quad_pass.color_attachments,
    };
  }
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Config uniform buffer */
  {
    state.lights.config_uniform_buffer_size = sizeof(uint32_t);
    state.lights.config_uniform_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label            = STRVIEW("Config - Uniform buffer"),
             .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
             .size             = state.lights.config_uniform_buffer_size,
             .mappedAtCreation = true,
      });
    ASSERT(state.lights.config_uniform_buffer);
    uint32_t* config_data = (uint32_t*)wgpuBufferGetMappedRange(
      state.lights.config_uniform_buffer, 0,
      state.lights.config_uniform_buffer_size);
    ASSERT(config_data);
    config_data[0] = state.settings.num_lights;
    wgpuBufferUnmap(state.lights.config_uniform_buffer);
  }

  /* Model uniform buffer */
  {
    state.model_uniform_buffer = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Model - Uniform buffer",
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = sizeof(mat4) * 2, /* two 4x4 matrix */
      });
    ASSERT(state.model_uniform_buffer.buffer);
  }

  /* Camera uniform buffer */
  {
    state.camera_uniform_buffer = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Camera - Uniform buffer",
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = sizeof(mat4) * 2, /* two 4x4 matrix */
      });
    ASSERT(state.camera_uniform_buffer.buffer);
  }

  /* Scene uniform bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.model_uniform_buffer.buffer,
        .size    = state.model_uniform_buffer.size, /* two 4x4 matrix */
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.camera_uniform_buffer.buffer,
        .size    = state.camera_uniform_buffer.size, /* two 4x4 matrix */
      },
    };
    state.scene_uniform_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Scene uniform - Bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.write_gbuffers_pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.scene_uniform_bind_group != NULL);
  }

  /* GBuffer textures bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = state.gbuffer.texture_views[0],
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = state.gbuffer.texture_views[1],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = state.gbuffer.texture_views[2],
      },
    };
    state.gbuffer_textures_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("GBuffer textures - Bind group"),
        .layout     = state.gbuffer_textures_bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.gbuffer_textures_bind_group != NULL);
  }
}

static void init_compute_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Light update compute pipeline layout */
  {
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = STRVIEW("Light update compute - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.lights.buffer_compute_bind_group_layout,
    };
    state.light_update_compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(state.light_update_compute_pipeline_layout != NULL);
  }
}

static void init_light_update_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Compute shader */
  WGPUShaderModule comp_shader_module
    = wgpu_create_shader_module(wgpu_context->device, light_update_wgsl);

  /* Create pipeline */
  state.light_update_compute_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Light update - Compute pipeline"),
      .layout  = state.light_update_compute_pipeline_layout,
      .compute = {
        .module     = comp_shader_module,
        .entryPoint = STRVIEW("main"),
      },
    });

  /* Shader module no longer needed */
  WGPU_RELEASE_RESOURCE(ShaderModule, comp_shader_module);
}

static void init_lights(wgpu_context_t* wgpu_context)
{
  /* Lights buffer */
  {
    // Lights data are uploaded in a storage buffer
    // which could be updated/culled/etc. with a compute shader
    vec3 extent = GLM_VEC3_ZERO_INIT;
    glm_vec3_sub(state.light_extent_max, state.light_extent_min, extent);
    state.lights.buffer_size
      = sizeof(float) * state.light_data_stride * state.max_num_lights;
    state.lights.buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device, &(WGPUBufferDescriptor){
                              .label = STRVIEW("Lights - Storage buffer"),
                              .usage = WGPUBufferUsage_Storage,
                              .size  = state.lights.buffer_size,
                              .mappedAtCreation = true,
                            });
    ASSERT(state.lights.buffer);

    // We randomly populate lights randomly in a box range
    // And simply move them along y-axis per frame to show they are dynamic
    // lightings
    float* light_data = (float*)wgpuBufferGetMappedRange(
      state.lights.buffer, 0, state.lights.buffer_size);
    ASSERT(light_data);
    vec4 tmp_vec4 = GLM_VEC4_ZERO_INIT;
    for (uint32_t i = 0, offset = 0; i < state.max_num_lights; ++i) {
      offset = state.light_data_stride * i;
      // position
      for (uint8_t j = 0; j < 3; j++) {
        tmp_vec4[j] = random_float_min_max(0.0f, 1.0f) * extent[j]
                      + state.light_extent_min[j];
      }
      tmp_vec4[3] = 1.0f;
      memcpy(&light_data[offset], tmp_vec4, sizeof(vec4));
      // color
      tmp_vec4[0] = random_float_min_max(0.0f, 1.0f) * 2.0f;
      tmp_vec4[1] = random_float_min_max(0.0f, 1.0f) * 2.0f;
      tmp_vec4[2] = random_float_min_max(0.0f, 1.0f) * 2.0f;
      // radius
      tmp_vec4[3] = 20.0f;
      memcpy(&light_data[offset + 4], tmp_vec4, sizeof(vec4));
    }
    wgpuBufferUnmap(state.lights.buffer);
  }

  /* Lights extent buffer */
  {
    state.lights.extent_buffer_size = 4 * 8;
    state.lights.extent_buffer      = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
             .label = STRVIEW("Lights extent - Uniform buffer"),
             .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
             .size  = state.lights.extent_buffer_size,
      });
    float light_extent_data[8] = {0};
    memcpy(&light_extent_data[0], state.light_extent_min, sizeof(vec3));
    memcpy(&light_extent_data[4], state.light_extent_max, sizeof(vec3));
    wgpuQueueWriteBuffer(wgpu_context->queue, state.lights.extent_buffer, 0,
                         &light_extent_data, state.lights.extent_buffer_size);
  }

  /* Lights buffer bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.lights.buffer,
        .size    = state.lights.buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.lights.config_uniform_buffer,
        .size    = state.lights.config_uniform_buffer_size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.camera_uniform_buffer.buffer,
        .size    = state.camera_uniform_buffer.size,
      },
    };
    state.lights.buffer_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Lights buffer - Bind group"),
                              .layout = state.lights.buffer_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.lights.buffer_bind_group != NULL);
  }

  /* Lights buffer compute bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.lights.buffer,
        .size    = state.lights.buffer_size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.lights.config_uniform_buffer,
        .size    = state.lights.config_uniform_buffer_size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.lights.extent_buffer,
        .size    = state.lights.extent_buffer_size,
      },
    };
    state.lights.buffer_compute_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label  = STRVIEW("Lights buffer compute - Bind group"),
        .layout = wgpuComputePipelineGetBindGroupLayout(
          state.light_update_compute_pipeline, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.lights.buffer_compute_bind_group != NULL);
  }
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  float aspect_ratio = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Scene matrices */
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, state.view_matrices.up_vector);
  glm_vec3_copy((vec3){0.0f, 0.0f, 0.0f}, state.view_matrices.origin);

  glm_mat4_identity(state.view_matrices.projection_matrix);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                  state.view_matrices.projection_matrix);

  /* Move the model so it's centered. */
  mat4 model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_translate(model_matrix, (vec3){0.0f, -45.0f, 0.0f});

  /* Write data to buffers */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.model_uniform_buffer.buffer,
                       0, model_matrix, sizeof(mat4));

  /* Normal model data */
  mat4 invert_transpose_model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_inv(model_matrix, invert_transpose_model_matrix);
  glm_mat4_transpose(invert_transpose_model_matrix);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.model_uniform_buffer.buffer,
                       64, invert_transpose_model_matrix, sizeof(mat4));
}

/**
 * @brief Rotates the given 4-by-4 matrix around the y-axis by the given angle.
 * @param m - The matrix.
 * @param angle_in_radians - The angle by which to rotate (in radians).
 * @param dst - matrix to hold result.
 * @returns The rotated matrix.
 */
static mat4* glm_mat4_rotate_y(mat4 m, float angle_in_radians, mat4* dst)
{
  const float m00 = m[0][0];
  const float m01 = m[0][1];
  const float m02 = m[0][2];
  const float m03 = m[0][3];
  const float m20 = m[2][0];
  const float m21 = m[2][1];
  const float m22 = m[2][2];
  const float m23 = m[2][3];
  const float c   = cosf(angle_in_radians);
  const float s   = sinf(angle_in_radians);
  (*dst)[0][0]    = c * m00 - s * m20;
  (*dst)[0][1]    = c * m01 - s * m21;
  (*dst)[0][2]    = c * m02 - s * m22;
  (*dst)[0][3]    = c * m03 - s * m23;
  (*dst)[2][0]    = c * m20 + s * m00;
  (*dst)[2][1]    = c * m21 + s * m01;
  (*dst)[2][2]    = c * m22 + s * m02;
  (*dst)[2][3]    = c * m23 + s * m03;
  if (m != *dst) {
    (*dst)[1][0] = m[1][0];
    (*dst)[1][1] = m[1][1];
    (*dst)[1][2] = m[1][2];
    (*dst)[1][3] = m[1][3];
    (*dst)[3][0] = m[3][0];
    (*dst)[3][1] = m[3][1];
    (*dst)[3][2] = m[3][2];
    (*dst)[3][3] = m[3][3];
  }
  return dst;
}

/**
 * @brief Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by which to translate.
 * @param dst - matrix to hold result. If not passed a new one is created.
 * @returns The translation matrix.
 */
static mat4* glm_mat4_translation(vec3 v, mat4* dst)
{
  (*dst)[0][0] = 1.0f;
  (*dst)[0][1] = 0.0f;
  (*dst)[0][2] = 0.0f;
  (*dst)[0][3] = 0.0f;
  (*dst)[1][0] = 0.0f;
  (*dst)[1][1] = 1.0f;
  (*dst)[1][2] = 0.0f;
  (*dst)[1][3] = 0.0f;
  (*dst)[2][0] = 0.0f;
  (*dst)[2][1] = 0.0f;
  (*dst)[2][2] = 1.0f;
  (*dst)[2][3] = 0.0f;
  (*dst)[3][0] = v[0];
  (*dst)[3][1] = v[1];
  (*dst)[3][2] = v[2];
  (*dst)[3][3] = 1.0f;
  return dst;
}

/**
 * @brief Transform vec4 by 4x4 matrix.
 * @param v - the vector
 * @param m - The matrix.
 * @param dst - vec4 to store result.
 * @returns the transformed vector
 */
static vec4* glm_vec4_transform_mat4(vec4 v, mat4 m, vec4* dst)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];
  const float w = v[3];
  (*dst)[0]     = m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0] * w;
  (*dst)[1]     = m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1] * w;
  (*dst)[2]     = m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2] * w;
  (*dst)[3]     = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3] * w;
  return dst;
}

/* Rotates the camera around the origin based on time. */
static mat4* get_camera_view_proj_matrix(float time_ms)
{
  const float rad  = PI * (time_ms / 5000.0f);
  mat4 translation = GLM_MAT4_IDENTITY_INIT, rotation = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_translation(state.view_matrices.origin, &translation);
  glm_mat4_rotate_y(translation, rad, &rotation);
  vec4 rotated_eye_position = GLM_VEC4_ZERO_INIT;
  glm_vec4_transform_mat4(state.view_matrices.eye_position, rotation,
                          &rotated_eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(rotated_eye_position,          /* eye    */
             state.view_matrices.origin,    /* center */
             state.view_matrices.up_vector, /* up     */
             view_matrix                    /* dest   */
  );

  glm_mat4_mulN((mat4*[]){&state.view_matrices.projection_matrix, &view_matrix},
                2, state.view_matrices.view_proj_matrix);
  return &state.view_matrices.view_proj_matrix;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context, float time_ms)
{
  mat4* camera_view_proj = get_camera_view_proj_matrix(time_ms);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform_buffer.buffer,
                       0, *camera_view_proj, sizeof(mat4));
  mat4 camera_inv_view_proj = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_inv(*camera_view_proj, camera_inv_view_proj);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform_buffer.buffer,
                       64, camera_inv_view_proj, sizeof(mat4));
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stanford_dragon_mesh_init(&state.stanford_dragon_mesh);
    init_vertex_and_index_buffers(wgpu_context, &state.stanford_dragon_mesh);
    init_gbuffer_texture_render_targets(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_render_pipeline_layouts(wgpu_context);
    init_write_gbuffers_pipeline(wgpu_context);
    init_gbuffers_debug_view_pipeline(wgpu_context);
    init_deferred_render_pipeline(wgpu_context);
    init_render_passes();
    init_uniform_buffers(wgpu_context);
    init_compute_pipeline_layout(wgpu_context);
    init_light_update_compute_pipeline(wgpu_context);
    init_lights(wgpu_context);
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
    /* Recreate GBuffer textures and bind groups on resize */
    for (uint8_t i = 0; i < 3; ++i) {
      WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.texture_views[i])
    }
    WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_2d_float16)
    WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_albedo)
    WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_depth)
    WGPU_RELEASE_RESOURCE(BindGroup, state.gbuffer_textures_bind_group)
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.gbuffers_debug_view_pipeline)

    init_gbuffer_texture_render_targets(wgpu_context);
    init_gbuffers_debug_view_pipeline(wgpu_context);
    init_render_passes();

    /* Recreate GBuffer textures bind group */
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = state.gbuffer.texture_views[0],
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = state.gbuffer.texture_views[1],
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = state.gbuffer.texture_views[2],
      },
    };
    state.gbuffer_textures_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("GBuffer textures - Bind group"),
        .layout     = state.gbuffer_textures_bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.gbuffer_textures_bind_group != NULL);

    init_view_matrices(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update time */
  if (state.time.start_time == 0) {
    state.time.start_time = (uint64_t)(glfwGetTime() * 1000.0);
  }
  state.time.elapsed_ms
    = (float)((uint64_t)(glfwGetTime() * 1000.0) - state.time.start_time);

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context, state.time.elapsed_ms);

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    /* Write position, normal, albedo etc. data to gBuffers */
    WGPURenderPassEncoder gbuffer_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.write_gbuffer_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(gbuffer_pass,
                                     state.write_gbuffers_pipeline);
    wgpuRenderPassEncoderSetBindGroup(gbuffer_pass, 0,
                                      state.scene_uniform_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(gbuffer_pass, 0, state.vertex_buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(gbuffer_pass, state.index_buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(gbuffer_pass, state.index_count, 1, 0, 0,
                                     0);
    wgpuRenderPassEncoderEnd(gbuffer_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, gbuffer_pass)
  }

  {
    /* Update lights position */
    WGPUComputePassEncoder light_pass
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(light_pass,
                                      state.light_update_compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      light_pass, 0, state.lights.buffer_compute_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      light_pass, (uint32_t)ceilf(state.max_num_lights / 64.f), 1, 1);
    wgpuComputePassEncoderEnd(light_pass);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, light_pass)
  }

  {
    if (state.settings.current_render_mode == RenderMode_GBuffer_View) {
      // GBuffers debug view
      // Left: depth
      // Middle: normal
      // Right: albedo (use uv to mimic a checkerboard texture)
      state.texture_quad_pass.color_attachments[0].view
        = wgpu_context->swapchain_view;
      WGPURenderPassEncoder debug_view_pass = wgpuCommandEncoderBeginRenderPass(
        cmd_enc, &state.texture_quad_pass.descriptor);
      wgpuRenderPassEncoderSetPipeline(debug_view_pass,
                                       state.gbuffers_debug_view_pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        debug_view_pass, 0, state.gbuffer_textures_bind_group, 0, 0);
      wgpuRenderPassEncoderDraw(debug_view_pass, 6, 1, 0, 0);
      wgpuRenderPassEncoderEnd(debug_view_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, debug_view_pass)
    }
    else {
      // Deferred rendering */
      state.texture_quad_pass.color_attachments[0].view
        = wgpu_context->swapchain_view;
      WGPURenderPassEncoder deferred_rendering_pass
        = wgpuCommandEncoderBeginRenderPass(
          cmd_enc, &state.texture_quad_pass.descriptor);
      wgpuRenderPassEncoderSetPipeline(deferred_rendering_pass,
                                       state.deferred_render_pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        deferred_rendering_pass, 0, state.gbuffer_textures_bind_group, 0, 0);
      wgpuRenderPassEncoderSetBindGroup(deferred_rendering_pass, 1,
                                        state.lights.buffer_bind_group, 0, 0);
      wgpuRenderPassEncoderDraw(deferred_rendering_pass, 6, 1, 0, 0);
      wgpuRenderPassEncoderEnd(deferred_rendering_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, deferred_rendering_pass)
    }
  }

  /* Create command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* Clean up used resources */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_2d_float16)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_albedo)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.texture_depth)
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(state.gbuffer.texture_views);
       ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.texture_views[i])
  }
  wgpu_destroy_buffer(&state.model_uniform_buffer);
  wgpu_destroy_buffer(&state.camera_uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.lights.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.lights.extent_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.lights.config_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.lights.buffer_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.lights.buffer_compute_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.gbuffer_textures_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.lights.buffer_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.lights.buffer_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.scene_uniform_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.gbuffer_textures_bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.write_gbuffers_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gbuffers_debug_view_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.deferred_render_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.light_update_compute_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.write_gbuffers_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        state.gbuffers_debug_view_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.deferred_render_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        state.light_update_compute_pipeline_layout)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Deferred Rendering",
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
static const char* fragment_deferred_rendering_wgsl = CODE(
  @group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
  @group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
  @group(0) @binding(2) var gBufferDepth: texture_depth_2d;

  struct LightData {
    position : vec4<f32>,
    color : vec3<f32>,
    radius : f32,
  }
  struct LightsBuffer {
    lights: array<LightData>,
  }
  @group(1) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;

  struct Config {
    numLights : u32,
  }
  struct Camera {
    viewProjectionMatrix : mat4x4<f32>,
    invViewProjectionMatrix : mat4x4<f32>,
  }
  @group(1) @binding(1) var<uniform> config: Config;
  @group(1) @binding(2) var<uniform> camera: Camera;

  fn world_from_screen_coord(coord : vec2<f32>, depth_sample: f32) -> vec3<f32> {
    // reconstruct world-space position from the screen coordinate.
    let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
    let posWorldW = camera.invViewProjectionMatrix * posClip;
    let posWorld = posWorldW.xyz / posWorldW.www;
    return posWorld;
  }

  @fragment
  fn main(
    @builtin(position) coord : vec4<f32>
  ) -> @location(0) vec4<f32> {
    var result : vec3<f32>;

    let depth = textureLoad(
      gBufferDepth,
      vec2<i32>(floor(coord.xy)),
      0
    );

    // Don't light the sky.
    if (depth >= 1.0) {
      discard;
    }

    let bufferSize = textureDimensions(gBufferDepth);
    let coordUV = coord.xy / vec2<f32>(bufferSize);
    let position = world_from_screen_coord(coordUV, depth);

    let normal = textureLoad(
      gBufferNormal,
      vec2<i32>(floor(coord.xy)),
      0
    ).xyz;

    let albedo = textureLoad(
      gBufferAlbedo,
      vec2<i32>(floor(coord.xy)),
      0
    ).rgb;

    for (var i = 0u; i < config.numLights; i++) {
      let L = lightsBuffer.lights[i].position.xyz - position;
      let distance = length(L);
      if (distance > lightsBuffer.lights[i].radius) {
        continue;
      }
      let lambert = max(dot(normal, normalize(L)), 0.0);
      result += vec3<f32>(
        lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo
      );
    }

    // some manual ambient
    result += vec3(0.2);

    return vec4(result, 1.0);
  }
);

static const char* fragment_gbuffers_debug_view_wgsl = CODE(
  @group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
  @group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
  @group(0) @binding(2) var gBufferDepth: texture_depth_2d;

  override canvasSizeWidth: f32;
  override canvasSizeHeight: f32;

  @fragment
  fn main(
    @builtin(position) coord : vec4<f32>
  ) -> @location(0) vec4<f32> {
    var result : vec4<f32>;
    let c = coord.xy / vec2<f32>(canvasSizeWidth, canvasSizeHeight);
    if (c.x < 0.33333) {
      let rawDepth = textureLoad(
        gBufferDepth,
        vec2<i32>(floor(coord.xy)),
        0
      );
      // remap depth into something a bit more visible
      let depth = (1.0 - rawDepth) * 50.0;
      result = vec4(depth);
    } else if (c.x < 0.66667) {
      result = textureLoad(
        gBufferNormal,
        vec2<i32>(floor(coord.xy)),
        0
      );
      result.x = (result.x + 1.0) * 0.5;
      result.y = (result.y + 1.0) * 0.5;
      result.z = (result.z + 1.0) * 0.5;
    } else {
      result = textureLoad(
        gBufferAlbedo,
        vec2<i32>(floor(coord.xy)),
        0
      );
    }
    return result;
  }
);

static const char* fragment_write_gbuffers_wgsl = CODE(
  struct GBufferOutput {
    @location(0) normal : vec4<f32>,

    // Textures: diffuse color, specular color, smoothness, emissive etc. could go here
    @location(1) albedo : vec4<f32>,
  }

  @fragment
  fn main(
    @location(0) fragNormal: vec3<f32>,
    @location(1) fragUV : vec2<f32>
  ) -> GBufferOutput {
    // faking some kind of checkerboard texture
    let uv = floor(30.0 * fragUV);
    let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));

    var output : GBufferOutput;
    output.normal = vec4(normalize(fragNormal), 1.0);
    output.albedo = vec4(c, c, c, 1.0);

    return output;
  }
);

static const char* light_update_wgsl = CODE(
  struct LightData {
    position : vec4<f32>,
    color : vec3<f32>,
    radius : f32,
  }
  struct LightsBuffer {
    lights: array<LightData>,
  }
  @group(0) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;

  struct Config {
    numLights : u32,
  }
  @group(0) @binding(1) var<uniform> config: Config;

  struct LightExtent {
    min : vec4<f32>,
    max : vec4<f32>,
  }
  @group(0) @binding(2) var<uniform> lightExtent: LightExtent;

  @compute @workgroup_size(64, 1, 1)
  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var index = GlobalInvocationID.x;
    if (index >= config.numLights) {
      return;
    }

    lightsBuffer.lights[index].position.y = lightsBuffer.lights[index].position.y - 0.5 - 0.003 * (f32(index) - 64.0 * floor(f32(index) / 64.0));

    if (lightsBuffer.lights[index].position.y < lightExtent.min.y) {
      lightsBuffer.lights[index].position.y = lightExtent.max.y;
    }
  }
);

static const char* vertex_texture_quad_wgsl = CODE(
  @vertex
  fn main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> @builtin(position) vec4<f32> {
    const pos = array(
      vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
      vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );

    return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
  }
);

static const char* vertex_write_gbuffers_wgsl = CODE(
  struct Uniforms {
    modelMatrix : mat4x4<f32>,
    normalModelMatrix : mat4x4<f32>,
  }
  struct Camera {
    viewProjectionMatrix : mat4x4<f32>,
    invViewProjectionMatrix : mat4x4<f32>,
  }
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var<uniform> camera : Camera;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragNormal: vec3<f32>,    // normal in world space
    @location(1) fragUV: vec2<f32>,
  }

  @vertex
  fn main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    let worldPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
    output.Position = camera.viewProjectionMatrix * vec4(worldPosition, 1.0);
    output.fragNormal = normalize((uniforms.normalModelMatrix * vec4(normal, 1.0)).xyz);
    output.fragUV = uv;
    return output;
  }
);

// clang-format on
