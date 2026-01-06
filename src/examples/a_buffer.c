#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <cJSON.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - A-Buffer
 *
 * This example demonstrates order independent transparency using a per-pixel
 * linked-list of translucent fragments.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/a-buffer
 * teapot: https://github.com/mikolalysenko/teapot
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* opaque_shader_wgsl;
static const char* translucent_shader_wgsl;
static const char* composite_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * A-Buffer example
 * -------------------------------------------------------------------------- */

/* Determines how much memory is allocated to store linked-list elements */
static const uint32_t average_layers_per_fragment = 4;

/* State struct */
static struct {
  utah_teapot_mesh_t teapot_mesh;
  uint8_t file_buffer[256 * 1024]; /* 256KB buffer for JSON file */
  WGPULimits device_limits;
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
    wgpu_buffer_t heads;
    wgpu_buffer_t heads_init;
    wgpu_buffer_t linked_list;
    wgpu_buffer_t slice_info;
    wgpu_buffer_t uniform;
  } buffers;
  struct {
    vec3 up_vector;
    vec3 origin;
    vec3 eye_position;
    mat4 projection_matrix;
    mat4 view_proj_matrix;
  } view_matrices;
  struct {
    mat4 model_view_projection_matrix;
    uint32_t max_storable_fragments;
    uint32_t target_width;
  } ubo_data;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUTextureFormat format;
  } depth_texture;
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
    WGPURenderPipeline pipeline;
    WGPUBindGroup bind_group;
  } opaque_pass;
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDescriptor descriptor;
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
  } translucent_pass;
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDescriptor descriptor;
    WGPURenderPipeline pipeline;
    WGPUPipelineLayout pipeline_layout;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
  } composite_pass;
  uint32_t num_slices;
  uint32_t slice_height;
  WGPUBool mesh_loaded;
  WGPUBool initialized;
} state = {
  .view_matrices = {
    .up_vector         = {0.0f, 1.0f, 0.0f},
    .origin            = GLM_VEC3_ZERO_INIT,
    .eye_position      = {0.0f, 5.0f, -100.0f},
    .projection_matrix = GLM_MAT4_IDENTITY_INIT,
    .view_proj_matrix  = GLM_MAT4_IDENTITY_INIT,
  },
  .depth_texture = {
    .format = WGPUTextureFormat_Depth32Float,
  },
  .mesh_loaded = false,
  .initialized = false,
};

/* Callback for asynchronously loading the teapot JSON file */
static void teapot_json_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    fprintf(stderr, "File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* Parse JSON and create teapot mesh */
  const char* json_data = (const char*)response->data.ptr;
  if (utah_teapot_mesh_init(&state.teapot_mesh, json_data) == EXIT_SUCCESS) {
    utah_teapot_mesh_compute_normals(&state.teapot_mesh);
    state.mesh_loaded = true;
  }
  else {
    fprintf(stderr, "Failed to load Utah teapot mesh\n");
  }
}

static void init_device_limits(wgpu_context_t* wgpu_context)
{
  wgpuDeviceGetLimits(wgpu_context->device, &state.device_limits);
}

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Release previous depth texture if it exists */
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view)

  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = state.depth_texture.format,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  state.depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.depth_texture.texture != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Depth texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = state.depth_texture.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.texture, &texture_view_dec);
  ASSERT(state.depth_texture.view != NULL);
}

static uint32_t round_up(uint32_t n, uint32_t k)
{
  return ceil((float)n / (float)k) * k;
}

static void init_mesh_buffers(wgpu_context_t* wgpu_context)
{
  /* Create the model vertex buffer */
  state.buffers.vertex = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Utah teapot - Vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = 3 * state.teapot_mesh.positions.count * sizeof(float),
      .initial.data = state.teapot_mesh.positions.data,
    });

  /* Create the model index buffer */
  state.buffers.index = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Utah teapot - Index buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
      .size         = 3 * state.teapot_mesh.triangles.count * sizeof(uint16_t),
      .initial.data = state.teapot_mesh.triangles.data,
      .count        = 3 * state.teapot_mesh.triangles.count,
    });
}

static void init_size_dependent_buffers(wgpu_context_t* wgpu_context)
{
  const uint32_t canvas_width  = wgpu_context->width;
  const uint32_t canvas_height = wgpu_context->height;

  // Each element stores
  // * color : vec4<f32>
  // * depth : f32
  // * index of next element in the list : u32
  {
    const uint32_t linked_list_element_size
      = 5 * sizeof(float) + 1 * sizeof(uint32_t);

    // We want to keep the linked-list buffer size under the
    // maxStorageBufferBindingSize.
    // Split the frame into enough slices to meet that constraint.
    const uint32_t bytes_per_line
      = canvas_width * average_layers_per_fragment * linked_list_element_size;
    const uint32_t max_lines_supported = (uint32_t)floorf(
      state.device_limits.maxStorageBufferBindingSize / (float)bytes_per_line);
    state.num_slices
      = (uint32_t)ceilf(canvas_height / (float)max_lines_supported);
    state.slice_height
      = (uint32_t)ceilf(canvas_height / (float)state.num_slices);
    const uint32_t linked_list_buffer_size
      = state.slice_height * bytes_per_line;
    state.buffers.linked_list = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Linked list - Storage buffer",
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
        .size  = linked_list_buffer_size,
      });
  }

  // To slice up the frame we need to pass the starting fragment y position of
  // the slice. We do this using a uniform buffer with a dynamic offset.
  {
    WGPUBufferDescriptor buffer_desc = {
      .label = STRVIEW("Slice info - uniform buffer"),
      .usage = WGPUBufferUsage_Uniform,
      .size
      = state.num_slices * state.device_limits.minUniformBufferOffsetAlignment,
      .mappedAtCreation = true,
    };
    state.buffers.slice_info = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
    };
    ASSERT(state.buffers.slice_info.buffer);
    int32_t* mapping = (int32_t*)wgpuBufferGetMappedRange(
      state.buffers.slice_info.buffer, 0, buffer_desc.size);
    // This assumes minUniformBufferOffsetAlignment is a multiple of 4
    const int32_t stride
      = state.device_limits.minUniformBufferOffsetAlignment / sizeof(int32_t);
    for (int32_t i = 0; i < (int32_t)state.num_slices; ++i) {
      mapping[i * stride] = i * state.slice_height;
    }
    wgpuBufferUnmap(state.buffers.slice_info.buffer);
  }

  // `Heads` struct contains the start index of the linked-list of translucent
  // fragments for a given pixel.
  // * numFragments : u32
  // * data : array<u32>
  {
    const uint32_t buffer_count = 1 + canvas_width * state.slice_height;

    state.buffers.heads = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Heads struct - storage buffer",
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
        .size  = buffer_count * sizeof(uint32_t),
      });

    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Heads init - storage buffer"),
      .usage            = WGPUBufferUsage_CopySrc,
      .size             = buffer_count * sizeof(uint32_t),
      .mappedAtCreation = true,
    };
    state.buffers.heads_init = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
      .count  = buffer_count,
    };
    ASSERT(state.buffers.heads_init.buffer);
    uint32_t* buffer = (uint32_t*)wgpuBufferGetMappedRange(
      state.buffers.heads_init.buffer, 0, buffer_desc.size);
    for (uint32_t i = 0; i < buffer_count; ++i) {
      buffer[i] = 0xffffffff;
    }
    wgpuBufferUnmap(state.buffers.heads_init.buffer);
  }

  // Uniforms contains:
  // * modelViewProjectionMatrix: mat4x4<f32>
  // * maxStorableFragments: u32
  // * targetWidth: u32
  {
    const uint32_t uniforms_size = round_up(sizeof(state.ubo_data), 16);

    state.buffers.uniform
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Uniform buffer",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = uniforms_size,
                                         });
  }
}

/**
 * @brief Creates a 4-by-4 matrix which translates by the given vector v.
 * @param v - The vector by which to translate.
 * @param dst - matrix to hold result.
 * @returns The translation matrix.
 */
static mat4* glm_mat4_translation(vec3 v, mat4* dst)
{
  glm_mat4_identity(*dst);
  (*dst)[3][0] = v[0];
  (*dst)[3][1] = v[1];
  (*dst)[3][2] = v[2];
  return dst;
}

/**
 * @brief Rotates the given 4-by-4 matrix around the y-axis by the given angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix m.
 */
static mat4* glm_mat4_rotate_y(mat4* m, float angle_in_radians)
{
  const float m00 = (*m)[0][0];
  const float m01 = (*m)[0][1];
  const float m02 = (*m)[0][2];
  const float m03 = (*m)[0][3];
  const float m20 = (*m)[2][0];
  const float m21 = (*m)[2][1];
  const float m22 = (*m)[2][2];
  const float m23 = (*m)[2][3];
  const float c   = cos(angle_in_radians);
  const float s   = sin(angle_in_radians);
  (*m)[0][0]      = c * m00 - s * m20;
  (*m)[0][1]      = c * m01 - s * m21;
  (*m)[0][2]      = c * m02 - s * m22;
  (*m)[0][3]      = c * m03 - s * m23;
  (*m)[2][0]      = c * m20 + s * m00;
  (*m)[2][1]      = c * m21 + s * m01;
  (*m)[2][2]      = c * m22 + s * m02;
  (*m)[2][3]      = c * m23 + s * m03;
  return m;
}

/**
 * @brief Transform vec3 by 4x4 matrix.
 * @param v - the vector
 * @param m - The matrix.
 * @param dst - vec3 to store result.
 * @returns the transformed vector dst
 */
static vec3* glm_vec3_transform_mat4(vec3 v, mat4 m, vec3* dst)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];
  const float w = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3];
  (*dst)[0]     = (m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0]) / w;
  (*dst)[1]     = (m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1]) / w;
  (*dst)[2]     = (m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2]) / w;
  return dst;
}

static void init_opaque_pass(wgpu_context_t* wgpu_context)
{
  /* Render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    /* Color target state */
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = state.depth_texture.format,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    /* Vertex buffer layout */
    WGPU_VERTEX_BUFFER_LAYOUT(
      opaque, 3 * sizeof(float),
      /* Attribute location 0: Position */
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))

    /* Vertex state */
    WGPUShaderModule vert_shader_module
      = wgpu_create_shader_module(wgpu_context->device, opaque_shader_wgsl);
    WGPUVertexState vertex_state = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main_vs"),
      .bufferCount = 1,
      .buffers     = &opaque_vertex_buffer_layout,
    };

    /* Fragment state */
    WGPUShaderModule frag_shader_module
      = wgpu_create_shader_module(wgpu_context->device, opaque_shader_wgsl);
    WGPUFragmentState fragment_state = {
      .module      = frag_shader_module,
      .entryPoint  = STRVIEW("main_fs"),
      .targetCount = 1,
      .targets     = &color_target_state,
    };

    /* Multisample state */
    WGPUMultisampleState multisample_state = {
      .count = 1,
      .mask  = 0xffffffff,
    };

    /* Create rendering pipeline using the specified states */
    state.opaque_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label     = STRVIEW("Opaque - Render pipeline"),
                              .primitive = primitive_state,
                              .vertex    = vertex_state,
                              .fragment  = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(state.opaque_pass.pipeline != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    state.opaque_pass.color_attachment
      = (WGPURenderPassColorAttachment){
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

    /* Depth-stencil attachment */
    state.opaque_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = state.depth_texture.view,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
      };

    /* Pass descriptor */
    state.opaque_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("Opaque - Render pass descriptor"),
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.opaque_pass.color_attachment,
      .depthStencilAttachment = &state.opaque_pass.depth_stencil_attachment,
    };
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.buffers.uniform.buffer,
        .size    = state.buffers.uniform.size,
      },
    };
    state.opaque_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Opaque - Bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.opaque_pass.pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.opaque_pass.bind_group != NULL);
  }
}

static void init_translucent_pass(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[5] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.uniform.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.heads.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.linked_list.size,
        },
        .sampler = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Depth,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = state.device_limits.minUniformBufferOffsetAlignment,
        },
        .sampler = {0},
      }
    };
    state.translucent_pass.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Translucent - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(state.translucent_pass.bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    state.translucent_pass.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Translucent - Pipeline layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.translucent_pass.bind_group_layout,
      });
    ASSERT(state.translucent_pass.pipeline_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.buffers.uniform.buffer,
        .size    = state.buffers.uniform.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.buffers.heads.buffer,
        .size    = state.buffers.heads.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.buffers.linked_list.buffer,
        .size    = state.buffers.linked_list.size,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding     = 3,
        .textureView = state.depth_texture.view,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .buffer  = state.buffers.slice_info.buffer,
        .size    = state.device_limits.minUniformBufferOffsetAlignment,
      },
    };
    state.translucent_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Translucent - Bind group"),
        .layout     = state.translucent_pass.bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.translucent_pass.bind_group != NULL);
  }

  /* Render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    /* Color target state */
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = 0x0,
    };

    /* Vertex buffer layout */
    WGPU_VERTEX_BUFFER_LAYOUT(
      translucent, sizeof(float) * 3,
      /* Attribute location 0: Position */
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))

    /* Vertex state */
    WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
      wgpu_context->device, translucent_shader_wgsl);
    WGPUVertexState vertex_state = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main_vs"),
      .bufferCount = 1,
      .buffers     = &translucent_vertex_buffer_layout,
    };

    /* Fragment state */
    WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
      wgpu_context->device, translucent_shader_wgsl);
    WGPUFragmentState fragment_state = {
      .module      = frag_shader_module,
      .entryPoint  = STRVIEW("main_fs"),
      .targetCount = 1,
      .targets     = &color_target_state,
    };

    /* Multisample state */
    WGPUMultisampleState multisample_state = {
      .count = 1,
      .mask  = 0xffffffff,
    };

    /* Create rendering pipeline using the specified states */
    state.translucent_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label = STRVIEW("Translucent - Render pipeline"),
                              .layout = state.translucent_pass.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(state.translucent_pass.pipeline != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    state.translucent_pass.color_attachment = (WGPURenderPassColorAttachment){
      .view       = NULL, /* View is acquired and set in render loop. */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
    };

    /* Pass descriptor */
    state.translucent_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("Translucent - Render pass descriptor"),
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.translucent_pass.color_attachment,
      .depthStencilAttachment = NULL,
    };
  }
}

static void init_composite_pass(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.uniform.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.heads.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = state.buffers.linked_list.size,
        },
        .sampler = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = state.device_limits.minUniformBufferOffsetAlignment,
        },
        .sampler = {0},
      }
    };
    state.composite_pass.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Composite - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(state.composite_pass.bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    state.composite_pass.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Composite - Pipeline layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.composite_pass.bind_group_layout,
      });
    ASSERT(state.composite_pass.pipeline_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.buffers.uniform.buffer,
        .size    = state.buffers.uniform.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.buffers.heads.buffer,
        .size    = state.buffers.heads.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = state.buffers.linked_list.buffer,
        .size    = state.buffers.linked_list.size,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer  = state.buffers.slice_info.buffer,
        .size    = state.device_limits.minUniformBufferOffsetAlignment,
      },
    };
    state.composite_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Composite - Bind group"),
                              .layout = state.composite_pass.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.composite_pass.bind_group != NULL);
  }

  /* Render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    /* Color target state */
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    blend_state.color.srcFactor             = WGPUBlendFactor_One;
    blend_state.color.dstFactor             = WGPUBlendFactor_OneMinusSrcAlpha;
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUShaderModule vert_shader_module
      = wgpu_create_shader_module(wgpu_context->device, composite_shader_wgsl);
    WGPUVertexState vertex_state = {
      .module     = vert_shader_module,
      .entryPoint = STRVIEW("main_vs"),
    };

    /* Fragment state */
    WGPUShaderModule frag_shader_module
      = wgpu_create_shader_module(wgpu_context->device, composite_shader_wgsl);
    WGPUFragmentState fragment_state = {
      .module      = frag_shader_module,
      .entryPoint  = STRVIEW("main_fs"),
      .targetCount = 1,
      .targets     = &color_target_state,
    };

    /* Multisample state */
    WGPUMultisampleState multisample_state = {
      .count = 1,
      .mask  = 0xffffffff,
    };

    /* Create rendering pipeline using the specified states */
    state.composite_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label  = STRVIEW("Composite - Render pipeline"),
                              .layout = state.composite_pass.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(state.composite_pass.pipeline != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    state.composite_pass.color_attachment = (WGPURenderPassColorAttachment){
      .view       = NULL, /* View is acquired and set in render loop. */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Load,
      .storeOp    = WGPUStoreOp_Store,
    };

    /* Pass descriptor */
    state.composite_pass.descriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Composite - Render pass descriptor"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.composite_pass.color_attachment,
    };
  }
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context,
                                  float timestamp_millis)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.0f, 2000.f,
                  state.view_matrices.projection_matrix);

  glm_vec3_copy((vec3){0.0f, 5.0f, -100.0f}, state.view_matrices.eye_position);

  const float rad = PI * (timestamp_millis / 5000.0f);
  mat4 rotation   = GLM_MAT4_ZERO_INIT;
  glm_mat4_rotate_y(glm_mat4_translation(state.view_matrices.origin, &rotation),
                    rad);
  glm_vec3_transform_mat4(state.view_matrices.eye_position, rotation,
                          &state.view_matrices.eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(state.view_matrices.eye_position, /* eye vector    */
             state.view_matrices.origin,       /* center vector */
             state.view_matrices.up_vector,    /* up vector     */
             view_matrix                       /* result matrix */
  );

  glm_mat4_mulN((mat4*[]){&state.view_matrices.projection_matrix, &view_matrix},
                2, state.view_matrices.view_proj_matrix);

  glm_mat4_copy(state.view_matrices.view_proj_matrix,
                state.ubo_data.model_view_projection_matrix);
  state.ubo_data.max_storable_fragments
    = average_layers_per_fragment * wgpu_context->width * state.slice_height;
  state.ubo_data.target_width = wgpu_context->width;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.uniform.buffer, 0,
                       &state.ubo_data, sizeof(state.ubo_data));
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  /* Initialize sokol-fetch */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 1,
    .num_channels = 1,
    .num_lanes    = 1,
  });

  /* Initialize sokol-time */
  stm_setup();

  /* Get device limits */
  init_device_limits(wgpu_context);

  /* Start async loading of teapot JSON */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/meshes/teapot.json",
    .callback = teapot_json_fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });

  /* Initialize depth texture */
  init_depth_texture(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate depth texture with new dimensions */
    init_depth_texture(wgpu_context);

    /* Recreate buffers that depend on canvas size */
    if (state.mesh_loaded && state.buffers.vertex.buffer != NULL) {
      /* Update depth stencil attachment to use the new depth texture view */
      state.opaque_pass.depth_stencil_attachment.view
        = state.depth_texture.view;

      /* Destroy size-dependent and uniform buffers before recreating */
      wgpu_destroy_buffer(&state.buffers.uniform);
      wgpu_destroy_buffer(&state.buffers.heads);
      wgpu_destroy_buffer(&state.buffers.heads_init);
      wgpu_destroy_buffer(&state.buffers.linked_list);
      wgpu_destroy_buffer(&state.buffers.slice_info);

      /* Recreate size-dependent buffers */
      init_size_dependent_buffers(wgpu_context);

      /* Recreate opaque pass bind group with new uniform buffer */
      WGPU_RELEASE_RESOURCE(BindGroup, state.opaque_pass.bind_group)
      WGPUBindGroupEntry bg_entries[1] = {
        [0] = (WGPUBindGroupEntry){
          .binding = 0,
          .buffer  = state.buffers.uniform.buffer,
          .size    = state.buffers.uniform.size,
        },
      };
      state.opaque_pass.bind_group = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .label  = STRVIEW("Opaque - Bind group"),
                                .layout = wgpuRenderPipelineGetBindGroupLayout(
                                  state.opaque_pass.pipeline, 0),
                                .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                                .entries    = bg_entries,
                              });
      ASSERT(state.opaque_pass.bind_group != NULL);

      /* Recreate bind groups for translucent and composite passes */
      init_translucent_pass(wgpu_context);
      init_composite_pass(wgpu_context);
    }
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loading */
  sfetch_dowork();

  /* Initialize pipelines once mesh is loaded */
  if (state.mesh_loaded && state.buffers.vertex.buffer == NULL) {
    init_mesh_buffers(wgpu_context);
    init_size_dependent_buffers(wgpu_context);
    init_opaque_pass(wgpu_context);
    init_translucent_pass(wgpu_context);
    init_composite_pass(wgpu_context);
  }

  /* Only render if mesh and pipelines are ready */
  if (!state.mesh_loaded || state.buffers.vertex.buffer == NULL) {
    return EXIT_SUCCESS;
  }

  /* Update uniform buffer */
  update_uniform_buffer(wgpu_context, (float)stm_ms(stm_now()));

  const uint32_t canvas_width  = wgpu_context->width;
  const uint32_t canvas_height = wgpu_context->height;
  WGPUDevice device            = wgpu_context->device;
  WGPUQueue queue              = wgpu_context->queue;
  WGPUTextureView texture_view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Draw the opaque objects */
  {
    state.opaque_pass.color_attachment.view = texture_view;
    WGPURenderPassEncoder opaque_pass_encoder
      = wgpuCommandEncoderBeginRenderPass(cmd_enc,
                                          &state.opaque_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(opaque_pass_encoder,
                                     state.opaque_pass.pipeline);
    wgpuRenderPassEncoderSetBindGroup(opaque_pass_encoder, 0,
                                      state.opaque_pass.bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      opaque_pass_encoder, 0, state.buffers.vertex.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      opaque_pass_encoder, state.buffers.index.buffer, WGPUIndexFormat_Uint16,
      0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(opaque_pass_encoder,
                                     state.buffers.index.count, 8, 0, 0, 0);
    wgpuRenderPassEncoderEnd(opaque_pass_encoder);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, opaque_pass_encoder)
  }

  /* Process each slice */
  for (uint32_t slice = 0; slice < state.num_slices; ++slice) {
    /* Initialize the heads buffer */
    wgpuCommandEncoderCopyBufferToBuffer(
      cmd_enc, state.buffers.heads_init.buffer, 0, state.buffers.heads.buffer,
      0, state.buffers.heads_init.size);

    const uint32_t scissor_x     = 0;
    const uint32_t scissor_y     = slice * state.slice_height;
    const uint32_t scissor_width = canvas_width;
    const uint32_t scissor_height
      = MIN((slice + 1) * state.slice_height, canvas_height)
        - slice * state.slice_height;

    /* Draw the translucent objects */
    {
      state.translucent_pass.color_attachment.view = texture_view;
      WGPURenderPassEncoder translucent_pass_encoder
        = wgpuCommandEncoderBeginRenderPass(cmd_enc,
                                            &state.translucent_pass.descriptor);

      /* Set the scissor to only process a horizontal slice of the frame */
      wgpuRenderPassEncoderSetScissorRect(translucent_pass_encoder, scissor_x,
                                          scissor_y, scissor_width,
                                          scissor_height);

      wgpuRenderPassEncoderSetPipeline(translucent_pass_encoder,
                                       state.translucent_pass.pipeline);
      const uint32_t dynamic_offsets[1]
        = {slice * state.device_limits.minUniformBufferOffsetAlignment};
      wgpuRenderPassEncoderSetBindGroup(translucent_pass_encoder, 0,
                                        state.translucent_pass.bind_group, 1,
                                        dynamic_offsets);
      wgpuRenderPassEncoderSetVertexBuffer(translucent_pass_encoder, 0,
                                           state.buffers.vertex.buffer, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(
        translucent_pass_encoder, state.buffers.index.buffer,
        WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(translucent_pass_encoder,
                                       state.buffers.index.count, 8, 0, 0, 0);
      wgpuRenderPassEncoderEnd(translucent_pass_encoder);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, translucent_pass_encoder)
    }

    /* Composite the opaque and translucent objects */
    {
      state.composite_pass.color_attachment.view = texture_view;
      WGPURenderPassEncoder composite_pass_encoder
        = wgpuCommandEncoderBeginRenderPass(cmd_enc,
                                            &state.composite_pass.descriptor);

      /* Set the scissor to only process a horizontal slice of the frame */
      wgpuRenderPassEncoderSetScissorRect(composite_pass_encoder, scissor_x,
                                          scissor_y, scissor_width,
                                          scissor_height);

      wgpuRenderPassEncoderSetPipeline(composite_pass_encoder,
                                       state.composite_pass.pipeline);
      const uint32_t dynamic_offsets[1]
        = {slice * state.device_limits.minUniformBufferOffsetAlignment};
      wgpuRenderPassEncoderSetBindGroup(composite_pass_encoder, 0,
                                        state.composite_pass.bind_group, 1,
                                        dynamic_offsets);
      wgpuRenderPassEncoderDraw(composite_pass_encoder, 6, 1, 0, 0);
      wgpuRenderPassEncoderEnd(composite_pass_encoder);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, composite_pass_encoder)
    }
  }

  /* Finish command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  ASSERT(cmd_buffer != NULL);

  /* Submit and present */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer)
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc)

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  wgpu_destroy_buffer(&state.buffers.vertex);
  wgpu_destroy_buffer(&state.buffers.index);
  wgpu_destroy_buffer(&state.buffers.heads);
  wgpu_destroy_buffer(&state.buffers.heads_init);
  wgpu_destroy_buffer(&state.buffers.linked_list);
  wgpu_destroy_buffer(&state.buffers.slice_info);
  wgpu_destroy_buffer(&state.buffers.uniform);
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.opaque_pass.pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, state.opaque_pass.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.translucent_pass.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.translucent_pass.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.translucent_pass.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.translucent_pass.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composite_pass.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composite_pass.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composite_pass.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composite_pass.bind_group)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "A-Buffer",
    .init_cb        = init,
    .frame_cb       = frame,
    .input_event_cb = input_event_cb,
    .shutdown_cb    = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* opaque_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix: mat4x4<f32>,
  };

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) instance: u32
  };

  @vertex
  fn main_vs(@location(0) position: vec4<f32>, @builtin(instance_index) instance: u32) -> VertexOutput {
    var output: VertexOutput;

    // distribute instances into a staggered 4x4 grid
    const gridWidth = 125.0;
    const cellSize = gridWidth / 4.0;
    let row = instance / 2u;
    let col = instance % 2u;

    let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u != 0u) * cellSize;
    let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

    let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

    output.position = uniforms.modelViewProjectionMatrix * offsetPos;
    output.instance = instance;
    return output;
  }

  @fragment
  fn main_fs(@location(0) @interpolate(flat) instance: u32) -> @location(0) vec4<f32> {
    const colors = array<vec3<f32>,6>(
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 1.0),
        vec3(1.0, 1.0, 0.0),
        vec3(0.0, 1.0, 1.0),
    );

    return vec4(colors[instance % 6u], 1.0);
  }
);

static const char* translucent_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix: mat4x4<f32>,
    maxStorableFragments: u32,
    targetWidth: u32,
  };

  struct SliceInfo {
    sliceStartY: i32
  };

  struct Heads {
    numFragments: atomic<u32>,
    data: array<atomic<u32>>
  };

  struct LinkedListElement {
    color: vec4<f32>,
    depth: f32,
    next: u32
  };

  struct LinkedList {
    data: array<LinkedListElement>
  };

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;
  @binding(1) @group(0) var<storage, read_write> heads: Heads;
  @binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
  @binding(3) @group(0) var opaqueDepthTexture: texture_depth_2d;
  @binding(4) @group(0) var<uniform> sliceInfo: SliceInfo;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) instance: u32
  };

  @vertex
  fn main_vs(@location(0) position: vec4<f32>, @builtin(instance_index) instance: u32) -> VertexOutput {
    var output: VertexOutput;

    // distribute instances into a staggered 4x4 grid
    const gridWidth = 125.0;
    const cellSize = gridWidth / 4.0;
    let row = instance / 2u;
    let col = instance % 2u;

    let xOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 * cellSize * f32(col) + f32(row % 2u == 0u) * cellSize;
    let zOffset = -gridWidth / 2.0 + cellSize / 2.0 + 2.0 + f32(row) * cellSize;

    let offsetPos = vec4(position.x + xOffset, position.y, position.z + zOffset, position.w);

    output.position = uniforms.modelViewProjectionMatrix * offsetPos;
    output.instance = instance;

    return output;
  }

  @fragment
  fn main_fs(@builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) instance: u32) {
    const colors = array<vec3<f32>,6>(
      vec3(1.0, 0.0, 0.0),
      vec3(0.0, 1.0, 0.0),
      vec3(0.0, 0.0, 1.0),
      vec3(1.0, 0.0, 1.0),
      vec3(1.0, 1.0, 0.0),
      vec3(0.0, 1.0, 1.0),
    );

    let fragCoords = vec2<i32>(position.xy);
    let opaqueDepth = textureLoad(opaqueDepthTexture, fragCoords, 0);

    // reject fragments behind opaque objects
    if position.z >= opaqueDepth {
      discard;
    }

    // The index in the heads buffer corresponding to the head data for the fragment at
    // the current location.
    let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

    // The index in the linkedList buffer at which to store the new fragment
    let fragIndex = atomicAdd(&heads.numFragments, 1u);

    // If we run out of space to store the fragments, we just lose them
    if fragIndex < uniforms.maxStorableFragments {
      let lastHead = atomicExchange(&heads.data[headsIndex], fragIndex);
      linkedList.data[fragIndex].depth = position.z;
      linkedList.data[fragIndex].next = lastHead;
      linkedList.data[fragIndex].color = vec4(colors[(instance + 3u) % 6u], 0.3);
    }
  }
);

static const char* composite_shader_wgsl = CODE(
  struct Uniforms {
    modelViewProjectionMatrix: mat4x4<f32>,
    maxStorableFragments: u32,
    targetWidth: u32,
  };

  struct SliceInfo {
    sliceStartY: i32
  };

  struct Heads {
    numFragments: u32,
    data: array<u32>
  };

  struct LinkedListElement {
    color: vec4<f32>,
    depth: f32,
    next: u32
  };

  struct LinkedList {
    data: array<LinkedListElement>
  };

  @binding(0) @group(0) var<uniform> uniforms: Uniforms;
  @binding(1) @group(0) var<storage, read_write> heads: Heads;
  @binding(2) @group(0) var<storage, read_write> linkedList: LinkedList;
  @binding(3) @group(0) var<uniform> sliceInfo: SliceInfo;

  // Output a full screen quad
  @vertex
  fn main_vs(@builtin(vertex_index) vertIndex: u32) -> @builtin(position) vec4<f32> {
    const position = array<vec2<f32>, 6>(
      vec2(-1.0, -1.0),
      vec2(1.0, -1.0),
      vec2(1.0, 1.0),
      vec2(-1.0, -1.0),
      vec2(1.0, 1.0),
      vec2(-1.0, 1.0),
    );

    return vec4(position[vertIndex], 0.0, 1.0);
  }

  @fragment
  fn main_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let fragCoords = vec2<i32>(position.xy);
    let headsIndex = u32(fragCoords.y - sliceInfo.sliceStartY) * uniforms.targetWidth + u32(fragCoords.x);

    // The maximum layers we can process for any pixel
    const maxLayers = 24u;

    var layers: array<LinkedListElement, maxLayers>;

    var numLayers = 0u;
    var elementIndex = heads.data[headsIndex];

    // copy the list elements into an array up to the maximum amount of layers
    while elementIndex != 0xFFFFFFFFu && numLayers < maxLayers {
      layers[numLayers] = linkedList.data[elementIndex];
      numLayers++;
      elementIndex = linkedList.data[elementIndex].next;
    }

    if numLayers == 0u {
      discard;
    }

    // sort the fragments by depth
    for (var i = 1u; i < numLayers; i++) {
      let toInsert = layers[i];
      var j = i;

      while j > 0u && toInsert.depth > layers[j - 1u].depth {
        layers[j] = layers[j - 1u];
        j--;
      }

      layers[j] = toInsert;
    }

    // pre-multiply alpha for the first layer
    var color = vec4(layers[0].color.a * layers[0].color.rgb, layers[0].color.a);

    // blend the remaining layers
    for (var i = 1u; i < numLayers; i++) {
      let mixed = mix(color.rgb, layers[i].color.rgb, layers[i].color.aaa);
      color = vec4(mixed, color.a);
    }

    return color;
  }
);
// clang-format on
