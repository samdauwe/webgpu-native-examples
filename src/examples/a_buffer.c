#include "example_base.h"
#include "meshes.h"

#include "../webgpu/imgui_overlay.h"

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

static utah_teapot_mesh_t utah_teapot_mesh = {0};

static WGPUSupportedLimits device_limits = {0};

static struct {
  wgpu_buffer_t vertex;
  wgpu_buffer_t index;
  wgpu_buffer_t heads;
  wgpu_buffer_t heads_init;
  wgpu_buffer_t linked_list;
  wgpu_buffer_t slice_info;
  wgpu_buffer_t uniform;
} buffers = {0};

static struct {
  vec3 up_vector;
  vec3 origin;
  vec3 eye_position;
  mat4 projection_matrix;
  mat4 view_proj_matrix;
} view_matrices = {
  .up_vector         = {0.0f, 1.0f, 0.0f},
  .origin            = GLM_VEC3_ZERO_INIT,
  .eye_position      = {0.0f, 5.0f, -100.0f},
  .projection_matrix = GLM_MAT4_IDENTITY_INIT,
  .view_proj_matrix  = GLM_MAT4_IDENTITY_INIT,
};

static struct {
  mat4 model_view_projection_matrix;
  uint32_t max_storable_fragments;
  uint32_t target_width;
} ubo_data = {0};

/* Depth texture */
static struct {
  WGPUTexture texture;
  WGPUTextureView view;
  WGPUTextureFormat format;
} depth_texture = {0};

/* Render pass to draw the opaque objects */
static struct {
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } pass_desc;
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
} opaque_render_pass = {0};

/* Render pass to draw the translucent objects */
static struct {
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } pass_desc;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
} translucent_render_pass = {0};

/* Render pass to composite the opaque and translucent objects */
static struct {
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } pass_desc;
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
} composite_render_pass = {0};

/* Determines how much memory is allocated to store linked-list elements */
static const uint32_t average_layers_per_fragment = 4;

/* Slices info */
uint32_t num_slices   = 0u;
uint32_t slice_height = 0u;

// Other variables
static const char* example_title = "A-Buffer";
static bool prepared             = false;

static void init_device_limits(wgpu_context_t* wgpu_context)
{
  if (!wgpuAdapterGetLimits(wgpu_context->adapter, &device_limits)) {
    log_error("Could not query WebGPU adapter limits");
  }
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  depth_texture.format = WGPUTextureFormat_Depth32Float;

  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "Depth texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = depth_texture.format,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(depth_texture.texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Depth texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = depth_texture.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  depth_texture.view
    = wgpuTextureCreateView(depth_texture.texture, &texture_view_dec);
  ASSERT(depth_texture.view != NULL);
}

static uint32_t round_up(uint32_t n, uint32_t k)
{
  return ceil((float)n / (float)k) * k;
}

static void prepare_buffers(wgpu_context_t* wgpu_context,
                            utah_teapot_mesh_t* utah_teapot_mesh)
{
  /* Create the model vertex buffer */
  buffers.vertex = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Utah teapot mesh - vertex buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
      .size         = 3 * utah_teapot_mesh->positions.count * sizeof(float),
      .initial.data = utah_teapot_mesh->positions.data,
    });

  /* Create the model index buffer */
  buffers.index = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label        = "Utah teapot mesh - index buffer",
      .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
      .size         = 3 * utah_teapot_mesh->triangles.count * sizeof(uint16_t),
      .initial.data = utah_teapot_mesh->triangles.data,
      .count        = 3 * utah_teapot_mesh->triangles.count,
    });

  const uint32_t canvas_width  = wgpu_context->surface.width;
  const uint32_t canvas_height = wgpu_context->surface.height;

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
      device_limits.limits.maxStorageBufferBindingSize / (float)bytes_per_line);
    num_slices   = (uint32_t)ceilf(canvas_height / (float)max_lines_supported);
    slice_height = (uint32_t)ceilf(canvas_height / (float)num_slices);
    const uint32_t linked_list_buffer_size = slice_height * bytes_per_line;
    buffers.linked_list                    = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
                           .label = "Linked list - storage buffer",
                           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                           .size  = linked_list_buffer_size,
      });
  }

  // To slice up the frame we need to pass the starting fragment y position of
  // the slice. We do this using a uniform buffer with a dynamic offset.
  {
    WGPUBufferDescriptor buffer_desc = {
      .label = "Slice info - uniform buffer",
      .usage = WGPUBufferUsage_Uniform,
      .size = num_slices * device_limits.limits.minUniformBufferOffsetAlignment,
      .mappedAtCreation = true,
    };
    buffers.slice_info = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
    };
    ASSERT(buffers.slice_info.buffer);
    int32_t* mapping = (int32_t*)wgpuBufferGetMappedRange(
      buffers.slice_info.buffer, 0, buffer_desc.size);
    // This assumes minUniformBufferOffsetAlignment is a multiple of 4
    const int32_t stride
      = device_limits.limits.minUniformBufferOffsetAlignment / sizeof(int32_t);
    for (int32_t i = 0; i < (int32_t)num_slices; ++i) {
      mapping[i * stride] = i * slice_height;
    }
    wgpuBufferUnmap(buffers.slice_info.buffer);
  }

  // `Heads` struct contains the start index of the linked-list of translucent
  // fragments for a given pixel.
  // * numFragments : u32
  // * data : array<u32>
  {
    const uint32_t buffer_count = 1 + canvas_width * slice_height;

    buffers.heads = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Heads struct - storage buffer",
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
        .size  = buffer_count * sizeof(uint32_t),
      });

    WGPUBufferDescriptor buffer_desc = {
      .label            = "Heads init - storage buffer",
      .usage            = WGPUBufferUsage_CopySrc,
      .size             = buffer_count * sizeof(uint32_t),
      .mappedAtCreation = true,
    };
    buffers.heads_init = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
      .count  = buffer_count,
    };
    ASSERT(buffers.heads_init.buffer);
    uint32_t* buffer = (uint32_t*)wgpuBufferGetMappedRange(
      buffers.heads_init.buffer, 0, buffer_desc.size);
    for (uint32_t i = 0; i < buffer_count; ++i) {
      buffer[i] = 0xffffffff;
    }
    wgpuBufferUnmap(buffers.heads_init.buffer);
  }

  // Uniforms contains:
  // * modelViewProjectionMatrix: mat4x4<f32>
  // * maxStorableFragments: u32
  // * targetWidth: u32
  {
    const uint32_t uniforms_size = round_up(sizeof(ubo_data), 16);

    buffers.uniform
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

/* Rotates the camera around the origin based on time. */
static mat4* get_camera_view_proj_matrix(wgpu_example_context_t* context)
{
  const float aspect_ratio = (float)context->wgpu_context->surface.width
                             / (float)context->wgpu_context->surface.height;

  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.0f, 2000.f,
                  view_matrices.projection_matrix);

  glm_vec3_copy((vec3){0.0f, 5.0f, -100.0f}, view_matrices.eye_position);

  const float rad = PI * (context->frame.timestamp_millis / 5000.0f);
  mat4 rotation   = GLM_MAT4_ZERO_INIT;
  glm_mat4_rotate_y(glm_mat4_translation(view_matrices.origin, &rotation), rad);
  glm_vec3_transform_mat4(view_matrices.eye_position, rotation,
                          &view_matrices.eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(view_matrices.eye_position, /* eye vector    */
             view_matrices.origin,       /* center vector */
             view_matrices.up_vector,    /* up vector     */
             view_matrix                 /* result matrix */
  );

  glm_mat4_mulN((mat4*[]){&view_matrices.projection_matrix, &view_matrix}, 2,
                view_matrices.view_proj_matrix);
  return &view_matrices.view_proj_matrix;
}

static void prepare_opaque_render_pass(wgpu_context_t* wgpu_context)
{
  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = depth_texture.format,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex buffer layout
    WGPU_VERTEX_BUFFER_LAYOUT(
      translucent, 3 * sizeof(float),
      // Attribute location 0: Position
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL
                        .label            = "Opaque vertex shader WGSL",
                        .wgsl_code.source = opaque_shader_wgsl,
                        .entry            = "main_vs",
                      },
                      .buffer_count = 1,
                      .buffers      = &translucent_vertex_buffer_layout,
                    });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL
                        .label            = "Opaque fragment shader WGSL",
                        .wgsl_code.source = opaque_shader_wgsl,
                        .entry            = "main_fs",
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
    opaque_render_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Opaque render pipeline",
                              .primitive    = primitive_state,
                              .vertex       = vertex_state,
                              .fragment     = &fragment_state,
                              .depthStencil = &depth_stencil_state,
                              .multisample  = multisample_state,
                            });
    ASSERT(opaque_render_pass.pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    opaque_render_pass.pass_desc.color_attachments[0]
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
    opaque_render_pass.pass_desc.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = depth_texture.view,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
      };

    /* Pass descriptor */
    opaque_render_pass.pass_desc.descriptor = (WGPURenderPassDescriptor){
      .label                = "Opaque render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = opaque_render_pass.pass_desc.color_attachments,
      .depthStencilAttachment
      = &opaque_render_pass.pass_desc.depth_stencil_attachment,
    };
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = buffers.uniform.buffer,
        .size    = buffers.uniform.size,
      },
    };
    opaque_render_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Opaque bind group",
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                opaque_render_pass.pipeline, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(opaque_render_pass.bind_group != NULL);
  }
}

static void prepare_translucent_render_pass(wgpu_context_t* wgpu_context)
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
          .minBindingSize   = buffers.uniform.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = buffers.heads.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = buffers.linked_list.size,
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
          .minBindingSize   = device_limits.limits.minUniformBufferOffsetAlignment,
        },
        .sampler = {0},
      }
    };
    translucent_render_pass.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Translucent bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(translucent_render_pass.bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    translucent_render_pass.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = "Translucent pipeline layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &translucent_render_pass.bind_group_layout,
      });
    ASSERT(translucent_render_pass.pipeline_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.uniform.buffer,
        .size    = buffers.uniform.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = buffers.heads.buffer,
        .size    = buffers.heads.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = buffers.linked_list.buffer,
        .size    = buffers.linked_list.size,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding     = 3,
        .textureView = depth_texture.view,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .buffer  = buffers.slice_info.buffer,
        .size    = device_limits.limits.minUniformBufferOffsetAlignment,
      },
    };
    translucent_render_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = "Translucent bind group",
        .layout     = translucent_render_pass.bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(translucent_render_pass.bind_group != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = 0x0,
    };

    // Vertex buffer layout
    WGPU_VERTEX_BUFFER_LAYOUT(
      translucent, sizeof(float) * 3,
      // Attribute location 0: Position
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0))

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL
                        .label            = "Translucent vertex shader WGSL",
                        .wgsl_code.source = translucent_shader_wgsl,
                        .entry            = "main_vs",
                      },
                      .buffer_count = 1,
                      .buffers      = &translucent_vertex_buffer_layout,
                    });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL
                        .label            = "Translucent fragment shader WGSL",
                        .wgsl_code.source = translucent_shader_wgsl,
                        .entry            = "main_fs",
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
    translucent_render_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label  = "Translucent pipeline",
                              .layout = translucent_render_pass.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(translucent_render_pass.pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    translucent_render_pass.pass_desc.color_attachments[0]
      = (WGPURenderPassColorAttachment){
        .view       = NULL, /* View is acquired and set in render loop. */
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Load,
        .storeOp    = WGPUStoreOp_Store,
      };

    /* Pass descriptor */
    translucent_render_pass.pass_desc.descriptor = (WGPURenderPassDescriptor){
      .label                = "Translucent render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments = translucent_render_pass.pass_desc.color_attachments,
      .depthStencilAttachment = NULL,
    };
  }
}

static void prepare_composite_render_pass(wgpu_context_t* wgpu_context)
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
          .minBindingSize   = buffers.uniform.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = buffers.heads.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Storage,
          .hasDynamicOffset = false,
          .minBindingSize   = buffers.linked_list.size,
        },
        .sampler = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = device_limits.limits.minUniformBufferOffsetAlignment,
        },
        .sampler = {0},
      }
    };
    composite_render_pass.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Composite bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(composite_render_pass.bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    composite_render_pass.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = "Composite pipeline layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &composite_render_pass.bind_group_layout,
      });
    ASSERT(composite_render_pass.pipeline_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = buffers.uniform.buffer,
        .size    = buffers.uniform.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = buffers.heads.buffer,
        .size    = buffers.heads.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = buffers.linked_list.buffer,
        .size    = buffers.linked_list.size,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer  = buffers.slice_info.buffer,
        .size    = device_limits.limits.minUniformBufferOffsetAlignment,
      },
    };
    composite_render_pass.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Composite bind group",
                              .layout = composite_render_pass.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(composite_render_pass.bind_group != NULL);
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
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL */
                        .label            = "Composite vertex shader WGSL",
                        .wgsl_code.source = composite_shader_wgsl,
                        .entry            = "main_vs",
                      },
                    });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
                      .shader_desc = (wgpu_shader_desc_t){
                        // Vertex shader WGSL
                        .label            = "Composite fragment shader WGSL",
                        .wgsl_code.source = composite_shader_wgsl,
                        .entry            = "main_fs",
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
    composite_render_pass.pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label  = "Composite render pipeline",
                              .layout = composite_render_pass.pipeline_layout,
                              .primitive   = primitive_state,
                              .vertex      = vertex_state,
                              .fragment    = &fragment_state,
                              .multisample = multisample_state,
                            });
    ASSERT(composite_render_pass.pipeline != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Render pass descriptor */
  {
    /* Color attachment */
    composite_render_pass.pass_desc.color_attachments[0]
      = (WGPURenderPassColorAttachment){
        .view       = NULL, /* view is acquired and set in render loop. */
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Load,
        .storeOp    = WGPUStoreOp_Store,
      };

    /* Pass descriptor */
    composite_render_pass.pass_desc.descriptor = (WGPURenderPassDescriptor){
      .label                = "Composite render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = composite_render_pass.pass_desc.color_attachments,
    };
  }
}

static void update_uniform_buffer(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  glm_mat4_copy(*get_camera_view_proj_matrix(context),
                ubo_data.model_view_projection_matrix);
  ubo_data.max_storable_fragments
    = average_layers_per_fragment * wgpu_context->surface.width * slice_height;
  ubo_data.target_width = wgpu_context->surface.width;

  wgpuQueueWriteBuffer(wgpu_context->queue, buffers.uniform.buffer, 0,
                       &ubo_data, sizeof(ubo_data));
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    init_device_limits(context->wgpu_context);
    utah_teapot_mesh_init(&utah_teapot_mesh);
    prepare_depth_texture(context->wgpu_context);
    prepare_buffers(context->wgpu_context, &utah_teapot_mesh);
    prepare_opaque_render_pass(context->wgpu_context);
    prepare_translucent_render_pass(context->wgpu_context);
    prepare_composite_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  const uint32_t canvas_width  = wgpu_context->surface.width;
  const uint32_t canvas_height = wgpu_context->surface.height;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPUTextureView texture_view = wgpu_context->swap_chain.frame_buffer;

  /* Draw the opaque objects */
  {
    opaque_render_pass.pass_desc.color_attachments[0].view = texture_view;
    WGPURenderPassEncoder opaque_pass_encoder
      = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc, &opaque_render_pass.pass_desc.descriptor);
    wgpuRenderPassEncoderSetPipeline(opaque_pass_encoder,
                                     opaque_render_pass.pipeline);
    wgpuRenderPassEncoderSetBindGroup(opaque_pass_encoder, 0,
                                      opaque_render_pass.bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      opaque_pass_encoder, 0, buffers.vertex.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      opaque_pass_encoder, buffers.index.buffer, WGPUIndexFormat_Uint16, 0,
      WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(opaque_pass_encoder, buffers.index.count,
                                     8, 0, 0, 0);
    wgpuRenderPassEncoderEnd(opaque_pass_encoder);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, opaque_pass_encoder)
  }

  for (uint32_t slice = 0; slice < num_slices; ++slice) {
    /* Initialize the heads buffer */
    wgpuCommandEncoderCopyBufferToBuffer(
      wgpu_context->cmd_enc, buffers.heads_init.buffer, 0, buffers.heads.buffer,
      0, buffers.heads_init.size);

    const uint32_t scissor_x     = 0;
    const uint32_t scissor_y     = slice * slice_height;
    const uint32_t scissor_width = canvas_width;
    const uint32_t scissor_height
      = MIN((slice + 1) * slice_height, canvas_height) - slice * slice_height;

    /* Draw the translucent objects */
    {
      translucent_render_pass.pass_desc.color_attachments[0].view
        = texture_view;
      WGPURenderPassEncoder translucent_pass_encoder
        = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc, &translucent_render_pass.pass_desc.descriptor);

      /* Set the scissor to only process a horizontal slice of the frame */
      wgpuRenderPassEncoderSetScissorRect(translucent_pass_encoder, scissor_x,
                                          scissor_y, scissor_width,
                                          scissor_height);

      wgpuRenderPassEncoderSetPipeline(translucent_pass_encoder,
                                       translucent_render_pass.pipeline);
      const uint32_t dynamic_offsets[1]
        = {slice * device_limits.limits.minUniformBufferOffsetAlignment};
      wgpuRenderPassEncoderSetBindGroup(translucent_pass_encoder, 0,
                                        translucent_render_pass.bind_group, 1,
                                        dynamic_offsets);
      wgpuRenderPassEncoderSetVertexBuffer(
        translucent_pass_encoder, 0, buffers.vertex.buffer, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(
        translucent_pass_encoder, buffers.index.buffer, WGPUIndexFormat_Uint16,
        0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(translucent_pass_encoder,
                                       buffers.index.count, 8, 0, 0, 0);
      wgpuRenderPassEncoderEnd(translucent_pass_encoder);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, translucent_pass_encoder)
    }

    /* Composite the opaque and translucent objects */
    {
      composite_render_pass.pass_desc.color_attachments[0].view = texture_view;
      WGPURenderPassEncoder composite_pass_encoder
        = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc, &composite_render_pass.pass_desc.descriptor);

      /* Set the scissor to only process a horizontal slice of the frame */
      wgpuRenderPassEncoderSetScissorRect(composite_pass_encoder, scissor_x,
                                          scissor_y, scissor_width,
                                          scissor_height);

      wgpuRenderPassEncoderSetPipeline(composite_pass_encoder,
                                       composite_render_pass.pipeline);
      const uint32_t dynamic_offsets[1]
        = {slice * device_limits.limits.minUniformBufferOffsetAlignment};
      wgpuRenderPassEncoderSetBindGroup(composite_pass_encoder, 0,
                                        composite_render_pass.bind_group, 1,
                                        dynamic_offsets);
      wgpuRenderPassEncoderDraw(composite_pass_encoder, 6, 1, 0, 0);
      wgpuRenderPassEncoderEnd(composite_pass_encoder);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, composite_pass_encoder)
    }
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
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
  if (!context->paused) {
    update_uniform_buffer(context);
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_buffer(&buffers.vertex);
  wgpu_destroy_buffer(&buffers.index);
  wgpu_destroy_buffer(&buffers.heads);
  wgpu_destroy_buffer(&buffers.heads_init);
  wgpu_destroy_buffer(&buffers.linked_list);
  wgpu_destroy_buffer(&buffers.slice_info);
  wgpu_destroy_buffer(&buffers.uniform);
  WGPU_RELEASE_RESOURCE(Texture, depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, depth_texture.view)
  WGPU_RELEASE_RESOURCE(RenderPipeline, opaque_render_pass.pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, opaque_render_pass.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, translucent_render_pass.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, translucent_render_pass.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        translucent_render_pass.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, translucent_render_pass.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, composite_render_pass.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, composite_render_pass.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        composite_render_pass.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, composite_render_pass.bind_group)
}

void example_a_buffer(int argc, char* argv[])
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
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
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
