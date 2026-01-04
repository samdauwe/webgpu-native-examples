#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <cJSON.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Primitive Picking
 *
 * This example demonstrates use of the primitive_index WGSL builtin.
 * It is used to render a unique ID for each primitive to a buffer, which is
 * then read at the current cursor/touch location to determine which primitive
 * has been selected. That primitive is then highlighted when rendering the
 * next frame.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/primitivePicking
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_forward_rendering_wgsl;
static const char* fragment_forward_rendering_wgsl;
static const char* vertex_texture_quad_wgsl;
static const char* fragment_primitives_debug_view_wgsl;
static const char* compute_pick_primitive_wgsl;

/* -------------------------------------------------------------------------- *
 * Primitive Picking example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  utah_teapot_mesh_t teapot_mesh;
  uint8_t file_buffer[256 * 1024]; /* 256KB buffer for JSON file */
  struct {
    wgpu_buffer_t vertex;
    wgpu_buffer_t index;
  } buffers;
  struct {
    WGPUBuffer model;
    WGPUBuffer frame;
  } uniform_buffers;
  struct {
    WGPUTexture primitive_index;
    WGPUTextureView primitive_index_view;
    WGPUTexture depth;
    WGPUTextureView depth_view;
  } textures;
  struct {
    vec3 eye_position;
    vec3 up_vector;
    vec3 origin;
    mat4 projection_matrix;
    mat4 model_matrix;
    mat4 normal_model_matrix;
  } view_matrices;
  struct {
    WGPURenderPipeline forward_rendering;
    WGPURenderPipeline primitives_debug_view;
    WGPUComputePipeline pick;
  } pipelines;
  struct {
    WGPUBindGroup scene_uniform;
    WGPUBindGroup primitive_texture;
    WGPUBindGroup pick;
  } bind_groups;
  struct {
    bool show_primitive_indexes;
    bool rotate;
  } settings;
  struct {
    float x;
    float y;
  } pick_coord;
  float rad;
  WGPUBool mesh_loaded;
  WGPUBool initialized;
} state = {
  .view_matrices = {
    .eye_position = {0.0f, 12.0f, -25.0f},
    .up_vector    = {0.0f, 1.0f, 0.0f},
    .origin       = {0.0f, 0.0f, 0.0f},
  },
  .settings = {
    .show_primitive_indexes = false,
    .rotate                 = true,
  },
  .pick_coord = {
    .x = 0.0f,
    .y = 0.0f,
  },
  .rad           = 0.0f,
  .mesh_loaded   = false,
  .initialized   = false,
};

/* Teapot JSON file fetch callback */
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

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Release previous textures if they exist */
  WGPU_RELEASE_RESOURCE(Texture, state.textures.primitive_index)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.primitive_index_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.depth)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.depth_view)

  /* Primitive index texture */
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };
  state.textures.primitive_index = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Primitive index texture"),
                            .size          = texture_extent,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = WGPUTextureFormat_R32Uint,
                            .usage         = WGPUTextureUsage_RenderAttachment
                                     | WGPUTextureUsage_TextureBinding,
                          });
  ASSERT(state.textures.primitive_index != NULL);

  state.textures.primitive_index_view
    = wgpuTextureCreateView(state.textures.primitive_index,
                            &(WGPUTextureViewDescriptor){
                              .label  = STRVIEW("Primitive index texture view"),
                              .format = WGPUTextureFormat_R32Uint,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });
  ASSERT(state.textures.primitive_index_view != NULL);

  /* Depth texture */
  state.textures.depth = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth texture"),
                            .size          = texture_extent,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = WGPUTextureFormat_Depth24Plus,
                            .usage         = WGPUTextureUsage_RenderAttachment
                                     | WGPUTextureUsage_TextureBinding,
                          });
  ASSERT(state.textures.depth != NULL);

  state.textures.depth_view = wgpuTextureCreateView(
    state.textures.depth, &(WGPUTextureViewDescriptor){
                            .label           = STRVIEW("Depth texture view"),
                            .format          = WGPUTextureFormat_Depth24Plus,
                            .dimension       = WGPUTextureViewDimension_2D,
                            .baseMipLevel    = 0,
                            .mipLevelCount   = 1,
                            .baseArrayLayer  = 0,
                            .arrayLayerCount = 1,
                            .aspect          = WGPUTextureAspect_All,
                          });
  ASSERT(state.textures.depth_view != NULL);
}

static void init_buffers(wgpu_context_t* wgpu_context)
{
  /* Convert indexed geometry to non-indexed for correct primitive index
   * calculation Since we can't use primitive_id builtin, we compute it from
   * vertex_index / 3. This only works correctly with non-indexed drawing where
   * vertices are sequential.
   */
  const uint32_t triangle_count = state.teapot_mesh.triangles.count;
  const uint32_t vertex_count   = triangle_count * 3;
  const uint32_t vertex_stride  = 6; // position: vec3, normal: vec3
  const uint32_t vertex_buffer_size
    = vertex_count * vertex_stride * sizeof(float);

  float* vertex_data = malloc(vertex_buffer_size);
  ASSERT(vertex_data != NULL);

  /* Expand indexed geometry into non-indexed by duplicating vertices per
   * triangle */
  for (uint32_t tri_idx = 0; tri_idx < triangle_count; ++tri_idx) {
    const uint16_t* triangle = state.teapot_mesh.triangles.data[tri_idx];

    for (uint32_t v = 0; v < 3; ++v) {
      const uint16_t vert_idx = triangle[v];
      const uint32_t dst_idx  = (tri_idx * 3 + v) * vertex_stride;

      const float* pos = &state.teapot_mesh.positions.data[vert_idx][0];
      const float* nor = &state.teapot_mesh.normals.data[vert_idx][0];

      memcpy(&vertex_data[dst_idx], pos, 3 * sizeof(float));
      memcpy(&vertex_data[dst_idx + 3], nor, 3 * sizeof(float));
    }
  }

  state.buffers.vertex = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Teapot - Vertex buffer (non-indexed)",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                    .initial.data = vertex_data,
                    .count        = vertex_count,
                  });

  free(vertex_data);

  /* No index buffer needed - we're using non-indexed drawing */
  state.buffers.index.buffer = NULL;
  state.buffers.index.count  = 0;
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_perspective((2.0f * PI) / 5.0f, aspect, 1.0f, 2000.0f,
                  state.view_matrices.projection_matrix);

  /* Model matrix - move the model so it's centered */
  glm_mat4_identity(state.view_matrices.model_matrix);

  /* Normal model matrix */
  glm_mat4_inv(state.view_matrices.model_matrix,
               state.view_matrices.normal_model_matrix);
  glm_mat4_transpose(state.view_matrices.normal_model_matrix);

  /* Model uniform buffer (2 matrices: model + normal model) */
  state.uniform_buffers.model = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Model uniform buffer"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = 2 * sizeof(mat4),
    });
  ASSERT(state.uniform_buffers.model != NULL);

  /* Write model matrices to buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.model, 0,
                       state.view_matrices.model_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.model,
                       sizeof(mat4), state.view_matrices.normal_model_matrix,
                       sizeof(mat4));

  /* Frame uniform buffer (2 matrices + pick uniforms) */
  /* viewProjectionMatrix + invViewProjectionMatrix + pickCoord (vec2) +
   * pickedPrimitive (u32) */
  const uint32_t frame_buffer_size = 2 * sizeof(mat4) + 4 * sizeof(float);
  state.uniform_buffers.frame      = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
           .label = STRVIEW("Frame uniform buffer"),
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform
               | WGPUBufferUsage_Storage,
           .size = frame_buffer_size,
    });
  ASSERT(state.uniform_buffers.frame != NULL);
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Forward rendering pipeline */
  {
    WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
      wgpu_context->device, vertex_forward_rendering_wgsl);
    WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
      wgpu_context->device, fragment_forward_rendering_wgsl);

    /* Vertex buffer layout */
    WGPU_VERTEX_BUFFER_LAYOUT(
      teapot, sizeof(float) * 6,
      /* Attribute location 0: Position */
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
      /* Attribute location 1: Normal */
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUColorTargetState primitive_target_state = {
      .format    = WGPUTextureFormat_R32Uint,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUColorTargetState color_targets[2]
      = {color_target_state, primitive_target_state};

    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24Plus,
        .depth_write_enabled = true,
      });

    state.pipelines.forward_rendering = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Forward rendering pipeline"),
        .layout = NULL,
        .vertex
        = (WGPUVertexState){
          .module      = vert_shader_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 1,
          .buffers     = &teapot_vertex_buffer_layout,
        },
        .primitive
        = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_state,
        .multisample
        = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment
        = &(WGPUFragmentState){
          .module      = frag_shader_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 2,
          .targets     = color_targets,
        },
      });
    ASSERT(state.pipelines.forward_rendering != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
    WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  }

  /* Primitives debug view pipeline */
  {
    WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
      wgpu_context->device, vertex_texture_quad_wgsl);
    WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
      wgpu_context->device, fragment_primitives_debug_view_wgsl);

    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.pipelines.primitives_debug_view = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Primitives debug view pipeline"),
        .layout = NULL,
        .vertex
        = (WGPUVertexState){
          .module     = vert_shader_module,
          .entryPoint = STRVIEW("main"),
        },
        .primitive
        = (WGPUPrimitiveState){
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .multisample
        = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment
        = &(WGPUFragmentState){
          .module      = frag_shader_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets     = &color_target_state,
        },
      });
    ASSERT(state.pipelines.primitives_debug_view != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vert_shader_module)
    WGPU_RELEASE_RESOURCE(ShaderModule, frag_shader_module)
  }

  /* Pick compute pipeline */
  {
    WGPUShaderModule comp_shader_module = wgpu_create_shader_module(
      wgpu_context->device, compute_pick_primitive_wgsl);

    state.pipelines.pick = wgpuDeviceCreateComputePipeline(
      wgpu_context->device, &(WGPUComputePipelineDescriptor){
                              .label   = STRVIEW("Pick compute pipeline"),
                              .layout  = NULL,
                              .compute = (WGPUComputeState){
                                .module     = comp_shader_module,
                                .entryPoint = STRVIEW("main"),
                              },
                            });
    ASSERT(state.pipelines.pick != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, comp_shader_module)
  }
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Scene uniform bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffers.model,
        .size    = 2 * sizeof(mat4),
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = state.uniform_buffers.frame,
        .size    = 2 * sizeof(mat4) + 4 * sizeof(float),
      },
    };

    state.bind_groups.scene_uniform = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Scene uniform bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.pipelines.forward_rendering, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.scene_uniform != NULL);
  }

  /* Primitive texture bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = state.textures.primitive_index_view,
      },
    };

    state.bind_groups.primitive_texture = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Primitive texture bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.pipelines.primitives_debug_view, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.primitive_texture != NULL);
  }

  /* Pick bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffers.frame,
        .size    = 2 * sizeof(mat4) + 4 * sizeof(float),
      },
      [1] = (WGPUBindGroupEntry){
        .binding     = 1,
        .textureView = state.textures.primitive_index_view,
      },
    };

    state.bind_groups.pick = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Pick bind group"),
                              .layout = wgpuComputePipelineGetBindGroupLayout(
                                state.pipelines.pick, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.pick != NULL);
  }
}

static void recreate_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Release old bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.primitive_texture)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.pick)

  /* Primitive texture bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = state.textures.primitive_index_view,
      },
    };

    state.bind_groups.primitive_texture = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Primitive texture bind group"),
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                state.pipelines.primitives_debug_view, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.primitive_texture != NULL);
  }

  /* Pick bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffers.frame,
        .size    = 2 * sizeof(mat4) + 4 * sizeof(float),
      },
      [1] = (WGPUBindGroupEntry){
        .binding     = 1,
        .textureView = state.textures.primitive_index_view,
      },
    };

    state.bind_groups.pick = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Pick bind group"),
                              .layout = wgpuComputePipelineGetBindGroupLayout(
                                state.pipelines.pick, 0),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.pick != NULL);
  }
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context && state.mesh_loaded) {
    stm_setup();
    init_textures(wgpu_context);
    init_buffers(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void get_camera_view_proj_matrix(mat4 dest)
{
  if (state.settings.rotate) {
    state.rad = PI * (stm_sec(stm_now()) / 10.0f);
  }

  mat4 rotation;
  glm_mat4_identity(rotation);
  glm_translate(rotation, state.view_matrices.origin);
  glm_rotate_y(rotation, state.rad, rotation);

  vec3 rotated_eye_position;
  glm_mat4_mulv3(rotation, state.view_matrices.eye_position, 1.0f,
                 rotated_eye_position);

  mat4 view_matrix;
  glm_lookat(rotated_eye_position, state.view_matrices.origin,
             state.view_matrices.up_vector, view_matrix);

  glm_mat4_mul(state.view_matrices.projection_matrix, view_matrix, dest);
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Update frame uniforms */
  mat4 camera_view_proj;
  get_camera_view_proj_matrix(camera_view_proj);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.frame, 0,
                       camera_view_proj, sizeof(mat4));

  mat4 camera_inv_view_proj;
  glm_mat4_inv(camera_view_proj, camera_inv_view_proj);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.frame,
                       sizeof(mat4), camera_inv_view_proj, sizeof(mat4));

  /* Write pick coordinates */
  float pick_data[2] = {state.pick_coord.x, state.pick_coord.y};
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.frame,
                       2 * sizeof(mat4), pick_data, 2 * sizeof(float));

  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Forward rendering pass */
  {
    WGPURenderPassColorAttachment color_attachments[2] = {
      [0] =
        (WGPURenderPassColorAttachment){
          .view       = wgpu_context->swapchain_view,
          .loadOp     = WGPULoadOp_Clear,
          .storeOp    = WGPUStoreOp_Store,
          .clearValue = (WGPUColor){0.0f, 0.0f, 1.0f, 1.0f},
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
      [1] =
        (WGPURenderPassColorAttachment){
          .view       = state.textures.primitive_index_view,
          .loadOp     = WGPULoadOp_Clear,
          .storeOp    = WGPUStoreOp_Store,
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        },
    };

    WGPURenderPassDepthStencilAttachment depth_stencil_attachment = {
      .view            = state.textures.depth_view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };

    WGPURenderPassDescriptor render_pass_desc = {
      .label                  = STRVIEW("Forward rendering pass"),
      .colorAttachmentCount   = 2,
      .colorAttachments       = color_attachments,
      .depthStencilAttachment = &depth_stencil_attachment,
    };

    WGPURenderPassEncoder render_pass
      = wgpuCommandEncoderBeginRenderPass(cmd_encoder, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(render_pass,
                                     state.pipelines.forward_rendering);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                      state.bind_groups.scene_uniform, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 0, state.buffers.vertex.buffer, 0, WGPU_WHOLE_SIZE);
    /* Use non-indexed drawing for correct primitive index calculation */
    wgpuRenderPassEncoderDraw(render_pass, state.buffers.vertex.count, 1, 0, 0);
    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }

  /* Primitive index debug view pass (optional) */
  if (state.settings.show_primitive_indexes) {
    WGPURenderPassColorAttachment color_attachment = {
      .view       = wgpu_context->swapchain_view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor){0.0f, 0.0f, 0.0f, 1.0f},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };

    WGPURenderPassDescriptor render_pass_desc = {
      .label                = STRVIEW("Primitive index debug view pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &color_attachment,
    };

    WGPURenderPassEncoder render_pass
      = wgpuCommandEncoderBeginRenderPass(cmd_encoder, &render_pass_desc);
    wgpuRenderPassEncoderSetPipeline(render_pass,
                                     state.pipelines.primitives_debug_view);
    wgpuRenderPassEncoderSetBindGroup(
      render_pass, 0, state.bind_groups.primitive_texture, 0, NULL);
    wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }

  /* Pick compute pass */
  {
    WGPUComputePassDescriptor compute_pass_desc = {
      .label = STRVIEW("Pick compute pass"),
    };

    WGPUComputePassEncoder compute_pass
      = wgpuCommandEncoderBeginComputePass(cmd_encoder, &compute_pass_desc);
    wgpuComputePassEncoderSetPipeline(compute_pass, state.pipelines.pick);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, state.bind_groups.pick,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(compute_pass);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, compute_pass)
  }

  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  return command_buffer;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  /* Process async file loading */
  sfetch_dowork();

  /* Initialize pipelines once mesh is loaded */
  if (state.mesh_loaded && !state.initialized) {
    init(wgpu_context);
  }

  /* Only render if mesh and pipelines are ready */
  if (!state.initialized) {
    return EXIT_SUCCESS;
  }

  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer */
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  /* Handle mouse move events */
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    state.pick_coord.x = input_event->mouse_x;
    state.pick_coord.y = input_event->mouse_y;
  }
  /* Handle resize events */
  else if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    if (state.initialized) {
      init_textures(wgpu_context);
      recreate_bind_groups(wgpu_context);

      /* Update projection matrix */
      const float aspect
        = (float)wgpu_context->width / (float)wgpu_context->height;
      glm_perspective((2.0f * PI) / 5.0f, aspect, 1.0f, 2000.0f,
                      state.view_matrices.projection_matrix);
    }
  }
}

static int setup(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Initialize sokol_fetch */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 1,
    .num_channels = 1,
    .num_lanes    = 1,
  });

  /* Start loading the teapot mesh */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/meshes/teapot.json",
    .callback = teapot_json_fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Release buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.vertex.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.index.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.model)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.frame)

  /* Release textures */
  WGPU_RELEASE_RESOURCE(Texture, state.textures.primitive_index)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.primitive_index_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.depth)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.depth_view)

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.forward_rendering)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.primitives_debug_view)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.pipelines.pick)

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene_uniform)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.primitive_texture)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.pick)

  /* Shutdown sokol_fetch */
  sfetch_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Primitive Picking",
    .init_cb        = setup,
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
static const char* vertex_forward_rendering_wgsl = CODE(
  struct Uniforms {
    modelMatrix : mat4x4f,
    normalModelMatrix : mat4x4f,
  }
  struct Frame {
    viewProjectionMatrix : mat4x4f,
    invViewProjectionMatrix : mat4x4f,
    pickCoord : vec2u,
    pickedPrimitive : u32,
  }
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var<uniform> frame : Frame;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) fragNormal : vec3f, // normal in world space
    @location(1) @interpolate(flat) vertexIndex : u32,
  }

  @vertex
  fn main(
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @builtin(vertex_index) vertexIndex : u32,
  ) -> VertexOutput {
    var output : VertexOutput;
    let worldPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
    output.Position = frame.viewProjectionMatrix * vec4(worldPosition, 1.0);
    output.fragNormal = normalize((uniforms.normalModelMatrix * vec4(normal, 1.0)).xyz);
    output.vertexIndex = vertexIndex;
    return output;
  }
);

static const char* fragment_forward_rendering_wgsl = CODE(
  struct Frame {
    viewProjectionMatrix : mat4x4f,
    invViewProjectionMatrix : mat4x4f,
    pickCoord : vec2f,
    pickedPrimitive : u32,
  }
  
  @group(0) @binding(1) var<uniform> frame : Frame;

  struct PassOutput {
    @location(0) color : vec4f,
    @location(1) primitive : u32,
  }

  @fragment fn main(@location(0) fragNormal : vec3f,
                    @location(1) @interpolate(flat) vertexIndex : u32
  ) -> PassOutput {
    // Compute primitive index from vertex index (3 vertices per triangle)
    let primIndex = vertexIndex / 3u;

    // Very simple N-dot-L lighting model
    let lightDirection = normalize(vec3f(4, 10, 6));
    let light        = dot(normalize(fragNormal), lightDirection) * 0.5 + 0.5;
    let surfaceColor = vec4f(0.8, 0.8, 0.8, 1.0);

    var output : PassOutput;

    // Highlight the primitive if it's the selected one, otherwise shade
    // normally.
    if (primIndex + 1 == frame.pickedPrimitive) {
      output.color = vec4f(1.0, 1.0, 0.0, 1.0);
    }
    else {
      output.color = vec4f(surfaceColor.xyz * light, surfaceColor.a);
    }

    // Adding one to each primitive index so that 0 can mean "nothing picked"
    output.primitive = primIndex + 1;
    return output;
  }
);

static const char* vertex_texture_quad_wgsl = CODE(
  @vertex
  fn main(@builtin(vertex_index) VertexIndex : u32
  ) ->@builtin(position) vec4f {
    const pos
      = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
              vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), );

    return vec4f(pos[VertexIndex], 0.0, 1.0);
  }
);

static const char* fragment_primitives_debug_view_wgsl = CODE(
  @group(0) @binding(0) var primitiveTex: texture_2d<u32>;

  @fragment
  fn main(
    @builtin(position) coord : vec4f
  ) -> @location(0) vec4f {
    // Load the primitive index for this pixel from the picking texture.
    let primitiveIndex = textureLoad(primitiveTex, vec2i(floor(coord.xy)), 0).x;
    var result : vec4f;

    // Generate a color for the primitive index. If we only increment the color
    // channels by 1 for each primitive index we can show a very large range of
    // unique values but it can make the individual primitives hard to distinguish.
    // This code steps through 8 distinct values per-channel, which may end up
    // repeating some colors for larger meshes but makes the unique primitive
    // index values easier to see.
    result.r = f32(primitiveIndex % 8) / 8;
    result.g = f32((primitiveIndex / 8) % 8) / 8;
    result.b = f32((primitiveIndex / 64) % 8) / 8;
    result.a = 1.0;
    return result;
  }
);

static const char* compute_pick_primitive_wgsl = CODE(
  struct Frame {
    viewProjectionMatrix : mat4x4f,
    invViewProjectionMatrix : mat4x4f,
    pickCoord : vec2f,
    pickedPrimitive : u32,
  }
  @group(0) @binding(0) var<storage, read_write> frame : Frame;
  @group(0) @binding(1) var primitiveTex : texture_2d<u32>;

  @compute @workgroup_size(1)
  fn main() {
    // Load the primitive index from the picking texture and store it in the
    // pickedPrimitive value (exposed to the rendering shaders as a uniform).
    let texel = vec2u(frame.pickCoord);
    frame.pickedPrimitive = textureLoad(primitiveTex, texel, 0).x;
  }
);
// clang-format on