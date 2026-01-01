#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Shadow Mapping
 *
 * This example shows how to sample from a depth texture to render shadows.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/pages/samples/shadowMapping.ts
 * stanford-dragon: https://github.com/hughsk/stanford-dragon
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* fragment_wgsl;
static const char* vertex_wgsl;
static const char* vertex_shadow_wgsl;

/* -------------------------------------------------------------------------- *
 * Shadow Mapping example
 * -------------------------------------------------------------------------- */

static const uint32_t shadow_depth_texture_size = 1024;

/* State struct */
static struct {
  stanford_dragon_mesh_t dragon_mesh;
  /* Vertex and index buffers */
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } vertices;
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } indices;
  /* Uniform buffers */
  struct {
    WGPUBuffer model;
    WGPUBuffer scene;
  } uniform_buffers;
  /* View matrices */
  struct {
    vec3 up_vector;
    vec3 origin;
    mat4 projection_matrix;
    mat4 view_proj_matrix;
  } view_matrices;
  /* Time tracking */
  uint64_t time_offset;
  /* The pipeline layouts */
  struct {
    WGPUPipelineLayout shadow;
    WGPUPipelineLayout color;
  } pipeline_layouts;
  /* Pipelines */
  struct {
    WGPURenderPipeline shadow;
    WGPURenderPipeline color;
  } render_pipelines;
  /* Render pass descriptors */
  struct {
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } shadow_render_pass;
  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } color_render_pass;
  /* Bind groups */
  struct {
    WGPUBindGroup scene_shadow;
    WGPUBindGroup scene_render;
    WGPUBindGroup model;
  } bind_groups;
  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout uniform_buffer_scene;
    WGPUBindGroupLayout uniform_buffer_model;
    WGPUBindGroupLayout render;
  } bind_group_layouts;
  /* Textures and sampler */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } shadow_depth_texture;
  WGPUSampler sampler;
  WGPUBool initialized;
} state = {
  .color_render_pass = {
    .color_attachment = {
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.5, 0.5, 0.5, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    },
    .depth_stencil_attachment = {
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    },
  },
  .shadow_render_pass = {
    .depth_stencil_attachment = {
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    },
  },
};

/* Prepare vertex and index buffers for the Stanford dragon mesh */
static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Create the model vertex buffer */
  {
    const uint8_t ground_plane_vertex_count = 4;
    uint64_t vertex_buffer_size
      = (state.dragon_mesh.positions.count + ground_plane_vertex_count) * 3 * 2
        * sizeof(float);
    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Model - Vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex,
      .size             = vertex_buffer_size,
      .mappedAtCreation = true,
    };
    state.vertices.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(state.vertices.buffer);
    float* mapping = (float*)wgpuBufferGetMappedRange(state.vertices.buffer, 0,
                                                      vertex_buffer_size);
    ASSERT(mapping);
    for (uint64_t i = 0; i < state.dragon_mesh.positions.count; ++i) {
      memcpy(&mapping[6 * i], state.dragon_mesh.positions.data[i],
             sizeof(vec3));
      memcpy(&mapping[6 * i + 3], state.dragon_mesh.normals.data[i],
             sizeof(vec3));
    }
    /* Push vertex attributes for an additional ground plane */
    static const vec3 ground_plane_positions[4] = {
      {-100.0f, 20.0f, -100.0f}, //
      {100.0f, 20.0f, 100.0f},   //
      {-100.0f, 20.0f, 100.0f},  //
      {100.0f, 20.0f, -100.0f},  //
    };
    static const vec3 ground_plane_normals[4] = {
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f}, //
    };
    const uint64_t offset = state.dragon_mesh.positions.count * 6;
    for (uint64_t i = 0; i < ground_plane_vertex_count; ++i) {
      memcpy(&mapping[offset + 6 * i], ground_plane_positions[i], sizeof(vec3));
      memcpy(&mapping[offset + 6 * i + 3], ground_plane_normals[i],
             sizeof(vec3));
    }
    wgpuBufferUnmap(state.vertices.buffer);
  }

  /* Create the model index buffer */
  {
    const uint8_t ground_plane_index_count = 2;
    state.indices.count
      = (state.dragon_mesh.triangles.count + ground_plane_index_count) * 3;
    uint64_t index_buffer_size       = state.indices.count * sizeof(uint16_t);
    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Model - Index buffer"),
      .usage            = WGPUBufferUsage_Index,
      .size             = index_buffer_size,
      .mappedAtCreation = true,
    };
    state.indices.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(state.indices.buffer);
    uint16_t* mapping = (uint16_t*)wgpuBufferGetMappedRange(
      state.indices.buffer, 0, index_buffer_size);
    ASSERT(mapping)
    for (uint64_t i = 0; i < state.dragon_mesh.triangles.count; ++i) {
      memcpy(&mapping[3 * i], state.dragon_mesh.triangles.data[i],
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
    const uint64_t offset = state.dragon_mesh.triangles.count * 3;
    for (uint64_t i = 0; i < ground_plane_index_count; ++i) {
      memcpy(&mapping[offset + 3 * i], ground_plane_indices[i],
             sizeof(uint16_t) * 3);
    }
    wgpuBufferUnmap(state.indices.buffer);
  }
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Create the depth texture for rendering/sampling the shadow map */
  {
    WGPUExtent3D texture_extent = {
      .width              = shadow_depth_texture_size,
      .height             = shadow_depth_texture_size,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Shadow depth - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth32Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    state.shadow_depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.shadow_depth_texture.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Depth32Float,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    state.shadow_depth_texture.view = wgpuTextureCreateView(
      state.shadow_depth_texture.texture, &texture_view_dec);
    ASSERT(state.shadow_depth_texture.view != NULL);
  }

  /* Create a depth/stencil texture for the color rendering pipeline */
  {
    WGPUExtent3D texture_extent = {
      .width              = wgpu_context->width,
      .height             = wgpu_context->height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Depth stencil - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    state.depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.depth_texture.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Depth24PlusStencil8,
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
}

static void prepare_sampler(wgpu_context_t* wgpu_context)
{
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Shadow - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Nearest,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .compare       = WGPUCompareFunction_Less,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.sampler != NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Bind group layout for uniform buffers in shadow pipeline */
  {
    /* Bind group layout for scene uniform */
    {
      WGPUBindGroupLayoutEntry bgl_entries[1] = {
        [0] = (WGPUBindGroupLayoutEntry) {
          /* Binding 0: Uniform */
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer = (WGPUBufferBindingLayout) {
            .type             = WGPUBufferBindingType_Uniform,
            .hasDynamicOffset = false,
            .minBindingSize   = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
          },
          .sampler = {0},
        },
      };
      state.bind_group_layouts.uniform_buffer_scene
        = wgpuDeviceCreateBindGroupLayout(
          wgpu_context->device,
          &(WGPUBindGroupLayoutDescriptor){
            .label      = STRVIEW("Scene uniform - Bind group layout"),
            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
            .entries    = bgl_entries,
          });
      ASSERT(state.bind_group_layouts.uniform_buffer_scene != NULL);
    }

    /* Bind group layout for model uniform */
    {
      WGPUBindGroupLayoutEntry bgl_entries[1] = {
        [0] = (WGPUBindGroupLayoutEntry) {
          /* Binding 0: Uniform */
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer = (WGPUBufferBindingLayout) {
            .type             = WGPUBufferBindingType_Uniform,
            .hasDynamicOffset = false,
            .minBindingSize   = sizeof(mat4),
          },
          .sampler = {0},
        },
      };
      state.bind_group_layouts.uniform_buffer_model
        = wgpuDeviceCreateBindGroupLayout(
          wgpu_context->device,
          &(WGPUBindGroupLayoutDescriptor){
            .label      = STRVIEW("Model uniform - Bind group layout"),
            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
            .entries    = bgl_entries,
          });
      ASSERT(state.bind_group_layouts.uniform_buffer_model != NULL);
    }

    WGPUBindGroupLayout bind_group_layouts[2] = {
      state.bind_group_layouts.uniform_buffer_scene,
      state.bind_group_layouts.uniform_buffer_model,
    };
    state.pipeline_layouts.shadow = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Shadow - Pipeline layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(state.pipeline_layouts.shadow != NULL);
  }

  /* Create a bind group layout which holds the scene uniforms and
   * the texture+sampler for depth. We create it manually because the WebGPU
   * implementation doesn't infer this from the shader (yet). */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Uniform */
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Texture view */
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Depth,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Sampler */
        .binding    = 2,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Comparison,
        },
        .texture = {0},
      }
    };
    state.bind_group_layouts.render = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Render - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(state.bind_group_layouts.render != NULL);

    /* Specify the pipeline layout. The layout for the model is the same, so
     * reuse it from the shadow pipeline. */
    WGPUBindGroupLayout bind_group_layouts[2] = {
      state.bind_group_layouts.render,               /* Group 0 */
      state.bind_group_layouts.uniform_buffer_model, /* Group 1 */
    };
    state.pipeline_layouts.color = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Color - Pipeline layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(state.pipeline_layouts.color != NULL);
  }
}

static void setup_render_pass(void)
{
  /* Shadow rendering */
  {
    /* Shadow pass descriptor */
    state.shadow_render_pass.depth_stencil_attachment.view
      = state.shadow_depth_texture.view;

    state.shadow_render_pass.descriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Shadow - Render pass"),
      .colorAttachmentCount = 0,
      .colorAttachments     = NULL,
      .depthStencilAttachment
      = &state.shadow_render_pass.depth_stencil_attachment,
      .occlusionQuerySet = NULL,
    };
  }

  /* Color rendering */
  {
    /* Depth stencil attachment */
    state.color_render_pass.depth_stencil_attachment.view
      = state.depth_texture.view;

    /* Render pass descriptor */
    state.color_render_pass.descriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Color - Render pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.color_render_pass.color_attachment,
      .depthStencilAttachment
      = &state.color_render_pass.depth_stencil_attachment,
      .occlusionQuerySet = NULL,
    };
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  vec3 eye_position = {0.0f, 50.0f, -100.0f};
  memcpy(state.view_matrices.up_vector, (vec3){0.0f, 1.0f, 0.0f}, sizeof(vec3));
  memcpy(state.view_matrices.origin, (vec3){0.0f, 0.0f, 0.0f}, sizeof(vec3));

  glm_mat4_identity(state.view_matrices.projection_matrix);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                  state.view_matrices.projection_matrix);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,                  /* eye vector */
             state.view_matrices.origin,    /* center vector */
             state.view_matrices.up_vector, /* up vector */
             view_matrix                    /* result matrix */
  );

  vec3 light_position    = {50.0f, 100.0f, -100.0f};
  mat4 light_view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(light_position,                /* eye vector */
             state.view_matrices.origin,    /* center vector */
             state.view_matrices.up_vector, /* up vector */
             light_view_matrix              /* result matrix */
  );

  mat4 light_projection_matrix = GLM_MAT4_IDENTITY_INIT;
  {
    const float left   = -80.0f;
    const float right  = 80.0f;
    const float bottom = -80.0f;
    const float top    = 80.0f;
    const float near   = -200.0f;
    const float far    = 300.0f;
    glm_ortho(left, right, bottom, top, near, far, light_projection_matrix);
  }

  mat4 light_view_proj_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_mulN((mat4*[]){&light_projection_matrix, &light_view_matrix}, 2,
                light_view_proj_matrix);

  glm_mat4_identity(state.view_matrices.view_proj_matrix);
  glm_mat4_mulN((mat4*[]){&state.view_matrices.projection_matrix, &view_matrix},
                2, state.view_matrices.view_proj_matrix);

  /* Move the model so it's centered. */
  mat4 model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_translate(model_matrix, (vec3){0.0f, -5.0f, 0.0f});
  glm_translate(model_matrix, (vec3){0.0f, -40.0f, 0.0f});

  /* The camera/light aren't moving, so write them into buffers now. */
  {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.scene, 0,
                         light_view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.scene, 64,
                         state.view_matrices.view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.scene, 128,
                         light_position, sizeof(vec3));

    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.model, 0,
                         model_matrix, sizeof(mat4));
  }
}

/* Rotate a 3D vector around the y-axis
 * Ref: https://glmatrix.net/docs/vec3.js.html#line593 */

static void glm_vec3_rotate_y(vec3 a, vec3 b, float rad, vec3* out)
{
  vec3 p, r;

  /* Translate point to the origin */
  p[0] = a[0] - b[0];
  p[1] = a[1] - b[1];
  p[2] = a[2] - b[2];

  /* perform rotation */

  r[0] = p[2] * sin(rad) + p[0] * cos(rad);
  r[1] = p[1];
  r[2] = p[2] * cos(rad) - p[0] * sin(rad);

  /* translate to correct position */
  (*out)[0] = r[0] + b[0];
  (*out)[1] = r[1] + b[1];
  (*out)[2] = r[2] + b[2];
}

/* Rotates the camera around the origin based on time. */
static mat4* get_camera_view_proj_matrix(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  vec3 eye_position = {0.0f, 50.0f, -100.0f};

  const uint64_t now  = stm_now();
  const float time_ms = (float)stm_ms(stm_diff(now, state.time_offset));
  float rad           = PI * (time_ms / 2000.0f);
  glm_vec3_rotate_y(eye_position, state.view_matrices.origin, rad,
                    &eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,                  /* eye vector    */
             state.view_matrices.origin,    /* center vector */
             state.view_matrices.up_vector, /* up vector     */
             view_matrix                    /* result matrix */
  );

  glm_mat4_mulN((mat4*[]){&state.view_matrices.projection_matrix, &view_matrix},
                2, state.view_matrices.view_proj_matrix);
  return &state.view_matrices.view_proj_matrix;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  mat4* camera_view_proj = get_camera_view_proj_matrix(wgpu_context);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.scene, 64,
                       *camera_view_proj, sizeof(mat4));
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Model uniform buffer */
  {
    const WGPUBufferDescriptor buffer_desc = {
      .label = STRVIEW("Model - Uniform buffer"),
      .size  = sizeof(mat4), /* 4x4 matrix */
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    state.uniform_buffers.model
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(state.uniform_buffers.model)
  }

  /* Scene uniform buffer */
  {
    state.uniform_buffers.scene = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Scene - Uniform buffer"),
        /* Two 4x4 viewProj matrices, one for the camera and one for the light.
         * Then a vec3 for the light position. */
        .size  = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
    ASSERT(state.uniform_buffers.scene);
  }

  /* Scene bind group for shadow */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.uniform_buffers.scene,
        .size    = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
      },
    };
    state.bind_groups.scene_shadow = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Scene shadow - Bind group"),
        .layout     = state.bind_group_layouts.uniform_buffer_scene,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.bind_groups.scene_shadow != NULL);
  }

  /* Scene bind group for render */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.uniform_buffers.scene,
        .size    = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = state.shadow_depth_texture.view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = state.sampler,
      },
    };
    state.bind_groups.scene_render = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Scene render - Bind group"),
                              .layout = state.bind_group_layouts.render,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.scene_render != NULL);
  }

  /* Model bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.uniform_buffers.model,
        .size    = sizeof(mat4),
      },
    };
    state.bind_groups.model = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Model - Bind group"),
        .layout     = state.bind_group_layouts.uniform_buffer_model,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(state.bind_groups.model != NULL);
  }
}

/* Create the shadow pipeline */
static void prepare_shadow_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state = {
    .format              = WGPUTextureFormat_Depth32Float,
    .depthWriteEnabled   = true,
    .depthCompare        = WGPUCompareFunction_Less,
    .stencilFront        = {
      .compare     = WGPUCompareFunction_Always,
      .failOp      = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp      = WGPUStencilOperation_Keep,
    },
    .stencilBack = {
      .compare     = WGPUCompareFunction_Always,
      .failOp      = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp      = WGPUStencilOperation_Keep,
    },
    .stencilReadMask  = 0xFFFFFFFF,
    .stencilWriteMask = 0xFFFFFFFF,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    shadow, sizeof(float) * 6,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shadow_wgsl),
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 1,
    .buffers     = &shadow_vertex_buffer_layout,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count                  = 1,
    .mask                   = 0xFFFFFFFF,
    .alphaToCoverageEnabled = false,
  };

  /* Create rendering pipeline using the specified states */
  state.render_pipelines.shadow = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = STRVIEW("Shadow - Render pipeline"),
                            .layout       = state.pipeline_layouts.shadow,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = NULL,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(state.render_pipelines.shadow != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
}

/* Create the color rendering pipeline */
static void prepare_color_rendering_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color blend state */
  WGPUBlendComponent blend_component = {
    .operation = WGPUBlendOperation_Add,
    .srcFactor = WGPUBlendFactor_One,
    .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
  };
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->render_format,
    .blend     = &(WGPUBlendState){
      .color = blend_component,
      .alpha = blend_component,
    },
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Constants */
  WGPUConstantEntry constant_entries[1] = {
    [0] = (WGPUConstantEntry){
      .key   = STRVIEW("shadowDepthTextureSize"),
      .value = shadow_depth_texture_size,
    },
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state = {
    .depthWriteEnabled = true,
    .format            = WGPUTextureFormat_Depth24PlusStencil8,
    .depthCompare      = WGPUCompareFunction_Less,
    .stencilFront = {
      .compare     = WGPUCompareFunction_Always,
      .failOp      = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp      = WGPUStencilOperation_Keep,
    },
    .stencilBack = {
      .compare     = WGPUCompareFunction_Always,
      .failOp      = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp      = WGPUStencilOperation_Keep,
    },
    .stencilReadMask  = 0xFFFFFFFF,
    .stencilWriteMask = 0xFFFFFFFF,
  };

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    color, sizeof(float) * 6,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  /* Vertex state */
  WGPUVertexState vertex_state = {
    .module      = wgpu_create_shader_module(wgpu_context->device, vertex_wgsl),
    .entryPoint  = STRVIEW("main"),
    .bufferCount = 1,
    .buffers     = &color_vertex_buffer_layout,
  };

  /* Fragment state */
  WGPUFragmentState fragment_state = {
    .module = wgpu_create_shader_module(wgpu_context->device, fragment_wgsl),
    .entryPoint    = STRVIEW("main"),
    .constantCount = (uint32_t)ARRAY_SIZE(constant_entries),
    .constants     = constant_entries,
    .targetCount   = 1,
    .targets       = &color_target_state,
  };

  /* Multisample state */
  WGPUMultisampleState multisample_state = {
    .count                  = 1,
    .mask                   = 0xFFFFFFFF,
    .alphaToCoverageEnabled = false,
  };

  /* Create rendering pipeline using the specified states */
  state.render_pipelines.color = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = STRVIEW("Color - Render pipeline"),
                            .layout       = state.pipeline_layouts.color,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(state.render_pipelines.color != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

/* Recreate depth texture on window resize */
static void recreate_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Release old depth texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture);

  /* Create new depth/stencil texture for the color rendering pipeline */
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->width,
    .height             = wgpu_context->height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Depth stencil - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth24PlusStencil8,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.depth_texture.texture != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = WGPUTextureFormat_Depth24PlusStencil8,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.texture, &texture_view_dec);
  ASSERT(state.depth_texture.view != NULL);

  /* Update render pass depth attachment */
  state.color_render_pass.depth_stencil_attachment.view
    = state.depth_texture.view;
}

/* Handle window resize */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    recreate_depth_texture(wgpu_context);

    /* Update projection matrix */
    const float aspect_ratio
      = (float)wgpu_context->width / (float)wgpu_context->height;
    glm_mat4_identity(state.view_matrices.projection_matrix);
    glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                    state.view_matrices.projection_matrix);
  }
}

static int init_cb(wgpu_context_t* wgpu_context)
{
  stanford_dragon_mesh_init(&state.dragon_mesh);
  prepare_vertex_and_index_buffers(wgpu_context);
  prepare_textures(wgpu_context);
  prepare_sampler(wgpu_context);
  setup_pipeline_layout(wgpu_context);
  prepare_shadow_pipeline(wgpu_context);
  prepare_color_rendering_pipeline(wgpu_context);
  prepare_uniform_buffers(wgpu_context);
  prepare_view_matrices(wgpu_context);
  setup_render_pass();

  /* Initialize time tracking */
  stm_setup();
  state.time_offset = stm_now();

  state.initialized = true;
  return 0;
}

static int frame_cb(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return 1;
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Shadow pass */
  {
    WGPURenderPassEncoder shadow_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_encoder, &state.shadow_render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(shadow_pass,
                                     state.render_pipelines.shadow);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 0,
                                      state.bind_groups.scene_shadow, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 1, state.bind_groups.model,
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(shadow_pass, 0, state.vertices.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(shadow_pass, state.indices.buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(shadow_pass, state.indices.count, 1, 0, 0,
                                     0);

    wgpuRenderPassEncoderEnd(shadow_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, shadow_pass)
  }

  /* Color render pass */
  {
    state.color_render_pass.color_attachment.view
      = wgpu_context->swapchain_view;
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_encoder, &state.color_render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(render_pass, state.render_pipelines.color);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                      state.bind_groups.scene_render, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, state.bind_groups.model,
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, state.vertices.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(render_pass, state.indices.buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(render_pass, state.indices.count, 1, 0, 0,
                                     0);

    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  /* Submit to queue */
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer);

  return 0;
}

/* Clean up used resources */
static void shutdown_cb(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.model)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffers.scene)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.shadow)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layouts.color)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipelines.shadow);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipelines.color)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene_shadow)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.scene_render)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.bind_group_layouts.uniform_buffer_scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.bind_group_layouts.uniform_buffer_model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.render)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_depth_texture.view)
}

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "Shadow Mapping",
    .init_cb        = init_cb,
    .frame_cb       = frame_cb,
    .shutdown_cb    = shutdown_cb,
    .input_event_cb = input_event_cb,
  });

  return 0;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* fragment_wgsl = CODE(
  override shadowDepthTextureSize: f32 = 1024.0;

  struct Scene {
    lightViewProjMatrix : mat4x4f,
    cameraViewProjMatrix : mat4x4f,
    lightPos : vec3f,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(0) @binding(1) var shadowMap: texture_depth_2d;
  @group(0) @binding(2) var shadowSampler: sampler_comparison;

  struct FragmentInput {
    @location(0) shadowPos : vec3f,
    @location(1) fragPos : vec3f,
    @location(2) fragNorm : vec3f,
  }

  const albedo = vec3f(0.9);
  const ambientFactor = 0.2;

  @fragment
  fn main(input : FragmentInput) -> @location(0) vec4f {
    // Percentage-closer filtering. Sample texels in the region
    // to smooth the result.
    var visibility = 0.0;
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let offset = vec2f(vec2(x, y)) * oneOverShadowDepthTextureSize;

        visibility += textureSampleCompare(
          shadowMap, shadowSampler,
          input.shadowPos.xy + offset, input.shadowPos.z - 0.007
        );
      }
    }
    visibility /= 9.0;

    let lambertFactor = max(dot(normalize(scene.lightPos - input.fragPos), normalize(input.fragNorm)), 0.0);
    let lightingFactor = min(ambientFactor + visibility * lambertFactor, 1.0);

    return vec4(lightingFactor * albedo, 1.0);
  }
);

static const char* vertex_wgsl = CODE(
  struct Scene {
    lightViewProjMatrix: mat4x4f,
    cameraViewProjMatrix: mat4x4f,
    lightPos: vec3f,
  }

  struct Model {
    modelMatrix: mat4x4f,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(1) @binding(0) var<uniform> model : Model;

  struct VertexOutput {
    @location(0) shadowPos: vec3f,
    @location(1) fragPos: vec3f,
    @location(2) fragNorm: vec3f,

    @builtin(position) Position: vec4f,
  }

  @vertex
  fn main(
    @location(0) position: vec3f,
    @location(1) normal: vec3f
  ) -> VertexOutput {
    var output : VertexOutput;

    // XY is in (-1, 1) space, Z is in (0, 1) space
    let posFromLight = scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);

    // Convert XY to (0, 1)
    // Y is flipped because texture coords are Y-down.
    output.shadowPos = vec3(
      posFromLight.xy * vec2(0.5, -0.5) + vec2(0.5),
      posFromLight.z
    );

    output.Position = scene.cameraViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
    output.fragPos = output.Position.xyz;
    output.fragNorm = normal;
    return output;
  }
);

static const char* vertex_shadow_wgsl = CODE(
  struct Scene {
    lightViewProjMatrix: mat4x4f,
    cameraViewProjMatrix: mat4x4f,
    lightPos: vec3f,
  }

  struct Model {
    modelMatrix: mat4x4f,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(1) @binding(0) var<uniform> model : Model;

  @vertex
  fn main(
    @location(0) position: vec3f
  ) -> @builtin(position) vec4f {
    return scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
  }
);
// clang-format on
