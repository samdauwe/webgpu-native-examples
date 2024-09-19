#include "example_base.h"
#include "meshes.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

static struct {
  vec3 up_vector;
  vec3 origin;
  mat4 projection_matrix;
  mat4 view_proj_matrix;
} view_matrices = {0};

static stanford_dragon_mesh_t stanford_dragon_mesh = {0};
static const uint32_t shadow_depth_texture_size    = 1024;

// Vertex and index buffers
static WGPUBuffer vertex_buffer;
static WGPUBuffer index_buffer;
static uint32_t index_count;

// Uniform buffers
static struct {
  WGPUBuffer model;
  WGPUBuffer scene;
} uniform_buffers = {0};

// The pipeline layout
static struct {
  WGPUPipelineLayout shadow;
  WGPUPipelineLayout color;
} pipeline_layouts = {0};

// Pipelines
static struct {
  WGPURenderPipeline shadow;
  WGPURenderPipeline color;
} render_pipelines = {0};

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} shadow_render_pass = {0};
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} color_render_pass = {0};

// Bind groups
static struct {
  WGPUBindGroup scene_shadow;
  WGPUBindGroup scene_render;
  WGPUBindGroup model;
} bind_groups = {0};

// Bind group layouts
static struct {
  WGPUBindGroupLayout uniform_buffer_scene;
  WGPUBindGroupLayout uniform_buffer_model;
  WGPUBindGroupLayout render;
} bind_groups_layouts = {0};

// Texture and sampler
static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } shadow_depth_texture;
  WGPUSampler sampler;
} textures = {0};

// Other variables
static const char* example_title = "Shadow Mapping";
static bool prepared             = false;

// Prepare vertex and index buffers for the Stanford dragon mesh
static void
prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context,
                                 stanford_dragon_mesh_t* dragon_mesh)
{
  // Create the model vertex buffer
  {
    const uint8_t ground_plane_vertex_count = 4;
    uint64_t vertex_buffer_size
      = (dragon_mesh->positions.count + ground_plane_vertex_count) * 3 * 2
        * sizeof(float);
    WGPUBufferDescriptor buffer_desc = {
      .label            = "Model - Vertex buffer",
      .usage            = WGPUBufferUsage_Vertex,
      .size             = vertex_buffer_size,
      .mappedAtCreation = true,
    };
    vertex_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(vertex_buffer);
    float* mapping
      = (float*)wgpuBufferGetMappedRange(vertex_buffer, 0, vertex_buffer_size);
    ASSERT(mapping);
    for (uint64_t i = 0; i < dragon_mesh->positions.count; ++i) {
      memcpy(&mapping[6 * i], dragon_mesh->positions.data[i], sizeof(vec3));
      memcpy(&mapping[6 * i + 3], dragon_mesh->normals.data[i], sizeof(vec3));
    }
    // Push vertex attributes for an additional ground plane
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
    const uint64_t offset = dragon_mesh->positions.count * 6;
    for (uint64_t i = 0; i < ground_plane_vertex_count; ++i) {
      memcpy(&mapping[offset + 6 * i], ground_plane_positions[i], sizeof(vec3));
      memcpy(&mapping[offset + 6 * i + 3], ground_plane_normals[i],
             sizeof(vec3));
    }
    wgpuBufferUnmap(vertex_buffer);
  }

  // Create the model index buffer
  {
    const uint8_t ground_plane_index_count = 2;
    index_count = (dragon_mesh->triangles.count + ground_plane_index_count) * 3;
    uint64_t index_buffer_size       = index_count * sizeof(uint16_t);
    WGPUBufferDescriptor buffer_desc = {
      .label            = "Model - Index buffer",
      .usage            = WGPUBufferUsage_Index,
      .size             = index_buffer_size,
      .mappedAtCreation = true,
    };
    index_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(index_buffer);
    uint16_t* mapping
      = (uint16_t*)wgpuBufferGetMappedRange(index_buffer, 0, index_buffer_size);
    ASSERT(mapping)
    for (uint64_t i = 0; i < dragon_mesh->triangles.count; ++i) {
      memcpy(&mapping[3 * i], dragon_mesh->triangles.data[i],
             sizeof(uint16_t) * 3);
    }
    // Push indices for an additional ground plane
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
    wgpuBufferUnmap(index_buffer);
  }
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  // Create the depth texture for rendering/sampling the shadow map
  {
    WGPUExtent3D texture_extent = {
      .width              = shadow_depth_texture_size,
      .height             = shadow_depth_texture_size,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "Depth texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth32Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    textures.shadow_depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.shadow_depth_texture.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Depth32Float,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    textures.shadow_depth_texture.view = wgpuTextureCreateView(
      textures.shadow_depth_texture.texture, &texture_view_dec);
    ASSERT(textures.shadow_depth_texture.view != NULL);
  }

  // Create a depth/stencil texture for the color rendering pipeline
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
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    textures.depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.depth_texture.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Depth24PlusStencil8,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    textures.depth_texture.view = wgpuTextureCreateView(
      textures.depth_texture.texture, &texture_view_dec);
    ASSERT(textures.depth_texture.view != NULL);
  }
}

static void prepare_sampler(wgpu_context_t* wgpu_context)
{
  textures.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
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
  ASSERT(textures.sampler != NULL);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Bind group layout for unform buffers in shadow pipeline
  {
    // Bind group layout for scene uniform
    {
      WGPUBindGroupLayoutEntry bgl_entries[1] = {
        [0] = (WGPUBindGroupLayoutEntry) {
          // Binding 0: Uniform
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
      bind_groups_layouts.uniform_buffer_scene
        = wgpuDeviceCreateBindGroupLayout(
          wgpu_context->device,
          &(WGPUBindGroupLayoutDescriptor){
            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
            .entries    = bgl_entries,
          });
      ASSERT(bind_groups_layouts.uniform_buffer_scene != NULL);
    }

    // Bind group layout for model uniform
    {
      WGPUBindGroupLayoutEntry bgl_entries[1] = {
        [0] = (WGPUBindGroupLayoutEntry) {
          // Binding 0: Uniform
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
      bind_groups_layouts.uniform_buffer_model
        = wgpuDeviceCreateBindGroupLayout(
          wgpu_context->device,
          &(WGPUBindGroupLayoutDescriptor){
            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
            .entries    = bgl_entries,
          });
      ASSERT(bind_groups_layouts.uniform_buffer_model != NULL);
    }

    WGPUBindGroupLayout bind_group_layouts[2] = {
      bind_groups_layouts.uniform_buffer_scene,
      bind_groups_layouts.uniform_buffer_model,
    };
    pipeline_layouts.shadow = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(pipeline_layouts.shadow != NULL);
  }

  // Create a bind group layout which holds the scene uniforms and
  // the texture+sampler for depth. We create it manually because the WebPU
  // implementation doesn't infer this from the shader (yet).
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Uniform
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
        // Binding 1: Texture view
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
        // Binding 2: Sampler
        .binding    = 2,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Comparison,
        },
        .texture = {0},
      }
    };
    bind_groups_layouts.render = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_groups_layouts.render != NULL);

    // Specify the pipeline layout. The layout for the model is the same, so
    // reuse it from the shadow pipeline.
    WGPUBindGroupLayout bind_group_layouts[2] = {
      bind_groups_layouts.render,               // Group 0
      bind_groups_layouts.uniform_buffer_model, // Group 1
    };
    pipeline_layouts.color = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(pipeline_layouts.color != NULL);
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Shadow rendering
  {
    // Shadow pass descriptor
    shadow_render_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = textures.shadow_depth_texture.view,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
      };

    shadow_render_pass.descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 0,
      .colorAttachments       = shadow_render_pass.color_attachments,
      .depthStencilAttachment = &shadow_render_pass.depth_stencil_attachment,
      .occlusionQuerySet      = NULL,
    };
  }

  // Color rendering
  {
    // Color attachment
    color_render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // view is acquired and set in render loop.
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.5f,
        .g = 0.5f,
        .b = 0.5f,
        .a = 1.0f,
      },
    };

    // Render pass descriptor
    color_render_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = textures.depth_texture.view,
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };
    color_render_pass.descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = color_render_pass.color_attachments,
      .depthStencilAttachment = &color_render_pass.depth_stencil_attachment,
      .occlusionQuerySet      = NULL,
    };
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  vec3 eye_position = {0.0f, 50.0f, -100.0f};
  memcpy(view_matrices.up_vector, (vec3){0.0f, 1.0f, 0.0f}, sizeof(vec3));
  memcpy(view_matrices.origin, (vec3){0.0f, 0.0f, 0.0f}, sizeof(vec3));

  glm_mat4_identity(view_matrices.projection_matrix);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                  view_matrices.projection_matrix);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,            // eye vector
             view_matrices.origin,    // center vector
             view_matrices.up_vector, // up vector
             view_matrix              // result matrix
  );

  vec3 light_position    = {50.0f, 100.0f, -100.0f};
  mat4 light_view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(light_position,          // eye vector
             view_matrices.origin,    // center vector
             view_matrices.up_vector, // up vector
             light_view_matrix        // result matrix
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

  glm_mat4_identity(view_matrices.view_proj_matrix);
  glm_mat4_mulN((mat4*[]){&view_matrices.projection_matrix, &view_matrix}, 2,
                view_matrices.view_proj_matrix);

  // Move the model so it's centered.
  mat4 model_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_translate(model_matrix, (vec3){0.0f, -5.0f, 0.0f});
  glm_translate(model_matrix, (vec3){0.0f, -40.0f, 0.0f});

  // The camera/light aren't moving, so write them into buffers now.
  {
    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffers.scene, 0,
                         light_view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffers.scene, 64,
                         view_matrices.view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffers.scene, 128,
                         light_position, sizeof(vec3));

    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffers.model, 0,
                         model_matrix, sizeof(mat4));
  }
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

// Rotates the camera around the origin based on time.
static mat4* get_camera_view_proj_matrix(wgpu_example_context_t* context)
{
  vec3 eye_position = {0.0f, 50.0f, -100.0f};

  float rad = PI * (context->frame.timestamp_millis / 2000.0f);
  glm_vec3_rotate_y(eye_position, view_matrices.origin, rad, &eye_position);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,            // eye vector
             view_matrices.origin,    // center vector
             view_matrices.up_vector, // up vector
             view_matrix              // result matrix
  );

  glm_mat4_mulN((mat4*[]){&view_matrices.projection_matrix, &view_matrix}, 2,
                view_matrices.view_proj_matrix);
  return &view_matrices.view_proj_matrix;
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  mat4* camera_view_proj = get_camera_view_proj_matrix(context);
  wgpuQueueWriteBuffer(context->wgpu_context->queue, uniform_buffers.scene, 64,
                       *camera_view_proj, sizeof(mat4));
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Model uniform buffer
  {
    const WGPUBufferDescriptor buffer_desc = {
      .size  = sizeof(mat4), // 4x4 matrix
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    uniform_buffers.model
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(uniform_buffers.model)
  }

  // Scene uniform buffer
  {
    uniform_buffers.scene = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        // Two 4x4 viewProj matrices, one for the camera and one for the light.
        // Then a vec3 for the light position.
        .size  = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
    ASSERT(uniform_buffers.scene);
  }

  // Scene bind group for shadow
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniform_buffers.scene,
        .size    = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
      },
    };
    bind_groups.scene_shadow = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout     = bind_groups_layouts.uniform_buffer_scene,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(bind_groups.scene_shadow != NULL);
  }

  // Scene bind group for render
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniform_buffers.scene,
        .size    = sizeof(mat4) + sizeof(mat4) + sizeof(vec4),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = textures.shadow_depth_texture.view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = textures.sampler,
      },
    };
    bind_groups.scene_render = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_groups_layouts.render,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.scene_render != NULL);
  }

  // Model bind group
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniform_buffers.model,
        .size    = sizeof(mat4),
      },
    };
    bind_groups.model = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout     = bind_groups_layouts.uniform_buffer_model,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(bind_groups.model != NULL);
  }
}

// Create the shadow pipeline
static void prepare_shadow_pipeline(wgpu_context_t* wgpu_context)
{
  /// Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /// Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    shadow, sizeof(float) * 6,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "vertex_shadow_shader",
                  .wgsl_code.source = vertex_shadow_wgsl,
                  .entry            = "main",
                },
                .buffer_count = 1,
                .buffers      = &shadow_vertex_buffer_layout,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  render_pipelines.shadow = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "shadow_render_pipeline",
                            .layout       = pipeline_layouts.shadow,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = NULL,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(render_pipelines.shadow != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
}

// Create the color rendering pipeline
static void prepare_color_rendering_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color blend state
  WGPUBlendComponent blend_component = {
    .operation = WGPUBlendOperation_Add,
    .srcFactor = WGPUBlendFactor_One,
    .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
  };
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &(WGPUBlendState){
      .color = blend_component,
      .alpha = blend_component,
    },
    .writeMask = WGPUColorWriteMask_All,
  };

  // Constants
  WGPUConstantEntry constant_entries[1] = {
    [0] = (WGPUConstantEntry){
      .key   = "shadowDepthTextureSize",
      .value = shadow_depth_texture_size,
    },
  };

  // Depth stencil state
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

  // Vertex buffer layout
  // Create some common descriptors used for both the shadow pipeline and the
  // color rendering pipeline.
  WGPU_VERTEX_BUFFER_LAYOUT(
    color, sizeof(float) * 6,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "vertex_shader",
                  .wgsl_code.source = vertex_wgsl,
                  .entry            = "main",
                },
                .buffer_count = 1,
                .buffers      = &color_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "fragment_shader",
                  .wgsl_code.source = fragment_wgsl,
                  .entry            = "main",
                },
                .constant_count = (uint32_t)ARRAY_SIZE(constant_entries),
                .constants      = constant_entries,
                .target_count   = 1,
                .targets        = &color_target_state,
              });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  render_pipelines.color = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "color_render_pipeline",
                            .layout       = pipeline_layouts.color,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(render_pipelines.color != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    stanford_dragon_mesh_init(&stanford_dragon_mesh);
    prepare_vertex_and_index_buffers(context->wgpu_context,
                                     &stanford_dragon_mesh);
    prepare_texture(context->wgpu_context);
    prepare_sampler(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_shadow_pipeline(context->wgpu_context);
    prepare_color_rendering_pipeline(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    prepare_view_matrices(context->wgpu_context);
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
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Shadow pass
  {
    WGPURenderPassEncoder shadow_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &shadow_render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(shadow_pass, render_pipelines.shadow);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 0, bind_groups.scene_shadow,
                                      0, 0);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 1, bind_groups.model, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(shadow_pass, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      shadow_pass, index_buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(shadow_pass, index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEnd(shadow_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, shadow_pass)
  }

  // Color render pass
  {
    color_render_pass.color_attachments[0].view
      = wgpu_context->swap_chain.frame_buffer;
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &color_render_pass.descriptor);
    wgpuRenderPassEncoderSetPipeline(render_pass, render_pipelines.color);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_groups.scene_render,
                                      0, 0);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, bind_groups.model, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      render_pass, index_buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(render_pass, index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEnd(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
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
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.model)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.scene)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.shadow)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layouts.color)
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.shadow);
  WGPU_RELEASE_RESOURCE(RenderPipeline, render_pipelines.color)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.scene_shadow)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.scene_render)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        bind_groups_layouts.uniform_buffer_scene)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        bind_groups_layouts.uniform_buffer_model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_groups_layouts.render)
  WGPU_RELEASE_RESOURCE(Sampler, textures.sampler)
  WGPU_RELEASE_RESOURCE(Texture, textures.depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.depth_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, textures.shadow_depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.shadow_depth_texture.view)
}

void example_shadow_mapping(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
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
static const char* fragment_wgsl = CODE(
  override shadowDepthTextureSize: f32 = 1024.0;

  struct Scene {
    lightViewProjMatrix : mat4x4<f32>,
    cameraViewProjMatrix : mat4x4<f32>,
    lightPos : vec3<f32>,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(0) @binding(1) var shadowMap: texture_depth_2d;
  @group(0) @binding(2) var shadowSampler: sampler_comparison;

  struct FragmentInput {
    @location(0) shadowPos : vec3<f32>,
    @location(1) fragPos : vec3<f32>,
    @location(2) fragNorm : vec3<f32>,
  }

  const albedo = vec3<f32>(0.9);
  const ambientFactor = 0.2;

  @fragment
  fn main(input : FragmentInput) -> @location(0) vec4<f32> {
    // Percentage-closer filtering. Sample texels in the region
    // to smooth the result.
    var visibility = 0.0;
    let oneOverShadowDepthTextureSize = 1.0 / shadowDepthTextureSize;
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let offset = vec2<f32>(vec2(x, y)) * oneOverShadowDepthTextureSize;

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
    lightViewProjMatrix: mat4x4<f32>,
    cameraViewProjMatrix: mat4x4<f32>,
    lightPos: vec3<f32>,
  }

  struct Model {
    modelMatrix: mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(1) @binding(0) var<uniform> model : Model;

  struct VertexOutput {
    @location(0) shadowPos: vec3<f32>,
    @location(1) fragPos: vec3<f32>,
    @location(2) fragNorm: vec3<f32>,

    @builtin(position) Position: vec4<f32>,
  }

  @vertex
  fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>
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
    lightViewProjMatrix: mat4x4<f32>,
    cameraViewProjMatrix: mat4x4<f32>,
    lightPos: vec3<f32>,
  }

  struct Model {
    modelMatrix: mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> scene : Scene;
  @group(1) @binding(0) var<uniform> model : Model;

  @vertex
  fn main(
    @location(0) position: vec3<f32>
  ) -> @builtin(position) vec4<f32> {
    return scene.lightViewProjMatrix * model.modelMatrix * vec4(position, 1.0);
  }
);
// clang-format on
