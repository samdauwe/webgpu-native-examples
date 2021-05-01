#include "example_base.h"
#include "examples.h"
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

static struct view_matrices_t {
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
static WGPUBuffer model_uniform_buffer;
static WGPUBuffer scene_uniform_buffer;

// The pipeline layout
static WGPUPipelineLayout pipeline_layout;

// Pipelines
static WGPURenderPipeline shadow_pipeline;
static WGPURenderPipeline color_render_pipeline;

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachmentDescriptor sp_color_att_descriptors = {0};
static WGPURenderPassColorAttachmentDescriptor rp_color_att_descriptors[1];
static WGPURenderPassDepthStencilAttachmentDescriptor sp_ds_att_descriptor;
static WGPURenderPassDepthStencilAttachmentDescriptor rp_ds_att_descriptor;
static WGPURenderPassDescriptor render_pass_descriptor;
static WGPURenderPassDescriptor shadow_pass_descriptor;

// Bind groups
static WGPUBindGroup scene_bind_group_for_shadow;
static WGPUBindGroup scene_bind_group_for_render;
static WGPUBindGroup model_bind_group;

// Bind group layout
static WGPUBindGroupLayout bind_group_layout_for_render;

// Texture and sampler
static WGPUSampler sampler;
static WGPUTexture depth_texture;
static WGPUTextureView depth_texture_view;
static WGPUTexture shadow_depth_texture;
static WGPUTextureView shadow_depth_texture_view;

// Other variables
static const char* example_title = "Shadow Mapping";
static bool prepared             = false;

// Prepare vertex and index buffers for the Stanford dragon mesh
static void
prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context,
                                 stanford_dragon_mesh_t* dragon_mesh)
{
  /** Mesh indices **/
  static uint16_t mesh_indices[(CELL_COUNT_RES_4 + 2) * 3];
  index_count = (CELL_COUNT_RES_4 + 2) * 3;
  memcpy(mesh_indices, dragon_mesh->triangles.data,
         sizeof(dragon_mesh->triangles.data));
  // Push indices for an additional ground plane
  static const uint16_t ground_plane_indices[3 * 2] = {
    POSITION_COUNT_RES_4, POSITION_COUNT_RES_4 + 2, POSITION_COUNT_RES_4 + 1, //
    POSITION_COUNT_RES_4, POSITION_COUNT_RES_4 + 1, POSITION_COUNT_RES_4 + 3, //
  };
  memcpy(mesh_indices + CELL_COUNT_RES_4 * 3, ground_plane_indices,
         2 * 3 * sizeof(uint16_t));

  /** Mesh positions **/
  static float mesh_positions[(POSITION_COUNT_RES_4 + 4) * 3];
  memcpy(mesh_positions, dragon_mesh->vertices.data,
         sizeof(dragon_mesh->vertices.data));
  // Push positions for an additional ground plane
  static const float ground_plane_positions[4 * 3] = {
    -100.0f, 20.0f, -100.0f, //
    100.0f,  20.0f, 100.0f,  //
    -100.0f, 20.0f, 100.0f,  //
    100.0f,  20.0f, -100.0f  //
  };
  memcpy(mesh_positions + POSITION_COUNT_RES_4 * 3, ground_plane_positions,
         4 * 3 * sizeof(float));

  /** Surface normals **/
  static vec3 mesh_normals[POSITION_COUNT_RES_4 + 4] = {0};
  const uint16_t mesh_normals_length                 = POSITION_COUNT_RES_4 + 4;
  const uint16_t mesh_indices_length                 = CELL_COUNT_RES_4 + 2;
  mesh_compute_surface_normals(mesh_positions, mesh_indices,
                               mesh_indices_length, mesh_normals,
                               mesh_normals_length);

  // Create the model vertex buffer.
  float vertex_buffer_data[(POSITION_COUNT_RES_4 + 4) * 3 * 2] = {0};
  const uint16_t nb_mesh_positions = POSITION_COUNT_RES_4 + 4;
  for (uint16_t i = 0; i < nb_mesh_positions; ++i) {
    memcpy(vertex_buffer_data + 6 * i, &mesh_positions[i * 3],
           3 * sizeof(float));
    memcpy(vertex_buffer_data + 6 * i + 3, &mesh_normals[i], 3 * sizeof(float));
  }

  vertex_buffer = wgpu_create_buffer_from_data(wgpu_context, vertex_buffer_data,
                                               sizeof(vertex_buffer_data),
                                               WGPUBufferUsage_Vertex);

  // Create the model index buffer.
  index_buffer = wgpu_create_buffer_from_data(
    wgpu_context, mesh_indices, sizeof(mesh_indices), WGPUBufferUsage_Index);
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  // Create the depth texture for rendering/sampling the shadow map.
  {
    WGPUExtent3D texture_extent = {
      .width  = shadow_depth_texture_size,
      .height = shadow_depth_texture_size,
      .depth  = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth32Float,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_Sampled,
    };
    shadow_depth_texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = WGPUTextureFormat_Depth32Float,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    shadow_depth_texture_view
      = wgpuTextureCreateView(shadow_depth_texture, &texture_view_dec);
  }

  // Create a depth/stencil texture for the color rendering pipeline
  {
    WGPUExtent3D texture_extent = {
      .width  = wgpu_context->surface.width,
      .height = wgpu_context->surface.height,
      .depth  = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    depth_texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

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
    depth_texture_view
      = wgpuTextureCreateView(depth_texture, &texture_view_dec);
  }
}

static void prepare_sampler(wgpu_context_t* wgpu_context)
{
  sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Nearest,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUFilterMode_Nearest,
                            .compare       = WGPUCompareFunction_Less,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Create a bind group layout which holds the scene uniforms and
  // the texture+sampler for depth. We create it manually because the WebPU
  // implementation doesn't infer this from the shader (yet).
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Uniform
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize = 2 * 4 * 16 + 3 * 4,
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Binding 1: Texture view
      .binding = 1,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType = WGPUTextureSampleType_Depth,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Binding 2: Sampler
      .binding = 2,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type=WGPUSamplerBindingType_Comparison,
      },
      .texture = {0},
    }
  };
  bind_group_layout_for_render = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(bind_group_layout_for_render != NULL)

  // Specify the pipeline layout. The layout for the model is the same, so
  // reuse it from the shadow pipeline.
  WGPUBindGroupLayout bind_group_layouts[2] = {
    bind_group_layout_for_render,
    wgpuRenderPipelineGetBindGroupLayout(shadow_pipeline, 1),
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(pipeline_layout != NULL)
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Shadow rendering
  {
    // Shadow pass descriptor
    sp_ds_att_descriptor = (WGPURenderPassDepthStencilAttachmentDescriptor){
      .attachment     = shadow_depth_texture_view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .clearDepth     = 1.0f,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
      .clearStencil   = 0,
    };

    shadow_pass_descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 0,
      .colorAttachments       = &sp_color_att_descriptors,
      .depthStencilAttachment = &sp_ds_att_descriptor,
      .occlusionQuerySet      = NULL,
    };
  }

  // Color rendering
  {
    // Color attachment
    rp_color_att_descriptors[0] = (WGPURenderPassColorAttachmentDescriptor) {
      .attachment = NULL, // attachment is acquired and set in render loop.
      .loadOp = WGPULoadOp_Clear,
      .storeOp = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.5f,
        .g = 0.5f,
        .b = 0.5f,
        .a = 1.0f,
      },
    };

    // Render pass descriptor
    rp_ds_att_descriptor = (WGPURenderPassDepthStencilAttachmentDescriptor){
      .attachment     = depth_texture_view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .clearDepth     = 1.0f,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
      .clearStencil   = 0,
    };
    render_pass_descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = rp_color_att_descriptors,
      .depthStencilAttachment = &rp_ds_att_descriptor,
      .occlusionQuerySet      = NULL,
    };
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  vec3 eye_position = {0.0f, 50.0f, -100.0f};
  memcpy(view_matrices.up_vector, (vec3){0.0f, 1.0f, 0.0f}, sizeof(vec3));
  memcpy(view_matrices.origin, (vec3){0.0f, 0.0f, 0.0f}, sizeof(vec3));

  glm_mat4_identity(view_matrices.projection_matrix);
  glm_perspective((2.0f * PI) / 5.0f, aspect_ratio, 1.f, 2000.f,
                  view_matrices.projection_matrix);

  mat4 view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(eye_position,            //
             view_matrices.origin,    //
             view_matrices.up_vector, //
             view_matrix);

  vec3 light_position    = {50.0f, 100.0f, -100.0f};
  mat4 light_view_matrix = GLM_MAT4_IDENTITY_INIT;
  glm_lookat(light_position,          //
             view_matrices.origin,    //
             view_matrices.up_vector, //
             light_view_matrix);

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
    wgpuQueueWriteBuffer(wgpu_context->queue, scene_uniform_buffer, 0,
                         light_view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, scene_uniform_buffer, 64,
                         view_matrices.view_proj_matrix, sizeof(mat4));

    wgpuQueueWriteBuffer(wgpu_context->queue, scene_uniform_buffer, 128,
                         light_position, sizeof(vec3));

    wgpuQueueWriteBuffer(wgpu_context->queue, model_uniform_buffer, 0,
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

  float rad = PI * (context->frame.timestamp_millis / 2000.0f);
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
  wgpuQueueWriteBuffer(context->wgpu_context->queue, scene_uniform_buffer, 64,
                       *camera_view_proj, sizeof(mat4));
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Model uniform buffer
  {
    const WGPUBufferDescriptor buffer_desc = {
      .size  = 4 * 16, // 4x4 matrix
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    };
    model_uniform_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(model_uniform_buffer)
  }

  // Scene uniform buffer
  {
    scene_uniform_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        // Two 4x4 viewProj matrices,
        // one for the camera and one for the light.
        // Then a vec3 for the light position.
        .size  = 2 * 4 * 16 + 3 * 4,
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
    ASSERT(scene_uniform_buffer)
  }

  // Scene bind group for shadow
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = scene_uniform_buffer,
        .size = 2 * 4 * 16 + 3 * 4,
      },
    };
    scene_bind_group_for_shadow = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout     = wgpuRenderPipelineGetBindGroupLayout(shadow_pipeline, 0),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(scene_bind_group_for_shadow != NULL)
  }

  // Scene bind group for render
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = scene_uniform_buffer,
        .size = 2 * 4 * 16 + 3 * 4,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = shadow_depth_texture_view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = sampler,
      },
    };
    scene_bind_group_for_render = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = bind_group_layout_for_render,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(scene_bind_group_for_render != NULL)
  }

  // Model bind group
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = model_uniform_buffer,
        .size = 4 * 16,
      },
    };
    model_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .layout     = wgpuRenderPipelineGetBindGroupLayout(shadow_pipeline, 1),
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(model_bind_group != NULL)
  }
}

// Create the shadow pipeline
static void prepare_shadow_pipeline(wgpu_context_t* wgpu_context)
{
  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state
    = wgpu_create_rasterization_state_descriptor(
      &(create_rasterization_state_desc_t){
        .front_face = WGPUFrontFace_CCW,
        .cull_mode  = WGPUCullMode_Back,
      });

  // Color state
  WGPUColorStateDescriptor color_desc = {0};

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc
    = wgpu_create_depth_stencil_state_descriptor(
      &(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });

  // Create the vertex state for the shadow pipeline
  WGPU_VERTSTATE(
    shadow, sizeof(float) * 6,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/shadow_mapping/shadow.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/shadow_mapping/shadow.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  shadow_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 0,
      .colorStates            = &color_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_shadow,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

// Create the color rendering pipeline
static void prepare_color_rendering_pipeline(wgpu_context_t* wgpu_context)
{
  // Rasterization state
  WGPURasterizationStateDescriptor rasterization_state = {
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color blend state
  WGPUBlendDescriptor blend_desc = {
    .operation = WGPUBlendOperation_Add,
    .srcFactor = WGPUBlendFactor_One,
    .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
  };
  WGPUColorStateDescriptor color_desc = {
    .format     = wgpu_context->swap_chain.format,
    .alphaBlend = blend_desc,
    .colorBlend = blend_desc,
    .writeMask  = WGPUColorWriteMask_All,
  };

  // Depth and stencil state containing depth and stencil compare and test
  // operations
  WGPUDepthStencilStateDescriptor depth_stencil_state_desc = {
    .depthWriteEnabled = true,
    .format = WGPUTextureFormat_Depth24PlusStencil8,
    .depthCompare = WGPUCompareFunction_Less,
    .stencilFront = {
      .compare = WGPUCompareFunction_Always,
      .failOp = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp = WGPUStencilOperation_Keep,
    },
    .stencilBack = {
      .compare = WGPUCompareFunction_Always,
      .failOp = WGPUStencilOperation_Keep,
      .depthFailOp = WGPUStencilOperation_Keep,
      .passOp = WGPUStencilOperation_Keep,
    },
    .stencilReadMask = 0xFFFFFFFF,
    .stencilWriteMask = 0xFFFFFFFF,
  };

  // Create the vertex state for the color rendering pipeline.
  WGPU_VERTSTATE(
    color, sizeof(float) * 6,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    // Attribute location 1: Normal
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3, sizeof(float) * 3))

  // Shaders
  // Vertex shader
  wgpu_shader_t vert_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Vertex shader SPIR-V
                    .file = "shaders/shadow_mapping/shader.vert.spv",
                  });
  // Fragment shader
  wgpu_shader_t frag_shader = wgpu_shader_create(
    wgpu_context, &(wgpu_shader_desc_t){
                    // Fragment shader SPIR-V
                    .file = "shaders/shadow_mapping/shader.frag.spv",
                  });

  // Create rendering pipeline using the specified states
  color_render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .layout = pipeline_layout,
      // Vertex shader
      .vertexStage = vert_shader.programmable_stage_descriptor,
      // Fragment shader
      .fragmentStage = &frag_shader.programmable_stage_descriptor,
      // Rasterization state
      .rasterizationState     = &rasterization_state,
      .primitiveTopology      = WGPUPrimitiveTopology_TriangleList,
      .colorStateCount        = 1,
      .colorStates            = &color_desc,
      .depthStencilState      = &depth_stencil_state_desc,
      .vertexState            = &vert_state_color,
      .sampleCount            = 1,
      .sampleMask             = 0xFFFFFFFF,
      .alphaToCoverageEnabled = false,
    });

  // Shader modules are no longer needed once the graphics pipeline has been
  // created
  wgpu_shader_release(&frag_shader);
  wgpu_shader_release(&vert_shader);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    stanford_dragon_mesh_init(&stanford_dragon_mesh);
    prepare_vertex_and_index_buffers(context->wgpu_context,
                                     &stanford_dragon_mesh);
    prepare_texture(context->wgpu_context);
    prepare_sampler(context->wgpu_context);
    prepare_shadow_pipeline(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
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
  rp_color_att_descriptors[0].attachment
    = wgpu_context->swap_chain.frame_buffer;
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Shadow pass */
  {
    WGPURenderPassEncoder shadow_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &shadow_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(shadow_pass, shadow_pipeline);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 0,
                                      scene_bind_group_for_shadow, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(shadow_pass, 1, model_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(shadow_pass, 0, vertex_buffer, 0, 0);
    wgpuRenderPassEncoderSetIndexBuffer(shadow_pass, index_buffer,
                                        WGPUIndexFormat_Uint16, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(shadow_pass, index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEndPass(shadow_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, shadow_pass)
  }

  /* Render pass */
  {
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(render_pass, color_render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                      scene_bind_group_for_render, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, model_bind_group, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, vertex_buffer, 0, 0);
    wgpuRenderPassEncoderSetIndexBuffer(render_pass, index_buffer,
                                        WGPUIndexFormat_Uint16, 0, 0);
    wgpuRenderPassEncoderDrawIndexed(render_pass, index_count, 1, 0, 0, 0);

    wgpuRenderPassEncoderEndPass(render_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
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
  WGPU_RELEASE_RESOURCE(Buffer, model_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, scene_uniform_buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, shadow_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, color_render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, scene_bind_group_for_shadow)
  WGPU_RELEASE_RESOURCE(BindGroup, scene_bind_group_for_render)
  WGPU_RELEASE_RESOURCE(BindGroup, model_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout_for_render)
  WGPU_RELEASE_RESOURCE(Sampler, sampler)
  WGPU_RELEASE_RESOURCE(Texture, depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, depth_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, shadow_depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, shadow_depth_texture_view)
}

void example_shadow_mapping(int argc, char* argv[])
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
