#include "example_base.h"

#include "meshes.h"

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Normal Mapping
 *
 * This example demonstrates multiple different methods that employ fragment
 * shaders to achieve additional perceptual depth on the surface of a cube mesh.
 * Demonstrated methods include normal mapping, parallax mapping, and steep
 * parallax mapping.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/normalMap
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* normal_map_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Math functions
 * @ref https://github.com/toji/gl-matrix
 * -------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------- *
 * Normal Map example
 * -------------------------------------------------------------------------- */

/* The mesh to be rendered */
typedef struct renderable_t {
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;
} renderable_t;

static struct {
  renderable_t renderable;
  box_mesh_t mesh;
} box = {0};

/* The texture type */
typedef enum texture_atlas_t {
  TEXTURE_ATLAS_SPIRAL    = 0,
  TEXTURE_ATLAS_TOYBOX    = 1,
  TEXTURE_ATLAS_BRICKWALL = 2,
  TEXTURE_ATLAS_COUNT     = 3,
} texture_atlas_t;

/* The bump mode */
typedef enum bump_mode_t {
  BUMP_MODE_ALBEDO_TEXTURE = 0,
  BUMP_MODE_NORMAL_TEXTURE = 1,
  BUMP_MODE_DEPTH_TEXTURE  = 2,
  BUMP_MODE_NORMAL_MAP     = 3,
  BUMP_MODE_PARALLAX_SCALE = 4,
  BUMP_MODE_STEEP_PARALLAX = 5,
  BUMP_MODE_COUNT          = 6,
} bump_mode_t;

/* The textures */
static struct {
  texture_t wood_albedo;
  texture_t spiral_normal;
  texture_t spiral_height;
  texture_t toybox_normal;
  texture_t toybox_height;
  texture_t brickwall_albedo;
  texture_t brickwall_normal;
  texture_t brickwall_height;
  texture_t depth;
  WGPUSampler sampler;
} textures = {0};

static struct {
  const char* file;
  texture_t* texture;
} texture_mappings[8] = {
  // clang-format off
  { .file = "textures/wood_albedo.png",      .texture = &textures.wood_albedo      },
  { .file = "textures/spiral_normal.png",    .texture = &textures.spiral_normal    },
  { .file = "textures/spiral_height.png",    .texture = &textures.spiral_height    },
  { .file = "textures/toybox_normal.png",    .texture = &textures.toybox_normal    },
  { .file = "textures/toybox_height.png",    .texture = &textures.toybox_height    },
  { .file = "textures/brickwall_albedo.png", .texture = &textures.brickwall_albedo },
  { .file = "textures/brickwall_normal.png", .texture = &textures.brickwall_normal },
  { .file = "textures/brickwall_height.png", .texture = &textures.brickwall_height },
  // clang-format on
};

/* Uniforms data */
static struct {
  mat4 projection;
  mat4 view;
  mat4 model;
} view_matrices = {0};

static struct {
  mat4 world_view_proj_matrix;
  mat4 world_view_matrix;
} space_transforms = {0};

static struct {
  vec3 light_pos_vs; /* Light position in view space */
  uint32_t mode;
  float light_intensity;
  float depth_scale;
  float depth_layers;
  float padding;
} map_info = {0};

/* Uniforms buffer */
static struct {
  wgpu_buffer_t space_transforms;
  wgpu_buffer_t map_info;
} uniforms_bufers = {0};

/* The bind groups and layouts */
static struct {
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
} frame_bg_descriptor = {0};

static struct {
  WGPUBindGroup bind_groups[TEXTURE_ATLAS_COUNT];
  WGPUBindGroupLayout bind_group_layout;
} surface_bg_descriptor = {0};

/* The render pipeline + pipeline layout */
static WGPURenderPipeline textured_cube_pipeline        = NULL;
static WGPUPipelineLayout textured_cube_pipeline_layout = NULL;

/* Render pass descriptor for frame buffer writes */
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

/* GUI control */
static int32_t current_surface_bind_group = 0;

static struct {
  bump_mode_t bump_mode;
  float camera_pos_x;
  float camera_pos_y;
  float camera_pos_z;
  float light_pos_x;
  float light_pos_y;
  float light_pos_z;
  float light_intensity;
  float depth_scale;
  int32_t depth_layers;
  texture_atlas_t texture;
} settings = {
  .bump_mode       = BUMP_MODE_NORMAL_MAP,
  .camera_pos_x    = 0.0f,
  .camera_pos_y    = 0.8f,
  .camera_pos_z    = -1.4f,
  .light_pos_x     = 1.7f,
  .light_pos_y     = 0.7f,
  .light_pos_z     = -1.9f,
  .light_intensity = 5.0f,
  .depth_scale     = 0.05f,
  .depth_layers    = 16,
  .texture         = TEXTURE_ATLAS_SPIRAL,
};

static const char* texture_atlas_str[TEXTURE_ATLAS_COUNT] = {
  "Spiral",    /* */
  "Toybox",    /* */
  "BrickWall", /* */
};

static const char* bump_modes_str[BUMP_MODE_COUNT] = {
  "Albedo Texture", /* */
  "Normal Texture", /* */
  "Depth Texture",  /* */
  "Normal Map",     /* */
  "Parallax Scale", /* */
  "Steep Parallax", /* */
};

/* Other variables */
static const char* example_title = "Normal Mapping";
static bool prepared             = false;

static void create_box_mesh_renderable(wgpu_context_t* wgpu_context)
{
  box_mesh_create_with_tangents(&box.mesh, 1.0f, 1.0f, 1.0f);

  // Create vertex buffers
  box.renderable.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box mesh - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = box.mesh.vertex_count * sizeof(float),
                    .initial.data = box.mesh.vertex_array,
                  });

  // Create index buffer
  box.renderable.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box mesh - Indices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = box.mesh.index_count * sizeof(uint32_t),
                    .initial.data = box.mesh.index_array,
                  });
  box.renderable.index_count = box.mesh.index_count;
}

static mat4* get_projection_matrix(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 0.1f, 10.0f,
                  view_matrices.projection);

  return &view_matrices.projection;
}

static mat4* get_view_matrix(void)
{
  glm_lookat((vec3){settings.camera_pos_x, settings.camera_pos_y,
                    settings.camera_pos_z}, /* eye vector    */
             (vec3){0.0f, 0.0f, 0.0f},      /* center vector */
             (vec3){0.0f, 1.0f, 0.0f},      /* up vector     */
             view_matrices.view             /* result matrix */
  );

  return &view_matrices.view;
}

static mat4* get_model_matrix(wgpu_example_context_t* context)
{
  glm_mat4_identity(view_matrices.model);
  const float now = context->frame.timestamp_millis / 1000.0f;
  glm_rotate_y(view_matrices.model, now * -0.5f, view_matrices.model);

  return &view_matrices.model;
}

static uint32_t get_bump_mode(void)
{
  return (uint32_t)settings.bump_mode;
}

static void update_space_transforms_buffer(wgpu_example_context_t* context)
{
  /* Update matrices */
  glm_mat4_mul(*get_view_matrix(), *get_model_matrix(context),
               space_transforms.world_view_matrix);
  glm_mat4_mul(*get_projection_matrix(context->wgpu_context),
               space_transforms.world_view_matrix,
               space_transforms.world_view_proj_matrix);

  /* Update GPU buffer*/
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniforms_bufers.space_transforms.buffer, 0,
                          &space_transforms, sizeof(space_transforms));
}

static void update_map_info_buffer(wgpu_context_t* wgpu_context)
{
  /* Update map info data */
  vec3 light_pos_ws
    = {settings.light_pos_x, settings.light_pos_y, settings.light_pos_z};
  glm_vec3_transform_mat4(light_pos_ws, *get_view_matrix(),
                          &map_info.light_pos_vs);
  map_info.mode            = get_bump_mode();
  map_info.light_intensity = settings.light_intensity;
  map_info.depth_scale     = settings.depth_scale;
  map_info.depth_layers    = settings.depth_layers;

  /* Update GPU buffer*/
  wgpu_queue_write_buffer(wgpu_context, uniforms_bufers.map_info.buffer, 0,
                          &map_info, sizeof(map_info));
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  update_space_transforms_buffer(context);
  update_map_info_buffer(context->wgpu_context);
}

static void prepare_uniforms_buffers(wgpu_context_t* wgpu_context)
{
  /* Space transforms buffer */
  uniforms_bufers.space_transforms = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      // Buffer holding projection, view, and model matrices plus padding bytes
      .label = "Space transforms - Uniform buffer",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(space_transforms),
    });

  /* Space transforms buffer */
  uniforms_bufers.map_info = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      // Buffer holding mapping type, light uniforms, and depth uniforms
      .label = "Space transforms - Uniform buffer",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(map_info),
    });
}

static void prepare_textures(wgpu_context_t* wgpu_context)
{
  /* Create the depth texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = wgpu_context->surface.width,
      .height             = wgpu_context->surface.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "Depth - Texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    textures.depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.depth.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Depth - Texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    textures.depth.view
      = wgpuTextureCreateView(textures.depth.texture, &texture_view_dec);
    ASSERT(textures.depth.view != NULL);
  }

  /* Fetch the images and upload them into a GPUTextures. */
  {
    for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(texture_mappings); ++i) {
      *(texture_mappings[i].texture) = wgpu_create_texture_from_file(
        wgpu_context, texture_mappings[i].file,
        &(struct wgpu_texture_load_options_t){
          .label = "Texture",
          .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                   | WGPUTextureUsage_RenderAttachment,
        });
    }
  }
}

/* Create a sampler with linear filtering for smooth interpolation. */
static void create_sampler(wgpu_context_t* wgpu_context)
{
  textures.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Texture - Sampler",
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(textures.sampler != NULL);
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Uniform bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = uniforms_bufers.space_transforms.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = uniforms_bufers.map_info.size,
        },
        .sampler = {0},
      },
    };
    frame_bg_descriptor.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Frame - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(frame_bg_descriptor.bind_group_layout != NULL);
  }

  /* Texture bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Sampler */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    for (uint8_t i = 1; i < (uint32_t)ARRAY_SIZE(bgl_entries); ++i) {
      bgl_entries[i] = (WGPUBindGroupLayoutEntry) {
        /* Texture view */
        .binding    = i,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      };
    }
    surface_bg_descriptor.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Texture - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(surface_bg_descriptor.bind_group_layout != NULL);
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Uniform bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = uniforms_bufers.space_transforms.buffer,
        .size    = uniforms_bufers.space_transforms.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = uniforms_bufers.map_info.buffer,
        .size    = uniforms_bufers.map_info.size,
      },
    };
    frame_bg_descriptor.bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Frame - Bind group",
                              .layout = frame_bg_descriptor.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(frame_bg_descriptor.bind_group != NULL);
  }

  /* Multiple bindgroups that accord to the layout defined above */
  {
    WGPUTextureView texture_views[TEXTURE_ATLAS_COUNT][3] = {
      // clang-format off
      {textures.wood_albedo.view,      textures.spiral_normal.view,    textures.spiral_height.view},
      {textures.wood_albedo.view,      textures.toybox_normal.view,    textures.toybox_height.view},
      {textures.brickwall_albedo.view, textures.brickwall_normal.view, textures.brickwall_height.view},
      // clang-format on
    };
    for (uint8_t i = 0; i < TEXTURE_ATLAS_COUNT; ++i) {
      WGPUBindGroupEntry bg_entries[4] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .sampler = textures.sampler,
        },
      };
      for (uint8_t j = 1; j <= 3; ++j) {
        bg_entries[j] = (WGPUBindGroupEntry){
          .binding     = j,
          .textureView = texture_views[i][j - 1],
        };
      }
      WGPUBindGroupDescriptor bg_desc = {
        .label      = "Surface - Bind group",
        .layout     = surface_bg_descriptor.bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      };
      surface_bg_descriptor.bind_groups[i]
        = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
      ASSERT(surface_bg_descriptor.bind_groups[i] != NULL);
    }
  }
}

static void setup_render_pass(void)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment){
    .view       = NULL, /* Assigned later. */
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
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view              = textures.depth.view,
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  };

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
  };
}

static void create_3d_render_pipeline(
  wgpu_context_t* wgpu_context, const char* label,
  WGPUBindGroupLayout const* bg_layouts, uint32_t bg_layout_count,
  const char* vertex_shader,
  WGPUVertexBufferLayout const* vertex_buffer_layouts,
  uint32_t vertex_buffer_count, const char* fragment_shader,
  WGPUTextureFormat presentation_format, bool depth_test,
  WGPUPrimitiveTopology topology, WGPUCullMode cull_mode,
  WGPURenderPipeline* render_pipeline,
  WGPUPipelineLayout* render_pipeline_layout)
{
  // Create the pipeline layout
  *render_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = "Render - Pipeline layout",
                            .bindGroupLayoutCount = bg_layout_count,
                            .bindGroupLayouts     = bg_layouts,
                          });
  ASSERT(*render_pipeline_layout != NULL);

  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = topology,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = cull_mode,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = presentation_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "Vertex shader WGSL",
                      .wgsl_code.source = vertex_shader,
                      .entry            = "vertexMain",
                    },
                    .buffer_count = vertex_buffer_count,
                    .buffers = vertex_buffer_layouts,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "Fragment shader WGSL",
                      .wgsl_code.source = fragment_shader,
                      .entry            = "fragmentMain",
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

  // Create rendering pipeline descriptor using the specified states
  WGPURenderPipelineDescriptor render_pipeline_descriptor = {
    .label       = label,
    .layout      = *render_pipeline_layout,
    .primitive   = primitive_state,
    .vertex      = vertex_state,
    .fragment    = &fragment_state,
    .multisample = multisample_state,
  };
  if (depth_test) {
    render_pipeline_descriptor.depthStencil = &depth_stencil_state;
  }

  // Create rendering pipeline using the pipeline descriptor
  *render_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &render_pipeline_descriptor);
  ASSERT(*render_pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bg_layouts[2] = {
    frame_bg_descriptor.bind_group_layout,
    surface_bg_descriptor.bind_group_layout,
  };

  typedef struct v_buffer_layout_t {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec3 tangent;
    vec3 bitangent;
  } v_buffer_layout_t;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    box, sizeof(v_buffer_layout_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                       offsetof(v_buffer_layout_t, position)),
    /* Attribute location 1: Normal */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                       offsetof(v_buffer_layout_t, normal)),
    /* Attribute location 2: UV */
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x2,
                       offsetof(v_buffer_layout_t, uv)),
    /* Attribute location 3: Tangent */
    WGPU_VERTATTR_DESC(3, WGPUVertexFormat_Float32x3,
                       offsetof(v_buffer_layout_t, tangent)),
    /* Attribute location 4: Bitangent */
    WGPU_VERTATTR_DESC(4, WGPUVertexFormat_Float32x3,
                       offsetof(v_buffer_layout_t, bitangent)))

  create_3d_render_pipeline(
    wgpu_context, "Normal mapping - Render pipeline", bg_layouts,
    (uint32_t)ARRAY_SIZE(bg_layouts), normal_map_shader_wgsl,
    &box_vertex_buffer_layout, 1, normal_map_shader_wgsl,
    wgpu_context->swap_chain.format, true, WGPUPrimitiveTopology_TriangleList,
    WGPUCullMode_Back, &textured_cube_pipeline, &textured_cube_pipeline_layout);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    create_box_mesh_renderable(context->wgpu_context);
    prepare_uniforms_buffers(context->wgpu_context);
    prepare_textures(context->wgpu_context);
    create_sampler(context->wgpu_context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void reset_light(void)
{
  settings.light_pos_x     = 1.7f;
  settings.light_pos_y     = 0.7f;
  settings.light_pos_z     = -1.9f;
  settings.light_intensity = 5.0f;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    int32_t current_bump_mode = (int32_t)settings.bump_mode;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Bump Mode",
                                &current_bump_mode, bump_modes_str,
                                (uint32_t)ARRAY_SIZE(bump_modes_str))) {
      settings.bump_mode = (bump_mode_t)current_bump_mode;
    }
    imgui_overlay_combo_box(context->imgui_overlay, "Texture",
                            &current_surface_bind_group, texture_atlas_str,
                            (uint32_t)ARRAY_SIZE(texture_atlas_str));
    if (imgui_overlay_header("Light")) {
      if (imgui_overlay_button(context->imgui_overlay, "Reset Light")) {
        reset_light();
      }
      imgui_overlay_slider_float(context->imgui_overlay, "lightPosX",
                                 &settings.light_pos_x, -5.0f, 5.0f, "%.1f");
      imgui_overlay_slider_float(context->imgui_overlay, "lightPosY",
                                 &settings.light_pos_y, -5.0f, 5.0f, "%.1f");
      imgui_overlay_slider_float(context->imgui_overlay, "lightPosZ",
                                 &settings.light_pos_z, -5.0f, 5.0f, "%.1f");
      imgui_overlay_slider_float(context->imgui_overlay, "lightIntensity",
                                 &settings.light_intensity, 0.0f, 10.0f,
                                 "%.1f");
    }
    if (imgui_overlay_header("Depth")) {
      imgui_overlay_slider_float(context->imgui_overlay, "depthScale",
                                 &settings.depth_scale, 0.0f, 0.1f, "%.01f");
      imgui_overlay_slider_int(context->imgui_overlay, "depthLayers",
                               &settings.depth_layers, 1, 32);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  /* Draw textured Cube */
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                   textured_cube_pipeline);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    frame_bg_descriptor.bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(
    wgpu_context->rpass_enc, 1,
    surface_bg_descriptor.bind_groups[current_surface_bind_group], 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       box.renderable.vertex_buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, box.renderable.index_buffer.buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   box.renderable.index_count, 1, 0, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  /* Get next image in the swap chain (back/front buffer) */
  wgpu_swap_chain_get_current_image(wgpu_context);

  /* Create command buffer */
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer to the queue */
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  /* Present the current buffer to the swap chain */
  wgpu_swap_chain_present(wgpu_context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return example_draw(context->wgpu_context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  wgpu_destroy_buffer(&box.renderable.vertex_buffer);
  wgpu_destroy_buffer(&box.renderable.index_buffer);
  wgpu_destroy_buffer(&uniforms_bufers.space_transforms);
  wgpu_destroy_buffer(&uniforms_bufers.map_info);
  wgpu_destroy_texture(&textures.wood_albedo);
  wgpu_destroy_texture(&textures.spiral_normal);
  wgpu_destroy_texture(&textures.spiral_height);
  wgpu_destroy_texture(&textures.toybox_normal);
  wgpu_destroy_texture(&textures.toybox_height);
  wgpu_destroy_texture(&textures.brickwall_albedo);
  wgpu_destroy_texture(&textures.brickwall_normal);
  wgpu_destroy_texture(&textures.brickwall_height);
  wgpu_destroy_texture(&textures.depth);
  WGPU_RELEASE_RESOURCE(Sampler, textures.sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, frame_bg_descriptor.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        surface_bg_descriptor.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, frame_bg_descriptor.bind_group)
  for (uint8_t i = 0; i < TEXTURE_ATLAS_COUNT; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, surface_bg_descriptor.bind_groups[i])
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, textured_cube_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, textured_cube_pipeline_layout)
}

void example_normal_map(int argc, char* argv[])
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
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* normal_map_shader_wgsl = CODE(
  const modeAlbedoTexture = 0;
  const modeNormalTexture = 1;
  const modeDepthTexture  = 2;
  const modeNormalMap     = 3;
  const modeParallaxScale = 4;
  const modeSteepParallax = 5;

  struct SpaceTransforms {
    worldViewProjMatrix: mat4x4f,
    worldViewMatrix: mat4x4f,
  }

  struct MapInfo {
    lightPosVS: vec3f, // Light position in view space
    mode: u32,
    lightIntensity: f32,
    depthScale: f32,
    depthLayers: f32,
  }

  struct VertexInput {
    // Shader assumes the missing 4th float is 1.0
    @location(0) position : vec4f,
    @location(1) normal : vec3f,
    @location(2) uv : vec2f,
    @location(3) vert_tan: vec3f,
    @location(4) vert_bitan: vec3f,
  }

  struct VertexOutput {
    @builtin(position) posCS : vec4f, // vertex position in clip space
    @location(0) posVS : vec3f,       // vertex position in view space
    @location(1) tangentVS: vec3f,    // vertex tangent in view space
    @location(2) bitangentVS: vec3f,  // vertex tangent in view space
    @location(3) normalVS: vec3f,     // vertex normal in view space
    @location(5) uv : vec2f,          // vertex texture coordinate
  }

  // Uniforms
  @group(0) @binding(0) var<uniform> spaceTransform : SpaceTransforms;
  @group(0) @binding(1) var<uniform> mapInfo: MapInfo;

  // Texture info
  @group(1) @binding(0) var textureSampler: sampler;
  @group(1) @binding(1) var albedoTexture: texture_2d<f32>;
  @group(1) @binding(2) var normalTexture: texture_2d<f32>;
  @group(1) @binding(3) var depthTexture: texture_2d<f32>;


  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output : VertexOutput;

    output.posCS = spaceTransform.worldViewProjMatrix * input.position;
    output.posVS = (spaceTransform.worldViewMatrix * input.position).xyz;
    output.tangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_tan, 0)).xyz;
    output.bitangentVS = (spaceTransform.worldViewMatrix * vec4(input.vert_bitan, 0)).xyz;
    output.normalVS = (spaceTransform.worldViewMatrix * vec4(input.normal, 0)).xyz;
    output.uv = input.uv;

    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    // Build the matrix to convert from tangent space to view space
    let tangentToView = mat3x3f(
        input.tangentVS,
        input.bitangentVS,
        input.normalVS,
    );

    // The inverse of a non-scaling affine 3x3 matrix is it's transpose
    let viewToTangent = transpose(tangentToView);

    // Calculate the normalized vector in tangent space from the camera to the fragment
    let viewDirTS = normalize(viewToTangent * input.posVS);

    // Apply parallax to the texture coordinate, if parallax is enabled
    var uv : vec2f;
    switch (mapInfo.mode) {
      case modeParallaxScale: {
        uv = parallaxScale(input.uv, viewDirTS);
        break;
      }
      case modeSteepParallax: {
        uv = parallaxSteep(input.uv, viewDirTS);
        break;
      }
      default: {
        uv = input.uv;
        break;
      }
    }

    // Sample the albedo texture
    let albedoSample = textureSample(albedoTexture, textureSampler, uv);

    // Sample the normal texture
    let normalSample = textureSample(normalTexture, textureSampler, uv);

    switch (mapInfo.mode) {
      case modeAlbedoTexture: { // Output the albedo sample
        return albedoSample;
      }
      case modeNormalTexture: { // Output the normal sample
        return normalSample;
      }
      case modeDepthTexture: { // Output the depth map
        return textureSample(depthTexture, textureSampler, input.uv);
      }
      default: {
        // Transform the normal sample to a tangent space normal
        let normalTS = normalSample.xyz * 2 - 1;

        // Convert normal from tangent space to view space, and normalize
        let normalVS = normalize(tangentToView * normalTS);

        // Calculate the vector in view space from the light position to the fragment
        let fragToLightVS = mapInfo.lightPosVS - input.posVS;

        // Calculate the square distance from the light to the fragment
        let lightSqrDist = dot(fragToLightVS, fragToLightVS);

        // Calculate the normalized vector in view space from the fragment to the light
        let lightDirVS = fragToLightVS * inverseSqrt(lightSqrDist);

        // Light strength is inversely proportional to square of distance from light
        let diffuseLight = mapInfo.lightIntensity * max(dot(lightDirVS, normalVS), 0) / lightSqrDist;

        // The diffuse is the albedo color multiplied by the diffuseLight
        let diffuse = albedoSample.rgb * diffuseLight;

        return vec4f(diffuse, 1.0);
      }
    }
  }

  // Returns the uv coordinate displaced in the view direction by a magnitude calculated by the depth
  // sampled from the depthTexture and the angle between the surface normal and view direction.
  fn parallaxScale(uv: vec2f, viewDirTS: vec3f) -> vec2f {
    let depthSample = textureSample(depthTexture, textureSampler, uv).r;
    return uv + viewDirTS.xy * (depthSample * mapInfo.depthScale) / -viewDirTS.z;
  }

  // Returns the uv coordinates displaced in the view direction by ray-tracing the depth map.
  fn parallaxSteep(startUV: vec2f, viewDirTS: vec3f) -> vec2f {
    // Calculate derivatives of the texture coordinate, so we can sample the texture with non-uniform
    // control flow.
    let ddx = dpdx(startUV);
    let ddy = dpdy(startUV);

    // Calculate the delta step in UV and depth per iteration
    let uvDelta = viewDirTS.xy * mapInfo.depthScale / (-viewDirTS.z * mapInfo.depthLayers);
    let depthDelta = 1.0 / f32(mapInfo.depthLayers);
    let posDelta = vec3(uvDelta, depthDelta);

    // Walk the depth texture, and stop when the ray intersects the depth map
    var pos = vec3(startUV, 0);
    for (var i = 0; i < 32; i++) {
      if (pos.z >= textureSampleGrad(depthTexture, textureSampler, pos.xy, ddx, ddy).r) {
        break; // Hit the surface
      }
      pos += posDelta;
    }

    return pos.xy;
  }
);
// clang-format on
