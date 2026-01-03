#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

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

#define TEXTURE_COUNT (8u)
#define DEPTH_TEXTURE_FORMAT (WGPUTextureFormat_Depth24PlusStencil8)

/* The mesh to be rendered */
typedef struct renderable_t {
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;
} renderable_t;

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

/* State struct */
static struct {
  /* Geometry */
  struct {
    renderable_t renderable;
    box_mesh_t mesh;
  } box;
  /* The textures */
  struct {
    wgpu_texture_t wood_albedo;
    wgpu_texture_t spiral_normal;
    wgpu_texture_t spiral_height;
    wgpu_texture_t toybox_normal;
    wgpu_texture_t toybox_height;
    wgpu_texture_t brickwall_albedo;
    wgpu_texture_t brickwall_normal;
    wgpu_texture_t brickwall_height;
    wgpu_texture_t depth;
    WGPUSampler sampler;
  } textures;
  struct {
    const char* file;
    wgpu_texture_t* texture;
  } texture_mappings[TEXTURE_COUNT];
  uint8_t file_buffer[512 * 512 * 4];
  /* Uniforms data */
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
  } view_matrices;
  struct {
    mat4 world_view_proj_matrix;
    mat4 world_view_matrix;
  } space_transforms;
  struct {
    vec3 light_pos_vs; /* Light position in view space */
    uint32_t mode;
    float light_intensity;
    float depth_scale;
    float depth_layers;
    float padding;
  } map_info;
  /* Uniforms buffer */
  struct {
    wgpu_buffer_t space_transforms;
    wgpu_buffer_t map_info;
  } uniforms_bufers;
  /* The bind groups and layouts */
  struct {
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout bind_group_layout;
  } frame_bg_descriptor;
  struct {
    WGPUBindGroup bind_groups[TEXTURE_ATLAS_COUNT];
    WGPUBindGroupLayout bind_group_layout;
  } surface_bg_descriptor;
  /* The render pipeline + pipeline layout */
  WGPURenderPipeline textured_cube_pipeline;
  WGPUPipelineLayout textured_cube_pipeline_layout;
  /* Render pass descriptor for frame buffer writes */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* GUI control */
  int32_t current_surface_bind_group;
  struct {
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
    bool paused;
  } settings;
  const char* texture_atlas_str[TEXTURE_ATLAS_COUNT];
  const char* bump_modes_str[BUMP_MODE_COUNT];
  WGPUBool initialized;
} state = {
  // clang-format off
  .texture_mappings = {
    { .file = "assets/textures/wood_albedo.png",      .texture = &state.textures.wood_albedo      },
    { .file = "assets/textures/spiral_normal.png",    .texture = &state.textures.spiral_normal    },
    { .file = "assets/textures/spiral_height.png",    .texture = &state.textures.spiral_height    },
    { .file = "assets/textures/toybox_normal.png",    .texture = &state.textures.toybox_normal    },
    { .file = "assets/textures/toybox_height.png",    .texture = &state.textures.toybox_height    },
    { .file = "assets/textures/brickwall_albedo.png", .texture = &state.textures.brickwall_albedo },
    { .file = "assets/textures/brickwall_normal.png", .texture = &state.textures.brickwall_normal },
    { .file = "assets/textures/brickwall_height.png", .texture = &state.textures.brickwall_height },
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
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
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  // clang-format on
  .settings = {
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
  },
  .texture_atlas_str = {
    "Spiral",    /* */
    "Toybox",    /* */
    "BrickWall", /* */
  },
  .bump_modes_str = {
    "Albedo Texture", /* */
    "Normal Texture", /* */
    "Depth Texture",  /* */
    "Normal Map",     /* */
    "Parallax Scale", /* */
    "Steep Parallax", /* */
  },
};

static void init_box_mesh_renderable(wgpu_context_t* wgpu_context)
{
  box_mesh_create_with_tangents(&state.box.mesh, 1.0f, 1.0f, 1.0f);

  /* Create vertex buffers */
  state.box.renderable.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box mesh - Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = state.box.mesh.vertex_count * sizeof(float),
                    .initial.data = state.box.mesh.vertex_array,
                  });

  /* Create index buffer */
  state.box.renderable.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box mesh - Indices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = state.box.mesh.index_count * sizeof(uint32_t),
                    .initial.data = state.box.mesh.index_array,
                  });
  state.box.renderable.index_count = state.box.mesh.index_count;
}

static mat4* get_projection_matrix(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 0.1f, 10.0f,
                  state.view_matrices.projection);

  return &state.view_matrices.projection;
}

static mat4* get_view_matrix(void)
{
  glm_lookat((vec3){state.settings.camera_pos_x, state.settings.camera_pos_y,
                    state.settings.camera_pos_z}, /* eye vector    */
             (vec3){0.0f, 0.0f, 0.0f},            /* center vector */
             (vec3){0.0f, 1.0f, 0.0f},            /* up vector     */
             state.view_matrices.view             /* result matrix */
  );

  return &state.view_matrices.view;
}

static mat4* get_model_matrix(void)
{
  glm_mat4_identity(state.view_matrices.model);
  const float now = stm_sec(stm_now());
  glm_rotate_y(state.view_matrices.model, now * -0.5f,
               state.view_matrices.model);

  return &state.view_matrices.model;
}

static uint32_t get_bump_mode(void)
{
  return (uint32_t)state.settings.bump_mode;
}

static void update_space_transforms_buffer(wgpu_context_t* wgpu_context)
{
  /* Update matrices */
  glm_mat4_mul(*get_view_matrix(), *get_model_matrix(),
               state.space_transforms.world_view_matrix);
  glm_mat4_mul(*get_projection_matrix(wgpu_context),
               state.space_transforms.world_view_matrix,
               state.space_transforms.world_view_proj_matrix);

  /* Update GPU buffer*/
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.uniforms_bufers.space_transforms.buffer, 0,
                       &state.space_transforms, sizeof(state.space_transforms));
}

static void update_map_info_buffer(wgpu_context_t* wgpu_context)
{
  /* Update map info data */
  vec3 light_pos_ws = {state.settings.light_pos_x, state.settings.light_pos_y,
                       state.settings.light_pos_z};
  glm_vec3_transform_mat4(light_pos_ws, *get_view_matrix(),
                          &state.map_info.light_pos_vs);
  state.map_info.mode            = get_bump_mode();
  state.map_info.light_intensity = state.settings.light_intensity;
  state.map_info.depth_scale     = state.settings.depth_scale;
  state.map_info.depth_layers    = state.settings.depth_layers;

  /* Update GPU buffer*/
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.uniforms_bufers.map_info.buffer, 0,
                       &state.map_info, sizeof(state.map_info));
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  update_space_transforms_buffer(wgpu_context);
  update_map_info_buffer(wgpu_context);
}

static void init_uniforms_buffers(wgpu_context_t* wgpu_context)
{
  /* Space transforms buffer */
  state.uniforms_bufers.space_transforms = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      // Buffer holding projection, view, and model matrices plus padding bytes
      .label = "Space transforms - Uniform buffer",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.space_transforms),
    });

  /* Space transforms buffer */
  state.uniforms_bufers.map_info = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      // Buffer holding mapping type, light uniforms, and depth uniforms
      .label = "Space transforms - Uniform buffer",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.map_info),
    });
}

static void init_depth_textures(wgpu_context_t* wgpu_context)
{
  /* Cleanup */
  wgpu_destroy_texture(&state.textures.depth);

  /* Create the depth texture */
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
    .format        = DEPTH_TEXTURE_FORMAT,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  state.textures.depth.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(state.textures.depth.handle != NULL);

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Depth - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.textures.depth.view
    = wgpuTextureCreateView(state.textures.depth.handle, &texture_view_dec);
  ASSERT(state.textures.depth.view != NULL);
}

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
        .extent = (WGPUExtent3D) {
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 1,
      },
        .format = WGPUTextureFormat_RGBA8Unorm,
        .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_surface_bg_textures(wgpu_context_t* wgpu_context)
{
  /* Fetch the images and upload them into a GPUTextures. */
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(state.texture_mappings); ++i) {
    wgpu_texture_t* texture = state.texture_mappings[i].texture;
    /* Create dummy texture */
    *(texture) = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });
    /* Start loading the image file */
    sfetch_send(&(sfetch_request_t){
      .path      = state.texture_mappings[i].file,
      .callback  = fetch_callback,
      .buffer    = SFETCH_RANGE(state.file_buffer),
      .user_data = {
        .ptr  = &texture,
        .size = sizeof(wgpu_texture_t*),
      },
    });
  }
}

/* Init a sampler with linear filtering for smooth interpolation. */
static void init_sampler(wgpu_context_t* wgpu_context)
{
  state.textures.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Texture - Sampler"),
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
  ASSERT(state.textures.sampler != NULL);
}

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
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
          .minBindingSize   = state.uniforms_bufers.space_transforms.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = state.uniforms_bufers.map_info.size,
        },
        .sampler = {0},
      },
    };
    state.frame_bg_descriptor.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                                .label = STRVIEW("Frame - Bind group layout"),
                                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                                .entries    = bgl_entries,
                              });
    ASSERT(state.frame_bg_descriptor.bind_group_layout != NULL);
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
    state.surface_bg_descriptor.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                                .label = STRVIEW("Texture - Bind group layout"),
                                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                                .entries    = bgl_entries,
                              });
    ASSERT(state.surface_bg_descriptor.bind_group_layout != NULL);
  }
}

static void init_frame_bg_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.uniforms_bufers.space_transforms.buffer,
        .size    = state.uniforms_bufers.space_transforms.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = state.uniforms_bufers.map_info.buffer,
        .size    = state.uniforms_bufers.map_info.size,
      },
    };
  state.frame_bg_descriptor.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Frame - Bind group"),
      .layout     = state.frame_bg_descriptor.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    });
  ASSERT(state.frame_bg_descriptor.bind_group != NULL);
}

/* Multiple bindgroups that accord to the layout defined above */
static void init_surface_bg_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Cleanup */
  for (uint8_t i = 0; i < TEXTURE_ATLAS_COUNT; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.surface_bg_descriptor.bind_groups[i])
  }

  WGPUTextureView texture_views[TEXTURE_ATLAS_COUNT][3] = {
    // clang-format off
    {state.textures.wood_albedo.view,      state.textures.spiral_normal.view,    state.textures.spiral_height.view},
    {state.textures.wood_albedo.view,      state.textures.toybox_normal.view,    state.textures.toybox_height.view},
    {state.textures.brickwall_albedo.view, state.textures.brickwall_normal.view, state.textures.brickwall_height.view},
    // clang-format on
  };
  for (uint8_t i = 0; i < TEXTURE_ATLAS_COUNT; ++i) {
    WGPUBindGroupEntry bg_entries[4] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .sampler = state.textures.sampler,
        },
      };
    for (uint8_t j = 1; j <= 3; ++j) {
      bg_entries[j] = (WGPUBindGroupEntry){
        .binding     = j,
        .textureView = texture_views[i][j - 1],
      };
    }
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Surface - Bind group"),
      .layout     = state.surface_bg_descriptor.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    state.surface_bg_descriptor.bind_groups[i]
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(state.surface_bg_descriptor.bind_groups[i] != NULL);
  }
}

static void
init_3d_render_pipeline(wgpu_context_t* wgpu_context, const char* label,
                        WGPUBindGroupLayout const* bg_layouts,
                        uint32_t bg_layout_count, const char* vertex_shader,
                        WGPUVertexBufferLayout const* vertex_buffer_layouts,
                        uint32_t vertex_buffer_count,
                        const char* fragment_shader,
                        WGPUTextureFormat presentation_format, bool depth_test,
                        WGPUPrimitiveTopology topology, WGPUCullMode cull_mode,
                        WGPURenderPipeline* render_pipeline,
                        WGPUPipelineLayout* render_pipeline_layout)
{
  /* Pipeline layout */
  {
    *render_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("Render - Pipeline layout"),
                              .bindGroupLayoutCount = bg_layout_count,
                              .bindGroupLayouts     = bg_layouts,
                            });
    ASSERT(*render_pipeline_layout != NULL);
  }

  /* Render pipeline layout */
  {
    WGPUShaderModule vert_shader_module
      = wgpu_create_shader_module(wgpu_context->device, vertex_shader);
    WGPUShaderModule frag_shader_module
      = wgpu_create_shader_module(wgpu_context->device, fragment_shader);

    /* Color blend state */
    WGPUBlendState blend_state = wgpu_create_blend_state(true);

    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_TEXTURE_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW(label),
      .layout = *render_pipeline_layout,
      .vertex = {
        .module      = vert_shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = vertex_buffer_count,
        .buffers     = vertex_buffer_layouts,
      },
      .fragment = &(WGPUFragmentState) {
        .entryPoint  = STRVIEW("fragmentMain"),
        .module      = frag_shader_module,
        .targetCount = 1,
        .targets = &(WGPUColorTargetState) {
          .format    = presentation_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = topology,
        .cullMode  = cull_mode,
        .frontFace = WGPUFrontFace_CCW
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    if (depth_test) {
      rp_desc.depthStencil = &depth_stencil_state;
    }

    *render_pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
    ASSERT(*render_pipeline != NULL);

    wgpuShaderModuleRelease(vert_shader_module);
    wgpuShaderModuleRelease(frag_shader_module);
  }
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bg_layouts[2] = {
    state.frame_bg_descriptor.bind_group_layout,
    state.surface_bg_descriptor.bind_group_layout,
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

  init_3d_render_pipeline(wgpu_context, "Normal mapping - Render pipeline",
                          bg_layouts, (uint32_t)ARRAY_SIZE(bg_layouts),
                          normal_map_shader_wgsl, &box_vertex_buffer_layout, 1,
                          normal_map_shader_wgsl, wgpu_context->render_format,
                          true, WGPUPrimitiveTopology_TriangleList,
                          WGPUCullMode_Back, &state.textured_cube_pipeline,
                          &state.textured_cube_pipeline_layout);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = TEXTURE_COUNT,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    init_box_mesh_renderable(wgpu_context);
    init_uniforms_buffers(wgpu_context);
    init_depth_textures(wgpu_context);
    init_surface_bg_textures(wgpu_context);
    init_sampler(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_frame_bg_bind_group(wgpu_context);
    init_surface_bg_bind_groups(wgpu_context);
    init_pipelines(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void reset_light(void)
{
  state.settings.light_pos_x     = 1.7f;
  state.settings.light_pos_y     = 0.7f;
  state.settings.light_pos_z     = -1.9f;
  state.settings.light_intensity = 5.0f;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_textures(wgpu_context);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'a') {
    state.current_surface_bind_group
      = (state.current_surface_bind_group + 1)
        % (uint32_t)ARRAY_SIZE(state.texture_atlas_str);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'b') {
    state.settings.bump_mode
      = (state.settings.bump_mode + 1) % ARRAY_SIZE(state.bump_modes_str);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'p') {
    state.settings.paused = !state.settings.paused;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'r') {
    reset_light();
  }
}

static void update_textures(struct wgpu_context_t* wgpu_context)
{
  bool is_dirty         = true;
  uint8_t texture_count = (uint8_t)ARRAY_SIZE(state.texture_mappings);
  for (uint8_t i = 0; i < texture_count; ++i) {
    is_dirty = is_dirty && state.texture_mappings[i].texture->desc.is_dirty;
  }

  if (is_dirty) {
    /* Recreate textures */
    for (uint8_t i = 0; i < texture_count; ++i) {
      wgpu_recreate_texture(wgpu_context, state.texture_mappings[i].texture);
      FREE_TEXTURE_PIXELS(*state.texture_mappings[i].texture);
    }
    /* Upddate the bind group */
    init_surface_bg_bind_groups(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update texture when pixel data loaded */
  update_textures(wgpu_context);

  /* Update matrix data */
  if (!state.settings.paused) {
    update_uniform_buffers(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.textures.depth.view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.textured_cube_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.frame_bg_descriptor.bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(
    rpass_enc, 1,
    state.surface_bg_descriptor.bind_groups[state.current_surface_bind_group],
    0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.box.renderable.vertex_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass_enc, state.box.renderable.index_buffer.buffer, WGPUIndexFormat_Uint16,
    0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.box.renderable.index_count,
                                   1, 0, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  wgpu_destroy_buffer(&state.box.renderable.vertex_buffer);
  wgpu_destroy_buffer(&state.box.renderable.index_buffer);
  wgpu_destroy_buffer(&state.uniforms_bufers.space_transforms);
  wgpu_destroy_buffer(&state.uniforms_bufers.map_info);
  wgpu_destroy_texture(&state.textures.wood_albedo);
  wgpu_destroy_texture(&state.textures.spiral_normal);
  wgpu_destroy_texture(&state.textures.spiral_height);
  wgpu_destroy_texture(&state.textures.toybox_normal);
  wgpu_destroy_texture(&state.textures.toybox_height);
  wgpu_destroy_texture(&state.textures.brickwall_albedo);
  wgpu_destroy_texture(&state.textures.brickwall_normal);
  wgpu_destroy_texture(&state.textures.brickwall_height);
  wgpu_destroy_texture(&state.textures.depth);
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.frame_bg_descriptor.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.surface_bg_descriptor.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.frame_bg_descriptor.bind_group)
  for (uint8_t i = 0; i < TEXTURE_ATLAS_COUNT; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.surface_bg_descriptor.bind_groups[i])
  }
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.textured_cube_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.textured_cube_pipeline_layout)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Normal Mapping",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
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
