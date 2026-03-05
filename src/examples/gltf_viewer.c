/* -------------------------------------------------------------------------- *
 * WebGPU Example - glTF PBR Viewer
 *
 * A physically based glTF 2.0 model viewer with Image Based Lighting (IBL).
 * Features:
 *   - Metallic-roughness PBR workflow
 *   - IBL: irradiance, prefiltered specular, BRDF LUT
 *   - Environment skybox rendering
 *   - Normal mapping, occlusion, emissive
 *   - Alpha mask & alpha blend (transparent sorting)
 *   - Orbit camera controls (tumble, pan, zoom)
 *   - Turntable rotation animation
 *   - GUI controls for animation and camera
 *
 * Based on: https://github.com/ArnCarve);
 * -------------------------------------------------------------------------- */

#include "webgpu/wgpu_common.h"

/* GUI overlay */
#include "webgpu/imgui_overlay.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Math library */
#include <cglm/cglm.h>

/* Timer */
#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* Async file loading */
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

/* glTF model */
#include "core/gltf_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define GLTF_VIEWER_MAX_MATERIALS 64
#define GLTF_VIEWER_MAX_SUBMESHES 256

#define GLB_FILE_PATH "assets/models/DamagedHelmet.glb"
#define HDR_FILE_PATH "assets/textures/venice_sunset_1k.hdr"

/* File buffer sizes for sokol_fetch */
#define GLB_FILE_BUFFER_SIZE (8 * 1024 * 1024) /* 8 MB */
#define HDR_FILE_BUFFER_SIZE (4 * 1024 * 1024) /* 4 MB */

/* Camera defaults */
#define CAMERA_FOV 45.0f
#define CAMERA_NEAR_FACTOR 0.01f
#define CAMERA_FAR_FACTOR 100.0f
#define TUMBLE_SPEED 0.004f
#define PAN_SPEED 0.01f
#define ZOOM_SPEED 0.01f
#define ZOOM_SCROLL_SENS 30.0f
#define TILT_CLAMP 0.98f

/* -------------------------------------------------------------------------- *
 * Forward-declared WGSL shaders (defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* environment_shader_wgsl;
static const char* gltf_pbr_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Uniform structures (match WGSL layout)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 view_matrix;               /* offset 0   */
  mat4 projection_matrix;         /* offset 64  */
  mat4 inverse_view_matrix;       /* offset 128 */
  mat4 inverse_projection_matrix; /* offset 192 */
  vec3 camera_position;           /* offset 256 */
  float _pad;                     /* offset 268 */
} global_uniforms_t;              /* size: 272  */

typedef struct {
  mat4 model_matrix;  /* offset 0  */
  mat4 normal_matrix; /* offset 64 */
} model_uniforms_t;   /* size: 128 */

typedef struct {
  vec4 base_color_factor;   /* offset 0  */
  vec3 emissive_factor;     /* offset 16 */
  float metallic_factor;    /* offset 28 */
  float roughness_factor;   /* offset 32 */
  float normal_scale;       /* offset 36 */
  float occlusion_strength; /* offset 40 */
  float alpha_cutoff;       /* offset 44 */
  int32_t alpha_mode;       /* offset 48, 0=Opaque, 1=Mask, 2=Blend */
  float _pad[3];            /* pad to 64 */
} material_uniforms_t;      /* size: 64  */

/* -------------------------------------------------------------------------- *
 * Per-material GPU data
 * -------------------------------------------------------------------------- */

typedef struct {
  material_uniforms_t uniforms;
  WGPUBuffer uniform_buffer;
  WGPUTexture base_color_texture;
  WGPUTexture metallic_roughness_texture;
  WGPUTexture normal_texture;
  WGPUTexture occlusion_texture;
  WGPUTexture emissive_texture;
  WGPUBindGroup bind_group;
} viewer_material_t;

/* -------------------------------------------------------------------------- *
 * Sub-mesh for draw calls
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t first_index;
  uint32_t index_count;
  int32_t material_index;
  vec3 centroid;
} viewer_sub_mesh_t;

/* Transparent mesh sorting helper */
typedef struct {
  float depth;
  uint32_t mesh_index;
} sub_mesh_depth_info_t;

/* -------------------------------------------------------------------------- *
 * Orbit camera
 * -------------------------------------------------------------------------- */

typedef struct {
  int width;
  int height;
  float near_clip;
  float far_clip;
  float pan_factor;
  float zoom_factor;
  vec3 position;
  vec3 target;
  vec3 forward;
  vec3 right;
  vec3 up;
} orbit_camera_t;

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Initialization flags */
  WGPUBool initialized;
  WGPUBool glb_loaded;
  WGPUBool hdr_loaded;
  WGPUBool resources_ready;
  WGPUBool model_resources_created;

  /* File loading buffers */
  uint8_t* glb_file_buffer;
  uint8_t* hdr_file_buffer;

  /* Timer */
  uint64_t last_frame_time;

  /* Camera */
  orbit_camera_t camera;

  /* Mouse state */
  struct {
    bool tumble;
    bool pan;
    float last_x;
    float last_y;
  } mouse;

  /* Model data */
  gltf_model_t model;
  bool model_loaded;
  bool animate_model;
  float rotation_angle;
  mat4 model_transform;
  mat4 node_base_transform;

  /* Environment data */
  wgpu_environment_t environment;
  wgpu_ibl_textures_t ibl;
  bool environment_loaded;

  /* GPU resources */
  struct {
    /* Global bind group (group 0) */
    WGPUBindGroupLayout global_bind_group_layout;
    WGPUBindGroup global_bind_group;
    WGPUBuffer global_uniform_buffer;

    /* Model bind group layout (group 1) */
    WGPUBindGroupLayout model_bind_group_layout;
    WGPUBuffer model_uniform_buffer;

    /* Vertex/index buffers */
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;

    /* Samplers */
    WGPUSampler model_texture_sampler;

    /* Default textures */
    WGPUTexture default_srgb_texture;
    WGPUTextureView default_srgb_view;
    WGPUTexture default_unorm_texture;
    WGPUTextureView default_unorm_view;
    WGPUTexture default_normal_texture;
    WGPUTextureView default_normal_view;
    WGPUTexture default_cube_texture;
    WGPUTextureView default_cube_view;

    /* Pipelines */
    WGPUShaderModule env_shader_module;
    WGPURenderPipeline env_pipeline;
    WGPUShaderModule model_shader_module;
    WGPURenderPipeline model_pipeline_opaque;
    WGPURenderPipeline model_pipeline_transparent;

    /* Depth texture */
    WGPUTexture depth_texture;
    WGPUTextureView depth_texture_view;
  } gpu;

  /* Materials and meshes */
  viewer_material_t materials[GLTF_VIEWER_MAX_MATERIALS];
  uint32_t material_count;

  viewer_sub_mesh_t opaque_meshes[GLTF_VIEWER_MAX_SUBMESHES];
  uint32_t opaque_mesh_count;

  viewer_sub_mesh_t transparent_meshes[GLTF_VIEWER_MAX_SUBMESHES];
  uint32_t transparent_mesh_count;

  sub_mesh_depth_info_t transparent_sorted[GLTF_VIEWER_MAX_SUBMESHES];
  uint32_t transparent_sorted_count;

  /* Render pass (pre-initialized) */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI */
  struct {
    bool show_gui;
  } settings;

} state = {
  .animate_model = true,
  .settings.show_gui = true,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.2f, 0.4f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
    .stencilLoadOp   = WGPULoadOp_Clear,
    .stencilStoreOp  = WGPUStoreOp_Store,
  },
};

/* -------------------------------------------------------------------------- *
 * Orbit Camera Implementation
 * -------------------------------------------------------------------------- */

static void camera_update_vectors(orbit_camera_t* cam)
{
  vec3 base_up = {0.0f, 1.0f, 0.0f};
  glm_vec3_sub(cam->target, cam->position, cam->forward);
  glm_vec3_normalize(cam->forward);
  glm_vec3_cross(cam->forward, base_up, cam->right);
  glm_vec3_normalize(cam->right);
  glm_vec3_cross(cam->right, cam->forward, cam->up);
  glm_vec3_normalize(cam->up);
}

static void camera_init(orbit_camera_t* cam, int w, int h)
{
  cam->width       = w;
  cam->height      = h;
  cam->near_clip   = 0.1f;
  cam->far_clip    = 100.0f;
  cam->pan_factor  = 1.0f;
  cam->zoom_factor = 1.0f;
  glm_vec3_copy((vec3){0.0f, 0.0f, 5.0f}, cam->position);
  glm_vec3_copy((vec3){0.0f, 0.0f, 0.0f}, cam->target);
  camera_update_vectors(cam);
}

static void camera_resize(orbit_camera_t* cam, int w, int h)
{
  if (w > 0 && h > 0) {
    cam->width  = w;
    cam->height = h;
  }
}

static void camera_get_view_matrix(const orbit_camera_t* cam, mat4 dest)
{
  glm_lookat((float*)cam->position, (float*)cam->target, (float*)cam->up, dest);
}

static void camera_get_projection_matrix(const orbit_camera_t* cam, mat4 dest)
{
  float ratio = (float)cam->width / (float)cam->height;
  glm_perspective(glm_rad(CAMERA_FOV), ratio, cam->near_clip, cam->far_clip,
                  dest);
}

static void camera_tumble(orbit_camera_t* cam, int dx, int dy)
{
  /* Rotate around world Y-axis */
  {
    vec3 offset;
    glm_vec3_sub(cam->position, cam->target, offset);
    float angle = (float)dx * TUMBLE_SPEED;
    float cos_a = cosf(angle), sin_a = sinf(angle);
    float new_x = offset[0] * cos_a - offset[2] * sin_a;
    float new_z = offset[0] * sin_a + offset[2] * cos_a;
    offset[0]   = new_x;
    offset[2]   = new_z;
    glm_vec3_add(cam->target, offset, cam->position);
    camera_update_vectors(cam);
  }

  /* Tilt around local X-axis (right) */
  {
    vec3 orig_pos, orig_fwd;
    glm_vec3_copy(cam->position, orig_pos);
    glm_vec3_copy(cam->forward, orig_fwd);

    vec3 offset;
    glm_vec3_sub(cam->position, cam->target, offset);
    float angle = (float)dy * TUMBLE_SPEED;

    float right_comp   = glm_vec3_dot(offset, cam->right);
    float up_comp      = glm_vec3_dot(offset, cam->up);
    float forward_comp = glm_vec3_dot(offset, cam->forward);

    float cos_a = cosf(angle), sin_a = sinf(angle);
    float new_up  = up_comp * cos_a - forward_comp * sin_a;
    float new_fwd = up_comp * sin_a + forward_comp * cos_a;

    /* Reconstruct */
    vec3 r_part, u_part, f_part;
    glm_vec3_scale(cam->right, right_comp, r_part);
    glm_vec3_scale(cam->up, new_up, u_part);
    glm_vec3_scale(cam->forward, new_fwd, f_part);

    glm_vec3_add(r_part, u_part, offset);
    glm_vec3_add(offset, f_part, offset);
    glm_vec3_add(cam->target, offset, cam->position);

    /* Clamp to prevent gimbal lock */
    vec3 fwd_test;
    glm_vec3_sub(cam->target, cam->position, fwd_test);
    glm_vec3_normalize(fwd_test);
    if (fabsf(fwd_test[1]) > TILT_CLAMP) {
      glm_vec3_copy(orig_pos, cam->position);
      glm_vec3_copy(orig_fwd, cam->forward);
    }
    camera_update_vectors(cam);
  }
}

static void camera_zoom(orbit_camera_t* cam, int dx, int dy)
{
  float delta = (float)(-dx + dy) * cam->zoom_factor;
  vec3 movement;
  glm_vec3_scale(cam->forward, delta, movement);
  glm_vec3_add(cam->position, movement, cam->position);
}

static void camera_pan(orbit_camera_t* cam, int dx, int dy)
{
  float delta_x = (float)(-dx) * cam->pan_factor;
  float delta_y = (float)(dy)*cam->pan_factor;

  vec3 up_move, right_move, total;
  glm_vec3_scale(cam->up, delta_y, up_move);
  glm_vec3_scale(cam->right, delta_x, right_move);
  glm_vec3_add(up_move, right_move, total);

  glm_vec3_add(cam->position, total, cam->position);
  glm_vec3_add(cam->target, total, cam->target);
}

static void camera_reset_to_model(orbit_camera_t* cam, vec3 min_b, vec3 max_b)
{
  /* Validate bounds */
  if (max_b[0] <= min_b[0] || max_b[1] <= min_b[1] || max_b[2] <= min_b[2]) {
    glm_vec3_copy((vec3){-0.5f, -0.5f, -0.5f}, min_b);
    glm_vec3_copy((vec3){0.5f, 0.5f, 0.5f}, max_b);
  }

  vec3 center, extent;
  glm_vec3_add(min_b, max_b, center);
  glm_vec3_scale(center, 0.5f, center);
  glm_vec3_sub(max_b, min_b, extent);
  float radius   = glm_vec3_norm(extent) * 0.5f;
  float distance = radius / sinf(glm_rad(CAMERA_FOV * 0.5f));

  glm_vec3_copy((vec3){center[0], center[1], center[2] + distance},
                cam->position);
  glm_vec3_copy(center, cam->target);
  cam->near_clip   = radius * CAMERA_NEAR_FACTOR;
  cam->far_clip    = distance + radius * CAMERA_FAR_FACTOR;
  cam->pan_factor  = radius * PAN_SPEED;
  cam->zoom_factor = radius * ZOOM_SPEED;
  camera_update_vectors(cam);
}

/* -------------------------------------------------------------------------- *
 * Default texture creation helpers
 * -------------------------------------------------------------------------- */

static void create_1x1_texture(WGPUDevice device, const uint8_t pixel[4],
                               WGPUTextureFormat format, WGPUTexture* out_tex,
                               WGPUTextureView* out_view)
{
  WGPUTextureDescriptor desc = {
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .size          = {1, 1, 1},
    .format        = format,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  *out_tex = wgpuDeviceCreateTexture(device, &desc);

  WGPUTexelCopyTextureInfo dst     = {.texture = *out_tex};
  WGPUTexelCopyBufferLayout layout = {.bytesPerRow = 4, .rowsPerImage = 1};
  WGPUExtent3D size                = {1, 1, 1};
  wgpuQueueWriteTexture(wgpuDeviceGetQueue(device), &dst, pixel, 4, &layout,
                        &size);
  *out_view = wgpuTextureCreateView(*out_tex, NULL);
}

static void create_default_textures(WGPUDevice device)
{
  const uint8_t white[4]  = {255, 255, 255, 255};
  const uint8_t normal[4] = {128, 128, 255, 255};

  /* 1x1 white sRGB */
  create_1x1_texture(device, white, WGPUTextureFormat_RGBA8UnormSrgb,
                     &state.gpu.default_srgb_texture,
                     &state.gpu.default_srgb_view);

  /* 1x1 white UNorm */
  create_1x1_texture(device, white, WGPUTextureFormat_RGBA8Unorm,
                     &state.gpu.default_unorm_texture,
                     &state.gpu.default_unorm_view);

  /* 1x1 flat normal */
  create_1x1_texture(device, normal, WGPUTextureFormat_RGBA8Unorm,
                     &state.gpu.default_normal_texture,
                     &state.gpu.default_normal_view);

  /* 1x1x6 white cube */
  {
    WGPUTextureDescriptor desc = {
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .size   = {1, 1, 6},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.gpu.default_cube_texture = wgpuDeviceCreateTexture(device, &desc);

    WGPUTexelCopyBufferLayout layout = {.bytesPerRow = 4, .rowsPerImage = 1};
    WGPUExtent3D size                = {1, 1, 1};
    for (uint32_t face = 0; face < 6; ++face) {
      WGPUTexelCopyTextureInfo dst = {
        .texture = state.gpu.default_cube_texture,
        .origin  = {0, 0, face},
      };
      wgpuQueueWriteTexture(wgpuDeviceGetQueue(device), &dst, white, 4, &layout,
                            &size);
    }
    WGPUTextureViewDescriptor vd = {
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .dimension       = WGPUTextureViewDimension_Cube,
      .mipLevelCount   = 1,
      .arrayLayerCount = 6,
    };
    state.gpu.default_cube_view
      = wgpuTextureCreateView(state.gpu.default_cube_texture, &vd);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void create_bind_group_layouts(WGPUDevice device)
{
  /* Global bind group layout (group 0): 7 entries */
  WGPUBindGroupLayoutEntry global_entries[7] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(global_uniforms_t)},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_Cube},
    },
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_Cube},
    },
    [4] = {
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_Cube},
    },
    [5] = {
      .binding    = 5,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [6] = {
      .binding    = 6,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
  };
  WGPUBindGroupLayoutDescriptor global_desc = {
    .entryCount = ARRAY_SIZE(global_entries),
    .entries    = global_entries,
  };
  state.gpu.global_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(device, &global_desc);

  /* Model bind group layout (group 1): 8 entries */
  WGPUBindGroupLayoutEntry model_entries[8] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(model_uniforms_t)},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(material_uniforms_t)},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
  };
  /* Bindings 3..7: material textures (2D, float) */
  for (int t = 0; t < 5; ++t) {
    model_entries[3 + t] = (WGPUBindGroupLayoutEntry){
      .binding    = (uint32_t)(3 + t),
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    };
  }
  WGPUBindGroupLayoutDescriptor model_desc = {
    .entryCount = ARRAY_SIZE(model_entries),
    .entries    = model_entries,
  };
  state.gpu.model_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(device, &model_desc);
}

/* -------------------------------------------------------------------------- *
 * Samplers
 * -------------------------------------------------------------------------- */

static void create_samplers(WGPUDevice device)
{
  WGPUSamplerDescriptor sd = {
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .maxAnisotropy = 1,
  };
  state.gpu.model_texture_sampler = wgpuDeviceCreateSampler(device, &sd);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void create_uniform_buffers(WGPUDevice device)
{
  /* Global uniforms */
  {
    WGPUBufferDescriptor bd = {
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(global_uniforms_t),
    };
    state.gpu.global_uniform_buffer = wgpuDeviceCreateBuffer(device, &bd);
  }

  /* Model uniforms */
  {
    WGPUBufferDescriptor bd = {
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(model_uniforms_t),
    };
    state.gpu.model_uniform_buffer = wgpuDeviceCreateBuffer(device, &bd);
  }
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void create_depth_texture(wgpu_context_t* ctx, uint32_t w, uint32_t h)
{
  /* Release old */
  if (state.gpu.depth_texture_view) {
    wgpuTextureViewRelease(state.gpu.depth_texture_view);
    state.gpu.depth_texture_view = NULL;
  }
  if (state.gpu.depth_texture) {
    wgpuTextureDestroy(state.gpu.depth_texture);
    wgpuTextureRelease(state.gpu.depth_texture);
    state.gpu.depth_texture = NULL;
  }

  WGPUTextureDescriptor td = {
    .usage         = WGPUTextureUsage_RenderAttachment,
    .size          = {w, h, 1},
    .format        = WGPUTextureFormat_Depth24PlusStencil8,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.gpu.depth_texture = wgpuDeviceCreateTexture(ctx->device, &td);
  state.gpu.depth_texture_view
    = wgpuTextureCreateView(state.gpu.depth_texture, NULL);
}

/* -------------------------------------------------------------------------- *
 * Global bind group (uses IBL textures or defaults)
 * -------------------------------------------------------------------------- */

static void create_global_bind_group(wgpu_context_t* ctx)
{
  /* Release old */
  WGPU_RELEASE_RESOURCE(BindGroup, state.gpu.global_bind_group)

  WGPUTextureView env_view  = state.environment_loaded ?
                                state.ibl.environment_view :
                                state.gpu.default_cube_view;
  WGPUTextureView irr_view  = state.environment_loaded ?
                                state.ibl.irradiance_view :
                                state.gpu.default_cube_view;
  WGPUTextureView spec_view = state.environment_loaded ?
                                state.ibl.prefiltered_view :
                                state.gpu.default_cube_view;
  WGPUTextureView brdf_view = state.environment_loaded ?
                                state.ibl.brdf_lut_view :
                                state.gpu.default_unorm_view;
  WGPUSampler env_sampler   = state.environment_loaded ?
                                state.ibl.environment_sampler :
                                state.gpu.model_texture_sampler;
  WGPUSampler brdf_sampler  = state.environment_loaded ?
                                state.ibl.brdf_lut_sampler :
                                state.gpu.model_texture_sampler;

  WGPUBindGroupEntry entries[7] = {
    [0] = {.binding = 0,
           .buffer  = state.gpu.global_uniform_buffer,
           .size    = sizeof(global_uniforms_t)},
    [1] = {.binding = 1, .sampler = env_sampler},
    [2] = {.binding = 2, .textureView = env_view},
    [3] = {.binding = 3, .textureView = irr_view},
    [4] = {.binding = 4, .textureView = spec_view},
    [5] = {.binding = 5, .textureView = brdf_view},
    [6] = {.binding = 6, .sampler = brdf_sampler},
  };
  WGPUBindGroupDescriptor desc = {
    .layout     = state.gpu.global_bind_group_layout,
    .entryCount = ARRAY_SIZE(entries),
    .entries    = entries,
  };
  state.gpu.global_bind_group = wgpuDeviceCreateBindGroup(ctx->device, &desc);
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void create_model_render_pipelines(wgpu_context_t* ctx)
{
  WGPUDevice device = ctx->device;

  /* Shader module */
  state.gpu.model_shader_module
    = wgpu_create_shader_module(device, gltf_pbr_shader_wgsl);

  /* Vertex buffer layout matching gltf_vertex_t */
  WGPUVertexAttribute vertex_attrs[] = {
    {.format         = WGPUVertexFormat_Float32x3,
     .offset         = offsetof(gltf_vertex_t, position),
     .shaderLocation = 0},
    {.format         = WGPUVertexFormat_Float32x3,
     .offset         = offsetof(gltf_vertex_t, normal),
     .shaderLocation = 1},
    {.format         = WGPUVertexFormat_Float32x4,
     .offset         = offsetof(gltf_vertex_t, tangent),
     .shaderLocation = 2},
    {.format         = WGPUVertexFormat_Float32x2,
     .offset         = offsetof(gltf_vertex_t, uv0),
     .shaderLocation = 3},
    {.format         = WGPUVertexFormat_Float32x2,
     .offset         = offsetof(gltf_vertex_t, uv1),
     .shaderLocation = 4},
    {.format         = WGPUVertexFormat_Float32x4,
     .offset         = offsetof(gltf_vertex_t, color),
     .shaderLocation = 5},
  };

  WGPUVertexBufferLayout vbl = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  WGPUColorTargetState color_target = {
    .format    = ctx->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUFragmentState fragment = {
    .module      = state.gpu.model_shader_module,
    .entryPoint  = STRVIEW("fs_main"),
    .targetCount = 1,
    .targets     = &color_target,
  };

  WGPUDepthStencilState depth_stencil = {
    .format            = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled = true,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  WGPUBindGroupLayout layouts[2] = {
    state.gpu.global_bind_group_layout,
    state.gpu.model_bind_group_layout,
  };

  WGPUPipelineLayoutDescriptor pl_desc = {
    .bindGroupLayoutCount = 2,
    .bindGroupLayouts     = layouts,
  };
  WGPUPipelineLayout pipeline_layout
    = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPURenderPipelineDescriptor rp_desc = {
    .layout    = pipeline_layout,
    .vertex    = {
      .module      = state.gpu.model_shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &vbl,
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .depthStencil = &depth_stencil,
    .fragment     = &fragment,
    .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
  };

  /* Opaque pipeline */
  state.gpu.model_pipeline_opaque
    = wgpuDeviceCreateRenderPipeline(device, &rp_desc);

  /* Transparent pipeline: enable blending, disable depth write */
  WGPUBlendComponent blend_comp = {
    .operation = WGPUBlendOperation_Add,
    .srcFactor = WGPUBlendFactor_SrcAlpha,
    .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
  };
  WGPUBlendState blend_state = {
    .color = blend_comp,
    .alpha = blend_comp,
  };
  color_target.blend              = &blend_state;
  depth_stencil.depthWriteEnabled = false;

  state.gpu.model_pipeline_transparent
    = wgpuDeviceCreateRenderPipeline(device, &rp_desc);

  wgpuPipelineLayoutRelease(pipeline_layout);
}

static void create_environment_pipeline(wgpu_context_t* ctx)
{
  WGPUDevice device = ctx->device;

  state.gpu.env_shader_module
    = wgpu_create_shader_module(device, environment_shader_wgsl);

  WGPUColorTargetState color_target = {
    .format    = ctx->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUFragmentState fragment = {
    .module      = state.gpu.env_shader_module,
    .entryPoint  = STRVIEW("fs_main"),
    .targetCount = 1,
    .targets     = &color_target,
  };

  WGPUDepthStencilState depth_stencil = {
    .format            = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled = false,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  WGPUPipelineLayoutDescriptor pl_desc = {
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = &state.gpu.global_bind_group_layout,
  };
  WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPURenderPipelineDescriptor rp_desc = {
    .layout    = layout,
    .vertex    = {
      .module     = state.gpu.env_shader_module,
      .entryPoint = STRVIEW("vs_main"),
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .depthStencil = &depth_stencil,
    .fragment     = &fragment,
    .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
  };

  state.gpu.env_pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
  wgpuPipelineLayoutRelease(layout);
}

/* -------------------------------------------------------------------------- *
 * Model texture creation (with mipmap support)
 * -------------------------------------------------------------------------- */

static WGPUTexture create_model_texture(wgpu_context_t* ctx,
                                        const gltf_texture_t* tex,
                                        WGPUTextureFormat format)
{
  if (!tex || !tex->data || tex->width == 0 || tex->height == 0) {
    return NULL;
  }

  uint32_t w         = tex->width;
  uint32_t h         = tex->height;
  uint32_t mip_count = wgpu_texture_mip_level_count(w, h);

  bool is_srgb = (format == WGPUTextureFormat_RGBA8UnormSrgb);

  if (is_srgb) {
    /* SRGB textures: create directly with RenderAttachment for render-based
     * mipmap generation. The GPU handles sRGB↔linear conversion during
     * sampling and render target writes automatically. */
    WGPUTextureDescriptor td = {
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_RenderAttachment,
      .size          = {w, h, 1},
      .format        = format,
      .mipLevelCount = mip_count,
      .sampleCount   = 1,
    };
    WGPUTexture texture = wgpuDeviceCreateTexture(ctx->device, &td);

    /* Upload level 0 */
    WGPUTexelCopyTextureInfo dst_info = {.texture = texture, .mipLevel = 0};
    WGPUTexelCopyBufferLayout src_layout
      = {.bytesPerRow = 4 * w, .rowsPerImage = h};
    WGPUExtent3D extent = {w, h, 1};
    wgpuQueueWriteTexture(ctx->queue, &dst_info, tex->data, (size_t)4 * w * h,
                          &src_layout, &extent);

    /* Generate mipmaps via render passes */
    wgpu_generate_mipmaps(ctx, texture, WGPU_MIPMAP_VIEW_2D);
    return texture;
  }
  else {
    /* UNORM textures (metallic-roughness, normal, occlusion): use a two-stage
     * approach matching the C++ reference. Create an intermediate texture with
     * RenderAttachment for mipmap generation, then copy all mips to the final
     * texture which only has TextureBinding | CopyDst. This makes RenderDoc
     * correctly classify these as '2D Image' instead of '2D Color Attachment'.
     */

    /* Stage 1: intermediate texture with RenderAttachment for mipmap gen */
    WGPUTextureDescriptor tmp_desc = {
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_CopySrc | WGPUTextureUsage_RenderAttachment,
      .size          = {w, h, 1},
      .format        = format,
      .mipLevelCount = mip_count,
      .sampleCount   = 1,
    };
    WGPUTexture tmp_tex = wgpuDeviceCreateTexture(ctx->device, &tmp_desc);

    /* Upload level 0 to intermediate */
    WGPUTexelCopyTextureInfo dst_info = {.texture = tmp_tex, .mipLevel = 0};
    WGPUTexelCopyBufferLayout src_layout
      = {.bytesPerRow = 4 * w, .rowsPerImage = h};
    WGPUExtent3D extent = {w, h, 1};
    wgpuQueueWriteTexture(ctx->queue, &dst_info, tex->data, (size_t)4 * w * h,
                          &src_layout, &extent);

    /* Generate mipmaps on intermediate */
    wgpu_generate_mipmaps(ctx, tmp_tex, WGPU_MIPMAP_VIEW_2D);

    /* Stage 2: final texture without RenderAttachment */
    WGPUTextureDescriptor final_desc = {
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .size   = {w, h, 1},
      .format = format,
      .mipLevelCount = mip_count,
      .sampleCount   = 1,
    };
    WGPUTexture final_tex = wgpuDeviceCreateTexture(ctx->device, &final_desc);

    /* Copy all mip levels from intermediate to final */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
    uint32_t mip_w = w, mip_h = h;
    for (uint32_t mip = 0; mip < mip_count; ++mip) {
      WGPUTexelCopyTextureInfo src = {.texture = tmp_tex, .mipLevel = mip};
      WGPUTexelCopyTextureInfo dst = {.texture = final_tex, .mipLevel = mip};
      WGPUExtent3D mip_size        = {mip_w, mip_h, 1};
      wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &mip_size);
      if (mip_w > 1)
        mip_w /= 2;
      if (mip_h > 1)
        mip_h /= 2;
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    /* Release intermediate texture */
    wgpuTextureDestroy(tmp_tex);
    wgpuTextureRelease(tmp_tex);

    return final_tex;
  }
}

/* -------------------------------------------------------------------------- *
 * Model buffers + submeshes + materials (called after glb loading)
 * -------------------------------------------------------------------------- */

static void create_model_buffers(wgpu_context_t* ctx)
{
  gltf_model_t* m   = &state.model;
  WGPUDevice device = ctx->device;

  /* Vertex buffer */
  {
    size_t size             = m->vertex_count * sizeof(gltf_vertex_t);
    WGPUBufferDescriptor bd = {
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = size,
      .mappedAtCreation = true,
    };
    state.gpu.vertex_buffer = wgpuDeviceCreateBuffer(device, &bd);
    memcpy(wgpuBufferGetMappedRange(state.gpu.vertex_buffer, 0, size),
           m->vertices, size);
    wgpuBufferUnmap(state.gpu.vertex_buffer);
  }

  /* Index buffer */
  {
    size_t size             = m->index_count * sizeof(uint32_t);
    WGPUBufferDescriptor bd = {
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = size,
      .mappedAtCreation = true,
    };
    state.gpu.index_buffer = wgpuDeviceCreateBuffer(device, &bd);
    memcpy(wgpuBufferGetMappedRange(state.gpu.index_buffer, 0, size),
           m->indices, size);
    wgpuBufferUnmap(state.gpu.index_buffer);
  }
}

static void create_submeshes(void)
{
  gltf_model_t* m = &state.model;

  state.opaque_mesh_count      = 0;
  state.transparent_mesh_count = 0;

  for (uint32_t ni = 0; ni < m->linear_node_count; ++ni) {
    gltf_node_t* node = m->linear_nodes[ni];
    if (!node->mesh)
      continue;

    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t pi = 0; pi < mesh->primitive_count; ++pi) {
      gltf_primitive_t* prim = &mesh->primitives[pi];

      vec3 centroid;
      glm_vec3_add(prim->bb.min, prim->bb.max, centroid);
      glm_vec3_scale(centroid, 0.5f, centroid);

      viewer_sub_mesh_t sm = {
        .first_index    = prim->first_index,
        .index_count    = prim->index_count,
        .material_index = prim->material_index,
      };
      glm_vec3_copy(centroid, sm.centroid);

      int mat_idx         = prim->material_index;
      bool is_transparent = false;
      if (mat_idx >= 0 && (uint32_t)mat_idx < m->material_count) {
        is_transparent
          = (m->materials[mat_idx].alpha_mode == GltfAlphaMode_Blend);
      }

      if (is_transparent) {
        if (state.transparent_mesh_count < GLTF_VIEWER_MAX_SUBMESHES) {
          state.transparent_meshes[state.transparent_mesh_count++] = sm;
        }
      }
      else {
        if (state.opaque_mesh_count < GLTF_VIEWER_MAX_SUBMESHES) {
          state.opaque_meshes[state.opaque_mesh_count++] = sm;
        }
      }
    }
  }
}

static void create_materials(wgpu_context_t* ctx)
{
  gltf_model_t* m   = &state.model;
  WGPUDevice device = ctx->device;

  state.material_count = 0;
  if (m->material_count == 0)
    return;

  uint32_t count = m->material_count;
  if (count > GLTF_VIEWER_MAX_MATERIALS)
    count = GLTF_VIEWER_MAX_MATERIALS;
  state.material_count = count;

  for (uint32_t i = 0; i < count; ++i) {
    const gltf_material_t* src = &m->materials[i];
    viewer_material_t* dst     = &state.materials[i];

    /* Uniform buffer */
    {
      WGPUBufferDescriptor bd = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = sizeof(material_uniforms_t),
      };
      dst->uniform_buffer = wgpuDeviceCreateBuffer(device, &bd);

      /* Fill uniform data */
      glm_vec4_copy((float*)src->base_color_factor,
                    dst->uniforms.base_color_factor);
      glm_vec3_copy((float*)src->emissive_factor,
                    dst->uniforms.emissive_factor);
      dst->uniforms.metallic_factor    = src->metallic_factor;
      dst->uniforms.roughness_factor   = src->roughness_factor;
      dst->uniforms.normal_scale       = src->normal_scale;
      dst->uniforms.occlusion_strength = src->occlusion_strength;
      dst->uniforms.alpha_cutoff       = src->alpha_cutoff;
      dst->uniforms.alpha_mode         = (int32_t)src->alpha_mode;

      wgpuQueueWriteBuffer(ctx->queue, dst->uniform_buffer, 0, &dst->uniforms,
                           sizeof(material_uniforms_t));
    }

    /* Base color texture */
    if (src->base_color_tex_index >= 0
        && (uint32_t)src->base_color_tex_index < m->texture_count) {
      dst->base_color_texture
        = create_model_texture(ctx, &m->textures[src->base_color_tex_index],
                               WGPUTextureFormat_RGBA8UnormSrgb);
      if (!dst->base_color_texture) {
        printf("[gltf_viewer] WARN: base color texture %d failed to create\n",
               src->base_color_tex_index);
      }
    }

    /* Metallic-roughness texture */
    if (src->metallic_roughness_tex_index >= 0
        && (uint32_t)src->metallic_roughness_tex_index < m->texture_count) {
      dst->metallic_roughness_texture = create_model_texture(
        ctx, &m->textures[src->metallic_roughness_tex_index],
        WGPUTextureFormat_RGBA8Unorm);
    }

    /* Normal texture */
    if (src->normal_tex_index >= 0
        && (uint32_t)src->normal_tex_index < m->texture_count) {
      dst->normal_texture = create_model_texture(
        ctx, &m->textures[src->normal_tex_index], WGPUTextureFormat_RGBA8Unorm);
    }

    /* Occlusion texture */
    if (src->occlusion_tex_index >= 0
        && (uint32_t)src->occlusion_tex_index < m->texture_count) {
      dst->occlusion_texture
        = create_model_texture(ctx, &m->textures[src->occlusion_tex_index],
                               WGPUTextureFormat_RGBA8Unorm);
    }

    /* Emissive texture */
    if (src->emissive_tex_index >= 0
        && (uint32_t)src->emissive_tex_index < m->texture_count) {
      dst->emissive_texture
        = create_model_texture(ctx, &m->textures[src->emissive_tex_index],
                               WGPUTextureFormat_RGBA8UnormSrgb);
    }

    /* Create material bind group */
    WGPUTextureView bc_view
      = dst->base_color_texture ?
          wgpuTextureCreateView(dst->base_color_texture, NULL) :
          state.gpu.default_srgb_view;
    WGPUTextureView mr_view
      = dst->metallic_roughness_texture ?
          wgpuTextureCreateView(dst->metallic_roughness_texture, NULL) :
          state.gpu.default_unorm_view;
    WGPUTextureView nm_view
      = dst->normal_texture ? wgpuTextureCreateView(dst->normal_texture, NULL) :
                              state.gpu.default_normal_view;
    WGPUTextureView ao_view
      = dst->occlusion_texture ?
          wgpuTextureCreateView(dst->occlusion_texture, NULL) :
          state.gpu.default_unorm_view;
    WGPUTextureView em_view
      = dst->emissive_texture ?
          wgpuTextureCreateView(dst->emissive_texture, NULL) :
          state.gpu.default_srgb_view;

    WGPUBindGroupEntry entries[8] = {
      [0] = {.binding = 0,
             .buffer  = state.gpu.model_uniform_buffer,
             .size    = sizeof(model_uniforms_t)},
      [1] = {.binding = 1,
             .buffer  = dst->uniform_buffer,
             .size    = sizeof(material_uniforms_t)},
      [2] = {.binding = 2, .sampler = state.gpu.model_texture_sampler},
      [3] = {.binding = 3, .textureView = bc_view},
      [4] = {.binding = 4, .textureView = mr_view},
      [5] = {.binding = 5, .textureView = nm_view},
      [6] = {.binding = 6, .textureView = ao_view},
      [7] = {.binding = 7, .textureView = em_view},
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = state.gpu.model_bind_group_layout,
      .entryCount = ARRAY_SIZE(entries),
      .entries    = entries,
    };
    dst->bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

    /* Release temporary views (they are now referenced by the bind group) */
    if (dst->base_color_texture && bc_view != state.gpu.default_srgb_view)
      wgpuTextureViewRelease(bc_view);
    if (dst->metallic_roughness_texture
        && mr_view != state.gpu.default_unorm_view)
      wgpuTextureViewRelease(mr_view);
    if (dst->normal_texture && nm_view != state.gpu.default_normal_view)
      wgpuTextureViewRelease(nm_view);
    if (dst->occlusion_texture && ao_view != state.gpu.default_unorm_view)
      wgpuTextureViewRelease(ao_view);
    if (dst->emissive_texture && em_view != state.gpu.default_srgb_view)
      wgpuTextureViewRelease(em_view);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform update per frame
 * -------------------------------------------------------------------------- */

static void update_uniforms(wgpu_context_t* ctx)
{
  orbit_camera_t* cam = &state.camera;

  /* Global uniforms */
  global_uniforms_t gu;
  camera_get_view_matrix(cam, gu.view_matrix);
  camera_get_projection_matrix(cam, gu.projection_matrix);
  glm_mat4_inv(gu.view_matrix, gu.inverse_view_matrix);
  glm_mat4_inv(gu.projection_matrix, gu.inverse_projection_matrix);
  glm_vec3_copy(cam->position, gu.camera_position);
  gu._pad = 0.0f;

  wgpuQueueWriteBuffer(ctx->queue, state.gpu.global_uniform_buffer, 0, &gu,
                       sizeof(global_uniforms_t));

  /* Model uniforms */
  model_uniforms_t mu;
  glm_mat4_copy(state.model_transform, mu.model_matrix);

  /* Normal matrix = transpose(inverse(model_matrix[3x3])) as mat4 */
  mat3 normal_mat3;
  glm_mat4_pick3(mu.model_matrix, normal_mat3);
  glm_mat3_inv(normal_mat3, normal_mat3);
  glm_mat3_transpose(normal_mat3);

  glm_mat4_identity(mu.normal_matrix);
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r)
      mu.normal_matrix[c][r] = normal_mat3[c][r];

  wgpuQueueWriteBuffer(ctx->queue, state.gpu.model_uniform_buffer, 0, &mu,
                       sizeof(model_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Transparent mesh sorting (back-to-front)
 * -------------------------------------------------------------------------- */

static int depth_compare(const void* a, const void* b)
{
  const sub_mesh_depth_info_t* da = (const sub_mesh_depth_info_t*)a;
  const sub_mesh_depth_info_t* db = (const sub_mesh_depth_info_t*)b;
  if (da->depth < db->depth)
    return -1;
  if (da->depth > db->depth)
    return 1;
  return 0;
}

static void sort_transparent_meshes(void)
{
  mat4 view, model_view;
  camera_get_view_matrix(&state.camera, view);
  glm_mat4_mul(view, state.model_transform, model_view);

  state.transparent_sorted_count = 0;

  for (uint32_t i = 0; i < state.transparent_mesh_count; ++i) {
    viewer_sub_mesh_t* sm = &state.transparent_meshes[i];
    vec4 centroid4 = {sm->centroid[0], sm->centroid[1], sm->centroid[2], 1.0f};
    vec4 view_pos;
    glm_mat4_mulv(model_view, centroid4, view_pos);

    float depth = view_pos[2];
    if (depth < 0.0f) {
      state.transparent_sorted[state.transparent_sorted_count++]
        = (sub_mesh_depth_info_t){.depth = depth, .mesh_index = i};
    }
  }

  if (state.transparent_sorted_count > 1) {
    qsort(state.transparent_sorted, state.transparent_sorted_count,
          sizeof(sub_mesh_depth_info_t), depth_compare);
  }
}

/* -------------------------------------------------------------------------- *
 * Async file loading callbacks
 * -------------------------------------------------------------------------- */

static void glb_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    printf("[gltf_viewer] GLB file loaded: %zu bytes\n", response->data.size);

    if (gltf_model_load_from_memory(&state.model, response->data.ptr,
                                    response->data.size, "", 1.0f)) {
      state.model_loaded = true;
      state.glb_loaded   = true;
      printf(
        "[gltf_viewer] Model loaded: %u vertices, %u indices, "
        "%u materials, %u textures\n",
        state.model.vertex_count, state.model.index_count,
        state.model.material_count, state.model.texture_count);
    }
    else {
      printf("[gltf_viewer] ERROR: Failed to parse GLB file\n");
    }
  }
  else if (response->failed) {
    printf("[gltf_viewer] ERROR: Failed to fetch GLB file\n");
  }
}

static void hdr_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    printf("[gltf_viewer] HDR file loaded: %zu bytes\n", response->data.size);

    if (wgpu_environment_load_from_memory(&state.environment,
                                          response->data.ptr,
                                          (uint32_t)response->data.size)) {
      state.hdr_loaded = true;
      printf("[gltf_viewer] HDR loaded: %ux%u\n", state.environment.width,
             state.environment.height);
    }
    else {
      printf("[gltf_viewer] ERROR: Failed to parse HDR file\n");
    }
  }
  else if (response->failed) {
    printf("[gltf_viewer] ERROR: Failed to fetch HDR file\n");
  }
}

/* -------------------------------------------------------------------------- *
 * Process loaded assets (called each frame until all resources ready)
 * -------------------------------------------------------------------------- */

static void process_loaded_assets(wgpu_context_t* ctx)
{
  if (state.resources_ready)
    return;

  bool need_rebuild = false;

  /* Process HDR environment */
  if (state.hdr_loaded && !state.environment_loaded) {
    printf("[gltf_viewer] Processing IBL textures...\n");
    wgpu_ibl_textures_desc_t ibl_desc = {0};
    if (wgpu_ibl_textures_from_environment(ctx, &state.environment, &ibl_desc,
                                           &state.ibl)) {
      state.environment_loaded = true;
      wgpu_environment_release(&state.environment);
      create_global_bind_group(ctx);
      printf("[gltf_viewer] IBL textures created\n");
      need_rebuild = true;
    }
  }

  /* Process GLB model */
  if (state.glb_loaded && state.model_loaded
      && !state.model_resources_created) {
    state.model_resources_created = true;
    printf("[gltf_viewer] Creating model GPU resources...\n");

    /* Release previous model resources */
    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.vertex_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.index_buffer)

    create_model_buffers(ctx);
    create_submeshes();
    create_materials(ctx);

    /* Compute node base transform (first mesh-bearing node's world matrix).
     * The C++ reference bakes node transforms into vertex positions at load
     * time. Since gltf_model.c does NOT bake transforms, we must multiply
     * the first mesh node's world matrix into the model transform so that
     * vertices are rendered in world space. */
    glm_mat4_identity(state.node_base_transform);
    for (uint32_t ni = 0; ni < state.model.linear_node_count; ++ni) {
      gltf_node_t* nd = state.model.linear_nodes[ni];
      if (nd->mesh) {
        gltf_node_get_world_matrix(nd, state.node_base_transform);
        break;
      }
    }

    /* Initial model transform = node base transform (no rotation yet) */
    glm_mat4_copy(state.node_base_transform, state.model_transform);

    /* Position camera to view model (dimensions already in world space) */
    camera_reset_to_model(&state.camera, state.model.dimensions.min,
                          state.model.dimensions.max);

    printf(
      "[gltf_viewer] Model resources created: %u opaque, "
      "%u transparent meshes\n",
      state.opaque_mesh_count, state.transparent_mesh_count);
    need_rebuild = true;
  }

  if (state.model_resources_created && state.environment_loaded) {
    state.resources_ready = true;
  }

  UNUSED_VAR(need_rebuild);
}

/* -------------------------------------------------------------------------- *
 * GUI rendering
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* ctx)
{
  UNUSED_VAR(ctx);
  if (!state.settings.show_gui)
    return;

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){260.0f, 160.0f}, ImGuiCond_FirstUseEver);

  if (igBegin("glTF PBR Viewer", NULL, ImGuiWindowFlags_None)) {
    if (!state.resources_ready) {
      igText("Loading assets...");
      if (state.glb_loaded)
        igText("  Model: loaded");
      else
        igText("  Model: loading...");
      if (state.hdr_loaded)
        igText("  HDR: loaded");
      else
        igText("  HDR: loading...");
    }
    else {
      igText("Model: %u verts, %u tris", state.model.vertex_count,
             state.model.index_count / 3);
      igText("Materials: %u", state.material_count);
      igText("Meshes: %u opaque, %u transparent", state.opaque_mesh_count,
             state.transparent_mesh_count);
      igSeparator();
      igCheckbox("Animate", &state.animate_model);

      igSeparator();
      igText("Camera:");
      igText("  Pos: %.1f, %.1f, %.1f", state.camera.position[0],
             state.camera.position[1], state.camera.position[2]);

      if (igButton("Reset Camera", (ImVec2){0, 0})) {
        camera_reset_to_model(&state.camera, state.model.dimensions.min,
                              state.model.dimensions.max);
        state.rotation_angle = 0.0f;
      }
    }
  }
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* ctx,
                           const input_event_t* input_event)
{
  UNUSED_VAR(ctx);

  /* Forward to ImGui */
  imgui_overlay_handle_input(ctx, input_event);
  if (imgui_overlay_want_capture_mouse()) {
    state.mouse.tumble = false;
    state.mouse.pan    = false;
    return;
  }

  switch (input_event->type) {
    case INPUT_EVENT_TYPE_MOUSE_DOWN: {
      state.mouse.last_x = input_event->mouse_x;
      state.mouse.last_y = input_event->mouse_y;
      if (input_event->mouse_button == BUTTON_LEFT) {
        if (input_event->keys_down[KEY_LEFT_SHIFT]
            || input_event->keys_down[KEY_RIGHT_SHIFT]) {
          state.mouse.pan = true;
        }
        else {
          state.mouse.tumble = true;
        }
      }
      else if (input_event->mouse_button == BUTTON_MIDDLE) {
        state.mouse.pan = true;
      }
      break;
    }
    case INPUT_EVENT_TYPE_MOUSE_UP: {
      state.mouse.tumble = false;
      state.mouse.pan    = false;
      break;
    }
    case INPUT_EVENT_TYPE_MOUSE_MOVE: {
      if (state.mouse.tumble || state.mouse.pan) {
        float dx           = input_event->mouse_x - state.mouse.last_x;
        float dy           = input_event->mouse_y - state.mouse.last_y;
        state.mouse.last_x = input_event->mouse_x;
        state.mouse.last_y = input_event->mouse_y;

        if (state.mouse.tumble) {
          camera_tumble(&state.camera, (int)dx, (int)dy);
        }
        else if (state.mouse.pan) {
          camera_pan(&state.camera, (int)dx, (int)dy);
        }
      }
      break;
    }
    case INPUT_EVENT_TYPE_MOUSE_SCROLL: {
      camera_zoom(&state.camera, 0,
                  (int)(input_event->scroll_y * ZOOM_SCROLL_SENS));
      break;
    }
    case INPUT_EVENT_TYPE_KEY_DOWN: {
      if (input_event->key_code == KEY_A) {
        if (input_event->keys_down[KEY_LEFT_SHIFT]
            || input_event->keys_down[KEY_RIGHT_SHIFT]) {
          /* Reset model orientation */
          glm_mat4_identity(state.model_transform);
        }
        else {
          state.animate_model = !state.animate_model;
        }
      }
      else if (input_event->key_code == KEY_HOME) {
        if (state.model_loaded) {
          camera_reset_to_model(&state.camera, state.model.dimensions.min,
                                state.model.dimensions.max);
        }
      }
      break;
    }
    case INPUT_EVENT_TYPE_RESIZED: {
      camera_resize(&state.camera, input_event->window_width,
                    input_event->window_height);
      create_depth_texture(ctx, (uint32_t)input_event->window_width,
                           (uint32_t)input_event->window_height);
      break;
    }
    default:
      break;
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* ctx)
{
  /* Timer */
  stm_setup();
  state.last_frame_time = stm_now();

  /* Async file loading */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 2,
    .num_channels = 2,
    .num_lanes    = 1,
  });

  /* Allocate file buffers */
  state.glb_file_buffer = (uint8_t*)malloc(GLB_FILE_BUFFER_SIZE);
  state.hdr_file_buffer = (uint8_t*)malloc(HDR_FILE_BUFFER_SIZE);

  /* Camera */
  camera_init(&state.camera, ctx->width, ctx->height);

  /* Model transform */
  glm_mat4_identity(state.model_transform);

  /* Create GPU resources */
  create_default_textures(ctx->device);
  create_bind_group_layouts(ctx->device);
  create_samplers(ctx->device);
  create_uniform_buffers(ctx->device);
  create_depth_texture(ctx, (uint32_t)ctx->width, (uint32_t)ctx->height);
  create_global_bind_group(ctx);
  create_model_render_pipelines(ctx);
  create_environment_pipeline(ctx);

  /* Render pass descriptor */
  state.render_pass_descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  };

  /* Initialize GUI */
  imgui_overlay_init(ctx);

  /* Start async file loading */
  sfetch_send(&(sfetch_request_t){
    .path     = GLB_FILE_PATH,
    .callback = glb_fetch_callback,
    .buffer   = {.ptr = state.glb_file_buffer, .size = GLB_FILE_BUFFER_SIZE},
    .channel  = 0,
  });
  sfetch_send(&(sfetch_request_t){
    .path     = HDR_FILE_PATH,
    .callback = hdr_fetch_callback,
    .buffer   = {.ptr = state.hdr_file_buffer, .size = HDR_FILE_BUFFER_SIZE},
    .channel  = 1,
  });

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* ctx)
{
  if (!state.initialized)
    return EXIT_FAILURE;

  /* Pump async I/O */
  sfetch_dowork();

  /* Process loaded assets */
  process_loaded_assets(ctx);

  /* Timer */
  uint64_t now          = stm_now();
  float delta_time      = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;
  if (delta_time <= 0.0f || delta_time > 0.1f)
    delta_time = 1.0f / 60.0f;

  /* Animate model (turntable rotation around Y-axis) */
  if (state.model_loaded && state.animate_model) {
    state.rotation_angle += delta_time;
    if (state.rotation_angle > 2.0f * GLM_PIf)
      state.rotation_angle -= 2.0f * GLM_PIf;
  }
  if (state.model_loaded) {
    mat4 turntable;
    glm_rotate_make(turntable, -state.rotation_angle, (vec3){0.0f, 1.0f, 0.0f});
    glm_mat4_mul(turntable, state.node_base_transform, state.model_transform);
  }

  /* GUI */
  imgui_overlay_new_frame(ctx, delta_time);
  render_gui(ctx);

  /* Update uniforms */
  if (state.resources_ready) {
    update_uniforms(ctx);
    sort_transparent_meshes();
  }

  /* Update render pass attachments */
  state.color_attachment.view         = ctx->swapchain_view;
  state.depth_stencil_attachment.view = state.gpu.depth_texture_view;

  /* Create command encoder and begin render pass */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  if (state.resources_ready) {
    /* Set global bind group */
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.gpu.global_bind_group, 0,
                                      NULL);

    /* Draw environment skybox (fullscreen triangle) */
    wgpuRenderPassEncoderSetPipeline(rpass, state.gpu.env_pipeline);
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    /* Set vertex/index buffers */
    wgpuRenderPassEncoderSetVertexBuffer(rpass, 0, state.gpu.vertex_buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass, state.gpu.index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);

    /* Draw opaque meshes */
    wgpuRenderPassEncoderSetPipeline(rpass, state.gpu.model_pipeline_opaque);
    for (uint32_t i = 0; i < state.opaque_mesh_count; ++i) {
      viewer_sub_mesh_t* sm = &state.opaque_meshes[i];
      if (sm->material_index >= 0
          && (uint32_t)sm->material_index < state.material_count) {
        wgpuRenderPassEncoderSetBindGroup(
          rpass, 1, state.materials[sm->material_index].bind_group, 0, NULL);
        wgpuRenderPassEncoderDrawIndexed(rpass, sm->index_count, 1,
                                         sm->first_index, 0, 0);
      }
    }

    /* Draw transparent meshes (sorted back-to-front) */
    wgpuRenderPassEncoderSetPipeline(rpass,
                                     state.gpu.model_pipeline_transparent);
    for (uint32_t i = 0; i < state.transparent_sorted_count; ++i) {
      uint32_t mi           = state.transparent_sorted[i].mesh_index;
      viewer_sub_mesh_t* sm = &state.transparent_meshes[mi];
      if (sm->material_index >= 0
          && (uint32_t)sm->material_index < state.material_count) {
        wgpuRenderPassEncoderSetBindGroup(
          rpass, 1, state.materials[sm->material_index].bind_group, 0, NULL);
        wgpuRenderPassEncoderDrawIndexed(rpass, sm->index_count, 1,
                                         sm->first_index, 0, 0);
      }
    }
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass);

  /* Submit */
  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &cmd_buf);

  /* Release per-frame resources */
  wgpuRenderPassEncoderRelease(rpass);
  wgpuCommandBufferRelease(cmd_buf);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render GUI overlay (after queue submit) */
  imgui_overlay_render(ctx);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* ctx)
{
  UNUSED_VAR(ctx);

  /* GUI */
  imgui_overlay_shutdown();

  /* Async I/O */
  sfetch_shutdown();

  /* Free file buffers */
  free(state.glb_file_buffer);
  free(state.hdr_file_buffer);

  /* Release materials */
  for (uint32_t i = 0; i < state.material_count; ++i) {
    viewer_material_t* mat = &state.materials[i];
    WGPU_RELEASE_RESOURCE(Buffer, mat->uniform_buffer)
    WGPU_RELEASE_RESOURCE(BindGroup, mat->bind_group)
    if (mat->base_color_texture) {
      wgpuTextureDestroy(mat->base_color_texture);
      wgpuTextureRelease(mat->base_color_texture);
    }
    if (mat->metallic_roughness_texture) {
      wgpuTextureDestroy(mat->metallic_roughness_texture);
      wgpuTextureRelease(mat->metallic_roughness_texture);
    }
    if (mat->normal_texture) {
      wgpuTextureDestroy(mat->normal_texture);
      wgpuTextureRelease(mat->normal_texture);
    }
    if (mat->occlusion_texture) {
      wgpuTextureDestroy(mat->occlusion_texture);
      wgpuTextureRelease(mat->occlusion_texture);
    }
    if (mat->emissive_texture) {
      wgpuTextureDestroy(mat->emissive_texture);
      wgpuTextureRelease(mat->emissive_texture);
    }
  }

  /* Release GPU resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.gpu.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.gpu.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.gpu.global_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.gpu.model_uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.gpu.global_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.gpu.global_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.gpu.model_bind_group_layout)
  WGPU_RELEASE_RESOURCE(Sampler, state.gpu.model_texture_sampler)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gpu.env_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gpu.model_pipeline_opaque)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gpu.model_pipeline_transparent)
  WGPU_RELEASE_RESOURCE(ShaderModule, state.gpu.env_shader_module)
  WGPU_RELEASE_RESOURCE(ShaderModule, state.gpu.model_shader_module)

  /* Depth texture */
  if (state.gpu.depth_texture_view)
    wgpuTextureViewRelease(state.gpu.depth_texture_view);
  if (state.gpu.depth_texture) {
    wgpuTextureDestroy(state.gpu.depth_texture);
    wgpuTextureRelease(state.gpu.depth_texture);
  }

  /* Default textures */
  WGPU_RELEASE_RESOURCE(TextureView, state.gpu.default_srgb_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gpu.default_unorm_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gpu.default_normal_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gpu.default_cube_view)
  if (state.gpu.default_srgb_texture) {
    wgpuTextureDestroy(state.gpu.default_srgb_texture);
    wgpuTextureRelease(state.gpu.default_srgb_texture);
  }
  if (state.gpu.default_unorm_texture) {
    wgpuTextureDestroy(state.gpu.default_unorm_texture);
    wgpuTextureRelease(state.gpu.default_unorm_texture);
  }
  if (state.gpu.default_normal_texture) {
    wgpuTextureDestroy(state.gpu.default_normal_texture);
    wgpuTextureRelease(state.gpu.default_normal_texture);
  }
  if (state.gpu.default_cube_texture) {
    wgpuTextureDestroy(state.gpu.default_cube_texture);
    wgpuTextureRelease(state.gpu.default_cube_texture);
  }

  /* IBL textures */
  wgpu_ibl_textures_destroy(&state.ibl);

  /* Model */
  gltf_model_destroy(&state.model);
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);
  wgpu_start(&(wgpu_desc_t){
    .title          = "glTF PBR Viewer",
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
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
static const char* environment_shader_wgsl = CODE(
  struct GlobalUniforms {
    viewMatrix : mat4x4<f32>,
    projectionMatrix : mat4x4<f32>,
    inverseViewMatrix : mat4x4<f32>,
    inverseProjectionMatrix : mat4x4<f32>,
    cameraPositionWorld : vec3<f32>,
  };

  @group(0) @binding(0) var<uniform> globalUniforms : GlobalUniforms;
  @group(0) @binding(1) var environmentCubeSampler : sampler;
  @group(0) @binding(2) var environmentTexture : texture_cube<f32>;

  const pi = 3.141592653589793;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  };

  fn toneMapPBRNeutral(colorIn : vec3f) -> vec3f {
    let startCompression : f32 = 0.8 - 0.04;
    let desaturation : f32 = 0.15;
    let x : f32 = min(colorIn.r, min(colorIn.g, colorIn.b));
    let offset : f32 = select(0.04, x - 6.25 * x * x, x < 0.08);
    var color = colorIn - offset;
    let peak : f32 = max(color.r, max(color.g, color.b));
    if (peak < startCompression) { return color; }
    let d : f32 = 1.0 - startCompression;
    let newPeak : f32 = 1.0 - d * d / (peak + d - startCompression);
    color = color * (newPeak / peak);
    let g : f32 = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3f(1.0, 1.0, 1.0), g);
  }

  fn toneMap(colorIn : vec3f) -> vec3f {
    const gamma = 2.2;
    const invGamma = 1.0 / gamma;
    const exposure = 1.0;
    var color = colorIn * exposure;
    color = toneMapPBRNeutral(color);
    color = pow(color, vec3f(invGamma));
    return color;
  }

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var positions : array<vec2f, 3> = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f( 3.0, -1.0),
      vec2f(-1.0,  3.0)
    );
    var uvs : array<vec2f, 3> = array<vec2f, 3>(
      vec2f(0.0, 0.0),
      vec2f(2.0, 0.0),
      vec2f(0.0, 2.0)
    );
    var output : VertexOutput;
    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    output.uv = uvs[vertexIndex];
    return output;
  }

  @fragment
  fn fs_main(input : VertexOutput) -> @location(0) vec4f {
    let ndc = input.uv * 2.0 - 1.0;
    let viewSpacePos = globalUniforms.inverseProjectionMatrix * vec4f(ndc.xy, 1.0, 1.0);
    var dir = normalize(viewSpacePos.xyz);
    let invRotMatrix = mat3x3f(
      globalUniforms.inverseViewMatrix[0].xyz,
      globalUniforms.inverseViewMatrix[1].xyz,
      globalUniforms.inverseViewMatrix[2].xyz
    );
    dir = normalize(invRotMatrix * dir);
    let iblSample = textureSample(environmentTexture, environmentCubeSampler, dir).rgb;
    let color = toneMap(iblSample);
    return vec4f(color, 1.0);
  }
);

static const char* gltf_pbr_shader_wgsl = CODE(
  struct GlobalUniforms {
    viewMatrix : mat4x4<f32>,
    projectionMatrix : mat4x4<f32>,
    inverseViewMatrix : mat4x4<f32>,
    inverseProjectionMatrix : mat4x4<f32>,
    cameraPositionWorld : vec3<f32>,
  };

  struct ModelUniforms {
    modelMatrix : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
  };

  struct MaterialUniforms {
    baseColorFactor : vec4<f32>,
    emissiveFactor : vec3<f32>,
    metallicFactor : f32,
    roughnessFactor : f32,
    normalScale : f32,
    occlusionStrength : f32,
    alphaCutoff : f32,
    alphaMode : i32,
  };

  @group(0) @binding(0) var<uniform> globalUniforms : GlobalUniforms;
  @group(0) @binding(1) var iblSampler : sampler;
  @group(0) @binding(2) var environmentTexture : texture_cube<f32>;
  @group(0) @binding(3) var iblIrradianceTexture : texture_cube<f32>;
  @group(0) @binding(4) var iblSpecularTexture : texture_cube<f32>;
  @group(0) @binding(5) var iblBRDFIntegrationLUTTexture : texture_2d<f32>;
  @group(0) @binding(6) var iblBRDFIntegrationLUTSampler : sampler;

  @group(1) @binding(0) var<uniform> modelUniforms : ModelUniforms;
  @group(1) @binding(1) var<uniform> materialUniforms : MaterialUniforms;
  @group(1) @binding(2) var textureSampler : sampler;
  @group(1) @binding(3) var baseColorTexture : texture_2d<f32>;
  @group(1) @binding(4) var metallicRoughnessTexture : texture_2d<f32>;
  @group(1) @binding(5) var normalTexture : texture_2d<f32>;
  @group(1) @binding(6) var occlusionTexture : texture_2d<f32>;
  @group(1) @binding(7) var emissiveTexture : texture_2d<f32>;

  const pi = 3.141592653589793;

  struct MaterialInfo {
    baseColor : vec4f,
    metallic : f32,
    perceptualRoughness : f32,
    f0_dielectric : vec3f,
    alphaRoughness : f32,
    f0 : vec3f,
    f90 : vec3f,
    cDiffuse : vec3f,
    specularWeight : f32,
  };

  struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) tangent : vec4<f32>,
    @location(3) texCoord0 : vec2<f32>,
    @location(4) texCoord1 : vec2<f32>,
    @location(5) color : vec4<f32>,
  };

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) texCoord0 : vec2<f32>,
    @location(2) texCoord1 : vec2<f32>,
    @location(3) normalWorld : vec3<f32>,
    @location(4) tangentWorld : vec4<f32>,
    @location(5) viewDirectionWorld : vec3<f32>,
  };

  fn clampedDot(a : vec3f, b : vec3f) -> f32 {
    return clamp(dot(a, b), 0.0, 1.0);
  }

  fn getNormal(in : VertexOutput) -> vec3f {
    let N = normalize(in.normalWorld);
    let T = normalize(in.tangentWorld.xyz);
    let B = cross(N, T) * in.tangentWorld.w;
    let TBN = mat3x3f(T, B, N);
    var sampledNormal = textureSample(normalTexture, textureSampler, in.texCoord0).xyz * 2.0 - 1.0;
    sampledNormal *= materialUniforms.normalScale;
    return normalize(TBN * sampledNormal);
  }

  fn FSchlick(f0 : vec3f, f90 : vec3f, vDotH : f32) -> vec3f {
    return f0 + (f90 - f0) * pow(clamp(1.0 - vDotH, 0.0, 1.0), 5.0);
  }

  fn BRDFLambertian(f0 : vec3f, f90 : vec3f, diffuseColor : vec3f, specularWeight : f32, vDotH : f32) -> vec3f {
    return (1.0 - specularWeight * FSchlick(f0, f90, vDotH)) * (diffuseColor / pi);
  }

  fn samplePrefilteredSpecularIBL(reflection : vec3<f32>, lod : f32) -> vec4<f32> {
    let sampleColor = textureSampleLevel(iblSpecularTexture, iblSampler, reflection, lod);
    return sampleColor;
  }

  fn getIBLRadianceGGX(n : vec3<f32>, v : vec3<f32>, roughness : f32) -> vec3<f32> {
    let NdotV = max(dot(n, v), 0.0);
    let lod = roughness * (f32(10) - 1.0);
    let reflection = normalize(reflect(-v, n));
    let specularSample = samplePrefilteredSpecularIBL(reflection, lod);
    return specularSample.rgb;
  }

  fn getIBLGGXFresnel(n : vec3<f32>, v : vec3<f32>, roughness : f32, F0 : vec3<f32>, specularWeight : f32) -> vec3<f32> {
    let NdotV = max(dot(n, v), 0.0);
    let brdfLUTCoords = vec2<f32>(NdotV, roughness);
    let brdfLUTSample = textureSample(iblBRDFIntegrationLUTTexture, iblBRDFIntegrationLUTSampler, brdfLUTCoords);
    let brdfLUT = brdfLUTSample.rg;
    let fresnelPivot = max(vec3<f32>(1.0 - roughness), F0) - F0;
    let fresnelSingleScatter = F0 + fresnelPivot * pow(1.0 - NdotV, 5.0);
    let FssEss = specularWeight * (fresnelSingleScatter * brdfLUT.x + brdfLUT.y);
    let Ems = 1.0 - (brdfLUT.x + brdfLUT.y);
    let F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
    let FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    return FssEss + FmsEms;
  }

  fn VGGX(nDotL : f32, nDotV : f32, alphaRoughness : f32) -> f32 {
    let a2 = alphaRoughness * alphaRoughness;
    let ggxV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    let ggxL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    let ggx = ggxV + ggxL;
    if (ggx > 0.0) { return 0.5 / ggx; }
    return 0.0;
  }

  fn DGGX(nDotH : f32, alphaRoughness : f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;
    let f = (nDotH * nDotH) * (alphaRoughnessSq - 1.0) + 1.0;
    return alphaRoughnessSq / (pi * f * f);
  }

  fn BRDFSpecularGGX(f0 : vec3f, f90 : vec3f, alphaRoughness : f32, specularWeight : f32, vDotH : f32, nDotL : f32, nDotV : f32, nDotH : f32) -> vec3f {
    let F = FSchlick(f0, f90, vDotH);
    let V = VGGX(nDotL, nDotV, alphaRoughness);
    let D = DGGX(nDotH, alphaRoughness);
    return specularWeight * F * V * D;
  }

  fn toneMapPBRNeutral(colorIn : vec3f) -> vec3f {
    let startCompression : f32 = 0.8 - 0.04;
    let desaturation : f32 = 0.15;
    let x : f32 = min(colorIn.r, min(colorIn.g, colorIn.b));
    let offset : f32 = select(0.04, x - 6.25 * x * x, x < 0.08);
    var color = colorIn - offset;
    let peak : f32 = max(color.r, max(color.g, color.b));
    if (peak < startCompression) { return color; }
    let d : f32 = 1.0 - startCompression;
    let newPeak : f32 = 1.0 - d * d / (peak + d - startCompression);
    color = color * (newPeak / peak);
    let g : f32 = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3f(1.0, 1.0, 1.0), g);
  }

  fn toneMap(colorIn : vec3f) -> vec3f {
    const gamma = 2.2;
    const invGamma = 1.0 / gamma;
    const exposure = 1.0;
    var color = colorIn * exposure;
    color = toneMapPBRNeutral(color);
    color = pow(color, vec3f(invGamma));
    return color;
  }

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    let worldPosition = modelUniforms.modelMatrix * vec4<f32>(in.position, 1.0);
    let worldNormal = normalize((modelUniforms.normalMatrix * vec4<f32>(in.normal, 0.0)).xyz);
    let worldTangent = vec4<f32>(
      normalize((modelUniforms.normalMatrix * vec4<f32>(in.tangent.xyz, 0.0)).xyz),
      in.tangent.w
    );
    var output : VertexOutput;
    output.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * worldPosition;
    output.color = in.color;
    output.texCoord0 = in.texCoord0;
    output.texCoord1 = in.texCoord1;
    output.normalWorld = worldNormal;
    output.tangentWorld = worldTangent;
    output.viewDirectionWorld = globalUniforms.cameraPositionWorld - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let baseColor = textureSample(baseColorTexture, textureSampler, in.texCoord0).rgba;
    let metallicRoughness = textureSample(metallicRoughnessTexture, textureSampler, in.texCoord0).rgb;

    var materialInfo : MaterialInfo;
    materialInfo.baseColor = baseColor * in.color * materialUniforms.baseColorFactor;
    materialInfo.metallic = metallicRoughness.b * materialUniforms.metallicFactor;
    materialInfo.perceptualRoughness = metallicRoughness.g * materialUniforms.roughnessFactor;
    materialInfo.f0_dielectric = vec3f(0.04);
    materialInfo.specularWeight = 1.0;
    materialInfo.alphaRoughness = metallicRoughness.g * metallicRoughness.g;
    materialInfo.f0 = mix(vec3f(0.04), materialInfo.baseColor.rgb, materialInfo.metallic);
    materialInfo.f90 = vec3f(1.0);
    materialInfo.cDiffuse = mix(materialInfo.baseColor.rgb * 0.5, vec3f(0.0), materialInfo.metallic);

    let n = getNormal(in);
    let v = normalize(in.viewDirectionWorld);

    var color = vec3f(0.0);

    {
      let diffuseEnv = textureSample(iblIrradianceTexture, iblSampler, in.normalWorld).rgb;
      let iblDiffuse = diffuseEnv * materialInfo.baseColor.rgb;
      let iblSpecular = getIBLRadianceGGX(n, v, materialInfo.perceptualRoughness);
      let fresnelDielectric = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness, materialInfo.f0_dielectric, materialInfo.specularWeight);
      let iblDielectric = mix(iblDiffuse, iblSpecular, fresnelDielectric);
      let fresnelMetal = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness, materialInfo.baseColor.rgb, 1.0);
      let iblMetal = fresnelMetal * iblSpecular;
      color += mix(iblDielectric, iblMetal, materialInfo.metallic);
    }

    let ao = textureSample(occlusionTexture, textureSampler, in.texCoord0).r * materialUniforms.occlusionStrength;
    color *= vec3f(ao);

    var emissive = textureSample(emissiveTexture, textureSampler, in.texCoord0).rgb;
    emissive *= materialUniforms.emissiveFactor;
    color += emissive;

    if (materialUniforms.alphaMode == 1) {
      if (baseColor.a < materialUniforms.alphaCutoff) { discard; }
    }

    color = toneMap(color);

    var alpha = select(materialInfo.baseColor.a, 1.0, materialUniforms.alphaMode == 0);
    return vec4f(color, alpha);
  }
);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// clang-format on
