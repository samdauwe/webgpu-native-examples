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

/* PBR workflow */
#include "webgpu/pbr.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define GLTF_VIEWER_MAX_MATERIALS 64
#define GLTF_VIEWER_MAX_SUBMESHES 256
#define GLTF_VIEWER_MAX_TEXTURES 64
#define GLTF_VIEWER_MAX_EXTERNAL_IMAGES 32

/* File paths — change MODEL_FILE_PATH to load a different model */
#define MODEL_FILE_PATH "assets/models/DamagedHelmet.glb"
#define HDR_FILE_PATH "assets/textures/venice_sunset_1k.hdr"

/* sokol_fetch configuration — tuned for parallel external resource loading */
#define SFETCH_MAX_REQUESTS 48
#define SFETCH_NUM_CHANNELS 4
#define SFETCH_NUM_LANES 8

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
  mat4 view_matrix;                  /* offset 0   */
  mat4 projection_matrix;            /* offset 64  */
  mat4 inverse_view_matrix;          /* offset 128 */
  mat4 inverse_projection_matrix;    /* offset 192 */
  vec3 camera_position;              /* offset 256 */
  float exposure;                    /* offset 268 */
  vec4 light_dir;                    /* offset 272 */
  float gamma;                       /* offset 288 */
  float prefiltered_cube_mip_levels; /* offset 292 */
  float scale_ibl_ambient;           /* offset 296 */
  float debug_view_inputs;           /* offset 300 */
  float debug_view_equation;         /* offset 304 */
  int32_t tone_mapping_type;         /* offset 308:
                                        0=PBRNeutral,1=Uncharted2,2=Reinhard,3=ACES */
  float _pad2[2];                    /* pad to 320 */
} global_uniforms_t;                 /* size: 320  */

typedef struct {
  mat4 model_matrix;  /* offset 0  */
  mat4 normal_matrix; /* offset 64 */
} model_uniforms_t;   /* size: 128 */

typedef struct {
  vec4 base_color_factor;   /* offset 0   */
  vec3 emissive_factor;     /* offset 16  */
  float metallic_factor;    /* offset 28  */
  float roughness_factor;   /* offset 32  */
  float normal_scale;       /* offset 36  */
  float occlusion_strength; /* offset 40  */
  float alpha_cutoff;       /* offset 44  */
  int32_t alpha_mode;       /* offset 48, 0=Opaque, 1=Mask, 2=Blend */
  float emissive_strength;  /* offset 52  */
  int32_t workflow; /* offset 56, 0=MetallicRoughness, 1=SpecGloss, 2=Unlit */
  int32_t double_sided; /* offset 60  */
  /* --- Clearcoat (KHR_materials_clearcoat) --- */
  float clearcoat_factor;    /* offset 64  */
  float clearcoat_roughness; /* offset 68  */
  /* --- Sheen (KHR_materials_sheen) --- */
  float sheen_roughness_factor; /* offset 72  */
  float _pad0;                  /* offset 76  */
  vec3 sheen_color_factor;      /* offset 80  */
  float _pad1;                  /* offset 92, pad to 96 */
} material_uniforms_t;          /* size: 96  */

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
  WGPUBool hdr_loaded;
  WGPUBool resources_ready;
  WGPUBool model_resources_created;

  /* File type detection */
  bool is_gltf; /* true = .gltf (separate files), false = .glb (single file) */

  /* ---- GLB loading (single-file path) ---- */
  WGPUBool glb_loaded;
  uint8_t* glb_file_buffer;
  size_t glb_file_buffer_size;

  /* ---- glTF multi-phase loading state ---- */
  struct {
    /* Phase 1: main .gltf JSON */
    uint8_t* json_buffer;
    size_t json_buffer_size;
    bool json_loaded;

    /* Phase 2: external .bin buffer(s) */
    uint8_t* bin_buffer;
    size_t bin_buffer_size;
    bool bin_loaded;
    bool has_external_bin; /* false if buffer is embedded (data URI) */
    char bin_uri[GLTF_MODEL_MAX_URI_LENGTH];

    /* Phase 3: external image files */
    struct {
      char path[GLTF_MODEL_MAX_URI_LENGTH];
      uint8_t* buffer;
      size_t buffer_size;
      bool loaded;
      uint32_t texture_index; /* index into model->textures[] */
    } images[GLTF_VIEWER_MAX_EXTERNAL_IMAGES];
    uint32_t num_images;
    uint32_t num_images_loaded;

    /* Geometry loaded flag (after .bin available) */
    bool geometry_loaded;
  } gltf;

  /* HDR environment loading */
  uint8_t* hdr_file_buffer;
  size_t hdr_file_buffer_size;

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

  /* Texture store: GPU textures for model, with progressive loading */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    bool created;
    WGPUTextureFormat format;
  } texture_store[GLTF_VIEWER_MAX_TEXTURES];
  uint32_t texture_store_count;
  uint32_t textures_uploaded; /* count of textures uploaded to GPU */

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

  /* PBR settings (GUI-controlled) */
  struct {
    float exposure;
    float gamma;
    float scale_ibl_ambient;
    float light_dir[4];        /* xyz = direction, w = unused */
    float debug_view_inputs;   /* 0=none, 1..6 = texture channels */
    float debug_view_equation; /* 0=none, 1..5 = BRDF terms */
    int tone_mapping_type;     /* 0=PBRNeutral, 1=Uncharted2, 2=Reinhard */
    bool enable_direct_light;  /* Toggle analytical directional light */
  } pbr;

  /* GUI */
  struct {
    bool show_gui;
  } settings;

} state = {
  .animate_model = true,
  .pbr = {
    .exposure           = 1.0f,
    .gamma              = 2.2f,
    .scale_ibl_ambient  = 1.0f,
    .light_dir          = {0.75f, 0.75f, 1.0f, 0.0f},
    .debug_view_inputs  = 0.0f,
    .debug_view_equation = 0.0f,
    .tone_mapping_type  = 0,
    .enable_direct_light = false,
  },
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
 * Pre-bake per-node world transforms into vertex positions
 *
 * glTF models can have per-node transforms (translation, rotation, scale)
 * that place each mesh part in the correct world-space position. Since the
 * viewer uses a single model matrix for all draw calls, we bake these
 * transforms into the CPU vertex data before creating GPU buffers.
 * -------------------------------------------------------------------------- */

static void apply_node_world_transforms(void)
{
  gltf_model_t* m = &state.model;
  if (!m->vertices || m->vertex_count == 0) {
    return;
  }

  /* Track which vertices have been transformed to avoid double-transforms */
  bool* transformed = (bool*)calloc(m->vertex_count, sizeof(bool));
  if (!transformed) {
    printf(
      "[gltf_viewer] WARNING: Could not allocate transform tracking "
      "array\n");
    return;
  }

  for (uint32_t ni = 0; ni < m->linear_node_count; ++ni) {
    gltf_node_t* node = m->linear_nodes[ni];
    if (!node->mesh) {
      continue;
    }

    /* Get this node's world matrix */
    mat4 world_mat;
    gltf_node_get_world_matrix(node, world_mat);

    /* Check if the world matrix is identity — skip if so */
    mat4 identity;
    glm_mat4_identity(identity);
    bool is_identity = true;
    for (int c = 0; c < 4 && is_identity; ++c) {
      for (int r = 0; r < 4 && is_identity; ++r) {
        if (fabsf(world_mat[c][r] - identity[c][r]) > 1e-6f) {
          is_identity = false;
        }
      }
    }
    if (is_identity) {
      continue;
    }

    /* Compute the normal matrix (transpose of inverse of upper-left 3x3) */
    mat3 normal_mat;
    glm_mat4_pick3(world_mat, normal_mat);
    glm_mat3_inv(normal_mat, normal_mat);
    glm_mat3_transpose(normal_mat);

    /* Transform vertices for each primitive of this node's mesh */
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t pi = 0; pi < mesh->primitive_count; ++pi) {
      gltf_primitive_t* prim = &mesh->primitives[pi];

      for (uint32_t ii = 0; ii < prim->index_count; ++ii) {
        uint32_t vi = m->indices[prim->first_index + ii];
        if (vi >= m->vertex_count || transformed[vi]) {
          continue;
        }
        transformed[vi] = true;

        gltf_vertex_t* vert = &m->vertices[vi];

        /* Transform position */
        vec4 pos4
          = {vert->position[0], vert->position[1], vert->position[2], 1.0f};
        vec4 result;
        glm_mat4_mulv(world_mat, pos4, result);
        vert->position[0] = result[0];
        vert->position[1] = result[1];
        vert->position[2] = result[2];

        /* Transform normal */
        vec3 n;
        glm_mat3_mulv(normal_mat, vert->normal, n);
        glm_vec3_normalize(n);
        glm_vec3_copy(n, vert->normal);

        /* Transform tangent (xyz only, w is handedness sign) */
        vec3 t = {vert->tangent[0], vert->tangent[1], vert->tangent[2]};
        vec3 t_out;
        glm_mat3_mulv(normal_mat, t, t_out);
        glm_vec3_normalize(t_out);
        vert->tangent[0] = t_out[0];
        vert->tangent[1] = t_out[1];
        vert->tangent[2] = t_out[2];
        /* vert->tangent[3] (handedness) remains unchanged */
      }
    }
  }

  free(transformed);

  /* Recompute scene dimensions from transformed vertices */
  vec3 scene_min = {FLT_MAX, FLT_MAX, FLT_MAX};
  vec3 scene_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  for (uint32_t vi = 0; vi < m->vertex_count; ++vi) {
    glm_vec3_minv(scene_min, m->vertices[vi].position, scene_min);
    glm_vec3_maxv(scene_max, m->vertices[vi].position, scene_max);
  }
  glm_vec3_copy(scene_min, m->dimensions.min);
  glm_vec3_copy(scene_max, m->dimensions.max);
  glm_vec3_sub(scene_max, scene_min, m->dimensions.size);
  glm_vec3_add(scene_min, scene_max, m->dimensions.center);
  glm_vec3_scale(m->dimensions.center, 0.5f, m->dimensions.center);
  m->dimensions.radius = glm_vec3_distance(scene_min, scene_max) * 0.5f;

  printf("[gltf_viewer] Applied per-node world transforms to vertices\n");
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

/* Forward declaration (defined after texture store helpers) */
static void rebuild_material_bind_group(wgpu_context_t* ctx, uint32_t mat_idx);

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
      dst->uniforms.emissive_strength
        = src->emissive_strength > 0.0f ? src->emissive_strength : 1.0f;
      dst->uniforms.workflow     = src->unlit ? 2 : 0; /* 0=MetRough, 2=Unlit */
      dst->uniforms.double_sided = src->double_sided ? 1 : 0;

      /* Clearcoat */
      dst->uniforms.clearcoat_factor    = src->clearcoat_factor;
      dst->uniforms.clearcoat_roughness = src->clearcoat_roughness_factor;

      /* Sheen */
      glm_vec3_copy((float*)src->sheen_color_factor,
                    dst->uniforms.sheen_color_factor);
      dst->uniforms.sheen_roughness_factor = src->sheen_roughness_factor;

      wgpuQueueWriteBuffer(ctx->queue, dst->uniform_buffer, 0, &dst->uniforms,
                           sizeof(material_uniforms_t));
    }

    /* Build bind group using texture store (with fallback to defaults) */
    rebuild_material_bind_group(ctx, i);
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
  memset(&gu, 0, sizeof(gu));
  camera_get_view_matrix(cam, gu.view_matrix);
  camera_get_projection_matrix(cam, gu.projection_matrix);
  glm_mat4_inv(gu.view_matrix, gu.inverse_view_matrix);
  glm_mat4_inv(gu.projection_matrix, gu.inverse_projection_matrix);
  glm_vec3_copy(cam->position, gu.camera_position);

  /* PBR parameters */
  gu.exposure = state.pbr.exposure;
  memcpy(gu.light_dir, state.pbr.light_dir, sizeof(vec4));
  /* Use w component to signal direct light on/off to shader */
  gu.light_dir[3]                = state.pbr.enable_direct_light ? 1.0f : -1.0f;
  gu.gamma                       = state.pbr.gamma;
  gu.prefiltered_cube_mip_levels = (float)state.ibl.prefiltered_mip_levels;
  gu.scale_ibl_ambient           = state.pbr.scale_ibl_ambient;
  gu.debug_view_inputs           = state.pbr.debug_view_inputs;
  gu.debug_view_equation         = state.pbr.debug_view_equation;
  gu.tone_mapping_type           = state.pbr.tone_mapping_type;

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
 * File type / path helpers
 * -------------------------------------------------------------------------- */

/**
 * Query the size of a file on disk using stat().
 * Returns the file size in bytes, or 0 on error.
 */
static size_t get_file_size(const char* path)
{
  struct stat st;
  if (stat(path, &st) != 0) {
    printf("[gltf_viewer] WARNING: Could not stat file '%s'\n", path);
    return 0;
  }
  return (size_t)st.st_size;
}

/**
 * Detect whether a path is a .gltf file (returns true) or .glb (returns false).
 */
static bool path_is_gltf(const char* path)
{
  const char* dot = strrchr(path, '.');
  if (dot && strcmp(dot, ".gltf") == 0) {
    return true;
  }
  return false;
}

/**
 * Extract directory from a file path (e.g., "a/b/file.gltf" -> "a/b/").
 */
static void extract_dir(const char* filepath, char* dir, size_t dir_size)
{
  strncpy(dir, filepath, dir_size - 1);
  dir[dir_size - 1] = '\0';
  char* last_sep    = strrchr(dir, '/');
  if (last_sep) {
    last_sep[1] = '\0';
  }
  else {
    dir[0] = '\0';
  }
}

/* -------------------------------------------------------------------------- *
 * Texture store helpers
 * -------------------------------------------------------------------------- */

/**
 * Get the texture view for a texture store entry, or the appropriate default.
 */
static WGPUTextureView get_texture_view(uint32_t tex_index,
                                        WGPUTextureView default_view)
{
  if (tex_index < state.texture_store_count
      && state.texture_store[tex_index].created) {
    return state.texture_store[tex_index].view;
  }
  return default_view;
}

/**
 * Upload a loaded texture to the GPU texture store.
 * Returns true if the GPU texture was created.
 */
static bool upload_texture_to_store(wgpu_context_t* ctx, uint32_t tex_index,
                                    WGPUTextureFormat format)
{
  if (tex_index >= state.texture_store_count) {
    return false;
  }

  const gltf_texture_t* tex = &state.model.textures[tex_index];
  if (!tex->data || tex->width == 0 || tex->height == 0) {
    return false;
  }

  /* Release any previous GPU texture for this slot */
  if (state.texture_store[tex_index].created) {
    if (state.texture_store[tex_index].view) {
      wgpuTextureViewRelease(state.texture_store[tex_index].view);
    }
    if (state.texture_store[tex_index].texture) {
      wgpuTextureDestroy(state.texture_store[tex_index].texture);
      wgpuTextureRelease(state.texture_store[tex_index].texture);
    }
  }

  WGPUTexture gpu_tex = create_model_texture(ctx, tex, format);
  if (!gpu_tex) {
    state.texture_store[tex_index].created = false;
    return false;
  }

  state.texture_store[tex_index].texture = gpu_tex;
  state.texture_store[tex_index].view    = wgpuTextureCreateView(gpu_tex, NULL);
  state.texture_store[tex_index].format  = format;
  state.texture_store[tex_index].created = true;
  return true;
}

/**
 * Determine the format for each texture in the texture store based on how
 * materials reference it, then upload all textures that have pixel data.
 */
static void initialize_texture_store(wgpu_context_t* ctx)
{
  gltf_model_t* m           = &state.model;
  state.texture_store_count = m->texture_count;
  state.textures_uploaded   = 0;

  if (m->texture_count == 0) {
    return;
  }

  /* Initialize all entries */
  for (uint32_t i = 0; i < m->texture_count && i < GLTF_VIEWER_MAX_TEXTURES;
       ++i) {
    memset(&state.texture_store[i], 0, sizeof(state.texture_store[i]));

    /* Default format — will be overridden by material references */
    state.texture_store[i].format = WGPUTextureFormat_RGBA8Unorm;
  }

  /* Determine format from material references */
  for (uint32_t mi = 0; mi < m->material_count; ++mi) {
    const gltf_material_t* mat = &m->materials[mi];
    if (mat->base_color_tex_index >= 0
        && (uint32_t)mat->base_color_tex_index < m->texture_count) {
      state.texture_store[mat->base_color_tex_index].format
        = WGPUTextureFormat_RGBA8UnormSrgb;
    }
    if (mat->emissive_tex_index >= 0
        && (uint32_t)mat->emissive_tex_index < m->texture_count) {
      state.texture_store[mat->emissive_tex_index].format
        = WGPUTextureFormat_RGBA8UnormSrgb;
    }
    /* metallic_roughness, normal, occlusion stay RGBA8Unorm */
  }

  /* Upload any textures that already have pixel data (GLB / embedded) */
  for (uint32_t i = 0; i < m->texture_count && i < GLTF_VIEWER_MAX_TEXTURES;
       ++i) {
    if (m->textures[i].data) {
      if (upload_texture_to_store(ctx, i, state.texture_store[i].format)) {
        state.textures_uploaded++;
      }
    }
  }
}

/**
 * Rebuild the bind group for a single material using current texture store
 * state. Uses default fallback textures for any slot not yet loaded.
 */
static void rebuild_material_bind_group(wgpu_context_t* ctx, uint32_t mat_idx)
{
  if (mat_idx >= state.material_count) {
    return;
  }

  WGPUDevice device          = ctx->device;
  viewer_material_t* dst     = &state.materials[mat_idx];
  const gltf_material_t* src = &state.model.materials[mat_idx];

  /* Release old bind group */
  if (dst->bind_group) {
    wgpuBindGroupRelease(dst->bind_group);
    dst->bind_group = NULL;
  }

  /* Get texture views — use texture store if available, else default */
  WGPUTextureView bc_view = state.gpu.default_srgb_view;
  WGPUTextureView mr_view = state.gpu.default_unorm_view;
  WGPUTextureView nm_view = state.gpu.default_normal_view;
  WGPUTextureView ao_view = state.gpu.default_unorm_view;
  WGPUTextureView em_view = state.gpu.default_srgb_view;

  if (src->base_color_tex_index >= 0) {
    bc_view = get_texture_view((uint32_t)src->base_color_tex_index,
                               state.gpu.default_srgb_view);
  }
  if (src->metallic_roughness_tex_index >= 0) {
    mr_view = get_texture_view((uint32_t)src->metallic_roughness_tex_index,
                               state.gpu.default_unorm_view);
  }
  if (src->normal_tex_index >= 0) {
    nm_view = get_texture_view((uint32_t)src->normal_tex_index,
                               state.gpu.default_normal_view);
  }
  if (src->occlusion_tex_index >= 0) {
    ao_view = get_texture_view((uint32_t)src->occlusion_tex_index,
                               state.gpu.default_unorm_view);
  }
  if (src->emissive_tex_index >= 0) {
    em_view = get_texture_view((uint32_t)src->emissive_tex_index,
                               state.gpu.default_srgb_view);
  }

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
}

/* -------------------------------------------------------------------------- *
 * Async file loading callbacks
 * -------------------------------------------------------------------------- */

/* Forward declarations for multi-phase loading */
static void gltf_start_external_loads(void);
static void gltf_process_geometry(void);

/**
 * GLB fetch callback — single-file loading path.
 */
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

/**
 * glTF JSON fetch callback — Phase 1 of multi-phase loading.
 *
 * After loading the .gltf JSON, we use gltf_model_discover_external_resources()
 * to enumerate external resources (.bin buffer + image files), then start
 * fetching them.
 */
static void gltf_json_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    printf("[gltf_viewer] glTF JSON loaded: %zu bytes\n", response->data.size);
    state.gltf.json_loaded = true;

    /* Discover external resources (buffers + images) from the glTF JSON */
    gltf_external_resources_t resources = {0};
    if (!gltf_model_discover_external_resources(
          response->data.ptr, response->data.size, &resources)) {
      printf("[gltf_viewer] ERROR: Failed to discover external resources\n");
      return;
    }

    /* Discover base directory for resolving relative URIs */
    char base_dir[GLTF_MODEL_MAX_URI_LENGTH];
    extract_dir(MODEL_FILE_PATH, base_dir, sizeof(base_dir));

    /* Process external buffer (.bin) */
    state.gltf.has_external_bin = false;
    if (resources.has_external_buffer) {
      char bin_path[GLTF_MODEL_MAX_URI_LENGTH * 2];
      snprintf(bin_path, sizeof(bin_path), "%s%s", base_dir,
               resources.buffer_uri);

      size_t bin_size = get_file_size(bin_path);
      if (bin_size == 0) {
        printf("[gltf_viewer] ERROR: Cannot stat external buffer '%s'\n",
               bin_path);
        return;
      }

      state.gltf.bin_buffer = (uint8_t*)malloc(bin_size);
      if (!state.gltf.bin_buffer) {
        printf(
          "[gltf_viewer] ERROR: Failed to allocate %zu bytes for "
          "buffer '%s'\n",
          bin_size, resources.buffer_uri);
        return;
      }
      state.gltf.bin_buffer_size  = bin_size;
      state.gltf.has_external_bin = true;
      strncpy(state.gltf.bin_uri, resources.buffer_uri,
              sizeof(state.gltf.bin_uri) - 1);
      printf("[gltf_viewer] External buffer: %s (%zu bytes)\n", bin_path,
             bin_size);
    }

    /* Process external image files */
    state.gltf.num_images        = 0;
    state.gltf.num_images_loaded = 0;

    for (uint32_t ri = 0; ri < resources.image_count; ri++) {
      if (state.gltf.num_images >= GLTF_VIEWER_MAX_EXTERNAL_IMAGES) {
        printf(
          "[gltf_viewer] WARNING: Too many external images, "
          "max %d supported\n",
          GLTF_VIEWER_MAX_EXTERNAL_IMAGES);
        break;
      }

      /* Build full path and stat the file */
      uint32_t idx = state.gltf.num_images;
      char img_path[GLTF_MODEL_MAX_URI_LENGTH * 2];
      snprintf(img_path, sizeof(img_path), "%s%s", base_dir,
               resources.images[ri].uri);
      strncpy(state.gltf.images[idx].path, img_path,
              sizeof(state.gltf.images[idx].path) - 1);
      state.gltf.images[idx].path[sizeof(state.gltf.images[idx].path) - 1]
        = '\0';

      size_t img_size = get_file_size(state.gltf.images[idx].path);
      if (img_size == 0) {
        printf("[gltf_viewer] WARNING: Cannot stat image '%s', skipping\n",
               state.gltf.images[idx].path);
        continue;
      }

      state.gltf.images[idx].buffer = (uint8_t*)malloc(img_size);
      if (!state.gltf.images[idx].buffer) {
        printf("[gltf_viewer] WARNING: Failed to allocate %zu bytes for '%s'\n",
               img_size, resources.images[ri].uri);
        continue;
      }
      state.gltf.images[idx].buffer_size   = img_size;
      state.gltf.images[idx].loaded        = false;
      state.gltf.images[idx].texture_index = resources.images[ri].texture_index;

      printf("[gltf_viewer] External image [%u]: %s (%zu bytes)\n", idx,
             resources.images[ri].uri, img_size);
      state.gltf.num_images++;
    }

    printf("[gltf_viewer] Discovered: %s external buffer, %u external images\n",
           state.gltf.has_external_bin ? "1" : "0", state.gltf.num_images);

    /* Start fetching external resources */
    gltf_start_external_loads();
  }
  else if (response->failed) {
    printf("[gltf_viewer] ERROR: Failed to fetch glTF JSON file\n");
  }
}

/**
 * glTF binary buffer (.bin) fetch callback — Phase 2.
 */
static void gltf_bin_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    printf("[gltf_viewer] Binary buffer loaded: %zu bytes\n",
           response->data.size);
    state.gltf.bin_loaded = true;

    /* Now we can load geometry */
    gltf_process_geometry();
  }
  else if (response->failed) {
    printf("[gltf_viewer] ERROR: Failed to fetch binary buffer\n");
  }
}

/**
 * glTF image fetch callback — Phase 3 (progressive).
 * Each image is decoded and uploaded to the texture store as it arrives.
 */
static void gltf_image_fetch_callback(const sfetch_response_t* response)
{
  if (response->fetched) {
    /* Find which image this response belongs to by matching buffer pointer */
    int found = -1;
    for (uint32_t i = 0; i < state.gltf.num_images; i++) {
      if (response->data.ptr >= (const void*)state.gltf.images[i].buffer
          && response->data.ptr
               < (const void*)(state.gltf.images[i].buffer
                               + state.gltf.images[i].buffer_size)) {
        found = (int)i;
        break;
      }
    }

    if (found < 0) {
      printf(
        "[gltf_viewer] WARNING: Received image data for unknown request\n");
      return;
    }

    uint32_t img_idx                  = (uint32_t)found;
    state.gltf.images[img_idx].loaded = true;
    state.gltf.num_images_loaded++;

    uint32_t tex_idx = state.gltf.images[img_idx].texture_index;
    printf("[gltf_viewer] Image [%u] loaded: %zu bytes (texture %u, %u/%u)\n",
           img_idx, response->data.size, tex_idx, state.gltf.num_images_loaded,
           state.gltf.num_images);

    /* Decode and store in model texture */
    if (gltf_model_load_texture_from_memory(
          &state.model, tex_idx, response->data.ptr, response->data.size)) {
      /* Image loaded flag will be checked in process_loaded_assets */
    }
    else {
      printf("[gltf_viewer] WARNING: Failed to decode image [%u]\n", img_idx);
    }
  }
  else if (response->failed) {
    printf("[gltf_viewer] ERROR: Failed to fetch an image file\n");
    state.gltf.num_images_loaded++; /* Count failures to not block forever */
  }
}

/**
 * HDR environment fetch callback.
 */
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
 * glTF multi-phase loading orchestration
 * -------------------------------------------------------------------------- */

/**
 * Start async fetches for external .bin buffer and image files.
 * Called after the .gltf JSON has been parsed and resources enumerated.
 */
static void gltf_start_external_loads(void)
{
  char base_dir[GLTF_MODEL_MAX_URI_LENGTH];
  extract_dir(MODEL_FILE_PATH, base_dir, sizeof(base_dir));

  /* Fetch .bin buffer if needed */
  if (state.gltf.has_external_bin) {
    char bin_path[GLTF_MODEL_MAX_URI_LENGTH * 2];
    snprintf(bin_path, sizeof(bin_path), "%s%s", base_dir, state.gltf.bin_uri);

    sfetch_send(&(sfetch_request_t){
      .path     = bin_path,
      .callback = gltf_bin_fetch_callback,
      .buffer
      = {.ptr = state.gltf.bin_buffer, .size = state.gltf.bin_buffer_size},
      .channel = 2, /* Dedicated channel for buffer data */
    });
  }
  else {
    /* No external buffer (embedded data URI) — geometry can proceed now */
    state.gltf.bin_loaded = true;
    gltf_process_geometry();
  }

  /* Fetch all external images in parallel (channel 3, multiple lanes) */
  for (uint32_t i = 0; i < state.gltf.num_images; i++) {
    sfetch_send(&(sfetch_request_t){
      .path     = state.gltf.images[i].path,
      .callback = gltf_image_fetch_callback,
      .buffer   = {.ptr  = state.gltf.images[i].buffer,
                   .size = state.gltf.images[i].buffer_size},
      .channel  = 3, /* Dedicated channel for images, with multiple lanes */
    });
  }
}

/**
 * Load geometry from the .gltf JSON + pre-loaded .bin buffer.
 * Called when binary buffer is available (either loaded or embedded).
 * This makes the model displayable with fallback textures.
 */
static void gltf_process_geometry(void)
{
  if (state.gltf.geometry_loaded) {
    return;
  }

  /* Prepare pre-loaded buffers */
  gltf_preloaded_buffer_t preloaded = {0};
  uint32_t num_preloaded            = 0;

  if (state.gltf.has_external_bin && state.gltf.bin_buffer) {
    preloaded.data = state.gltf.bin_buffer;
    preloaded.size = state.gltf.bin_buffer_size;
    num_preloaded  = 1;
  }

  /* Load model with deferred images (geometry + materials, NO texture data) */
  if (gltf_model_load_gltf_deferred(
        &state.model, state.gltf.json_buffer, state.gltf.json_buffer_size,
        MODEL_FILE_PATH, 1.0f, &preloaded, num_preloaded)) {
    state.model_loaded         = true;
    state.gltf.geometry_loaded = true;
    printf(
      "[gltf_viewer] Geometry loaded (deferred textures): %u vertices, "
      "%u indices, %u materials, %u textures\n",
      state.model.vertex_count, state.model.index_count,
      state.model.material_count, state.model.texture_count);
  }
  else {
    printf("[gltf_viewer] ERROR: Failed to load glTF model\n");
  }
}

/* -------------------------------------------------------------------------- *
 * Process loaded assets (called each frame until all resources ready)
 * -------------------------------------------------------------------------- */

static void process_loaded_assets(wgpu_context_t* ctx)
{
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
    }
  }

  /* Process model (GLB path: all-in-one) */
  if (!state.is_gltf && state.glb_loaded && state.model_loaded
      && !state.model_resources_created) {
    state.model_resources_created = true;
    printf("[gltf_viewer] Creating model GPU resources (GLB)...\n");

    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.vertex_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.index_buffer)

    apply_node_world_transforms();
    create_model_buffers(ctx);
    create_submeshes();
    initialize_texture_store(ctx);
    create_materials(ctx);

    /* Per-node transforms are baked into vertices, so use identity */
    glm_mat4_identity(state.node_base_transform);
    glm_mat4_identity(state.model_transform);
    camera_reset_to_model(&state.camera, state.model.dimensions.min,
                          state.model.dimensions.max);

    printf(
      "[gltf_viewer] Model resources created: %u opaque, "
      "%u transparent meshes\n",
      state.opaque_mesh_count, state.transparent_mesh_count);
  }

  /* Process model (glTF path: geometry first, then progressive textures) */
  if (state.is_gltf && state.model_loaded && !state.model_resources_created) {
    state.model_resources_created = true;
    printf("[gltf_viewer] Creating model GPU resources (glTF deferred)...\n");

    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.vertex_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.gpu.index_buffer)

    apply_node_world_transforms();
    create_model_buffers(ctx);
    create_submeshes();
    initialize_texture_store(ctx); /* Will upload any already-loaded textures */
    create_materials(ctx);

    /* Per-node transforms are baked into vertices, so use identity */
    glm_mat4_identity(state.node_base_transform);
    glm_mat4_identity(state.model_transform);
    camera_reset_to_model(&state.camera, state.model.dimensions.min,
                          state.model.dimensions.max);

    printf(
      "[gltf_viewer] Model resources created (deferred textures): "
      "%u opaque, %u transparent meshes\n",
      state.opaque_mesh_count, state.transparent_mesh_count);
  }

  /* Progressive texture loading (glTF path): upload newly loaded textures */
  if (state.is_gltf && state.model_resources_created
      && state.textures_uploaded < state.texture_store_count) {
    gltf_model_t* m = &state.model;

    for (uint32_t ti = 0;
         ti < m->texture_count && ti < GLTF_VIEWER_MAX_TEXTURES; ++ti) {
      if (state.texture_store[ti].created || !m->textures[ti].data) {
        continue; /* Already uploaded or not yet loaded */
      }

      /* New texture data available — upload to GPU */
      if (upload_texture_to_store(ctx, ti, state.texture_store[ti].format)) {
        state.textures_uploaded++;
        printf("[gltf_viewer] Texture %u uploaded to GPU (%u/%u)\n", ti,
               state.textures_uploaded, state.texture_store_count);

        /* Rebuild bind groups for all materials that reference this texture */
        for (uint32_t mi = 0; mi < state.material_count; ++mi) {
          const gltf_material_t* mat = &m->materials[mi];
          if (mat->base_color_tex_index == (int32_t)ti
              || mat->metallic_roughness_tex_index == (int32_t)ti
              || mat->normal_tex_index == (int32_t)ti
              || mat->occlusion_tex_index == (int32_t)ti
              || mat->emissive_tex_index == (int32_t)ti) {
            rebuild_material_bind_group(ctx, mi);
          }
        }
      }
    }
  }

  /* Check if all resources are ready */
  if (state.model_resources_created && state.environment_loaded) {
    state.resources_ready = true;
  }
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
  igSetNextWindowSize((ImVec2){300.0f, 420.0f}, ImGuiCond_FirstUseEver);

  if (igBegin("glTF PBR Viewer", NULL, ImGuiWindowFlags_None)) {
    if (!state.resources_ready) {
      igText("Loading assets...");
      if (state.is_gltf) {
        igText("  Format: glTF (multi-file)");
        if (state.gltf.json_loaded)
          igText("  JSON: loaded");
        else
          igText("  JSON: loading...");
        if (state.gltf.has_external_bin) {
          if (state.gltf.bin_loaded)
            igText("  Buffer: loaded");
          else
            igText("  Buffer: loading...");
        }
        if (state.gltf.num_images > 0) {
          igText("  Images: %u/%u", state.gltf.num_images_loaded,
                 state.gltf.num_images);
        }
        if (state.model_loaded)
          igText("  Geometry: ready");
      }
      else {
        igText("  Format: GLB (single file)");
        if (state.glb_loaded)
          igText("  Model: loaded");
        else
          igText("  Model: loading...");
      }
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
      if (state.texture_store_count > 0) {
        igText("Textures: %u/%u loaded", state.textures_uploaded,
               state.texture_store_count);
      }
      igSeparator();
      igCheckbox("Animate", &state.animate_model);

      /* --- PBR Settings --- */
      if (igCollapsingHeaderBoolPtr("PBR Settings", NULL,
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
        igSliderFloat("Exposure", &state.pbr.exposure, 0.1f, 10.0f, "%.1f", 0);
        igSliderFloat("Gamma", &state.pbr.gamma, 1.0f, 4.0f, "%.1f", 0);
        igSliderFloat("IBL Scale", &state.pbr.scale_ibl_ambient, 0.0f, 2.0f,
                      "%.2f", 0);
        igCheckbox("Direct Light", &state.pbr.enable_direct_light);

        /* Tone mapping selector */
        const char* tone_map_items[]
          = {"PBR Neutral", "Uncharted2", "Reinhard", "ACES"};
        igCombo("Tone Mapping", &state.pbr.tone_mapping_type, tone_map_items, 4,
                0);
      }

      /* --- Debug Visualization --- */
      if (igCollapsingHeaderBoolPtr("Debug Views", NULL, 0)) {
        const char* input_items[]
          = {"None",     "Base Color", "Normals",  "Occlusion",
             "Emissive", "Metallic",   "Roughness"};
        int debug_input = (int)state.pbr.debug_view_inputs;
        if (igCombo("Inputs", &debug_input, input_items, 7, 0)) {
          state.pbr.debug_view_inputs = (float)debug_input;
        }
        const char* equation_items[]
          = {"None",         "Diffuse",          "F (Fresnel)",
             "G (Geometry)", "D (Distribution)", "Specular"};
        int debug_eq = (int)state.pbr.debug_view_equation;
        if (igCombo("Equation", &debug_eq, equation_items, 6, 0)) {
          state.pbr.debug_view_equation = (float)debug_eq;
        }
      }

      /* --- Camera --- */
      if (igCollapsingHeaderBoolPtr("Camera", NULL, 0)) {
        igText("  Pos: %.1f, %.1f, %.1f", state.camera.position[0],
               state.camera.position[1], state.camera.position[2]);
        if (igButton("Reset Camera", (ImVec2){0, 0})) {
          camera_reset_to_model(&state.camera, state.model.dimensions.min,
                                state.model.dimensions.max);
          state.rotation_angle = 0.0f;
        }
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

  /* Detect file type */
  state.is_gltf = path_is_gltf(MODEL_FILE_PATH);
  printf("[gltf_viewer] Model file: %s (%s)\n", MODEL_FILE_PATH,
         state.is_gltf ? "glTF" : "GLB");

  /* Async file loading — optimized for parallel requests.
   * Channel 0: model file (GLB or glTF JSON)
   * Channel 1: HDR environment
   * Channel 2: glTF binary buffer (.bin)
   * Channel 3: glTF external images (multiple lanes for parallelism) */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = SFETCH_MAX_REQUESTS,
    .num_channels = SFETCH_NUM_CHANNELS,
    .num_lanes    = SFETCH_NUM_LANES,
  });

  /* Allocate HDR buffer (always needed) */
  state.hdr_file_buffer_size = get_file_size(HDR_FILE_PATH);
  if (state.hdr_file_buffer_size == 0) {
    printf("[gltf_viewer] ERROR: Cannot determine file size for '%s'\n",
           HDR_FILE_PATH);
    return EXIT_FAILURE;
  }
  state.hdr_file_buffer = (uint8_t*)malloc(state.hdr_file_buffer_size);
  if (!state.hdr_file_buffer) {
    printf("[gltf_viewer] ERROR: Failed to allocate HDR buffer (%zu bytes)\n",
           state.hdr_file_buffer_size);
    return EXIT_FAILURE;
  }

  /* Allocate model file buffer */
  size_t model_file_size = get_file_size(MODEL_FILE_PATH);
  if (model_file_size == 0) {
    printf("[gltf_viewer] ERROR: Cannot determine file size for '%s'\n",
           MODEL_FILE_PATH);
    free(state.hdr_file_buffer);
    state.hdr_file_buffer = NULL;
    return EXIT_FAILURE;
  }

  if (state.is_gltf) {
    /* glTF: allocate buffer for JSON file */
    state.gltf.json_buffer      = (uint8_t*)malloc(model_file_size);
    state.gltf.json_buffer_size = model_file_size;
    if (!state.gltf.json_buffer) {
      printf(
        "[gltf_viewer] ERROR: Failed to allocate glTF JSON buffer "
        "(%zu bytes)\n",
        model_file_size);
      free(state.hdr_file_buffer);
      state.hdr_file_buffer = NULL;
      return EXIT_FAILURE;
    }
    printf(
      "[gltf_viewer] Allocated buffers: glTF JSON=%zu bytes, "
      "HDR=%zu bytes\n",
      model_file_size, state.hdr_file_buffer_size);
  }
  else {
    /* GLB: allocate buffer for single binary file */
    state.glb_file_buffer      = (uint8_t*)malloc(model_file_size);
    state.glb_file_buffer_size = model_file_size;
    if (!state.glb_file_buffer) {
      printf(
        "[gltf_viewer] ERROR: Failed to allocate GLB buffer "
        "(%zu bytes)\n",
        model_file_size);
      free(state.hdr_file_buffer);
      state.hdr_file_buffer = NULL;
      return EXIT_FAILURE;
    }
    printf("[gltf_viewer] Allocated buffers: GLB=%zu bytes, HDR=%zu bytes\n",
           model_file_size, state.hdr_file_buffer_size);
  }

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
  if (state.is_gltf) {
    /* Phase 1: Fetch the .gltf JSON file */
    sfetch_send(&(sfetch_request_t){
      .path     = MODEL_FILE_PATH,
      .callback = gltf_json_fetch_callback,
      .buffer
      = {.ptr = state.gltf.json_buffer, .size = state.gltf.json_buffer_size},
      .channel = 0,
    });
  }
  else {
    /* GLB: single file fetch */
    sfetch_send(&(sfetch_request_t){
      .path     = MODEL_FILE_PATH,
      .callback = glb_fetch_callback,
      .buffer
      = {.ptr = state.glb_file_buffer, .size = state.glb_file_buffer_size},
      .channel = 0,
    });
  }

  /* HDR environment (always) */
  sfetch_send(&(sfetch_request_t){
    .path     = HDR_FILE_PATH,
    .callback = hdr_fetch_callback,
    .buffer
    = {.ptr = state.hdr_file_buffer, .size = state.hdr_file_buffer_size},
    .channel = 1,
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

  /* Free glTF loading buffers */
  free(state.gltf.json_buffer);
  free(state.gltf.bin_buffer);
  for (uint32_t i = 0; i < state.gltf.num_images; i++) {
    free(state.gltf.images[i].buffer);
  }

  /* Release texture store */
  for (uint32_t i = 0; i < state.texture_store_count; ++i) {
    if (state.texture_store[i].created) {
      if (state.texture_store[i].view) {
        wgpuTextureViewRelease(state.texture_store[i].view);
      }
      if (state.texture_store[i].texture) {
        wgpuTextureDestroy(state.texture_store[i].texture);
        wgpuTextureRelease(state.texture_store[i].texture);
      }
    }
  }

  /* Release materials */
  for (uint32_t i = 0; i < state.material_count; ++i) {
    viewer_material_t* mat = &state.materials[i];
    WGPU_RELEASE_RESOURCE(Buffer, mat->uniform_buffer)
    WGPU_RELEASE_RESOURCE(BindGroup, mat->bind_group)
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
    exposure : f32,
    lightDir : vec4<f32>,
    gamma : f32,
    prefilteredCubeMipLevels : f32,
    scaleIBLAmbient : f32,
    debugViewInputs : f32,
    debugViewEquation : f32,
    toneMappingType : i32,
  };

  @group(0) @binding(0) var<uniform> globalUniforms : GlobalUniforms;
  @group(0) @binding(1) var environmentCubeSampler : sampler;
  @group(0) @binding(2) var environmentTexture : texture_cube<f32>;

  const pi = 3.141592653589793;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  };

  fn Uncharted2Tonemap(colorIn : vec3f) -> vec3f {
    let A = 0.15; let B = 0.50; let C = 0.10;
    let D = 0.20; let E = 0.02; let F = 0.30;
    return ((colorIn * (A * colorIn + C * B) + D * E) / (colorIn * (A * colorIn + B) + D * F)) - E / F;
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
    let invGamma = 1.0 / globalUniforms.gamma;
    var color = colorIn * globalUniforms.exposure;
    if (globalUniforms.toneMappingType == 1) {
      let W = 11.2;
      color = Uncharted2Tonemap(color) * (1.0 / Uncharted2Tonemap(vec3f(W)));
    } else if (globalUniforms.toneMappingType == 2) {
      color = color / (color + vec3f(1.0));
    } else {
      color = toneMapPBRNeutral(color);
    }
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
    exposure : f32,
    lightDir : vec4<f32>,
    gamma : f32,
    prefilteredCubeMipLevels : f32,
    scaleIBLAmbient : f32,
    debugViewInputs : f32,
    debugViewEquation : f32,
    toneMappingType : i32,
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
    emissiveStrength : f32,
    workflow : i32,  // 0=MetallicRoughness, 1=SpecGloss, 2=Unlit
    doubleSided : i32,
    // Clearcoat (KHR_materials_clearcoat)
    clearcoatFactor : f32,
    clearcoatRoughness : f32,
    // Sheen (KHR_materials_sheen)
    sheenRoughnessFactor : f32,
    _pad0 : f32,
    sheenColorFactor : vec3<f32>,
    _pad1 : f32,
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
  const c_MinRoughness = 0.04;
  const PBR_WORKFLOW_METALLIC_ROUGHNESS = 0;
  const PBR_WORKFLOW_SPECULAR_GLOSSINESS = 1;

  // MaterialInfo following the Khronos glTF 2.0 Sample Viewer reference
  struct MaterialInfo {
    baseColor : vec4f,
    ior : f32,
    perceptualRoughness : f32,
    alphaRoughness : f32,
    metallic : f32,
    f0_dielectric : vec3f,
    f90 : vec3f,
    f90_dielectric : vec3f,
    specularWeight : f32,
    // Clearcoat
    clearcoatFactor : f32,
    clearcoatRoughness : f32,
    clearcoatF0 : vec3f,
    clearcoatF90 : vec3f,
    clearcoatNormal : vec3f,
    // Sheen
    sheenColorFactor : vec3f,
    sheenRoughnessFactor : f32,
    // Emissive
    emissiveStrength : f32,
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
    @location(6) worldPosition : vec3<f32>,
  };

  fn clampedDot(a : vec3f, b : vec3f) -> f32 {
    return clamp(dot(a, b), 0.0, 1.0);
  }

  // sRGB to linear conversion (accurate piecewise function per IEC 61966-2-1)
  fn SRGBtoLINEAR(srgbIn : vec4f) -> vec4f {
    let bLess = step(vec3f(0.04045), srgbIn.xyz);
    let linOut = mix(srgbIn.xyz / vec3f(12.92),
                     pow((srgbIn.xyz + vec3f(0.055)) / vec3f(1.055), vec3f(2.4)),
                     bLess);
    return vec4f(linOut, srgbIn.w);
  }

  // Normal mapping: construct TBN matrix and apply normal map
  fn getNormal(in : VertexOutput) -> vec3f {
    let N = normalize(in.normalWorld);
    let T = normalize(in.tangentWorld.xyz);
    let B = cross(N, T) * in.tangentWorld.w;
    let TBN = mat3x3f(T, B, N);
    var sampledNormal = textureSample(normalTexture, textureSampler, in.texCoord0).xyz * 2.0 - 1.0;
    sampledNormal = vec3f(sampledNormal.xy * materialUniforms.normalScale, sampledNormal.z);
    return normalize(TBN * sampledNormal);
  }

  // ========================================================================
  // Fresnel — Schlick approximation [Schlick 1994]
  // Implementation from Khronos glTF Sample Viewer reference
  // ========================================================================

  fn F_Schlick_vec3(f0 : vec3f, f90 : vec3f, VdotH : f32) -> vec3f {
    let x = clamp(1.0 - VdotH, 0.0, 1.0);
    let x2 = x * x;
    let x5 = x * x2 * x2;
    return f0 + (f90 - f0) * x5;
  }

  fn F_Schlick_scalar(f0 : f32, f90 : f32, VdotH : f32) -> f32 {
    let x = clamp(1.0 - VdotH, 0.0, 1.0);
    let x2 = x * x;
    let x5 = x * x2 * x2;
    return f0 + (f90 - f0) * x5;
  }

  // ========================================================================
  // Smith Joint GGX Visibility (height-correlated)
  // Vis = G / (4 * NdotL * NdotV)
  // [Heitz 2014] "Understanding the Masking-Shadowing Function"
  // ========================================================================

  fn V_GGX(NdotL : f32, NdotV : f32, alphaRoughness : f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;
    let GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
    let GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
    let GGX = GGXV + GGXL;
    if (GGX > 0.0) { return 0.5 / GGX; }
    return 0.0;
  }

  // ========================================================================
  // GGX/Trowbridge-Reitz Normal Distribution Function
  // [Trowbridge & Reitz 1975], recommended by [Epic Games, SIGGRAPH 2013]
  // ========================================================================

  fn D_GGX(NdotH : f32, alphaRoughness : f32) -> f32 {
    let alphaRoughnessSq = alphaRoughness * alphaRoughness;
    let f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
    return alphaRoughnessSq / (pi * f * f);
  }

  // ========================================================================
  // Lambertian diffuse BRDF (energy-conserving)
  // https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  // ========================================================================

  fn BRDF_lambertian(diffuseColor : vec3f) -> vec3f {
    return diffuseColor / pi;
  }

  // ========================================================================
  // Cook-Torrance specular microfacet BRDF
  // Combines GGX D and height-correlated Smith V
  // Fresnel is applied separately for dielectric/metallic split
  // ========================================================================

  fn BRDF_specularGGX(alphaRoughness : f32, NdotL : f32, NdotV : f32, NdotH : f32) -> vec3f {
    let Vis = V_GGX(NdotL, NdotV, alphaRoughness);
    let D = D_GGX(NdotH, alphaRoughness);
    return vec3f(Vis * D);
  }

  // ========================================================================
  // Sheen: Charlie NDF + Ashikhmin Visibility
  // [Estevez & Kulla, Sony ImageWorks, SIGGRAPH 2017]
  // ========================================================================

  fn lambdaSheenNumericHelper(x : f32, alphaG : f32) -> f32 {
    let oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
    let a = mix(21.5473, 25.3245, oneMinusAlphaSq);
    let b = mix(3.82987, 3.32435, oneMinusAlphaSq);
    let c = mix(0.19823, 0.16801, oneMinusAlphaSq);
    let d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
    let e = mix(-4.32054, -4.85967, oneMinusAlphaSq);
    return a / (1.0 + b * pow(x, c)) + d * x + e;
  }

  fn lambdaSheen(cosTheta : f32, alphaG : f32) -> f32 {
    if (abs(cosTheta) < 0.5) {
      return exp(lambdaSheenNumericHelper(cosTheta, alphaG));
    } else {
      return exp(2.0 * lambdaSheenNumericHelper(0.5, alphaG) - lambdaSheenNumericHelper(1.0 - cosTheta, alphaG));
    }
  }

  fn V_Sheen(NdotL : f32, NdotV : f32, sheenRoughness : f32) -> f32 {
    let sr = max(sheenRoughness, 0.000001);
    let alphaG = sr * sr;
    return clamp(1.0 / ((1.0 + lambdaSheen(NdotV, alphaG) + lambdaSheen(NdotL, alphaG)) *
        (4.0 * NdotV * NdotL)), 0.0, 1.0);
  }

  fn D_Charlie(sheenRoughness : f32, NdotH : f32) -> f32 {
    let sr = max(sheenRoughness, 0.000001);
    let alphaG = sr * sr;
    let invR = 1.0 / alphaG;
    let cos2h = NdotH * NdotH;
    let sin2h = 1.0 - cos2h;
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * pi);
  }

  fn BRDF_specularSheen(sheenColor : vec3f, sheenRoughness : f32, NdotL : f32, NdotV : f32, NdotH : f32) -> vec3f {
    let sheenDistribution = D_Charlie(sheenRoughness, NdotH);
    let sheenVisibility = V_Sheen(NdotL, NdotV, sheenRoughness);
    return sheenColor * sheenDistribution * sheenVisibility;
  }

  // ========================================================================
  // Specular-glossiness to metallic-roughness conversion
  // ========================================================================

  fn convertMetallic(diffuse : vec3f, specular : vec3f, maxSpecular : f32) -> f32 {
    let perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
    let perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
    if (perceivedSpecular < c_MinRoughness) { return 0.0; }
    let a = c_MinRoughness;
    let b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - c_MinRoughness) + perceivedSpecular - 2.0 * c_MinRoughness;
    let c = c_MinRoughness - perceivedSpecular;
    let D = max(b * b - 4.0 * a * c, 0.0);
    return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
  }

  // ========================================================================
  // IBL: Multi-scattering GGX Fresnel (Fdez-Aguera approximation)
  // https://bruop.github.io/ibl/#single_scattering_results
  // Energy-compensating multi-scattering from Kulla-Conty
  // ========================================================================

  fn getIBLRadianceGGX(n : vec3f, v : vec3f, roughness : f32) -> vec3f {
    let lod = roughness * globalUniforms.prefilteredCubeMipLevels;
    let reflection = normalize(reflect(-v, n));
    return textureSampleLevel(iblSpecularTexture, iblSampler, reflection, lod).rgb;
  }

  fn getIBLGGXFresnel(n : vec3f, v : vec3f, roughness : f32, F0 : vec3f, specularWeight : f32) -> vec3f {
    let NdotV = clampedDot(n, v);
    let brdfSamplePoint = vec2f(NdotV, roughness);
    let f_ab = textureSample(iblBRDFIntegrationLUTTexture, iblBRDFIntegrationLUTSampler, brdfSamplePoint).rg;

    // Single scattering: roughness-dependent Fresnel (Fdez-Aguera)
    let Fr = max(vec3f(1.0 - roughness), F0) - F0;
    let k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);
    let FssEss = specularWeight * (k_S * f_ab.x + f_ab.y);

    // Multi-scattering energy compensation (Kulla-Conty)
    let Ems = 1.0 - (f_ab.x + f_ab.y);
    let F_avg = specularWeight * (F0 + (1.0 - F0) / 21.0);
    let FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);

    return FssEss + FmsEms;
  }

  fn getDiffuseLight(n : vec3f) -> vec3f {
    return textureSample(iblIrradianceTexture, iblSampler, n).rgb;
  }

  // ========================================================================
  // Tone mapping operators
  // ========================================================================

  fn Uncharted2Tonemap(colorIn : vec3f) -> vec3f {
    let A = 0.15; let B = 0.50; let C = 0.10;
    let D = 0.20; let E = 0.02; let F = 0.30;
    return ((colorIn * (A * colorIn + C * B) + D * E) / (colorIn * (A * colorIn + B) + D * F)) - E / F;
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

  // ACES Filmic (Narkowicz 2015 fast approximation)
  fn toneMapACES(colorIn : vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((colorIn * (a * colorIn + b)) / (colorIn * (c * colorIn + d) + e), vec3f(0.0), vec3f(1.0));
  }

  fn toneMap(colorIn : vec3f) -> vec3f {
    let invGamma = 1.0 / globalUniforms.gamma;
    var color = colorIn * globalUniforms.exposure;
    if (globalUniforms.toneMappingType == 1) {
      let W = 11.2;
      color = Uncharted2Tonemap(color) * (1.0 / Uncharted2Tonemap(vec3f(W)));
    } else if (globalUniforms.toneMappingType == 2) {
      color = color / (color + vec3f(1.0));
    } else if (globalUniforms.toneMappingType == 3) {
      color = toneMapACES(color);
    } else {
      color = toneMapPBRNeutral(color);
    }
    color = pow(color, vec3f(invGamma));
    return color;
  }

  // ========================================================================
  // Vertex shader
  // ========================================================================

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
    output.worldPosition = worldPosition.xyz;
    return output;
  }

  // ========================================================================
  // Fragment shader — Khronos glTF 2.0 PBR reference pipeline
  //
  // Implements the full PBR material model from the glTF 2.0 specification
  // Appendix B with support for:
  //   - Metallic-roughness workflow (core)
  //   - Specular-glossiness workflow (legacy)
  //   - IBL with multi-scattering energy compensation (Fdez-Aguera / Kulla-Conty)
  //   - Height-correlated Smith GGX visibility
  //   - Clearcoat (KHR_materials_clearcoat)
  //   - Sheen (KHR_materials_sheen)
  //   - Emissive strength (KHR_materials_emissive_strength)
  //   - Unlit (KHR_materials_unlit)
  //   - Double-sided rendering
  //   - Alpha modes: Opaque, Mask, Blend
  //   - Tone mapping: PBR Neutral, Uncharted2, Reinhard, ACES
  //   - Debug visualization of inputs and BRDF terms
  // ========================================================================

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    // --- Unlit materials: return base color directly ---
    if (materialUniforms.workflow == 2) {
      var unlitColor = textureSample(baseColorTexture, textureSampler, in.texCoord0) * materialUniforms.baseColorFactor;
      unlitColor *= in.color;
      return vec4f(toneMap(unlitColor.rgb), unlitColor.a);
    }

    var baseColor : vec4f;

    // --- Alpha mask early discard ---
    if (materialUniforms.alphaMode == 1) {
      let earlyAlpha = textureSample(baseColorTexture, textureSampler, in.texCoord0).a;
      if (earlyAlpha * materialUniforms.baseColorFactor.a < materialUniforms.alphaCutoff) { discard; }
    }

    // --- Material parameter extraction ---
    var materialInfo : MaterialInfo;

    // Initialize defaults matching glTF 2.0 spec
    materialInfo.ior = 1.5;
    materialInfo.f0_dielectric = vec3f(0.04);
    materialInfo.specularWeight = 1.0;
    materialInfo.f90 = vec3f(1.0);
    materialInfo.f90_dielectric = vec3f(1.0);
    materialInfo.clearcoatFactor = 0.0;
    materialInfo.clearcoatRoughness = 0.0;
    materialInfo.clearcoatF0 = vec3f(0.04);
    materialInfo.clearcoatF90 = vec3f(1.0);
    materialInfo.sheenColorFactor = vec3f(0.0);
    materialInfo.sheenRoughnessFactor = 0.0;

    if (materialUniforms.workflow == PBR_WORKFLOW_METALLIC_ROUGHNESS) {
      // Metallic-Roughness workflow
      materialInfo.metallic = materialUniforms.metallicFactor;
      materialInfo.perceptualRoughness = materialUniforms.roughnessFactor;
      let mrSample = textureSample(metallicRoughnessTexture, textureSampler, in.texCoord0);
      materialInfo.perceptualRoughness *= mrSample.g;
      materialInfo.metallic *= mrSample.b;
      baseColor = textureSample(baseColorTexture, textureSampler, in.texCoord0) * materialUniforms.baseColorFactor;
    } else {
      // Specular-Glossiness workflow (legacy)
      let mrSample = textureSample(metallicRoughnessTexture, textureSampler, in.texCoord0);
      materialInfo.perceptualRoughness = 1.0 - mrSample.a;
      let diffuseSample = textureSample(baseColorTexture, textureSampler, in.texCoord0);
      let specularSample = mrSample.rgb;
      let maxSpecular = max(max(specularSample.r, specularSample.g), specularSample.b);
      materialInfo.metallic = convertMetallic(diffuseSample.rgb, specularSample, maxSpecular);
      let epsilon = 1e-6;
      let baseColorDiffuse = diffuseSample.rgb * ((1.0 - maxSpecular) / (1.0 - c_MinRoughness) / max(1.0 - materialInfo.metallic, epsilon));
      let baseColorSpecular = specularSample - (vec3f(c_MinRoughness) * (1.0 - materialInfo.metallic) * (1.0 / max(materialInfo.metallic, epsilon)));
      baseColor = vec4f(mix(baseColorDiffuse, baseColorSpecular, materialInfo.metallic * materialInfo.metallic), diffuseSample.a);
    }

    // Apply vertex color
    baseColor *= in.color;
    materialInfo.baseColor = baseColor;

    // Apply clearcoat from material uniforms
    materialInfo.clearcoatFactor = materialUniforms.clearcoatFactor;
    materialInfo.clearcoatRoughness = clamp(materialUniforms.clearcoatRoughness, 0.0, 1.0);
    materialInfo.clearcoatF0 = vec3f(pow((materialInfo.ior - 1.0) / (materialInfo.ior + 1.0), 2.0));
    materialInfo.clearcoatF90 = vec3f(1.0);

    // Apply sheen from material uniforms
    materialInfo.sheenColorFactor = materialUniforms.sheenColorFactor;
    materialInfo.sheenRoughnessFactor = materialUniforms.sheenRoughnessFactor;

    // Clamp material parameters
    materialInfo.perceptualRoughness = clamp(materialInfo.perceptualRoughness, 0.0, 1.0);
    materialInfo.metallic = clamp(materialInfo.metallic, 0.0, 1.0);

    // Roughness is authored as perceptual roughness; convert to alpha roughness
    // by squaring, as is convention [Burley 2012]
    materialInfo.alphaRoughness = materialInfo.perceptualRoughness * materialInfo.perceptualRoughness;

    // ========================================================================
    // Lighting computation following Khronos glTF 2.0 reference
    //
    // The material is decomposed into separate dielectric and metallic BRDFs:
    //   material = mix(dielectric_brdf, metal_brdf, metallic)
    //
    // Dielectric BRDF = fresnel_mix(diffuse, specular)
    // Metal BRDF = conductor_fresnel(baseColor, specular)
    // ========================================================================

    var n = getNormal(in);
    let v = normalize(in.viewDirectionWorld);

    // Handle double-sided: flip normal if back-facing
    if (materialUniforms.doubleSided > 0 && dot(n, v) < 0.0) {
      n = -n;
    }

    let NdotV = clampedDot(n, v);

    // Clearcoat normal (same as geometric normal for now — no separate clearcoat normal map)
    materialInfo.clearcoatNormal = n;

    // Accumulate lighting
    var f_specular_dielectric = vec3f(0.0);
    var f_specular_metal = vec3f(0.0);
    var f_diffuse = vec3f(0.0);
    var f_dielectric_brdf_ibl = vec3f(0.0);
    var f_metal_brdf_ibl = vec3f(0.0);
    var f_emissive = vec3f(0.0);
    var clearcoat_brdf = vec3f(0.0);
    var f_sheen = vec3f(0.0);

    var clearcoatFresnel = vec3f(0.0);
    var albedoSheenScaling : f32 = 1.0;

    // Clearcoat Fresnel (precomputed for both IBL and punctual)
    if (materialInfo.clearcoatFactor > 0.0) {
      clearcoatFresnel = F_Schlick_vec3(materialInfo.clearcoatF0, materialInfo.clearcoatF90,
                                         clampedDot(materialInfo.clearcoatNormal, v));
    }

    // ====================================================================
    // IBL contribution (Image-Based Lighting)
    // Split-sum approximation with multi-scattering energy compensation
    // ====================================================================

    // Diffuse IBL
    f_diffuse = getDiffuseLight(n) * baseColor.rgb;

    // Specular IBL (GGX importance-sampled prefiltered environment)
    f_specular_metal = getIBLRadianceGGX(n, v, materialInfo.perceptualRoughness);
    f_specular_dielectric = f_specular_metal;

    // Multi-scattering GGX Fresnel for metals (F0 = baseColor, weight = 1.0)
    let f_metal_fresnel_ibl = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness, baseColor.rgb, 1.0);
    f_metal_brdf_ibl = f_metal_fresnel_ibl * f_specular_metal;

    // Multi-scattering GGX Fresnel for dielectrics (F0 = 0.04, weight = specularWeight)
    let f_dielectric_fresnel_ibl = getIBLGGXFresnel(n, v, materialInfo.perceptualRoughness,
                                                     materialInfo.f0_dielectric, materialInfo.specularWeight);
    // Dielectric BRDF = mix(diffuse, specular, fresnel)
    f_dielectric_brdf_ibl = mix(f_diffuse, f_specular_dielectric, f_dielectric_fresnel_ibl);

    // Clearcoat IBL
    if (materialInfo.clearcoatFactor > 0.0) {
      clearcoat_brdf = getIBLRadianceGGX(materialInfo.clearcoatNormal, v, materialInfo.clearcoatRoughness);
    }

    // Compose: mix(dielectric, metal, metallic)
    var color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, materialInfo.metallic);

    // Apply sheen on top (energy-conserving scaling)
    color = f_sheen + color * albedoSheenScaling;

    // Apply clearcoat layer
    color = mix(color, clearcoat_brdf, materialInfo.clearcoatFactor * clearcoatFresnel);

    // Occlusion: only affects indirect (IBL) lighting
    let ao = textureSample(occlusionTexture, textureSampler, in.texCoord0).r;
    color = color * (1.0 + materialUniforms.occlusionStrength * (ao - 1.0));

    // Scale IBL ambient
    color *= globalUniforms.scaleIBLAmbient;

    // ====================================================================
    // Punctual light contribution (analytical directional light)
    // Following Khronos reference: separate dielectric/metal Fresnel
    // ====================================================================

    if (globalUniforms.lightDir.w >= 0.0) {
      let l = normalize(globalUniforms.lightDir.xyz);
      let h = normalize(l + v);
      let NdotL = clampedDot(n, l);
      let NdotH = clampedDot(n, h);
      let VdotH = clampedDot(v, h);

      if (NdotL > 0.0 || NdotV > 0.0) {
        // Separate dielectric and metallic Fresnel
        let dielectric_fresnel = F_Schlick_vec3(
          materialInfo.f0_dielectric * materialInfo.specularWeight,
          materialInfo.f90_dielectric, abs(VdotH));
        let metal_fresnel = F_Schlick_vec3(baseColor.rgb, vec3f(1.0), abs(VdotH));

        // Lambertian diffuse
        let l_diffuse = NdotL * BRDF_lambertian(baseColor.rgb);

        // Specular GGX (same lobe for both dielectric and metal)
        let l_specular = NdotL * BRDF_specularGGX(materialInfo.alphaRoughness, NdotL, NdotV, NdotH);

        // Metal BRDF = metalFresnel * specular
        let l_metal_brdf = metal_fresnel * l_specular;

        // Dielectric BRDF = mix(diffuse, specular, dielectricFresnel)
        let l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel);

        // Clearcoat contribution for punctual light
        var l_clearcoat_brdf = vec3f(0.0);
        if (materialInfo.clearcoatFactor > 0.0) {
          let clearcoatNdotH = clampedDot(materialInfo.clearcoatNormal, h);
          let clearcoatNdotL = clampedDot(materialInfo.clearcoatNormal, l);
          let clearcoatAlpha = materialInfo.clearcoatRoughness * materialInfo.clearcoatRoughness;
          let Dc = D_GGX(clearcoatNdotH, clearcoatAlpha);
          let Vc = V_GGX(clearcoatNdotL, clampedDot(materialInfo.clearcoatNormal, v), clearcoatAlpha);
          let Fc = F_Schlick_scalar(0.04, 1.0, VdotH);
          l_clearcoat_brdf = vec3f(Fc * Dc * Vc) * clearcoatNdotL;
        }

        // Sheen contribution for punctual light
        var l_sheen = vec3f(0.0);
        var l_albedoSheenScaling : f32 = 1.0;
        if (materialInfo.sheenRoughnessFactor > 0.0) {
          l_sheen = BRDF_specularSheen(materialInfo.sheenColorFactor, materialInfo.sheenRoughnessFactor,
                                       NdotL, NdotV, NdotH);
        }

        // Compose punctual: mix(dielectric, metal, metallic)
        var l_color = mix(l_dielectric_brdf, l_metal_brdf, materialInfo.metallic);
        l_color = l_sheen + l_color * l_albedoSheenScaling;
        l_color = mix(l_color, l_clearcoat_brdf, materialInfo.clearcoatFactor * clearcoatFresnel);

        color += l_color;
      }
    }

    // ====================================================================
    // Debug views: BRDF equation terms (uses punctual light for visualization)
    // ====================================================================

    if (globalUniforms.debugViewEquation > 0.0) {
      let l = normalize(globalUniforms.lightDir.xyz);
      let h = normalize(l + v);
      let NdotL = clampedDot(n, l);
      let NdotH = clampedDot(n, h);
      let VdotH = clampedDot(v, h);
      let F = F_Schlick_vec3(mix(materialInfo.f0_dielectric, baseColor.rgb, materialInfo.metallic),
                              materialInfo.f90, VdotH);
      let G = V_GGX(NdotL, NdotV, materialInfo.alphaRoughness);
      let D = D_GGX(NdotH, materialInfo.alphaRoughness);
      let debugIndex = i32(globalUniforms.debugViewEquation);
      var debugColor = vec3f(0.0);
      if (debugIndex == 1) { debugColor = BRDF_lambertian(baseColor.rgb); }
      else if (debugIndex == 2) { debugColor = F; }
      else if (debugIndex == 3) { debugColor = vec3f(G); }
      else if (debugIndex == 4) { debugColor = vec3f(D); }
      else if (debugIndex == 5) { debugColor = F * G * D; }
      return vec4f(toneMap(debugColor), 1.0);
    }

    // ====================================================================
    // Emissive
    // ====================================================================
    f_emissive = materialUniforms.emissiveFactor * materialUniforms.emissiveStrength;
    f_emissive *= SRGBtoLINEAR(textureSample(emissiveTexture, textureSampler, in.texCoord0)).rgb;

    // Clearcoat attenuates emissive: emissive * (1 - clearcoatFactor * clearcoatFresnel)
    color = f_emissive * (1.0 - materialInfo.clearcoatFactor * clearcoatFresnel) + color;

    // ====================================================================
    // Debug views: material inputs
    // ====================================================================
    if (globalUniforms.debugViewInputs > 0.0) {
      let idx = i32(globalUniforms.debugViewInputs);
      var debugColor = vec4f(0.0);
      if (idx == 1) { debugColor = baseColor; }
      else if (idx == 2) { debugColor = vec4f(n * 0.5 + 0.5, 1.0); }
      else if (idx == 3) { debugColor = vec4f(vec3f(ao), 1.0); }
      else if (idx == 4) { debugColor = vec4f(f_emissive, 1.0); }
      else if (idx == 5) { debugColor = vec4f(vec3f(materialInfo.metallic), 1.0); }
      else if (idx == 6) { debugColor = vec4f(vec3f(materialInfo.perceptualRoughness), 1.0); }
      return debugColor;
    }

    // ====================================================================
    // Tone mapping and final output
    // ====================================================================
    color = toneMap(color);

    var alpha = select(baseColor.a, 1.0, materialUniforms.alphaMode == 0);
    return vec4f(color, alpha);
  }
);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
// clang-format on
