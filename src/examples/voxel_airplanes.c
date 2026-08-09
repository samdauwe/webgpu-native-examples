/* ============================================================================
 * Voxel Airplanes
 *
 * C99/WebGPU port of the "webgl-voxel-airplanes" WebGL demo:
 * https://keaukraine.medium.com/voxel-airplanes-3d-webgl-demo-d210d5dfa54e
 *
 * Demonstrates a voxel airplane formation scene with:
 *  - 9 voxel airplane models with packed normal+color vertices
 *  - 6 formation types (Single, Triangle, LargeTriangle, Diamond, Cross, Line)
 *  - Animated propellers and plane wonder motion
 *  - Fly-in / fly-away transitions between plane models
 *  - Scrolling blocky-filtered terrain texture (13 terrain presets)
 *  - Terrain transition with blue-noise mask
 *  - Wind stripe effect (additive blend quads from hardcoded shader vertices)
 *  - Cloud billboard sprite
 *  - Rotating camera with optional fixed mode
 *  - ImGui GUI for scene controls
 *
 * Vertex formats:
 *   Plane/Glass/Prop: stride 4 = pos(Sint8x3) + normalColor(Uint8 packed)
 *     normalColor bits: [2:0] = normal_idx (0-5), [7:3] = color_idx (0-31)
 *   Quad/Cloud: stride 20 = pos(Float32x3) + uv(Float32x2)
 * ========================================================================== */

#ifdef __WAJIC__
#define WAJIC_IMAGE_IMPL
#include <wajic_image.h>
/* Bump max concurrent slots: 44 model + 19 texture requests = 63 total */
#define WAJIC_SFETCH_MAX_REQUESTS 128
#define WAJIC_SFETCH_IMPL
#include <wajic_sfetch.h>
#define WAJIC_TIME_IMPL
#include <wajic_time.h>
#else
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>
#define SOKOL_LOG_IMPL
#include <sokol_log.h>
#define SOKOL_TIME_IMPL
#include <sokol_time.h>
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

/* -------------------------------------------------------------------------- *
 * WGSL shader forward declarations
 * -------------------------------------------------------------------------- */

static const char* va_terrain_shader_wgsl;
static const char* va_terrain_trans_shader_wgsl;
static const char* va_plane_body_shader_wgsl;
static const char* va_glass_shader_wgsl;
static const char* va_wind_shader_wgsl;
static const char* va_cloud_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define VA_ASSETS "assets/models/VoxelAirplanes"

/* Camera */
#define VA_Z_NEAR (100.0f)
#define VA_Z_FAR (2000.0f)
#define VA_FOV_LANDSCAPE (25.0f)
#define VA_FOV_PORTRAIT (35.0f)

/* Animation periods (ms) */
#define VA_CAMERA_PERIOD (87000.0f)
#define VA_GROUND_PERIOD (15000.0f * 0.8f)
#define VA_GROUND2_PERIOD (74800.0f * 0.8f)
#define VA_WONDER1_PERIOD (25000.0f * 0.993f)
#define VA_WONDER2_PERIOD (6000.0f * 0.993f)
#define VA_PROP_PERIOD (300.0f * 0.993f)
#define VA_GLASS_PERIOD (1300.0f)
#define VA_GLASS2_PERIOD (9000.0f)
#define VA_WIND_PERIOD (2500.0f)
#define VA_WIND2_PERIOD (16000.0f)
#define VA_CLOUDS_PERIOD (13000.0f)
#define VA_PLANE_TRANS_PERIOD (1800.0f)
#define VA_TERRAIN_TRANS_PERIOD (1500.0f)

/* Scene constants */
#define VA_WIND_COUNT (10)
#define VA_WIND_SPREAD_X (300.0f)
#define VA_WIND_SPREAD_Y (1000.0f)
#define VA_WIND_WONDER (180.0f)

/* Model counts */
#define VA_PLANE_PRESET_COUNT (9)
#define VA_TERRAIN_COUNT (13)
#define VA_PALETTE_COUNT (3)

/* Max planes drawn in a single frame (LargeTriangle = 5 planes) */
#define VA_MAX_PLANE_DRAWS (5)
/* Max props per plane (2 engines per plane) */
#define VA_MAX_PROPS_PER_PLANE (2)
/* Total plane body UBO slots: plane draws + prop draws */
#define VA_BODY_UBO_TOTAL                                                      \
  (VA_MAX_PLANE_DRAWS + VA_MAX_PLANE_DRAWS * VA_MAX_PROPS_PER_PLANE)

/* UBO stride (must be >= 256 per WebGPU spec) */
#define VA_UBO_STRIDE (256u)

/* Fetch buffer sizes */
#define VA_MODEL_FETCH_BUF (65536u) /* 64 KB: max model file ~52 KB   */
#define VA_TEX_FETCH_BUF (131072u)  /* 128 KB: max texture PNG ~121 KB */

/* Depth format */
#define VA_DEPTH_FORMAT (ctx->depth_stencil_format)

/* -------------------------------------------------------------------------- *
 * Enums
 * -------------------------------------------------------------------------- */

typedef enum {
  VA_FORMATION_SINGLE         = 0,
  VA_FORMATION_TRIANGLE       = 1,
  VA_FORMATION_LARGE_TRIANGLE = 2,
  VA_FORMATION_DIAMOND        = 3,
  VA_FORMATION_CROSS          = 4,
  VA_FORMATION_LINE           = 5,
  VA_FORMATION_COUNT          = 6,
} va_formation_t;

typedef enum {
  VA_PLANE_NORMAL   = 0,
  VA_PLANE_FLY_AWAY = 1,
  VA_PLANE_FLY_IN   = 2,
} va_plane_state_t;

typedef enum {
  VA_TERRAIN_NORMAL     = 0,
  VA_TERRAIN_TRANSITION = 1,
} va_terrain_state_t;

typedef enum {
  VA_CAMERA_ROTATING = 0,
  VA_CAMERA_FIXED    = 1,
} va_camera_mode_t;

/* -------------------------------------------------------------------------- *
 * Plane preset data
 * -------------------------------------------------------------------------- */

typedef struct {
  const char* name;
  float banking;
  int prop_count;
  float props[2][3]; /* max 2 props: [i][0]=offsetX, [i][1]=offsetY,
                        [i][2]=offsetZ */
} va_plane_preset_t;

static const va_plane_preset_t va_plane_presets[VA_PLANE_PRESET_COUNT] = {
  {"01", 0.7f, 2, {{19.0f, 19.0f, 7.5f}, {-19.0f, 19.0f, 7.5f}}},
  {"02", 1.0f, 1, {{0.0f, 31.5f, 12.5f}, {0.0f, 0.0f, 0.0f}}},
  {"03", 0.2f, 2, {{21.0f, 29.0f, 8.0f}, {-21.0f, 29.0f, 8.0f}}},
  {"05", 0.2f, 2, {{18.0f, 15.5f, 24.0f}, {-18.0f, 15.5f, 24.0f}}},
  {"06", 1.0f, 1, {{0.0f, 35.5f, 6.0f}, {0.0f, 0.0f, 0.0f}}},
  {"07", 0.2f, 1, {{0.0f, 35.5f, -3.0f}, {0.0f, 0.0f, 0.0f}}},
  {"08", 0.2f, 2, {{19.0f, 11.5f, 1.0f}, {-19.0f, 11.5f, 1.0f}}},
  {"09", 0.1f, 2, {{24.0f, 18.5f, -1.0f}, {-24.0f, 18.5f, -1.0f}}},
  {"10", 0.8f, 2, {{21.0f, 12.5f, 0.0f}, {-21.0f, 12.5f, 0.0f}}},
};

/* -------------------------------------------------------------------------- *
 * UBO structures (each <= 256 bytes for single-stride dynamic UBOs)
 * -------------------------------------------------------------------------- */

/* MVP: precomputed proj*view*model, MV: view*model for normals */
typedef struct {
  mat4 mvp;            /* 64 bytes */
  mat4 mv;             /* 64 bytes */
  vec4 light_dir_view; /* 16 bytes  (light in view space) */
  vec4 ambient;        /* 16 bytes */
  vec4 diffuse;        /* 16 bytes */
  float diffuse_coef;  /*  4 bytes */
  float diffuse_exp;   /*  4 bytes */
  float _pad[2];       /*  8 bytes */
  /* total = 192 bytes */
} va_plane_body_ubo_t;

typedef struct {
  mat4 mvp;            /* 64 bytes */
  mat4 mv;             /* 64 bytes */
  vec4 light_dir_view; /* 16 bytes */
  vec4 ambient;        /* 16 bytes */
  vec4 diffuse;        /* 16 bytes */
  vec4 glass_color;    /* 16 bytes */
  float diffuse_coef;  /*  4 bytes */
  float diffuse_exp;   /*  4 bytes */
  float u_time;        /*  4 bytes */
  float _pad;          /*  4 bytes */
  /* total = 208 bytes */
} va_glass_ubo_t;

typedef struct {
  mat4 view_proj; /* 64 bytes */
  vec3 uv_offset; /* 12 bytes (x=side_offset, y=scroll, z=scale) */
  float _pad;     /*  4 bytes */
  /* total = 80 bytes */
} va_terrain_ubo_t;

typedef struct {
  mat4 view_proj;   /* 64 bytes */
  vec3 uv_offset;   /* 12 bytes */
  float transition; /*  4 bytes */
  /* total = 80 bytes */
} va_terrain_trans_ubo_t;

typedef struct {
  mat4 mvp;   /* 64 bytes */
  vec4 color; /* 16 bytes */
  /* total = 80 bytes */
} va_wind_ubo_t;

typedef struct {
  mat4 mvp; /* 64 bytes */
  /* total = 64 bytes */
} va_cloud_ubo_t;

/* -------------------------------------------------------------------------- *
 * GPU model
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBuffer vtx;
  WGPUBuffer idx;
  uint32_t index_count; /* number of uint16 indices */
  bool is_ready;
  /* pending upload data */
  uint8_t* pending_vtx;
  uint8_t* pending_idx;
  size_t pending_vtx_sz;
  size_t pending_idx_sz;
  bool vtx_pending;
  bool idx_pending;
} va_model_t;

/* -------------------------------------------------------------------------- *
 * Fetch helpers
 * -------------------------------------------------------------------------- */

typedef struct {
  va_model_t* model;
  bool is_indices;
  bool is_glass; /* glass model: uint8 → expand to uint16 indices */
} va_fetch_model_t;

typedef struct {
  wgpu_texture_t* tex;
  int* counter;
  bool nearest; /* use nearest filter */
} va_fetch_tex_t;

#define VA_MAX_MODEL_FETCHES (64u)
#define VA_MAX_TEX_FETCHES (32u)

static va_fetch_model_t va_fetch_model_states[VA_MAX_MODEL_FETCHES];
static int va_fetch_model_count = 0;
static va_fetch_tex_t va_fetch_tex_states[VA_MAX_TEX_FETCHES];
static int va_fetch_tex_count = 0;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  wgpu_context_t* wgpu_context;

  /* Models: 9 plane bodies + 9 glass + prop + cloud + quad */
  va_model_t model_plane[VA_PLANE_PRESET_COUNT];
  va_model_t model_glass[VA_PLANE_PRESET_COUNT];
  va_model_t model_prop;
  va_model_t model_cloud;
  va_model_t model_quad;

  /* Textures */
  wgpu_texture_t tex_terrain[VA_TERRAIN_COUNT]; /* ground0..ground12 */
  wgpu_texture_t tex_target_terrain; /* terrain being transitioned to */
  wgpu_texture_t tex_palette[VA_PALETTE_COUNT]; /* palette0..palette2 */
  wgpu_texture_t tex_glass;
  wgpu_texture_t tex_cloud;
  wgpu_texture_t tex_noise; /* blue-noise for terrain transition mask */

  /* Samplers */
  WGPUSampler sampler_linear;
  WGPUSampler sampler_nearest;

  /* Pipelines */
  WGPURenderPipeline terrain_pipeline;
  WGPURenderPipeline terrain_trans_pipeline;
  WGPURenderPipeline plane_body_pipeline;
  WGPURenderPipeline glass_pipeline;
  WGPURenderPipeline wind_pipeline;
  WGPURenderPipeline cloud_pipeline;

  /* Bind group layouts */
  WGPUBindGroupLayout bgl_terrain;
  WGPUBindGroupLayout bgl_terrain_trans;
  WGPUBindGroupLayout bgl_plane_body;
  WGPUBindGroupLayout bgl_glass;
  WGPUBindGroupLayout bgl_wind;
  WGPUBindGroupLayout bgl_cloud;

  /* Bind groups */
  WGPUBindGroup bg_terrain;
  WGPUBindGroup bg_terrain_trans;
  WGPUBindGroup bg_plane_body;
  WGPUBindGroup bg_glass;
  WGPUBindGroup bg_wind;
  WGPUBindGroup bg_cloud;

  /* UBO GPU buffers */
  WGPUBuffer terrain_ubo;
  WGPUBuffer terrain_trans_ubo;
  WGPUBuffer plane_body_ubo; /* VA_MAX_PLANE_DRAWS * VA_UBO_STRIDE bytes */
  WGPUBuffer glass_ubo;      /* VA_MAX_PLANE_DRAWS * VA_UBO_STRIDE bytes */
  WGPUBuffer wind_ubo;       /* VA_WIND_COUNT * VA_UBO_STRIDE bytes */
  WGPUBuffer cloud_ubo;

  /* UBO staging memory */
  va_terrain_ubo_t terrain_staging;
  va_terrain_trans_ubo_t terrain_trans_staging;
  va_plane_body_ubo_t plane_body_stagings[VA_MAX_PLANE_DRAWS];
  va_plane_body_ubo_t
    prop_stagings[VA_MAX_PLANE_DRAWS * VA_MAX_PROPS_PER_PLANE];
  int plane_prop_counts[VA_MAX_PLANE_DRAWS];
  va_glass_ubo_t glass_stagings[VA_MAX_PLANE_DRAWS];
  va_wind_ubo_t wind_stagings[VA_WIND_COUNT];
  va_cloud_ubo_t cloud_staging;

  /* Render pass */
  WGPURenderPassColorAttachment color_attach;
  WGPURenderPassDepthStencilAttachment depth_attach;
  WGPURenderPassDescriptor render_pass_desc;

  /* Matrices */
  mat4 proj_matrix;
  mat4 view_matrix;
  mat4 view_proj_matrix; /* proj * view */

  /* Timers (0..1 normalized) */
  float t_camera;
  float t_ground;        /* ground Y scroll */
  float t_ground2;       /* ground X side movement */
  float t_wonder1;       /* plane XY wonder */
  float t_wonder2;       /* plane Z wonder */
  float t_prop;          /* propeller spin */
  float t_glass;         /* glass shimmer 1 */
  float t_glass2;        /* glass shimmer 2 */
  float t_wind;          /* wind stripe Y position */
  float t_wind2;         /* wind stripe X wander */
  float t_clouds;        /* cloud Y position */
  float t_plane_trans;   /* plane fly transition (0..1) */
  float t_terrain_trans; /* terrain transition (0..1) */

  /* Plane state */
  va_plane_state_t state_plane;
  float fly_away_side; /* +1 or -1 */
  float fly_away_fwd;  /* ±0..1 forward speed */

  /* Terrain state */
  va_terrain_state_t state_terrain;
  int next_terrain;

  /* Cloud rotation (random, updated each cycle) */
  float clouds_rotation;

  /* GUI / config */
  struct {
    float plane_height;
    float plane_banking;
    float plane_wonder_xy;
    float plane_wonder_z;
    float prop_offset_x;
    float prop_offset_y;
    float prop_offset_z;
    int current_plane;
    int current_palette;
    int current_terrain;
    va_formation_t formation;
    va_camera_mode_t camera_mode;
  } cfg;

  /* Light direction (normalized, world space) */
  vec3 light_dir;

  /* Loading counters */
  int models_loaded; /* += 1 for each vtx+idx pair fully uploaded */
  int textures_loaded;
  int total_models;   /* expected total */
  int total_textures; /* expected total (set in init) */
  bool pipelines_ready;
  bool resources_dirty; /* bind groups need recreation */
  bool initialized;

  uint64_t last_time; /* stm_now() at last frame */
  float dt_sec;       /* last frame delta time in seconds */
} state = {
  /* clang-format off */
  .color_attach = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.56, 0.89, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_attach = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
    .stencilLoadOp   = WGPULoadOp_Clear,
    .stencilStoreOp  = WGPUStoreOp_Store,
  },
  .render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attach,
    .depthStencilAttachment = &state.depth_attach,
  },
  .cfg = {
    .plane_height    = 100.0f,
    .plane_banking   = 0.25f,
    .plane_wonder_xy = 4.0f,
    .plane_wonder_z  = 6.0f,
    .prop_offset_x   = 33.0f,
    .prop_offset_y   = 12.0f,
    .prop_offset_z   = 4.0f,
    .current_plane   = 0,
    .current_palette = 2,
    .current_terrain = 0,
    .formation       = VA_FORMATION_TRIANGLE,
    .camera_mode     = VA_CAMERA_ROTATING,
  },
  .light_dir     = {-1.0f / 1.7320508f, -1.0f / 1.7320508f, 1.0f / 1.7320508f},
  .state_plane   = VA_PLANE_FLY_IN,
  .fly_away_side = 1.0f,
  .total_models  = VA_PLANE_PRESET_COUNT * 2 + 3,  /* 9 planes + 9 glass + prop + cloud + quad */
  /* clang-format on */
};

/* -------------------------------------------------------------------------- *
 * Fetch callbacks
 * -------------------------------------------------------------------------- */

static void va_model_cb(const sfetch_response_t* r)
{
  va_fetch_model_t* fs = (va_fetch_model_t*)r->user_data;
  if (!r->fetched) {
    printf("[VA] Model fetch failed: %s (err %d)\n", r->path, r->error_code);
    free((void*)r->buffer.ptr);
    return;
  }

  size_t src_sz = r->data.size;
  va_model_t* m = fs->model;

  if (fs->is_indices) {
    if (fs->is_glass) {
      /* Glass uses uint8 indices - expand to uint16 for WebGPU */
      uint16_t* expanded = (uint16_t*)malloc(src_sz * sizeof(uint16_t));
      if (!expanded) {
        free((void*)r->buffer.ptr);
        return;
      }
      const uint8_t* src = (const uint8_t*)r->data.ptr;
      for (size_t i = 0; i < src_sz; i++)
        expanded[i] = (uint16_t)src[i];
      m->pending_idx    = (uint8_t*)expanded;
      m->pending_idx_sz = src_sz * sizeof(uint16_t);
      m->index_count    = (uint32_t)src_sz;
    }
    else {
      /* Standard uint16 indices */
      size_t aligned = (src_sz + 3u) & ~3u;
      uint8_t* data  = (uint8_t*)calloc(1, aligned);
      if (!data) {
        free((void*)r->buffer.ptr);
        return;
      }
      memcpy(data, r->data.ptr, src_sz);
      m->pending_idx    = data;
      m->pending_idx_sz = aligned;
      m->index_count    = (uint32_t)(src_sz / sizeof(uint16_t));
    }
    m->idx_pending = true;
  }
  else {
    /* Vertex data: raw copy, 4-byte aligned */
    size_t aligned = (src_sz + 3u) & ~3u;
    uint8_t* data  = (uint8_t*)calloc(1, aligned);
    if (!data) {
      free((void*)r->buffer.ptr);
      return;
    }
    memcpy(data, r->data.ptr, src_sz);
    m->pending_vtx    = data;
    m->pending_vtx_sz = aligned;
    m->vtx_pending    = true;
  }
  free((void*)r->buffer.ptr);
}

static void va_tex_cb(const sfetch_response_t* r)
{
  va_fetch_tex_t* ft = (va_fetch_tex_t*)r->user_data;
  if (!r->fetched) {
    printf("[VA] Texture fetch failed: %s (err %d)\n", r->path, r->error_code);
    free((void*)r->buffer.ptr);
    return;
  }
  int w, h, ch;
  uint8_t* pixels = image_pixels_from_memory((const uint8_t*)r->data.ptr,
                                             (int)r->data.size, &w, &h, &ch, 4);
  free((void*)r->buffer.ptr);
  if (!pixels)
    return;

  ft->tex->desc = (wgpu_texture_desc_t){
    .extent   = {(uint32_t)w, (uint32_t)h, 1},
    .format   = WGPUTextureFormat_RGBA8Unorm,
    .usage    = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .pixels   = {.ptr = pixels, .size = (size_t)(w * h * 4)},
    .is_dirty = true,
  };
}

/* -------------------------------------------------------------------------- *
 * GPU buffer helpers
 * -------------------------------------------------------------------------- */

static WGPUBuffer va_create_ubo(wgpu_context_t* ctx, size_t size,
                                const char* label)
{
  WGPUBufferDescriptor bd = {
    .label = {.data = label, .length = strlen(label)},
    .size  = size,
    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
  };
  return wgpuDeviceCreateBuffer(ctx->device, &bd);
}

static void va_alloc_model_buffers(wgpu_context_t* ctx, va_model_t* m,
                                   size_t vtx_sz, size_t idx_sz,
                                   const char* lbl)
{
  char buf[64];
  snprintf(buf, sizeof(buf), "%s vtx", lbl);
  WGPUBufferDescriptor vd = {
    .label = STRVIEW(buf),
    .size  = (vtx_sz + 3u) & ~3u,
    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
  };
  m->vtx = wgpuDeviceCreateBuffer(ctx->device, &vd);

  snprintf(buf, sizeof(buf), "%s idx", lbl);
  WGPUBufferDescriptor id = {
    .label = STRVIEW(buf),
    .size  = (idx_sz + 3u) & ~3u,
    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
  };
  m->idx = wgpuDeviceCreateBuffer(ctx->device, &id);
}

/* Upload pending model data to GPU and release CPU memory */
static void va_upload_model(wgpu_context_t* ctx, va_model_t* m)
{
  if (!m->vtx_pending || !m->idx_pending)
    return;

  if (!m->vtx) {
    va_alloc_model_buffers(ctx, m, m->pending_vtx_sz, m->pending_idx_sz, "VA");
  }
  wgpuQueueWriteBuffer(ctx->queue, m->vtx, 0, m->pending_vtx,
                       m->pending_vtx_sz);
  wgpuQueueWriteBuffer(ctx->queue, m->idx, 0, m->pending_idx,
                       m->pending_idx_sz);

  free(m->pending_vtx);
  m->pending_vtx = NULL;
  m->vtx_pending = false;
  free(m->pending_idx);
  m->pending_idx = NULL;
  m->idx_pending = false;

  m->is_ready = true;
  state.models_loaded++;
}

/* Upload pending texture to GPU and release CPU memory */
static void va_upload_texture(wgpu_context_t* ctx, wgpu_texture_t* t)
{
  if (!t->desc.is_dirty || !t->desc.pixels.ptr)
    return;

  /* Release old texture if any */
  if (t->handle)
    wgpu_destroy_texture(t);

  uint32_t w = t->desc.extent.width, h = t->desc.extent.height;

  WGPUTextureDescriptor td = {
    .label         = STRVIEW("Upload pending - Texture"),
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {w, h, 1},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  t->handle = wgpuDeviceCreateTexture(ctx->device, &td);
  t->view   = wgpuTextureCreateView(t->handle, NULL);

  wgpuQueueWriteTexture(ctx->queue,
                        &(WGPUTexelCopyTextureInfo){.texture = t->handle},
                        t->desc.pixels.ptr, t->desc.pixels.size,
                        &(WGPUTexelCopyBufferLayout){
                          .bytesPerRow  = w * 4,
                          .rowsPerImage = h,
                        },
                        &(WGPUExtent3D){w, h, 1});

  free((void*)t->desc.pixels.ptr);
  t->desc.pixels.ptr  = NULL;
  t->desc.pixels.size = 0;
  t->desc.is_dirty    = false;
  state.textures_loaded++;
  /* Trigger bind group recreation so the actual texture is picked up */
  if (state.initialized)
    state.resources_dirty = true;
}

/* -------------------------------------------------------------------------- *
 * Asset loading
 * -------------------------------------------------------------------------- */

static void va_load_model(va_model_t* m, const char* base, bool is_glass)
{
  char path[256];
  va_fetch_model_t* fi = &va_fetch_model_states[va_fetch_model_count++];
  fi->model            = m;
  fi->is_indices       = true;
  fi->is_glass         = is_glass;
  snprintf(path, sizeof(path), "%s/models/%s-indices.bin", VA_ASSETS, base);
  uint8_t* ibuf = (uint8_t*)malloc(VA_MODEL_FETCH_BUF);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = va_model_cb,
    .buffer    = {.ptr = ibuf, .size = VA_MODEL_FETCH_BUF},
    .user_data = {.ptr = fi, .size = sizeof(va_fetch_model_t)},
  });

  va_fetch_model_t* fv = &va_fetch_model_states[va_fetch_model_count++];
  fv->model            = m;
  fv->is_indices       = false;
  fv->is_glass         = false;
  snprintf(path, sizeof(path), "%s/models/%s-strides.bin", VA_ASSETS, base);
  uint8_t* vbuf = (uint8_t*)malloc(VA_MODEL_FETCH_BUF);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = va_model_cb,
    .buffer    = {.ptr = vbuf, .size = VA_MODEL_FETCH_BUF},
    .user_data = {.ptr = fv, .size = sizeof(va_fetch_model_t)},
  });
}

static void va_load_texture(wgpu_context_t* ctx, wgpu_texture_t* tex,
                            const char* path, bool nearest, int* counter)
{
  /* Placeholder until loaded */
  *tex = wgpu_create_color_bars_texture(
    ctx, &(wgpu_texture_desc_t){
           .format = WGPUTextureFormat_RGBA8Unorm,
           .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
         });

  va_fetch_tex_t* ft = &va_fetch_tex_states[va_fetch_tex_count++];
  ft->tex            = tex;
  ft->counter        = counter;
  ft->nearest        = nearest;

  uint8_t* buf = (uint8_t*)malloc(VA_TEX_FETCH_BUF);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = va_tex_cb,
    .buffer    = {.ptr = buf, .size = VA_TEX_FETCH_BUF},
    .user_data = {.ptr = ft, .size = sizeof(va_fetch_tex_t)},
  });
}

/* -------------------------------------------------------------------------- *
 * Pipeline / bind group creation
 * -------------------------------------------------------------------------- */

/* ---- Terrain pipeline (DiffuseScrollingFiltered) ---- */

static void va_create_terrain_pipeline(wgpu_context_t* ctx)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgle[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = sizeof(va_terrain_ubo_t)},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
  };
  state.bgl_terrain = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label      = STRVIEW("VA Terrain - Bind group layout"),
                   .entryCount = 3,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label = STRVIEW("VA Terrain - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_terrain,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_terrain_shader_wgsl);

  /* Vertex buffer: stride=20, float3 pos + float2 uv */
  WGPUVertexAttribute terrain_attrs[2] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x2, .offset = 12, .shaderLocation = 1},
  };
  WGPUVertexBufferLayout terrain_vbl = {
    .arrayStride    = 20,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = terrain_attrs,
  };

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("VA Terrain - Render pipeline"),
    .layout = pl,
    .vertex = {
      .module      = sm,
      .entryPoint  = STRVIEW("vertexMain"),
      .bufferCount = 1,
      .buffers     = &terrain_vbl,
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW,
    },
    .depthStencil = &(WGPUDepthStencilState){
      .format            = VA_DEPTH_FORMAT,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_Less,
    },
    .fragment = &(WGPUFragmentState){
      .module      = sm,
      .entryPoint  = STRVIEW("fragmentMain"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = ctx->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.terrain_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);

  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* ---- Terrain Transition pipeline ---- */

static void va_create_terrain_trans_pipeline(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry bgle[5] = {
    [0] = {.binding    = 0,
           .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
           .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                          .minBindingSize = sizeof(va_terrain_trans_ubo_t)}},
    [1] = {.binding    = 1,
           .visibility = WGPUShaderStage_Fragment,
           .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
    [2] = {.binding    = 2,
           .visibility = WGPUShaderStage_Fragment,
           .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                          .viewDimension = WGPUTextureViewDimension_2D}},
    [3] = {.binding    = 3,
           .visibility = WGPUShaderStage_Fragment,
           .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                          .viewDimension = WGPUTextureViewDimension_2D}},
    [4] = {.binding    = 4,
           .visibility = WGPUShaderStage_Fragment,
           .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                          .viewDimension = WGPUTextureViewDimension_2D}},
  };
  state.bgl_terrain_trans = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label = STRVIEW("VA Terrain Trans - Bind group layout"),
                   .entryCount = 5,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label = STRVIEW("VA Terrain Trans - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_terrain_trans,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_terrain_trans_shader_wgsl);

  WGPUVertexAttribute terrain_attrs[2] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x2, .offset = 12, .shaderLocation = 1},
  };
  WGPUVertexBufferLayout vbl = {
    .arrayStride    = 20,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = terrain_attrs,
  };

  WGPURenderPipelineDescriptor desc = {
    .label     = STRVIEW("VA Terrain Trans - Render pipeline"),
    .layout    = pl,
    .vertex    = {.module      = sm,
                  .entryPoint  = STRVIEW("vertexMain"),
                  .bufferCount = 1,
                  .buffers     = &vbl},
    .primitive = {.topology  = WGPUPrimitiveTopology_TriangleList,
                  .cullMode  = WGPUCullMode_Back,
                  .frontFace = WGPUFrontFace_CCW},
    .depthStencil
    = &(WGPUDepthStencilState){.format            = VA_DEPTH_FORMAT,
                               .depthWriteEnabled = WGPUOptionalBool_True,
                               .depthCompare      = WGPUCompareFunction_Less},
    .fragment
    = &(WGPUFragmentState){.module      = sm,
                           .entryPoint  = STRVIEW("fragmentMain"),
                           .targetCount = 1,
                           .targets     = &(
                             WGPUColorTargetState){.format = ctx->render_format,
                                                       .writeMask
                                                       = WGPUColorWriteMask_All}},
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.terrain_trans_pipeline
    = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* ---- Plane body pipeline ---- */

static void va_create_plane_body_pipeline(wgpu_context_t* ctx)
{
  /* Only 2 bindings: UBO (dynamic) + palette texture.
   * No sampler needed because the shader uses textureLoad only. */
  WGPUBindGroupLayoutEntry bgle[2] = {
    [0] = {.binding    = 0,
           .visibility = WGPUShaderStage_Vertex,
           .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                          .hasDynamicOffset = true,
                          .minBindingSize   = sizeof(va_plane_body_ubo_t)}},
    /* Use UnfilterableFloat + no sampler: textureLoad does not need a sampler
     * and Chrome WebGPU rejects NonFiltering sampler + Float texture pairs. */
    [1] = {.binding    = 1,
           .visibility = WGPUShaderStage_Vertex,
           .texture    = {.sampleType    = WGPUTextureSampleType_UnfilterableFloat,
                          .viewDimension = WGPUTextureViewDimension_2D}},
  };
  state.bgl_plane_body = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label      = STRVIEW("VA Plane Body - Bind group layout"),
                   .entryCount = 2,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label = STRVIEW("VA Plane Body - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_plane_body,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_plane_body_shader_wgsl);

  /* Vertex buffer: stride=4, Sint8x4 packed */
  WGPUVertexAttribute body_attr = {
    .format         = WGPUVertexFormat_Sint8x4,
    .offset         = 0,
    .shaderLocation = 0,
  };
  WGPUVertexBufferLayout vbl = {
    .arrayStride    = 4,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = &body_attr,
  };

  WGPURenderPipelineDescriptor desc = {
    .label     = STRVIEW("VA Plane Body - Render pipeline"),
    .layout    = pl,
    .vertex    = {.module      = sm,
                  .entryPoint  = STRVIEW("vertexMain"),
                  .bufferCount = 1,
                  .buffers     = &vbl},
    .primitive = {.topology  = WGPUPrimitiveTopology_TriangleList,
                  .cullMode  = WGPUCullMode_Back,
                  .frontFace = WGPUFrontFace_CCW},
    .depthStencil
    = &(WGPUDepthStencilState){.format            = VA_DEPTH_FORMAT,
                               .depthWriteEnabled = WGPUOptionalBool_True,
                               .depthCompare      = WGPUCompareFunction_Less},
    .fragment
    = &(WGPUFragmentState){.module      = sm,
                           .entryPoint  = STRVIEW("fragmentMain"),
                           .targetCount = 1,
                           .targets     = &(
                             WGPUColorTargetState){.format = ctx->render_format,
                                                       .writeMask
                                                       = WGPUColorWriteMask_All}},
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.plane_body_pipeline
    = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* ---- Glass pipeline ---- */

static void va_create_glass_pipeline(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry bgle[3] = {
    [0] = {.binding    = 0,
           .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
           .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                          .hasDynamicOffset = true,
                          .minBindingSize   = sizeof(va_glass_ubo_t)}},
    [1] = {.binding    = 1,
           .visibility = WGPUShaderStage_Fragment,
           .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
    [2] = {.binding    = 2,
           .visibility = WGPUShaderStage_Fragment,
           .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                          .viewDimension = WGPUTextureViewDimension_2D}},
  };
  state.bgl_glass = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label      = STRVIEW("VA Glass - Bind group layout"),
                   .entryCount = 3,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label = STRVIEW("VA Glass - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_glass,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_glass_shader_wgsl);

  WGPUVertexAttribute glass_attr = {
    .format         = WGPUVertexFormat_Sint8x4,
    .offset         = 0,
    .shaderLocation = 0,
  };
  WGPUVertexBufferLayout vbl = {
    .arrayStride    = 4,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = &glass_attr,
  };

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("VA Glass - Render pipeline"),
    .layout = pl,
    .vertex = { .module = sm, .entryPoint = STRVIEW("vertexMain"),
                .bufferCount = 1, .buffers = &vbl },
    .primitive = {.topology = WGPUPrimitiveTopology_TriangleList,
                  .cullMode = WGPUCullMode_None, .frontFace = WGPUFrontFace_CCW},
    .depthStencil = &(WGPUDepthStencilState){
      .format = VA_DEPTH_FORMAT, .depthWriteEnabled = WGPUOptionalBool_False,
      .depthCompare = WGPUCompareFunction_Less},
    .fragment = &(WGPUFragmentState){
      .module = sm, .entryPoint = STRVIEW("fragmentMain"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = ctx->render_format,
        .writeMask = WGPUColorWriteMask_All,
        .blend     = &(WGPUBlendState){
          .color = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrc},
          .alpha = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrcAlpha},
        },
      }},
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.glass_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* ---- Wind stripe pipeline ---- */

static void va_create_wind_pipeline(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry bgle[1] = {
    [0] = {.binding    = 0,
           .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
           .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                          .hasDynamicOffset = true,
                          .minBindingSize   = sizeof(va_wind_ubo_t)}},
  };
  state.bgl_wind = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label      = STRVIEW("VA Wind - Bind group layout"),
                   .entryCount = 1,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label                = STRVIEW("VA Wind - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_wind,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_wind_shader_wgsl);

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("VA Wind - Render pipeline"),
    .layout = pl,
    .vertex = { .module = sm, .entryPoint = STRVIEW("vertexMain"),
                .bufferCount = 0 }, /* no vertex buffer - hardcoded in shader */
    .primitive = {.topology = WGPUPrimitiveTopology_TriangleStrip,
                  .cullMode = WGPUCullMode_None},
    .depthStencil = &(WGPUDepthStencilState){
      .format = VA_DEPTH_FORMAT, .depthWriteEnabled = WGPUOptionalBool_False,
      .depthCompare = WGPUCompareFunction_Less},
    .fragment = &(WGPUFragmentState){
      .module = sm, .entryPoint = STRVIEW("fragmentMain"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = ctx->render_format,
        .writeMask = WGPUColorWriteMask_All,
        .blend     = &(WGPUBlendState){
          .color = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrc},
          .alpha = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrcAlpha},
        },
      }},
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.wind_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* ---- Cloud pipeline ---- */

static void va_create_cloud_pipeline(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry bgle[3] = {
    [0] = {.binding    = 0,
           .visibility = WGPUShaderStage_Vertex,
           .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                          .minBindingSize = sizeof(va_cloud_ubo_t)}},
    [1] = {.binding    = 1,
           .visibility = WGPUShaderStage_Fragment,
           .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
    [2] = {.binding    = 2,
           .visibility = WGPUShaderStage_Fragment,
           .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                          .viewDimension = WGPUTextureViewDimension_2D}},
  };
  state.bgl_cloud = wgpuDeviceCreateBindGroupLayout(
    ctx->device, &(WGPUBindGroupLayoutDescriptor){
                   .label      = STRVIEW("VA Cloud - Bind group layout"),
                   .entryCount = 3,
                   .entries    = bgle,
                 });

  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
    ctx->device, &(WGPUPipelineLayoutDescriptor){
                   .label = STRVIEW("VA Cloud - Pipeline layout"),
                   .bindGroupLayoutCount = 1,
                   .bindGroupLayouts     = &state.bgl_cloud,
                 });

  WGPUShaderModule sm
    = wgpu_create_shader_module(ctx->device, va_cloud_shader_wgsl);

  WGPUVertexAttribute cloud_attrs[2] = {
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    {.format = WGPUVertexFormat_Float32x2, .offset = 12, .shaderLocation = 1},
  };
  WGPUVertexBufferLayout vbl = {
    .arrayStride    = 20,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = cloud_attrs,
  };

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("VA Cloud - Render pipeline"),
    .layout = pl,
    .vertex = { .module = sm, .entryPoint = STRVIEW("vertexMain"),
                .bufferCount = 1, .buffers = &vbl },
    .primitive = {.topology = WGPUPrimitiveTopology_TriangleList,
                  .cullMode = WGPUCullMode_None},
    .depthStencil = &(WGPUDepthStencilState){
      .format = VA_DEPTH_FORMAT, .depthWriteEnabled = WGPUOptionalBool_False,
      .depthCompare = WGPUCompareFunction_Less},
    .fragment = &(WGPUFragmentState){
      .module = sm, .entryPoint = STRVIEW("fragmentMain"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = ctx->render_format,
        .writeMask = WGPUColorWriteMask_All,
        .blend     = &(WGPUBlendState){
          .color = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrc},
          .alpha = {.operation = WGPUBlendOperation_Add,
                    .srcFactor  = WGPUBlendFactor_One,
                    .dstFactor  = WGPUBlendFactor_OneMinusSrcAlpha},
        },
      }},
    .multisample = {.count = 1, .mask = 0xFFFFFFFF},
  };
  state.cloud_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  wgpuShaderModuleRelease(sm);
  wgpuPipelineLayoutRelease(pl);
}

/* -------------------------------------------------------------------------- *
 * Bind group (re)creation
 * -------------------------------------------------------------------------- */

static void va_create_bind_groups(wgpu_context_t* ctx)
{
  /* Terrain */
  if (state.bg_terrain)
    wgpuBindGroupRelease(state.bg_terrain);
  {
    WGPUBindGroupEntry bge[3] = {
      [0] = {.binding = 0,
             .buffer  = state.terrain_ubo,
             .size    = sizeof(va_terrain_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler_linear},
      [2] = {.binding     = 2,
             .textureView = state.tex_terrain[state.cfg.current_terrain].view},
    };
    state.bg_terrain = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Terrain - Bind group"),
                     .layout     = state.bgl_terrain,
                     .entryCount = 3,
                     .entries    = bge,
                   });
  }

  /* Terrain transition */
  if (state.bg_terrain_trans)
    wgpuBindGroupRelease(state.bg_terrain_trans);
  {
    WGPUTextureView target_view
      = state.tex_target_terrain.view ?
          state.tex_target_terrain.view :
          state.tex_terrain[state.cfg.current_terrain].view;
    WGPUBindGroupEntry bge[5] = {
      [0] = {.binding = 0,
             .buffer  = state.terrain_trans_ubo,
             .size    = sizeof(va_terrain_trans_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler_linear},
      [2] = {.binding     = 2,
             .textureView = state.tex_terrain[state.cfg.current_terrain].view},
      [3] = {.binding = 3, .textureView = target_view},
      [4] = {.binding = 4, .textureView = state.tex_noise.view},
    };
    state.bg_terrain_trans = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Terrain Trans - Bind group"),
                     .layout     = state.bgl_terrain_trans,
                     .entryCount = 5,
                     .entries    = bge,
                   });
  }

  /* Plane body — 2 bindings (no sampler: shader uses textureLoad only) */
  if (state.bg_plane_body)
    wgpuBindGroupRelease(state.bg_plane_body);
  {
    WGPUBindGroupEntry bge[2] = {
      [0] = {.binding = 0,
             .buffer  = state.plane_body_ubo,
             .size    = sizeof(va_plane_body_ubo_t)},
      [1] = {.binding     = 1,
             .textureView = state.tex_palette[state.cfg.current_palette].view},
    };
    state.bg_plane_body = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Plane Body - Bind group"),
                     .layout     = state.bgl_plane_body,
                     .entryCount = 2,
                     .entries    = bge,
                   });
  }

  /* Glass */
  if (state.bg_glass)
    wgpuBindGroupRelease(state.bg_glass);
  {
    WGPUBindGroupEntry bge[3] = {
      [0] = {.binding = 0,
             .buffer  = state.glass_ubo,
             .size    = sizeof(va_glass_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler_linear},
      [2] = {.binding = 2, .textureView = state.tex_glass.view},
    };
    state.bg_glass = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Glass - Bind group"),
                     .layout     = state.bgl_glass,
                     .entryCount = 3,
                     .entries    = bge,
                   });
  }

  /* Wind */
  if (state.bg_wind)
    wgpuBindGroupRelease(state.bg_wind);
  {
    WGPUBindGroupEntry bge[1] = {
      [0]
      = {.binding = 0, .buffer = state.wind_ubo, .size = sizeof(va_wind_ubo_t)},
    };
    state.bg_wind = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Wind - Bind group"),
                     .layout     = state.bgl_wind,
                     .entryCount = 1,
                     .entries    = bge,
                   });
  }

  /* Cloud */
  if (state.bg_cloud)
    wgpuBindGroupRelease(state.bg_cloud);
  {
    WGPUBindGroupEntry bge[3] = {
      [0] = {.binding = 0,
             .buffer  = state.cloud_ubo,
             .size    = sizeof(va_cloud_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler_linear},
      [2] = {.binding = 2, .textureView = state.tex_cloud.view},
    };
    state.bg_cloud = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .label      = STRVIEW("VA Cloud - Bind group"),
                     .layout     = state.bgl_cloud,
                     .entryCount = 3,
                     .entries    = bge,
                   });
  }
}

/* -------------------------------------------------------------------------- *
 * Camera & projection
 * -------------------------------------------------------------------------- */

static void va_position_camera(void)
{
  float t = state.t_camera * 2.0f * (float)M_PI;
  float x = sinf(t) * 300.0f;
  float y = cosf(t) * 300.0f;
  float z = 250.0f + sinf(t) * 50.0f + state.cfg.plane_height;

  glm_lookat((vec3){x, y, z}, (vec3){0.0f, 0.0f, state.cfg.plane_height},
             (vec3){0.0f, 0.0f, 1.0f}, state.view_matrix);
}

static void va_set_projection(wgpu_context_t* ctx)
{
  float w      = (float)ctx->width;
  float h      = (float)ctx->height;
  float aspect = (h > 0.0f) ? (w / h) : 1.0f;
  float fov    = (w >= h) ? VA_FOV_LANDSCAPE : VA_FOV_PORTRAIT;
  glm_perspective(glm_rad(fov), aspect, VA_Z_NEAR, VA_Z_FAR, state.proj_matrix);
}

/* -------------------------------------------------------------------------- *
 * Model matrix helpers
 * -------------------------------------------------------------------------- */

/* model = translate(tx,ty,tz) * rotateY(ry) * scale(sx,sy,sz) */
static void va_model_matrix(float tx, float ty, float tz, float ry, float sx,
                            float sy, float sz, mat4 out)
{
  glm_mat4_identity(out);
  glm_translate(out, (vec3){tx, ty, tz});
  if (ry != 0.0f)
    glm_rotate(out, ry, (vec3){0.0f, 1.0f, 0.0f});
  if (sx != 1.0f || sy != 1.0f || sz != 1.0f)
    glm_scale(out, (vec3){sx, sy, sz});
}

/* Compute model matrices from banking + position offset, populate stagings */
static int va_compute_plane_stagings(int plane_idx, float x, float y, float z,
                                     float banking, float time_glass,
                                     float time_glass2, int slot)
{
  /* Clamp slot */
  if (slot >= VA_MAX_PLANE_DRAWS)
    return slot;

  /* Precompute light in view space (same for all planes this frame) */
  vec4 ld4;
  glm_vec4_copy3(state.light_dir, ld4);
  ld4[3] = 0.0f;
  vec4 ldv;
  glm_mat4_mulv(state.view_matrix, ld4, ldv);

  /* === Plane body === */
  mat4 model;
  va_model_matrix(x, y, z + state.cfg.plane_height, banking, 1, 1, 1, model);
  va_plane_body_ubo_t* ub = &state.plane_body_stagings[slot];

  /* MVP = proj * view * model */
  glm_mat4_mul(state.view_proj_matrix, model, ub->mvp);
  /* MV = view * model */
  glm_mat4_mul(state.view_matrix, model, ub->mv);
  glm_vec4_copy(ldv, ub->light_dir_view);
  ub->ambient[0]   = 0.5f;
  ub->ambient[1]   = 0.5f;
  ub->ambient[2]   = 0.5f;
  ub->ambient[3]   = 0.0f;
  ub->diffuse[0]   = 1.0f;
  ub->diffuse[1]   = 1.0f;
  ub->diffuse[2]   = 1.0f;
  ub->diffuse[3]   = 0.0f;
  ub->diffuse_coef = 1.0f;
  ub->diffuse_exp  = 1.0f;

  /* === Glass === */
  va_glass_ubo_t* ug = &state.glass_stagings[slot];
  glm_mat4_copy(ub->mvp, ug->mvp);
  glm_mat4_copy(ub->mv, ug->mv);
  glm_vec4_copy(ldv, ug->light_dir_view);
  ug->ambient[0]     = 0.5f;
  ug->ambient[1]     = 0.5f;
  ug->ambient[2]     = 0.5f;
  ug->ambient[3]     = 0.0f;
  ug->diffuse[0]     = 1.0f;
  ug->diffuse[1]     = 1.0f;
  ug->diffuse[2]     = 1.0f;
  ug->diffuse[3]     = 0.0f;
  ug->glass_color[0] = 0.6f;
  ug->glass_color[1] = 0.6f;
  ug->glass_color[2] = 1.0f;
  ug->glass_color[3] = 0.0f;
  ug->diffuse_coef   = 1.0f;
  ug->diffuse_exp    = 1.0f;
  ug->u_time
    = (sinf(time_glass2 * 2.0f * (float)M_PI) > 0.0f) ? time_glass : 0.2f;

  /* === Props === */
  const va_plane_preset_t* preset = &va_plane_presets[plane_idx];
  state.plane_prop_counts[slot]   = preset->prop_count;
  for (int pi = 0; pi < preset->prop_count; pi++) {
    float pox    = preset->props[pi][0];
    float poy    = preset->props[pi][1];
    float poz    = preset->props[pi][2];
    float t_prop = fmodf(state.t_prop + (float)(slot * 3 + pi) * 0.23f, 1.0f);

    /* Rotate prop local XZ offset by banking angle (same as original JS) */
    float prop_world_x
      = sinf(banking + (float)M_PI * 0.5f) * pox + sinf(banking) * poz + x;
    float prop_world_y = poy + y;
    float prop_world_z = cosf(banking + (float)M_PI * 0.5f) * pox
                         + cosf(banking) * poz + z + state.cfg.plane_height;
    float prop_rot = t_prop * 2.0f * (float)M_PI;

    mat4 prop_model;
    va_model_matrix(prop_world_x, prop_world_y, prop_world_z, prop_rot, 1, 1, 1,
                    prop_model);

    int prop_slot           = slot * VA_MAX_PROPS_PER_PLANE + pi;
    va_plane_body_ubo_t* up = &state.prop_stagings[prop_slot];
    glm_mat4_mul(state.view_proj_matrix, prop_model, up->mvp);
    glm_mat4_mul(state.view_matrix, prop_model, up->mv);
    glm_vec4_copy(ldv, up->light_dir_view);
    up->ambient[0]   = 0.5f;
    up->ambient[1]   = 0.5f;
    up->ambient[2]   = 0.5f;
    up->ambient[3]   = 0.0f;
    up->diffuse[0]   = 1.0f;
    up->diffuse[1]   = 1.0f;
    up->diffuse[2]   = 1.0f;
    up->diffuse[3]   = 0.0f;
    up->diffuse_coef = 1.0f;
    up->diffuse_exp  = 1.0f;
  }

  return slot + 1;
}

/* -------------------------------------------------------------------------- *
 * UBO write helpers
 * -------------------------------------------------------------------------- */

static void va_write_ubos(wgpu_context_t* ctx)
{
  /* Terrain */
  mat4 quad_model;
  va_model_matrix(0, 0, 0, 0, 30.0f, 30.0f, 30.0f, quad_model);
  mat4 terrain_mvp;
  glm_mat4_mul(state.view_proj_matrix, quad_model, terrain_mvp);

  float side_coef = sinf(state.t_ground2 * 2.0f * (float)M_PI);

  state.terrain_staging.uv_offset[0] = side_coef * 0.2f; /* x side */
  state.terrain_staging.uv_offset[1] = -state.t_ground;  /* y scroll */
  state.terrain_staging.uv_offset[2] = 3.8f;             /* scale */
  glm_mat4_copy(terrain_mvp, state.terrain_staging.view_proj);
  wgpuQueueWriteBuffer(ctx->queue, state.terrain_ubo, 0, &state.terrain_staging,
                       sizeof(va_terrain_ubo_t));

  /* Terrain transition (same geometry + transition params) */
  state.terrain_trans_staging.uv_offset[0] = state.terrain_staging.uv_offset[0];
  state.terrain_trans_staging.uv_offset[1] = state.terrain_staging.uv_offset[1];
  state.terrain_trans_staging.uv_offset[2] = state.terrain_staging.uv_offset[2];
  state.terrain_trans_staging.transition   = state.t_terrain_trans;
  glm_mat4_copy(terrain_mvp, state.terrain_trans_staging.view_proj);
  wgpuQueueWriteBuffer(ctx->queue, state.terrain_trans_ubo, 0,
                       &state.terrain_trans_staging,
                       sizeof(va_terrain_trans_ubo_t));

  /* Cloud */
  float cloud_y = VA_WIND_SPREAD_Y - VA_WIND_SPREAD_Y * 2.0f * state.t_clouds;
  float cloud_z = state.cfg.plane_height * 1.7f;
  mat4 cloud_model;
  va_model_matrix(0.0f, cloud_y, cloud_z, 0.0f, 9.0f, 9.0f, 9.0f, cloud_model);
  /* Apply Z rotation for cloud spin */
  glm_rotate_z(cloud_model, state.clouds_rotation, cloud_model);
  mat4 cloud_mvp;
  glm_mat4_mul(state.view_proj_matrix, cloud_model, cloud_mvp);
  glm_mat4_copy(cloud_mvp, state.cloud_staging.mvp);
  wgpuQueueWriteBuffer(ctx->queue, state.cloud_ubo, 0, &state.cloud_staging,
                       sizeof(va_cloud_ubo_t));

  /* Wind stripes */
  for (int i = 0; i < VA_WIND_COUNT; i++) {
    float o
      = (sinf((float)i / (float)VA_WIND_COUNT * (float)M_PI * 22.0f) + 1.0f)
        * 0.5f;
    float t_w = fmodf(state.t_wind + (float)i / (float)VA_WIND_COUNT, 1.0f);
    float wx  = -VA_WIND_SPREAD_X + o * VA_WIND_SPREAD_X * 2.0f;
    wx += cosf(state.t_wind2 * 2.0f * (float)M_PI) * VA_WIND_WONDER;
    float wy = VA_WIND_SPREAD_Y - VA_WIND_SPREAD_Y * 2.0f * t_w;
    float wz = state.cfg.plane_height * ((i % 2) ? 0.75f : 1.25f);

    mat4 wind_model;
    va_model_matrix(wx, wy, wz, 0.0f, 0.03f, 0.6f, 1.0f, wind_model);
    mat4 wind_mvp;
    glm_mat4_mul(state.view_proj_matrix, wind_model, wind_mvp);
    glm_mat4_copy(wind_mvp, state.wind_stagings[i].mvp);
    state.wind_stagings[i].color[0] = 0.18f;
    state.wind_stagings[i].color[1] = 0.18f;
    state.wind_stagings[i].color[2] = 0.18f;
    state.wind_stagings[i].color[3] = 1.0f;
    wgpuQueueWriteBuffer(ctx->queue, state.wind_ubo,
                         (uint64_t)(i * VA_UBO_STRIDE), &state.wind_stagings[i],
                         sizeof(va_wind_ubo_t));
  }
}

/* -------------------------------------------------------------------------- *
 * Draw plane formation
 * Returns the slot index after all planes have been staged.
 * -------------------------------------------------------------------------- */

static int va_stage_plane(float ox, float oy, float oz, float timer_off,
                          int slot)
{
  if (slot >= VA_MAX_PLANE_DRAWS)
    return slot;

  float t1  = fmodf(state.t_wonder1 + timer_off, 1.0f);
  float t2  = fmodf(state.t_wonder2 + timer_off, 1.0f);
  float tg  = fmodf(state.t_glass + timer_off, 1.0f);
  float tg2 = fmodf(state.t_glass2 + timer_off, 1.0f);

  float px
    = ox + sinf(t1 * 3.0f * (float)M_PI * 2.0f) * state.cfg.plane_wonder_xy;
  float py
    = oy + cosf(t1 * 5.0f * (float)M_PI * 2.0f) * state.cfg.plane_wonder_xy;
  float pz = oz + cosf(t2 * (float)M_PI * 2.0f) * state.cfg.plane_wonder_z;

  float side_coef = sinf(state.t_ground2 * 2.0f * (float)M_PI);
  float banking   = side_coef * state.cfg.plane_banking
                  * va_plane_presets[state.cfg.current_plane].banking;

  if (state.state_plane == VA_PLANE_FLY_AWAY) {
    float c1 = powf(state.t_plane_trans, 1.5f);
    float c2 = powf(state.t_plane_trans, 2.5f);
    banking += c1 * 0.8f * state.fly_away_side;
    pz -= c2 * 50.0f;
    px += c2 * 600.0f * state.fly_away_side;
    py += c2 * 400.0f * state.fly_away_fwd;
  }
  else if (state.state_plane == VA_PLANE_FLY_IN) {
    float c = powf(1.0f - state.t_plane_trans, 2.5f);
    pz += c * 200.0f;
    py += c * -200.0f;
  }

  return va_compute_plane_stagings(state.cfg.current_plane, px, py, pz, banking,
                                   tg, tg2, slot);
}

static int va_stage_all_planes(void)
{
  int slot = 0;
  slot     = va_stage_plane(0, 0, 0, 0.0f, slot);

  switch (state.cfg.formation) {
    case VA_FORMATION_LARGE_TRIANGLE:
      slot = va_stage_plane(180, -180, 0, 0.1f, slot);
      slot = va_stage_plane(-180, -180, 0, 0.2f, slot);
      /* fallthrough */
    case VA_FORMATION_TRIANGLE:
      slot = va_stage_plane(90, -90, 0, 0.1f, slot);
      slot = va_stage_plane(-90, -90, 0, 0.2f, slot);
      break;
    case VA_FORMATION_CROSS:
      slot = va_stage_plane(110, -110, 0, 0.1f, slot);
      slot = va_stage_plane(-110, -110, 0, 0.2f, slot);
      slot = va_stage_plane(-110, 110, 0, 0.1f, slot);
      slot = va_stage_plane(110, 110, 0, 0.2f, slot);
      break;
    case VA_FORMATION_LINE:
      slot = va_stage_plane(140, 0, 0, 0.1f, slot);
      slot = va_stage_plane(-140, 0, 0, 0.2f, slot);
      break;
    case VA_FORMATION_DIAMOND:
      slot = va_stage_plane(90, -90, 0, 0.1f, slot);
      slot = va_stage_plane(-90, -90, 0, 0.2f, slot);
      slot = va_stage_plane(0, -190, 0, 0.2f, slot);
      break;
    default:
      break;
  }
  return slot;
}

static void va_upload_plane_ubos(wgpu_context_t* ctx, int plane_count)
{
  for (int i = 0; i < plane_count; i++) {
    /* Plane body at slot i */
    wgpuQueueWriteBuffer(
      ctx->queue, state.plane_body_ubo, (uint64_t)(i * VA_UBO_STRIDE),
      &state.plane_body_stagings[i], sizeof(va_plane_body_ubo_t));
    /* Glass at slot i */
    wgpuQueueWriteBuffer(ctx->queue, state.glass_ubo,
                         (uint64_t)(i * VA_UBO_STRIDE),
                         &state.glass_stagings[i], sizeof(va_glass_ubo_t));
    /* Props at slots VA_MAX_PLANE_DRAWS + i*VA_MAX_PROPS_PER_PLANE + pi */
    for (int pi = 0; pi < state.plane_prop_counts[i]; pi++) {
      int prop_slot = i * VA_MAX_PROPS_PER_PLANE + pi;
      uint64_t offset
        = (uint64_t)((VA_MAX_PLANE_DRAWS + prop_slot) * VA_UBO_STRIDE);
      wgpuQueueWriteBuffer(ctx->queue, state.plane_body_ubo, offset,
                           &state.prop_stagings[prop_slot],
                           sizeof(va_plane_body_ubo_t));
    }
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static const char* va_formation_names[VA_FORMATION_COUNT]
  = {"Single", "Triangle", "Large Triangle", "Diamond", "Cross", "Line"};

static void va_render_gui(wgpu_context_t* ctx)
{
  /* Note: imgui_overlay_render is called by frame() AFTER scene submission. */
  imgui_overlay_new_frame(ctx, state.dt_sec);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_Once, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_Once);
  igSetNextWindowCollapsed(false, ImGuiCond_Once);

  bool open = true;
  igBegin("Voxel Airplanes", &open,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

  /* Camera mode */
  igText("Camera");
  igSameLine(0, 8);
  if (igRadioButton_Bool("Rotating",
                         state.cfg.camera_mode == VA_CAMERA_ROTATING)) {
    state.cfg.camera_mode = VA_CAMERA_ROTATING;
  }
  igSameLine(0, 8);
  if (igRadioButton_Bool("Fixed", state.cfg.camera_mode == VA_CAMERA_FIXED)) {
    state.cfg.camera_mode = VA_CAMERA_FIXED;
  }

  igSeparator();

  /* Plane */
  igText("Plane:");
  for (int i = 0; i < VA_PLANE_PRESET_COUNT; i++) {
    if (i > 0)
      igSameLine(0, 4);
    igPushID_Int(i);
    bool sel = (state.cfg.current_plane == i);
    if (igRadioButton_Bool(va_plane_presets[i].name, sel)) {
      state.cfg.current_plane = i;
      state.state_plane       = VA_PLANE_FLY_IN;
      state.t_plane_trans     = 0.0f;
    }
    igPopID();
  }

  /* Palette */
  igText("Palette:");
  igSameLine(0, 4);
  for (int i = 0; i < VA_PALETTE_COUNT; i++) {
    if (i > 0)
      igSameLine(0, 4);
    char lbl[4];
    snprintf(lbl, sizeof(lbl), "%d", i);
    igPushID_Int(100 + i);
    if (igRadioButton_Bool(lbl, state.cfg.current_palette == i)) {
      state.cfg.current_palette = i;
      state.resources_dirty     = true;
    }
    igPopID();
  }

  /* Terrain */
  igText("Terrain: %d", state.cfg.current_terrain);
  igSameLine(0, 8);
  if (igButton("Next", (ImVec2){0, 0})) {
    if (state.state_terrain == VA_TERRAIN_NORMAL) {
      int next = (state.cfg.current_terrain + 1) % VA_TERRAIN_COUNT;
      /* Start transition if target texture loaded */
      if (state.tex_terrain[next].handle) {
        state.next_terrain    = next;
        state.t_terrain_trans = 0.0f;
        state.state_terrain   = VA_TERRAIN_TRANSITION;
        /* Copy target view to tex_target_terrain for bind group */
        state.tex_target_terrain = state.tex_terrain[next];
        state.resources_dirty    = true;
      }
    }
  }

  igSeparator();

  /* Formation */
  igText("Formation");
  for (int i = 0; i < VA_FORMATION_COUNT; i++) {
    if (i > 0)
      igSameLine(0, 4);
    igPushID_Int(200 + i);
    if (igRadioButton_Bool(va_formation_names[i],
                           state.cfg.formation == (va_formation_t)i)) {
      state.cfg.formation = (va_formation_t)i;
    }
    igPopID();
  }

  igSeparator();

  /* Sliders */
  igSliderFloat("Height", &state.cfg.plane_height, 0.0f, 300.0f, "%.0f", 0);
  igSliderFloat("Banking", &state.cfg.plane_banking, 0.0f, 2.0f, "%.2f", 0);
  igSliderFloat("Wonder XY", &state.cfg.plane_wonder_xy, 0.0f, 20.0f, "%.1f",
                0);
  igSliderFloat("Wonder Z", &state.cfg.plane_wonder_z, 0.0f, 20.0f, "%.1f", 0);

  /* Loading status */
  if (!state.initialized) {
    igSeparator();
    igText("Loading... (%d/%d models, %d textures)", state.models_loaded,
           state.total_models, state.textures_loaded);
  }

  igEnd();
  /* imgui_overlay_render is called by frame() after the scene command buffer */
}

/* -------------------------------------------------------------------------- *
 * Main render
 * -------------------------------------------------------------------------- */

static void va_draw_scene(WGPURenderPassEncoder rpe, int plane_count)
{
  int cur_plane       = state.cfg.current_plane;
  va_model_t* m_body  = &state.model_plane[cur_plane];
  va_model_t* m_glass = &state.model_glass[cur_plane];
  va_model_t* m_prop  = &state.model_prop;
  va_model_t* m_quad  = &state.model_quad;
  va_model_t* m_cloud = &state.model_cloud;

  /* 1. Terrain */
  if (state.state_terrain == VA_TERRAIN_TRANSITION) {
    wgpuRenderPassEncoderSetPipeline(rpe, state.terrain_trans_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_terrain_trans, 0, NULL);
  }
  else {
    wgpuRenderPassEncoderSetPipeline(rpe, state.terrain_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_terrain, 0, NULL);
  }
  if (m_quad->is_ready) {
    wgpuRenderPassEncoderSetVertexBuffer(rpe, 0, m_quad->vtx, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      rpe, m_quad->idx, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rpe, m_quad->index_count, 1, 0, 0, 0);
  }

  /* 2. Plane bodies + glass + props */
  if (m_body->is_ready && m_glass->is_ready && m_prop->is_ready) {
    for (int slot = 0; slot < plane_count; slot++) {
      uint32_t body_offset  = (uint32_t)(slot * VA_UBO_STRIDE);
      uint32_t glass_offset = (uint32_t)(slot * VA_UBO_STRIDE);

      /* Glass (drawn first, blended on top) */
      wgpuRenderPassEncoderSetPipeline(rpe, state.glass_pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_glass, 1,
                                        &glass_offset);
      wgpuRenderPassEncoderSetVertexBuffer(rpe, 0, m_glass->vtx, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(
        rpe, m_glass->idx, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(rpe, m_glass->index_count, 1, 0, 0, 0);

      /* Plane body */
      wgpuRenderPassEncoderSetPipeline(rpe, state.plane_body_pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_plane_body, 1,
                                        &body_offset);
      wgpuRenderPassEncoderSetVertexBuffer(rpe, 0, m_body->vtx, 0,
                                           WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderSetIndexBuffer(
        rpe, m_body->idx, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
      wgpuRenderPassEncoderDrawIndexed(rpe, m_body->index_count, 1, 0, 0, 0);

      /* Props: use pre-computed UBO at offset beyond plane body slots */
      for (int pi = 0; pi < state.plane_prop_counts[slot]; pi++) {
        int prop_slot = slot * VA_MAX_PROPS_PER_PLANE + pi;
        uint32_t prop_off
          = (uint32_t)((VA_MAX_PLANE_DRAWS + prop_slot) * VA_UBO_STRIDE);
        wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_plane_body, 1,
                                          &prop_off);
        wgpuRenderPassEncoderSetVertexBuffer(rpe, 0, m_prop->vtx, 0,
                                             WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(
          rpe, m_prop->idx, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(rpe, m_prop->index_count, 1, 0, 0, 0);
      }
    }
  }

  /* 3. Wind stripes */
  wgpuRenderPassEncoderSetPipeline(rpe, state.wind_pipeline);
  for (int i = 0; i < VA_WIND_COUNT; i++) {
    uint32_t offset = (uint32_t)(i * VA_UBO_STRIDE);
    wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_wind, 1, &offset);
    wgpuRenderPassEncoderDraw(rpe, 4, 1, 0, 0); /* no vertex buffer */
  }

  /* 4. Clouds */
  if (m_cloud->is_ready) {
    /* Update cloud rotation if cycle just started */
    if (state.t_clouds < 0.01f) {
      /* Random quarter-turn rotation */
      state.clouds_rotation = 2.0f * (float)M_PI * (float)(rand() % 4) / 4.0f;
    }
    wgpuRenderPassEncoderSetPipeline(rpe, state.cloud_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpe, 0, state.bg_cloud, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(rpe, 0, m_cloud->vtx, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      rpe, m_cloud->idx, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rpe, m_cloud->index_count, 1, 0, 0, 0);
  }
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static void va_check_and_upload_pending(wgpu_context_t* ctx)
{
  /* Upload any model data that arrived via fetch callbacks */
  for (int i = 0; i < VA_PLANE_PRESET_COUNT; i++) {
    va_upload_model(ctx, &state.model_plane[i]);
    va_upload_model(ctx, &state.model_glass[i]);
  }
  va_upload_model(ctx, &state.model_prop);
  va_upload_model(ctx, &state.model_cloud);
  va_upload_model(ctx, &state.model_quad);

  /* Upload textures */
  for (int i = 0; i < VA_TERRAIN_COUNT; i++)
    va_upload_texture(ctx, &state.tex_terrain[i]);
  for (int i = 0; i < VA_PALETTE_COUNT; i++)
    va_upload_texture(ctx, &state.tex_palette[i]);
  va_upload_texture(ctx, &state.tex_glass);
  va_upload_texture(ctx, &state.tex_cloud);
  va_upload_texture(ctx, &state.tex_noise);

  if (state.resources_dirty && state.initialized) {
    va_create_bind_groups(ctx);
    state.resources_dirty = false;
  }
}

static void va_update_timers(double dt_ms)
{
  uint64_t now_ms = stm_ms(stm_now());

  state.t_camera = (state.cfg.camera_mode == VA_CAMERA_FIXED) ?
                     0.13f :
                     (float)fmod((double)now_ms, (double)VA_CAMERA_PERIOD)
                       / VA_CAMERA_PERIOD;
  state.t_ground
    = (float)fmod((double)now_ms, (double)VA_GROUND_PERIOD) / VA_GROUND_PERIOD;
  state.t_ground2 = (float)fmod((double)now_ms, (double)VA_GROUND2_PERIOD)
                    / VA_GROUND2_PERIOD;
  state.t_wonder1 = (float)fmod((double)now_ms, (double)VA_WONDER1_PERIOD)
                    / VA_WONDER1_PERIOD;
  state.t_wonder2 = (float)fmod((double)now_ms, (double)VA_WONDER2_PERIOD)
                    / VA_WONDER2_PERIOD;
  state.t_prop
    = (float)fmod((double)now_ms, (double)VA_PROP_PERIOD) / VA_PROP_PERIOD;
  state.t_glass
    = (float)fmod((double)now_ms, (double)VA_GLASS_PERIOD) / VA_GLASS_PERIOD;
  state.t_glass2
    = (float)fmod((double)now_ms, (double)VA_GLASS2_PERIOD) / VA_GLASS2_PERIOD;
  state.t_wind
    = (float)fmod((double)now_ms, (double)VA_WIND_PERIOD) / VA_WIND_PERIOD;
  state.t_wind2
    = (float)fmod((double)now_ms, (double)VA_WIND2_PERIOD) / VA_WIND2_PERIOD;
  state.t_clouds
    = (float)fmod((double)now_ms, (double)VA_CLOUDS_PERIOD) / VA_CLOUDS_PERIOD;

  /* Transitions (delta-time based) */
  if (state.state_plane != VA_PLANE_NORMAL) {
    state.t_plane_trans += (float)(dt_ms / VA_PLANE_TRANS_PERIOD);
    if (state.t_plane_trans >= 1.0f) {
      state.t_plane_trans = 1.0f;
      if (state.state_plane == VA_PLANE_FLY_IN)
        state.state_plane = VA_PLANE_NORMAL;
      else if (state.state_plane == VA_PLANE_FLY_AWAY)
        state.state_plane = VA_PLANE_FLY_IN; /* switch to fly-in */
    }
  }

  if (state.state_terrain == VA_TERRAIN_TRANSITION) {
    state.t_terrain_trans += (float)(dt_ms / VA_TERRAIN_TRANS_PERIOD);
    if (state.t_terrain_trans >= 1.0f) {
      state.t_terrain_trans           = 1.0f;
      state.cfg.current_terrain       = state.next_terrain;
      state.state_terrain             = VA_TERRAIN_NORMAL;
      state.tex_target_terrain.handle = NULL;
      state.tex_target_terrain.view   = NULL;
      state.resources_dirty           = true;
    }
  }
}

static int frame(wgpu_context_t* ctx)
{
  sfetch_dowork();

  uint64_t now = stm_now();
  double dt_ms
    = (state.last_time > 0) ? stm_ms(stm_diff(now, state.last_time)) : 0.0;
  state.last_time = now;
  state.dt_sec    = (float)(dt_ms * 0.001);

  va_check_and_upload_pending(ctx);

  /* Initialize once the required GPU buffers (models) are ready.
   * Textures start as valid placeholder color-bars; they are replaced by
   * real textures as sfetch completes, which sets resources_dirty = true
   * and triggers a bind-group recreation next frame. */
  if (!state.initialized) {
    bool models_ready = state.model_quad.is_ready && state.model_cloud.is_ready
                        && state.model_prop.is_ready
                        && state.model_plane[0].is_ready
                        && state.model_glass[0].is_ready;
    if (models_ready) {
      va_create_bind_groups(ctx);
      state.resources_dirty = false;
      state.initialized     = true;
    }
  }

  va_update_timers(dt_ms);

  /* Recompute view/projection each frame */
  va_position_camera();
  va_set_projection(ctx);
  glm_mat4_mul(state.proj_matrix, state.view_matrix, state.view_proj_matrix);

  /* Stage all plane UBOs */
  int plane_count = va_stage_all_planes();

  /* Write all UBOs to GPU (before render pass) */
  va_write_ubos(ctx);
  va_upload_plane_ubos(ctx, plane_count);

  /* Build GUI — must happen before imgui_overlay_render */
  va_render_gui(ctx);

  /* Scene render pass */
  state.color_attach.view = ctx->swapchain_view;
  state.depth_attach.view = ctx->depth_stencil_view;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
  WGPURenderPassEncoder rpe
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_desc);

  if (state.initialized) {
    va_draw_scene(rpe, plane_count);
  }

  wgpuRenderPassEncoderEnd(rpe);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &cmd);

  /* ImGui overlay — must be submitted AFTER the scene command buffer */
  imgui_overlay_render(ctx);

  wgpuRenderPassEncoderRelease(rpe);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Input / resize
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* ctx, const input_event_t* ev)
{
  imgui_overlay_handle_input(ctx, ev);
  if (ev->type == INPUT_EVENT_TYPE_RESIZED) {
    va_set_projection(ctx);
  }
}

/* -------------------------------------------------------------------------- *
 * Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* ctx)
{
  stm_setup();
  srand((unsigned)stm_now());

  state.wgpu_context = ctx;

  /* Samplers */
  state.sampler_linear = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("VA Linear - Sampler"),
                   .minFilter     = WGPUFilterMode_Linear,
                   .magFilter     = WGPUFilterMode_Linear,
                   .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                   .addressModeU  = WGPUAddressMode_Repeat,
                   .addressModeV  = WGPUAddressMode_Repeat,
                   .maxAnisotropy = 1,
                 });
  state.sampler_nearest = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("VA Nearest - Sampler"),
                   .minFilter     = WGPUFilterMode_Nearest,
                   .magFilter     = WGPUFilterMode_Nearest,
                   .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                   .addressModeU  = WGPUAddressMode_ClampToEdge,
                   .addressModeV  = WGPUAddressMode_ClampToEdge,
                   .maxAnisotropy = 1,
                 });

  /* UBO buffers */
  state.terrain_ubo = va_create_ubo(ctx, VA_UBO_STRIDE, "VA Terrain UBO");
  state.terrain_trans_ubo
    = va_create_ubo(ctx, VA_UBO_STRIDE, "VA Terrain Trans UBO");
  state.plane_body_ubo = va_create_ubo(
    ctx, (size_t)VA_BODY_UBO_TOTAL * VA_UBO_STRIDE, "VA Plane Body UBO");
  state.glass_ubo = va_create_ubo(
    ctx, (size_t)VA_MAX_PLANE_DRAWS * VA_UBO_STRIDE, "VA Glass UBO");
  state.wind_ubo
    = va_create_ubo(ctx, (size_t)VA_WIND_COUNT * VA_UBO_STRIDE, "VA Wind UBO");
  state.cloud_ubo = va_create_ubo(ctx, VA_UBO_STRIDE, "VA Cloud UBO");

  /* Create pipelines */
  va_create_terrain_pipeline(ctx);
  va_create_terrain_trans_pipeline(ctx);
  va_create_plane_body_pipeline(ctx);
  va_create_glass_pipeline(ctx);
  va_create_wind_pipeline(ctx);
  va_create_cloud_pipeline(ctx);

  /* sfetch setup */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 128,
    .num_channels = 4,
    .num_lanes    = 4,
#ifndef __WAJIC__
    .logger = {.func = slog_func},
#endif
  });

  /* Load models */
  for (int i = 0; i < VA_PLANE_PRESET_COUNT; i++) {
    char base[32];
    snprintf(base, sizeof(base), "%s-plane", va_plane_presets[i].name);
    va_load_model(&state.model_plane[i], base, false);
    snprintf(base, sizeof(base), "%s-glass", va_plane_presets[i].name);
    va_load_model(&state.model_glass[i], base, true);
  }
  va_load_model(&state.model_prop, "prop", false);
  va_load_model(&state.model_cloud, "cloud", false);
  va_load_model(&state.model_quad, "quad", false);

  /* Load textures */
  char path[256];
  for (int i = 0; i < VA_TERRAIN_COUNT; i++) {
    snprintf(path, sizeof(path), "%s/textures/ground%d.png", VA_ASSETS, i);
    va_load_texture(ctx, &state.tex_terrain[i], path, false,
                    &state.textures_loaded);
  }
  for (int i = 0; i < VA_PALETTE_COUNT; i++) {
    snprintf(path, sizeof(path), "%s/textures/palette%d.png", VA_ASSETS, i);
    va_load_texture(ctx, &state.tex_palette[i], path, true,
                    &state.textures_loaded);
  }
  snprintf(path, sizeof(path), "%s/textures/glass.png", VA_ASSETS);
  va_load_texture(ctx, &state.tex_glass, path, true, &state.textures_loaded);
  snprintf(path, sizeof(path), "%s/textures/clouds.png", VA_ASSETS);
  va_load_texture(ctx, &state.tex_cloud, path, true, &state.textures_loaded);
  snprintf(path, sizeof(path), "%s/textures/blue-noise.png", VA_ASSETS);
  va_load_texture(ctx, &state.tex_noise, path, true, &state.textures_loaded);
  state.total_textures = VA_TERRAIN_COUNT + VA_PALETTE_COUNT + 3;

  /* Initialize ImGui */
  imgui_overlay_init(ctx);

  va_set_projection(ctx);
  state.clouds_rotation = 0.0f;

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* ctx)
{
  (void)ctx;

  sfetch_shutdown();
  imgui_overlay_shutdown();

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_terrain)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_terrain_trans)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_plane_body)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_glass)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_wind)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_cloud)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_terrain)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_terrain_trans)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_plane_body)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_glass)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_wind)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_cloud)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.terrain_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.terrain_trans_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.plane_body_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.glass_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.wind_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.cloud_pipeline)

  /* UBO buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.terrain_trans_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_body_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.glass_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.wind_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.cloud_ubo)

  /* Model buffers */
  for (int i = 0; i < VA_PLANE_PRESET_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(Buffer, state.model_plane[i].vtx)
    WGPU_RELEASE_RESOURCE(Buffer, state.model_plane[i].idx)
    WGPU_RELEASE_RESOURCE(Buffer, state.model_glass[i].vtx)
    WGPU_RELEASE_RESOURCE(Buffer, state.model_glass[i].idx)
  }
  WGPU_RELEASE_RESOURCE(Buffer, state.model_prop.vtx)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_prop.idx)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_cloud.vtx)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_cloud.idx)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_quad.vtx)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_quad.idx)

  /* Textures */
  for (int i = 0; i < VA_TERRAIN_COUNT; i++)
    wgpu_destroy_texture(&state.tex_terrain[i]);
  for (int i = 0; i < VA_PALETTE_COUNT; i++)
    wgpu_destroy_texture(&state.tex_palette[i]);
  wgpu_destroy_texture(&state.tex_glass);
  wgpu_destroy_texture(&state.tex_cloud);
  wgpu_destroy_texture(&state.tex_noise);

  /* Samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_linear)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_nearest)
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Voxel Airplanes",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });
  return EXIT_SUCCESS;
}

/* ============================================================================
 * WGSL Shaders
 * ========================================================================== */

// clang-format off

/* ---- Terrain shader (DiffuseScrollingFiltered) ---- */
static const char* va_terrain_shader_wgsl = CODE(
  struct Uniforms {
    view_proj : mat4x4f,
    uv_offset : vec3f,
    _pad      : f32,
  }

  @group(0) @binding(0) var<uniform> u    : Uniforms;
  @group(0) @binding(1) var          samp : sampler;
  @group(0) @binding(2) var          tex  : texture_2d<f32>;

  struct VIn  { @location(0) pos : vec3f, @location(1) uv : vec2f }
  struct VOut {
    @builtin(position) clip : vec4f,
    @location(0) tex_coord  : vec2f,
  }

  fn v2len(a : vec2f, b : vec2f) -> vec2f { return sqrt(a * a + b * b); }

  fn texture_blocky(uv : vec2f, res : vec2f) -> vec4f {
    var uv2 = uv * res;
    let seam = floor(uv2 + 0.5);
    uv2 = (uv2 - seam) / v2len(dpdx(uv2), dpdy(uv2)) + seam;
    uv2 = clamp(uv2, seam - 0.5, seam + 0.5);
    return textureSampleBias(tex, samp, uv2 / res, -1000.0);
  }

  @vertex fn vertexMain(in : VIn) -> VOut {
    var out : VOut;
    out.clip      = u.view_proj * vec4f(in.pos, 1.0);
    out.tex_coord = in.uv * u.uv_offset.z + u.uv_offset.xy;
    return out;
  }

  @fragment fn fragmentMain(in : VOut) -> @location(0) vec4f {
    return texture_blocky(in.tex_coord, vec2f(256.0));
  }
);

/* ---- Terrain transition shader ---- */
static const char* va_terrain_trans_shader_wgsl = CODE(
  struct Uniforms {
    view_proj  : mat4x4f,
    uv_offset  : vec3f,
    transition : f32,
  }

  @group(0) @binding(0) var<uniform> u          : Uniforms;
  @group(0) @binding(1) var          samp        : sampler;
  @group(0) @binding(2) var          tex_src     : texture_2d<f32>;
  @group(0) @binding(3) var          tex_dst     : texture_2d<f32>;
  @group(0) @binding(4) var          tex_mask    : texture_2d<f32>;

  struct VIn  { @location(0) pos : vec3f, @location(1) uv : vec2f }
  struct VOut {
    @builtin(position) clip : vec4f,
    @location(0) tex_coord  : vec2f,
  }

  fn v2len(a : vec2f, b : vec2f) -> vec2f { return sqrt(a * a + b * b); }

  fn texture_blocky_src(uv : vec2f, res : vec2f) -> vec4f {
    var uv2 = uv * res;
    let seam = floor(uv2 + 0.5);
    uv2 = (uv2 - seam) / v2len(dpdx(uv2), dpdy(uv2)) + seam;
    uv2 = clamp(uv2, seam - 0.5, seam + 0.5);
    return textureSampleBias(tex_src, samp, uv2 / res, -1000.0);
  }

  fn texture_blocky_dst(uv : vec2f, res : vec2f) -> vec4f {
    var uv2 = uv * res;
    let seam = floor(uv2 + 0.5);
    uv2 = (uv2 - seam) / v2len(dpdx(uv2), dpdy(uv2)) + seam;
    uv2 = clamp(uv2, seam - 0.5, seam + 0.5);
    return textureSampleBias(tex_dst, samp, uv2 / res, -1000.0);
  }

  @vertex fn vertexMain(in : VIn) -> VOut {
    var out : VOut;
    out.clip      = u.view_proj * vec4f(in.pos, 1.0);
    out.tex_coord = in.uv * u.uv_offset.z + u.uv_offset.xy;
    return out;
  }

  @fragment fn fragmentMain(in : VOut) -> @location(0) vec4f {
    let c1   = texture_blocky_src(in.tex_coord, vec2f(256.0));
    let c2   = texture_blocky_dst(in.tex_coord, vec2f(256.0));
    let mask = textureSampleLevel(tex_mask, samp, in.tex_coord, 0.0).r;
    let t    = smoothstep(0.4, 0.6, u.transition + u.transition * mask);
    return mix(c1, c2, t);
  }
);

/* ---- Plane body shader (PlaneBodyLit) ---- */
static const char* va_plane_body_shader_wgsl = CODE(
  struct Uniforms {
    mvp           : mat4x4f,
    mv            : mat4x4f,
    light_dir_view : vec4f,
    ambient       : vec4f,
    diffuse       : vec4f,
    diffuse_coef  : f32,
    diffuse_exp   : f32,
    _pad          : vec2f,
  }

  @group(0) @binding(0) var<uniform> u       : Uniforms;
  /* binding 1: palette texture — no sampler needed, we use textureLoad */
  @group(0) @binding(1) var          palette : texture_2d<f32>;

  const NORMALS = array<vec3f, 6>(
    vec3f( 1.0,  0.0,  0.0),
    vec3f(-1.0,  0.0,  0.0),
    vec3f( 0.0,  1.0,  0.0),
    vec3f( 0.0, -1.0,  0.0),
    vec3f( 0.0,  0.0,  1.0),
    vec3f( 0.0,  0.0, -1.0),
  );

  struct VIn  { @location(0) packed : vec4i }
  struct VOut {
    @builtin(position) clip  : vec4f,
    @location(0) diffuse_col : vec4f,
  }

  @vertex fn vertexMain(in : VIn) -> VOut {
    var out : VOut;
    let pos = vec3f(f32(in.packed.x), f32(in.packed.y), f32(in.packed.z));
    out.clip = u.mvp * vec4f(pos, 1.0);

    let nc        = bitcast<u32>(in.packed.w) & 0xffu;
    let normal_idx = nc & 7u;
    let color_idx  = nc >> 3u;

    let world_normal = vec4f(NORMALS[normal_idx], 0.0);
    let view_normal  = normalize((u.mv * world_normal).xyz);
    let d = pow(max(0.0, dot(view_normal, normalize(u.light_dir_view.xyz))),
                u.diffuse_exp);
    let lighting = mix(u.ambient, u.diffuse, d * u.diffuse_coef);

    let palette_col = textureLoad(palette, vec2u(color_idx, 0u), 0);
    out.diffuse_col = lighting * palette_col;
    return out;
  }

  @fragment fn fragmentMain(in : VOut) -> @location(0) vec4f {
    return in.diffuse_col;
  }
);

/* ---- Glass shader ---- */
static const char* va_glass_shader_wgsl = CODE(
  struct Uniforms {
    mvp           : mat4x4f,
    mv            : mat4x4f,
    light_dir_view : vec4f,
    ambient       : vec4f,
    diffuse       : vec4f,
    glass_color   : vec4f,
    diffuse_coef  : f32,
    diffuse_exp   : f32,
    u_time        : f32,
    _pad          : f32,
  }

  @group(0) @binding(0) var<uniform> u    : Uniforms;
  @group(0) @binding(1) var          samp : sampler;
  @group(0) @binding(2) var          tex  : texture_2d<f32>;

  const NORMALS = array<vec3f, 6>(
    vec3f( 1.0,  0.0,  0.0),
    vec3f(-1.0,  0.0,  0.0),
    vec3f( 0.0,  1.0,  0.0),
    vec3f( 0.0, -1.0,  0.0),
    vec3f( 0.0,  0.0,  1.0),
    vec3f( 0.0,  0.0, -1.0),
  );

  struct VIn  { @location(0) packed : vec4i }
  struct VOut {
    @builtin(position) clip  : vec4f,
    @location(0) diffuse_col : vec4f,
    @location(1) tex_coord   : vec2f,
  }

  @vertex fn vertexMain(in : VIn) -> VOut {
    var out : VOut;
    let pos = vec3f(f32(in.packed.x), f32(in.packed.y), f32(in.packed.z));
    out.clip = u.mvp * vec4f(pos, 1.0);

    let normal_idx = bitcast<u32>(in.packed.w) & 0x7u;
    let world_normal = vec4f(NORMALS[normal_idx], 0.0);
    let view_normal  = normalize((u.mv * world_normal).xyz);
    let d = pow(max(0.0, dot(view_normal, normalize(u.light_dir_view.xyz))),
                u.diffuse_exp);
    let lighting = mix(u.ambient, u.diffuse, d * u.diffuse_coef);
    out.diffuse_col = lighting * u.glass_color;

    out.tex_coord   = pos.xy * 0.02;
    out.tex_coord.y = out.tex_coord.y + u.u_time;
    return out;
  }

  @fragment fn fragmentMain(in : VOut) -> @location(0) vec4f {
    var col = in.diffuse_col;
    col = col + textureSample(tex, samp, in.tex_coord);
    return col;
  }
);

/* ---- Wind stripe shader (hardcoded quad) ---- */
static const char* va_wind_shader_wgsl = CODE(
  struct Uniforms {
    mvp   : mat4x4f,
    color : vec4f,
  }

  @group(0) @binding(0) var<uniform> u : Uniforms;

  const VERTS = array<vec3f, 4>(
    vec3f(-50.0, -50.0, 0.0),
    vec3f( 50.0, -50.0, 0.0),
    vec3f(-50.0,  50.0, 0.0),
    vec3f( 50.0,  50.0, 0.0),
  );

  @vertex fn vertexMain(@builtin(vertex_index) vid : u32) -> @builtin(position) vec4f {
    return u.mvp * vec4f(VERTS[vid], 1.0);
  }

  @fragment fn fragmentMain() -> @location(0) vec4f {
    return u.color;
  }
);

/* ---- Cloud shader (simple diffuse) ---- */
static const char* va_cloud_shader_wgsl = CODE(
  struct Uniforms {
    mvp : mat4x4f,
  }

  @group(0) @binding(0) var<uniform> u    : Uniforms;
  @group(0) @binding(1) var          samp : sampler;
  @group(0) @binding(2) var          tex  : texture_2d<f32>;

  struct VIn  { @location(0) pos : vec3f, @location(1) uv : vec2f }
  struct VOut {
    @builtin(position) clip : vec4f,
    @location(0) tex_coord  : vec2f,
  }

  @vertex fn vertexMain(in : VIn) -> VOut {
    var out : VOut;
    out.clip      = u.mvp * vec4f(in.pos, 1.0);
    out.tex_coord = in.uv;
    return out;
  }

  @fragment fn fragmentMain(in : VOut) -> @location(0) vec4f {
    return textureSample(tex, samp, in.tex_coord);
  }
);

// clang-format on
