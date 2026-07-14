/* ============================================================================
 * Floating Rock Pillars
 *
 * C99/WebGPU port of the "webgl-rock-pillars" WebGL demo by Kean Morel
 * (https://github.com/keaukraine/webgl-rock-pillars).
 *
 * Demonstrates instanced rendering of floating rock pillar formations with:
 *  - 5 groups of instanced rock pillars with vertex lighting and grass blending
 *  - Instanced foliage (ferns, pine trees) with alpha test
 *  - Animated instanced birds
 *  - Billboard cloud/smoke particles with soft-particle depth fade
 *  - Cubemap skybox with 4 presets (Sunrise / Day / Sunset / Night)
 *  - Height-based and distance-based fog sampling the cubemap
 *  - Procedurally placed instances along a spline path
 *  - Rotating spline camera
 *
 * Vertex formats (converted from WebGL half-float at load time):
 *   Rock models (with normal): stride 32 = pos(float3) + uv(float2) +
 * norm(float3) Foliage / tree / sky / cloud: stride 20 = pos(float3) +
 * uv(float2) Bird animation: stride 68 = 5×pos(float3) + uv(float2)  (float32)
 *
 * Instance data (per-instance vertex buffer, 16 bytes):
 *   vec4f(tx, ty, scale_rand, angle_rad)
 * ========================================================================== */

#ifdef __WAJIC__
#define WAJIC_IMAGE_IMPL
#include <wajic_image.h>
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define RP_ASSETS_BASE "assets/models/RockPillars"
#define RP_Z_NEAR 2.0f
#define RP_Z_FAR 1100.0f
#define RP_FOV_DEG 60.0f
#define RP_SMOKE_SOFT 0.012f
#define RP_UBO_STRIDE 256u

/* Instance group counts (matching TypeScript ObjectsPlacement.ts) */
#define RP_ROCKS1_COUNT 97
#define RP_ROCKS2_COUNT 119
#define RP_ROCKS3_COUNT 60
#define RP_ROCKS4_COUNT 40
#define RP_ROCKS5_COUNT 70
#define RP_TREES_COUNT 1360
#define RP_SMOKE_COUNT 40

#define RP_BIRD_FRAMES 5

/* Cubemap face order: +X,-X,+Y,-Y,+Z,-Z */
static const char* const RP_CUBEMAP_FACES[6] = {
  "sky-posx.png", "sky-negx.png", "sky-posy.png",
  "sky-negy.png", "sky-posz.png", "sky-negz.png",
};
#define RP_CUBEMAP_PRESET_COUNT 4

/* Camera spline period and fog period (milliseconds) */
#define RP_CAMERA_PERIOD 166000.0f
#define RP_FOG_PERIOD 42000.0f
#define RP_BIRD_ANIM_PERIOD 1200.0f
#define RP_ROCKS_MOVE_PERIOD 8000.0f

/* Timers (monotone) */
#define RP_PRESET_COUNT 4

/* -------------------------------------------------------------------------- *
 * Shader strings (forward declarations)
 * -------------------------------------------------------------------------- */

static const char* rp_sky_shader_wgsl;
static const char* rp_rocks_shader_wgsl;
static const char* rp_fern_shader_wgsl;
static const char* rp_birds_shader_wgsl;
static const char* rp_smoke_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Uniform buffer layout (shared by all draws, 248 bytes < 256 stride)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 view_proj; /* 64 */
  mat4 view;      /* 64 */
  vec4 color;     /* 16 ambient */
  vec4 color_sun; /* 16 sun diffuse */
  vec4 light_dir; /* 16 */
  /* x=fog_start, y=fog_dist, z=height_fog_offset, w=height_fog_mult */
  vec4 fog_params; /* 16 */
  /* x=height_rand_mult, y=height_fixed, z=scale_base, w=scale_range */
  vec4 placement_params; /* 16 */
  /* x=diffuse_exp, y=grass_amount, z=morph(birds), w=timer_camera */
  vec4 misc_params; /* 16 */
} rp_ubo_t;         /* = 224 bytes */

/* -------------------------------------------------------------------------- *
 * Model (index + vertex buffers)
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_buffer_t vtx;
  wgpu_buffer_t idx;
  uint32_t index_count;
  bool is_ready;
  bool vtx_pending, idx_pending;
  uint8_t* pending_vtx;
  uint8_t* pending_idx;
  size_t pending_vtx_sz;
  size_t pending_idx_sz;
} rp_model_t;

/* -------------------------------------------------------------------------- *
 * Preset (scene lighting + cubemap)
 * -------------------------------------------------------------------------- */

typedef struct {
  const char* name;
  const char* cubemap_dir;
  vec4 light_dir;
  float diffuse_exponent;
  vec4 clouds_color;
  vec4 color_sun;
  vec4 color_ambient;
} rp_preset_t;

static const rp_preset_t rp_presets[RP_PRESET_COUNT] = {
  {
    .name             = "Sunrise",
    .cubemap_dir      = "sunrise1",
    .light_dir        = {0, 1, 0, 1},
    .diffuse_exponent = 0.025f,
    .clouds_color     = {0.17f, 0.17f, 0.17f, 1},
    .color_sun        = {255 / 255.f, 167 / 255.f, 53 / 255.f, 1},
    .color_ambient    = {93 / 255.f, 81 / 255.f, 112 / 255.f, 1},
  },
  {
    .name             = "Day",
    .cubemap_dir      = "day1",
    .light_dir        = {0, 1, 0, 1},
    .diffuse_exponent = 0.025f,
    .clouds_color     = {0.17f, 0.17f, 0.17f, 1},
    .color_sun        = {1, 1, 1, 1},
    .color_ambient    = {61 / 255.f, 95 / 255.f, 156 / 255.f, 1},
  },
  {
    .name             = "Sunset",
    .cubemap_dir      = "sunset1",
    .light_dir        = {0, 1, 0, 1},
    .diffuse_exponent = 0.025f,
    .clouds_color     = {0.17f, 0.17f, 0.17f, 1},
    .color_sun        = {1.f, 0.78f, 0.2f, 1},
    .color_ambient    = {0.43f, 0.45f, 0.47f, 1},
  },
  {
    .name             = "Night",
    .cubemap_dir      = "night1",
    .light_dir        = {0, 1, 0, 1},
    .diffuse_exponent = 0.025f,
    .clouds_color     = {24 / 255.f, 28 / 255.f, 34 / 255.f, 1},
    .color_sun        = {44 / 255.f, 60 / 255.f, 93 / 255.f, 1},
    .color_ambient    = {37 / 255.f, 42 / 255.f, 51 / 255.f, 1},
  },
};

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

/* Number of draw slots for pre-computed UBOs:
 * sky(1) + 5*rock(1) + 5*grass(1) + trees(1) + 6*birds(2) + smoke(1) = 25 */
#define RP_UBO_DRAW_SLOTS 32u

typedef struct {
  wgpu_context_t* wgpu_context;

  /* Models */
  rp_model_t sky_model;
  rp_model_t rock1_model;  /* rock-8  pillar */
  rp_model_t rock2_model;  /* rock-11 narrow */
  rp_model_t rock3_model;  /* rock-6  wide   */
  rp_model_t rock1g_model; /* rock-8-grass  */
  rp_model_t rock2g_model; /* rock-11-grass */
  rp_model_t rock3g_model; /* rock-6-grass  */
  rp_model_t tree_model;   /* pinetree */
  rp_model_t bird_model;   /* bird-anim-uv */
  rp_model_t smoke_model;  /* cloud */

  /* Textures */
  wgpu_texture_t tex_rocks;
  wgpu_texture_t tex_trees;
  wgpu_texture_t tex_grass;
  wgpu_texture_t tex_fern;
  wgpu_texture_t tex_bird;
  wgpu_texture_t tex_smoke;
  wgpu_texture_t tex_white;
  wgpu_texture_t tex_cubemap; /* current preset cubemap (6 layers) */

  /* Cubemap face loading (6 faces * 4 pixels each = 24 loads) */
  uint8_t* cubemap_pixels[6]; /* decoded RGBA pixels per face */
  int cubemap_w, cubemap_h;
  int cubemap_faces_loaded;
  bool cubemap_dirty;

  /* Instance vertex buffers (one per object group) */
  WGPUBuffer inst_rocks1;
  WGPUBuffer inst_rocks2;
  WGPUBuffer inst_rocks3;
  WGPUBuffer inst_rocks4;
  WGPUBuffer inst_rocks5;
  WGPUBuffer inst_trees;
  WGPUBuffer inst_smoke;

  /* Smoke positions (for billboard computation) */
  float smoke_positions[RP_SMOKE_COUNT * 3]; /* x,y,z */

  /* Bird paths (6 paths) */
  struct {
    float start[3], end[3];
    float t;        /* 0..1 timer */
    float inv_dur;  /* 1/duration */
    float rotation; /* yaw angle */
    bool reverse;
    float last_time;
  } bird_paths[6];

  /* Pipelines */
  WGPURenderPipeline sky_pipeline;
  WGPURenderPipeline rocks_pipeline;
  WGPURenderPipeline fern_pipeline;  /* alpha test, no normal */
  WGPURenderPipeline trees_pipeline; /* alpha test, no normal */
  WGPURenderPipeline birds_pipeline;
  WGPURenderPipeline smoke_pipeline;
  WGPURenderPipeline depth_pipeline; /* depth-only for rocks */

  /* Bind group layouts */
  WGPUBindGroupLayout bgl_sky;
  WGPUBindGroupLayout bgl_rocks;
  WGPUBindGroupLayout bgl_fern;
  WGPUBindGroupLayout bgl_birds;
  WGPUBindGroupLayout bgl_smoke;

  /* Pipeline layouts */
  WGPUPipelineLayout pl_sky;
  WGPUPipelineLayout pl_rocks;
  WGPUPipelineLayout pl_fern;
  WGPUPipelineLayout pl_birds;
  WGPUPipelineLayout pl_smoke;
  WGPUPipelineLayout pl_depth;

  /* Bind groups */
  WGPUBindGroup bg_sky;
  WGPUBindGroup bg_rocks;
  WGPUBindGroup bg_fern;
  WGPUBindGroup bg_trees;
  WGPUBindGroup bg_birds;
  WGPUBindGroup bg_smoke;

  /* UBO dynamic buffer */
  wgpu_buffer_t ubo_buf;

  /* Sampler */
  WGPUSampler sampler;
  WGPUSampler sampler_clamp;
  WGPUSampler sampler_cube;
  WGPUSampler sampler_depth;

  /* Offscreen depth for soft particles */
  WGPUTexture depth_offscreen_tex;
  WGPUTextureView depth_offscreen_view;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attach;
  WGPURenderPassDepthStencilAttachment depth_attach;
  WGPURenderPassDescriptor render_pass_desc;

  /* Depth-only pass for soft particles */
  WGPURenderPassDepthStencilAttachment depth_only_attach;
  WGPURenderPassDescriptor depth_only_pass_desc;

  /* Timers */
  float timer_camera;
  float timer_fog;
  float timer_bird_anim;
  float timer_rocks_move;

  /* Animation state */
  int bird_frame1, bird_frame2;
  float bird_morph;

  /* Config (GUI-editable) */
  struct {
    float fog_start;
    float fog_distance;
    float fog_height_offset;
    float fog_height_mult;
    float height_offset;
    float trees_height_offset;
    float diffuse_exponent;
    float grass_amount;
    float clouds_scale;
    int preset;
  } cfg;

  /* State flags */
  int models_loaded; /* how many models are fully loaded */
  int textures_loaded;
  bool resources_dirty;
  bool initialized;
  bool cubemap_loading; /* currently loading cubemap faces */
  int pending_preset;   /* preset to switch to (-1 = no change) */
  uint64_t last_time;   /* for delta-time computation */
} rp_state_t;

static rp_state_t state;

/* -------------------------------------------------------------------------- *
 * Spline / camera math (port of ObjectsPlacement.ts)
 * -------------------------------------------------------------------------- */

#define RP_MAIN_RADIUS 120.0f
#define RP_SPIKE_LEN1 16.0f
#define RP_SPIKE_LEN2 3.0f
#define RP_SPIKES_XY1 7
#define RP_SPIKES_XY2 19
#define RP_SPIKES_Z1 4
#define RP_SPIKES_Z2 8
#define RP_SPLINE_AMP_Z1 1.0f
#define RP_SPLINE_AMP_Z2 3.0f
#define RP_CAM_HEIGHT_OFFSET 7.0f

static void rp_position_on_spline(float t, float radius_offset, float out[3])
{
  float a = GLM_PI * 2.0f * t;
  float r = RP_MAIN_RADIUS + RP_SPIKE_LEN1 * sinf(a * RP_SPIKES_XY1)
            + RP_SPIKE_LEN2 * sinf(a * RP_SPIKES_XY2) + radius_offset;
  float z = RP_SPLINE_AMP_Z1 * sinf(a * RP_SPIKES_Z1)
            + RP_SPLINE_AMP_Z2 * sinf(a * RP_SPIKES_Z2) + RP_CAM_HEIGHT_OFFSET;
  out[0] = sinf(a) * r;
  out[1] = cosf(a) * r;
  out[2] = z;
}

/* Position camera along spline using lookAt with banking */
static void rp_position_camera(float t, mat4 view_out)
{
  float p1[3], p2[3], p3[3];
  rp_position_on_spline(t, 0.0f, p1);
  rp_position_on_spline(t + 0.02f, 0.0f, p2);
  rp_position_on_spline(t + 0.0205f, 0.0f, p3);

  /* compute banking vector */
  float v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
  float v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
  float len1  = sqrtf(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  float len2  = sqrtf(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
  if (len1 > 1e-6f) {
    v1[0] /= len1;
    v1[1] /= len1;
    v1[2] /= len1;
  }
  if (len2 > 1e-6f) {
    v2[0] /= len2;
    v2[1] /= len2;
    v2[2] /= len2;
  }
  /* cross product for banking */
  /* cross product for banking */
  /* cx,cy unused; only cz matters for the up vector banking */
  float cz = v1[0] * v2[1] - v1[1] * v2[0];

  vec3 eye    = {p1[0], p1[1], p1[2]};
  vec3 center = {p2[0], p2[1], p2[2]};
  vec3 up     = {0.0f, cz * -6.0f, 1.0f};

  glm_lookat(eye, center, up, view_out);
}

/* -------------------------------------------------------------------------- *
 * Instance generation (port of ObjectsPlacement.ts initPositions)
 * Uses a deterministic LCG instead of Math.random() for repeatability.
 * -------------------------------------------------------------------------- */

/* Simple LCG RNG matching V8's Math.random() for seed reproducibility */
static uint64_t rp_rng_state = 0;

static void rp_rng_seed(uint64_t seed)
{
  rp_rng_state = seed;
}

/* Returns value in [0, 1) */
static float rp_rand(void)
{
  /* xorshift64 */
  rp_rng_state ^= rp_rng_state << 13;
  rp_rng_state ^= rp_rng_state >> 7;
  rp_rng_state ^= rp_rng_state << 17;
  return (float)((rp_rng_state & 0x0FFFFFFFu) / (double)0x10000000u);
}

/* Returns instance data buffer: [tx, ty, scale_rand, angle_rad] per instance.
 * Caller must free() the returned buffer. */
static float* rp_gen_instances(int count, float min_dist, float offset_scale,
                               float* smoke_positions_out,
                               int* actual_count_out)
{
  float* buf = (float*)malloc((size_t)(count * 4) * sizeof(float));
  if (!buf)
    return NULL;

  for (int i = 0; i < count; i++) {
    float o = rp_rand() - 0.5f;
    if (o > 0 && o < min_dist)
      o += min_dist;
    if (o < 0 && o > -min_dist)
      o -= min_dist;

    float pos[3];
    rp_position_on_spline((float)i / count, o * offset_scale, pos);

    float scale_rand = rp_rand();
    float angle      = rp_rand() * GLM_PI * 2.0f;

    buf[i * 4 + 0] = pos[0]; /* tx */
    buf[i * 4 + 1] = pos[1]; /* ty */
    buf[i * 4 + 2] = scale_rand;
    buf[i * 4 + 3] = angle;

    /* Store smoke/particle positions separately (including Z) */
    if (smoke_positions_out) {
      smoke_positions_out[i * 3 + 0] = pos[0];
      smoke_positions_out[i * 3 + 1] = pos[1];
      smoke_positions_out[i * 3 + 2] = pos[2];
    }
  }
  if (actual_count_out)
    *actual_count_out = count;
  return buf;
}

/* -------------------------------------------------------------------------- *
 * Half-float → float conversion
 * -------------------------------------------------------------------------- */

static float half_to_float(uint16_t h)
{
  uint32_t sign     = (uint32_t)(h >> 15) << 31;
  uint32_t exp      = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    /* denormal or zero */
    f = sign | ((mantissa) << 13);
  }
  else if (exp == 31) {
    /* inf or NaN */
    f = sign | 0x7F800000u | (mantissa << 13);
  }
  else {
    f = sign | ((exp + 112) << 23) | (mantissa << 13);
  }
  float result;
  memcpy(&result, &f, 4);
  return result;
}

/* -------------------------------------------------------------------------- *
 * Model file loading (stride 16 = half-float pos3+uv2+norm3,
 *                     stride 12 = half-float pos3+uv2,
 *                     stride 20 = float32 pos3+uv2,
 *                     stride 68 = float32 bird animation)
 * -------------------------------------------------------------------------- */

/* Stride IDs */
#define RP_STRIDE_HF16                                                         \
  16 /* half-float pos3+uv2+norm3 → expand to float32 stride32 */
#define RP_STRIDE_HF12                                                         \
  12                     /* half-float pos3+uv2 → expand to float32 stride20 \
                          */
#define RP_STRIDE_F20 20 /* float32 pos3+uv2 (sky, cloud) */
#define RP_STRIDE_F68 68 /* float32 bird animation */

/* Convert a stride-16 half-float model to stride-32 float32 */
static uint8_t* rp_convert_stride16(const uint8_t* src, size_t src_size,
                                    size_t* out_size)
{
  uint32_t n   = (uint32_t)(src_size / 16);
  *out_size    = n * 32u;
  uint8_t* dst = (uint8_t*)malloc(*out_size);
  if (!dst)
    return NULL;
  for (uint32_t i = 0; i < n; i++) {
    const uint16_t* s = (const uint16_t*)(src + i * 16);
    float* d          = (float*)(dst + i * 32);
    /* pos3 */
    d[0] = half_to_float(s[0]);
    d[1] = half_to_float(s[1]);
    d[2] = half_to_float(s[2]);
    /* uv2 */
    d[3] = half_to_float(s[3]);
    d[4] = half_to_float(s[4]);
    /* normal3 */
    d[5] = half_to_float(s[5]);
    d[6] = half_to_float(s[6]);
    d[7] = half_to_float(s[7]);
  }
  return dst;
}

/* Convert a stride-12 half-float model to stride-20 float32 */
static uint8_t* rp_convert_stride12(const uint8_t* src, size_t src_size,
                                    size_t* out_size)
{
  uint32_t n   = (uint32_t)(src_size / 12);
  *out_size    = n * 20u;
  uint8_t* dst = (uint8_t*)malloc(*out_size);
  if (!dst)
    return NULL;
  for (uint32_t i = 0; i < n; i++) {
    const uint16_t* s = (const uint16_t*)(src + i * 12);
    float* d          = (float*)(dst + i * 20);
    d[0]              = half_to_float(s[0]);
    d[1]              = half_to_float(s[1]);
    d[2]              = half_to_float(s[2]);
    d[3]              = half_to_float(s[3]);
    d[4]              = half_to_float(s[4]);
  }
  return dst;
}

/* -------------------------------------------------------------------------- *
 * Fetch state helpers
 * -------------------------------------------------------------------------- */

typedef struct {
  rp_model_t* model;
  bool is_indices;
  int src_stride; /* 0=raw indices, 16=hf16, 12=hf12, 20=f20, 68=f68 */
} rp_fetch_model_t;

typedef struct {
  wgpu_texture_t* tex;
  int* counter;
} rp_fetch_tex_t;

typedef struct {
  int face_idx; /* 0-5 for cubemap faces */
  bool is_cubemap;
} rp_fetch_cube_t;

#define RP_FETCH_STATES_MAX 64
static rp_fetch_model_t rp_fetch_model_states[RP_FETCH_STATES_MAX];
static rp_fetch_tex_t rp_fetch_tex_states[32];
static rp_fetch_cube_t rp_fetch_cube_states[6];
static int rp_fetch_model_count = 0;
static int rp_fetch_tex_count   = 0;

/* -------------------------------------------------------------------------- *
 * Fetch callbacks
 * -------------------------------------------------------------------------- */

static void rp_model_cb(const sfetch_response_t* r)
{
  rp_fetch_model_t* fs = (rp_fetch_model_t*)r->user_data;
  if (!r->fetched) {
    printf("[RP] Model fetch failed: %s (err %d)\n", r->path, r->error_code);
    free((void*)r->buffer.ptr);
    return;
  }

  size_t src_sz = r->data.size;
  size_t dst_sz;
  uint8_t* data;

  if (fs->src_stride == RP_STRIDE_HF16) {
    data = rp_convert_stride16(r->data.ptr, src_sz, &dst_sz);
  }
  else if (fs->src_stride == RP_STRIDE_HF12) {
    data = rp_convert_stride12(r->data.ptr, src_sz, &dst_sz);
  }
  else {
    /* raw copy (float32 or indices) */
    dst_sz = (src_sz + 3u) & ~3u;
    data   = (uint8_t*)calloc(1, dst_sz);
    if (data)
      memcpy(data, r->data.ptr, src_sz);
  }
  free((void*)r->buffer.ptr);
  if (!data)
    return;

  if (fs->is_indices) {
    fs->model->pending_idx    = data;
    fs->model->pending_idx_sz = dst_sz;
    fs->model->index_count    = (uint32_t)(src_sz / 2); /* uint16 */
    fs->model->idx_pending    = true;
  }
  else {
    fs->model->pending_vtx    = data;
    fs->model->pending_vtx_sz = dst_sz;
    fs->model->vtx_pending    = true;
  }
}

static void rp_tex_cb(const sfetch_response_t* r)
{
  rp_fetch_tex_t* ft = (rp_fetch_tex_t*)r->user_data;
  if (!r->fetched) {
    printf("[RP] Texture fetch failed: %s (err %d)\n", r->path, r->error_code);
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
    .extent = {(uint32_t)w, (uint32_t)h, 1},
    .format = WGPUTextureFormat_RGBA8Unorm,
    .pixels = {.ptr = pixels, .size = (size_t)(w * h * 4)},
  };
  ft->tex->desc.is_dirty = true;
  if (ft->counter)
    (*ft->counter)++;
}

static void rp_cube_cb(const sfetch_response_t* r)
{
  rp_fetch_cube_t* fc = (rp_fetch_cube_t*)r->user_data;
  if (!r->fetched) {
    printf("[RP] Cubemap face fetch failed: %s (err %d)\n", r->path,
           r->error_code);
    free((void*)r->buffer.ptr);
    return;
  }
  int w, h, ch;
  uint8_t* pixels = image_pixels_from_memory((const uint8_t*)r->data.ptr,
                                             (int)r->data.size, &w, &h, &ch, 4);
  free((void*)r->buffer.ptr);
  if (!pixels)
    return;

  if (state.cubemap_pixels[fc->face_idx]) {
    free(state.cubemap_pixels[fc->face_idx]);
  }
  state.cubemap_pixels[fc->face_idx] = pixels;
  state.cubemap_w                    = w;
  state.cubemap_h                    = h;
  state.cubemap_faces_loaded++;

  if (state.cubemap_faces_loaded == 6) {
    state.cubemap_dirty   = true;
    state.cubemap_loading = false;
    printf("[RP] All cubemap faces loaded (%dx%d)\n", w, h);
  }
}

/* -------------------------------------------------------------------------- *
 * Buffer / texture helpers
 * -------------------------------------------------------------------------- */

static WGPUBuffer rp_create_instance_buffer(wgpu_context_t* ctx,
                                            const float* data, int count,
                                            const char* label)
{
  size_t sz               = (size_t)(count * 4) * sizeof(float);
  WGPUBufferDescriptor bd = {
    .label = (WGPUStringView){.data = label, .length = strlen(label)},
    .size  = sz,
    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
  };
  WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx->device, &bd);
  if (buf)
    wgpuQueueWriteBuffer(ctx->queue, buf, 0, data, sz);
  return buf;
}

/* Upload cubemap from 6 RGBA pixel arrays */
static void rp_upload_cubemap(wgpu_context_t* ctx)
{
  if (state.cubemap_faces_loaded != 6)
    return;
  int w = state.cubemap_w, h = state.cubemap_h;

  /* Release old cubemap */
  if (state.tex_cubemap.handle) {
    wgpu_destroy_texture(&state.tex_cubemap);
  }

  WGPUTextureDescriptor td = {
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {(uint32_t)w, (uint32_t)h, 6},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.tex_cubemap.handle = wgpuDeviceCreateTexture(ctx->device, &td);
  state.tex_cubemap.desc.extent.width  = (uint32_t)w;
  state.tex_cubemap.desc.extent.height = (uint32_t)h;

  for (int face = 0; face < 6; face++) {
    wgpuQueueWriteTexture(ctx->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture  = state.tex_cubemap.handle,
                            .mipLevel = 0,
                            .origin   = {0, 0, (uint32_t)face},
                            .aspect   = WGPUTextureAspect_All,
                          },
                          state.cubemap_pixels[face], (size_t)(w * h * 4),
                          &(WGPUTexelCopyBufferLayout){
                            .offset       = 0,
                            .bytesPerRow  = (uint32_t)(w * 4),
                            .rowsPerImage = (uint32_t)h,
                          },
                          &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});
    free(state.cubemap_pixels[face]);
    state.cubemap_pixels[face] = NULL;
  }

  state.tex_cubemap.view = wgpuTextureCreateView(
    state.tex_cubemap.handle, &(WGPUTextureViewDescriptor){
                                .format         = WGPUTextureFormat_RGBA8Unorm,
                                .dimension      = WGPUTextureViewDimension_Cube,
                                .baseMipLevel   = 0,
                                .mipLevelCount  = 1,
                                .baseArrayLayer = 0,
                                .arrayLayerCount = 6,
                                .aspect          = WGPUTextureAspect_All,
                              });

  state.cubemap_faces_loaded = 0;
  state.cubemap_dirty        = false;
  state.resources_dirty      = true;
  printf("[RP] Cubemap uploaded to GPU\n");
}

/* -------------------------------------------------------------------------- *
 * Load asset helpers
 * -------------------------------------------------------------------------- */

static void rp_load_model(rp_model_t* model, const char* base, int vtx_stride,
                          size_t idx_sz, size_t vtx_sz)
{
  char path[256];

  rp_fetch_model_t* fi = &rp_fetch_model_states[rp_fetch_model_count++];
  fi->model            = model;
  fi->is_indices       = true;
  fi->src_stride       = 0; /* raw uint16 indices */
  snprintf(path, sizeof(path), "%s-indices.bin", base);
  uint8_t* ibuf = (uint8_t*)malloc(idx_sz);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = rp_model_cb,
    .buffer    = {.ptr = ibuf, .size = idx_sz},
    .user_data = {.ptr = fi, .size = sizeof(rp_fetch_model_t)},
  });

  rp_fetch_model_t* fv = &rp_fetch_model_states[rp_fetch_model_count++];
  fv->model            = model;
  fv->is_indices       = false;
  fv->src_stride       = vtx_stride;
  snprintf(path, sizeof(path), "%s-strides.bin", base);
  uint8_t* vbuf = (uint8_t*)malloc(vtx_sz);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = rp_model_cb,
    .buffer    = {.ptr = vbuf, .size = vtx_sz},
    .user_data = {.ptr = fv, .size = sizeof(rp_fetch_model_t)},
  });
}

static void rp_load_texture(wgpu_context_t* ctx, wgpu_texture_t* tex,
                            const char* path, size_t sz, int* counter)
{
  /* Initialize placeholder */
  *tex = wgpu_create_color_bars_texture(
    ctx, &(wgpu_texture_desc_t){
           .format = WGPUTextureFormat_RGBA8Unorm,
           .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
         });

  rp_fetch_tex_t* ft = &rp_fetch_tex_states[rp_fetch_tex_count++];
  ft->tex            = tex;
  ft->counter        = counter;

  uint8_t* buf = (uint8_t*)malloc(sz);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = rp_tex_cb,
    .buffer    = {.ptr = buf, .size = sz},
    .user_data = {.ptr = ft, .size = sizeof(rp_fetch_tex_t)},
  });
}

static void rp_load_cubemap_faces(const char* preset_dir)
{
  state.cubemap_faces_loaded = 0;
  state.cubemap_loading      = true;

  for (int i = 0; i < 6; i++) {
    char path[256];
    snprintf(path, sizeof(path), "%s/textures/cubemaps/%s/%s", RP_ASSETS_BASE,
             preset_dir, RP_CUBEMAP_FACES[i]);
    rp_fetch_cube_states[i].face_idx   = i;
    rp_fetch_cube_states[i].is_cubemap = true;
    uint8_t* buf                       = (uint8_t*)malloc(128 * 128 * 4);
    sfetch_send(&(sfetch_request_t){
      .path     = path,
      .callback = rp_cube_cb,
      .buffer   = {.ptr = buf, .size = 128 * 128 * 4},
      .user_data
      = {.ptr = &rp_fetch_cube_states[i], .size = sizeof(rp_fetch_cube_t)},
    });
  }
}

/* -------------------------------------------------------------------------- *
 * GPU resource init
 * -------------------------------------------------------------------------- */

static void rp_alloc_model_buffers(wgpu_context_t* ctx, rp_model_t* m,
                                   size_t idx_sz, size_t vtx_sz,
                                   const char* lbl)
{
  char buf[64];
  snprintf(buf, sizeof(buf), "%s idx", lbl);
  m->idx = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
           .size  = (idx_sz + 3u) & ~3u,
         });
  snprintf(buf, sizeof(buf), "%s vtx", lbl);
  m->vtx = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size  = (vtx_sz + 3u) & ~3u,
         });
}

static void rp_upload_pending_models(wgpu_context_t* ctx)
{
  rp_model_t* models[] = {
    &state.sky_model,    &state.rock1_model,  &state.rock2_model,
    &state.rock3_model,  &state.rock1g_model, &state.rock2g_model,
    &state.rock3g_model, &state.tree_model,   &state.bird_model,
    &state.smoke_model,
  };
  for (int i = 0; i < 10; i++) {
    rp_model_t* m = models[i];
    if (m->idx_pending && m->pending_idx) {
      size_t ws = (m->pending_idx_sz + 3u) & ~3u;
      wgpuQueueWriteBuffer(ctx->queue, m->idx.buffer, 0, m->pending_idx, ws);
      free(m->pending_idx);
      m->pending_idx = NULL;
      m->idx_pending = false;
    }
    if (m->vtx_pending && m->pending_vtx) {
      size_t ws = (m->pending_vtx_sz + 3u) & ~3u;
      wgpuQueueWriteBuffer(ctx->queue, m->vtx.buffer, 0, m->pending_vtx, ws);
      free(m->pending_vtx);
      m->pending_vtx = NULL;
      m->vtx_pending = false;
    }
    if (!m->is_ready && !m->idx_pending && !m->vtx_pending
        && m->index_count > 0) {
      m->is_ready = true;
      state.models_loaded++;
      printf("[RP] Model loaded (%u triangles)\n", m->index_count / 3);
    }
  }
}

static void rp_upload_pending_textures(wgpu_context_t* ctx)
{
  wgpu_texture_t* texes[] = {
    &state.tex_rocks, &state.tex_trees, &state.tex_grass, &state.tex_fern,
    &state.tex_bird,  &state.tex_smoke, &state.tex_white,
  };
  for (int i = 0; i < 7; i++) {
    if (texes[i]->desc.is_dirty) {
      wgpu_recreate_texture(ctx, texes[i]);
      FREE_TEXTURE_PIXELS(*texes[i]);
      state.resources_dirty = true;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Offscreen depth texture for soft particles
 * -------------------------------------------------------------------------- */

static void rp_init_offscreen_depth(wgpu_context_t* ctx)
{
  if (state.depth_offscreen_view)
    wgpuTextureViewRelease(state.depth_offscreen_view);
  if (state.depth_offscreen_tex)
    wgpuTextureRelease(state.depth_offscreen_tex);

  uint32_t w = (uint32_t)ctx->width;
  uint32_t h = (uint32_t)ctx->height;

  WGPUTextureDescriptor td = {
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {w, h, 1},
    .format        = WGPUTextureFormat_Depth32Float,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.depth_offscreen_tex  = wgpuDeviceCreateTexture(ctx->device, &td);
  state.depth_offscreen_view = wgpuTextureCreateView(
    state.depth_offscreen_tex, &(WGPUTextureViewDescriptor){
                                 .format    = WGPUTextureFormat_Depth32Float,
                                 .dimension = WGPUTextureViewDimension_2D,
                                 .mipLevelCount   = 1,
                                 .arrayLayerCount = 1,
                                 .aspect          = WGPUTextureAspect_DepthOnly,
                               });
}

/* -------------------------------------------------------------------------- *
 * Vertex buffer layouts
 * -------------------------------------------------------------------------- */

/* stride-32: pos(float3) + uv(float2) + normal(float3) — rocks */
static WGPUVertexAttribute rp_vtx_attrs32[3] = {
  {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
  {.format = WGPUVertexFormat_Float32x2, .offset = 12, .shaderLocation = 1},
  {.format = WGPUVertexFormat_Float32x3, .offset = 20, .shaderLocation = 2},
};
static WGPUVertexBufferLayout rp_vtx_layout32 = {
  .arrayStride    = 32,
  .stepMode       = WGPUVertexStepMode_Vertex,
  .attributeCount = 3,
  .attributes     = rp_vtx_attrs32,
};

/* stride-20: pos(float3) + uv(float2) — sky, fern, trees, cloud */
static WGPUVertexAttribute rp_vtx_attrs20[2] = {
  {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
  {.format = WGPUVertexFormat_Float32x2, .offset = 12, .shaderLocation = 1},
};
static WGPUVertexBufferLayout rp_vtx_layout20 = {
  .arrayStride    = 20,
  .stepMode       = WGPUVertexStepMode_Vertex,
  .attributeCount = 2,
  .attributes     = rp_vtx_attrs20,
};

/* Instance buffer: vec4f(tx, ty, scale_rand, angle) — step per instance */
static WGPUVertexAttribute rp_inst_attr[1] = {
  {.format = WGPUVertexFormat_Float32x4, .offset = 0, .shaderLocation = 3},
};
static WGPUVertexBufferLayout rp_inst_layout = {
  .arrayStride    = 16,
  .stepMode       = WGPUVertexStepMode_Instance,
  .attributeCount = 1,
  .attributes     = rp_inst_attr,
};

/* Bird: stride-68, 3 slots (pos1, pos2, uv) */
static WGPUVertexAttribute rp_bird_pos1_attr
  = {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0};
static WGPUVertexAttribute rp_bird_pos2_attr
  = {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 1};
static WGPUVertexAttribute rp_bird_uv_attr
  = {.format = WGPUVertexFormat_Float32x2, .offset = 0, .shaderLocation = 2};
static WGPUVertexBufferLayout rp_bird_vbls[3] = {
  {.arrayStride    = 68,
   .stepMode       = WGPUVertexStepMode_Vertex,
   .attributeCount = 1,
   .attributes     = &rp_bird_pos1_attr},
  {.arrayStride    = 68,
   .stepMode       = WGPUVertexStepMode_Vertex,
   .attributeCount = 1,
   .attributes     = &rp_bird_pos2_attr},
  {.arrayStride    = 68,
   .stepMode       = WGPUVertexStepMode_Vertex,
   .attributeCount = 1,
   .attributes     = &rp_bird_uv_attr},
};

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

/* STRVIEW and ASSERT are defined in wgpu_common.h */

static void rp_init_bind_group_layouts(wgpu_context_t* ctx)
{
  WGPUDevice d = ctx->device;

  /* SKY: UBO(dyn) + sampler_cube + cubemap */
  {
    WGPUBindGroupLayoutEntry e[3] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(rp_ubo_t)}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_Cube}},
    };
    state.bgl_sky = wgpuDeviceCreateBindGroupLayout(
      d, &(WGPUBindGroupLayoutDescriptor){
           .label = STRVIEW("RP Sky BGL"), .entryCount = 3, .entries = e});
  }

  /* ROCKS: UBO(dyn) + sampler + diffuse_tex + grass_tex + cubemap */
  {
    WGPUBindGroupLayoutEntry e[5] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(rp_ubo_t)}},
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
                            .viewDimension = WGPUTextureViewDimension_Cube}},
    };
    state.bgl_rocks = wgpuDeviceCreateBindGroupLayout(
      d, &(WGPUBindGroupLayoutDescriptor){
           .label = STRVIEW("RP Rocks BGL"), .entryCount = 5, .entries = e});
  }

  /* FERN / TREES: UBO(dyn) + sampler + color_tex + cubemap */
  {
    WGPUBindGroupLayoutEntry e[4] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(rp_ubo_t)}},
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
                            .viewDimension = WGPUTextureViewDimension_Cube}},
    };
    state.bgl_fern = wgpuDeviceCreateBindGroupLayout(
      d, &(WGPUBindGroupLayoutDescriptor){
           .label = STRVIEW("RP Fern BGL"), .entryCount = 4, .entries = e});
  }

  /* BIRDS: UBO(dyn) + sampler + bird_tex + cubemap */
  {
    WGPUBindGroupLayoutEntry e[4] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(rp_ubo_t)}},
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
                            .viewDimension = WGPUTextureViewDimension_Cube}},
    };
    state.bgl_birds = wgpuDeviceCreateBindGroupLayout(
      d, &(WGPUBindGroupLayoutDescriptor){
           .label = STRVIEW("RP Birds BGL"), .entryCount = 4, .entries = e});
  }

  /* SMOKE: UBO(dyn) + sampler + smoke_tex + cubemap + depth_tex + depth_samp */
  {
    WGPUBindGroupLayoutEntry e[6] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(rp_ubo_t)}},
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
                            .viewDimension = WGPUTextureViewDimension_Cube}},
      [4] = {.binding    = 4,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Depth,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [5] = {.binding    = 5,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_NonFiltering}},
    };
    state.bgl_smoke = wgpuDeviceCreateBindGroupLayout(
      d, &(WGPUBindGroupLayoutDescriptor){
           .label = STRVIEW("RP Smoke BGL"), .entryCount = 6, .entries = e});
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups (rebuilt when textures / cubemap change)
 * -------------------------------------------------------------------------- */

static void rp_init_bind_groups(wgpu_context_t* ctx)
{
  WGPUDevice d = ctx->device;

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_sky)
  {
    WGPUBindGroupEntry e[3] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler_cube},
      [2] = {.binding = 2, .textureView = state.tex_cubemap.view},
    };
    state.bg_sky = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Sky BG"),
                                    .layout     = state.bgl_sky,
                                    .entryCount = 3,
                                    .entries    = e});
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_rocks)
  {
    WGPUBindGroupEntry e[5] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_rocks.view},
      [3] = {.binding = 3, .textureView = state.tex_grass.view},
      [4] = {.binding = 4, .textureView = state.tex_cubemap.view},
    };
    state.bg_rocks = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Rocks BG"),
                                    .layout     = state.bgl_rocks,
                                    .entryCount = 5,
                                    .entries    = e});
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_fern)
  {
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_fern.view},
      [3] = {.binding = 3, .textureView = state.tex_cubemap.view},
    };
    state.bg_fern = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Fern BG"),
                                    .layout     = state.bgl_fern,
                                    .entryCount = 4,
                                    .entries    = e});
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_trees)
  {
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_trees.view},
      [3] = {.binding = 3, .textureView = state.tex_cubemap.view},
    };
    state.bg_trees = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Trees BG"),
                                    .layout     = state.bgl_fern,
                                    .entryCount = 4,
                                    .entries    = e});
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_birds)
  {
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_bird.view},
      [3] = {.binding = 3, .textureView = state.tex_cubemap.view},
    };
    state.bg_birds = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Birds BG"),
                                    .layout     = state.bgl_birds,
                                    .entryCount = 4,
                                    .entries    = e});
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_smoke)
  {
    WGPUBindGroupEntry e[6] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_smoke.view},
      [3] = {.binding = 3, .textureView = state.tex_cubemap.view},
      [4] = {.binding = 4, .textureView = state.depth_offscreen_view},
      [5] = {.binding = 5, .sampler = state.sampler_depth},
    };
    state.bg_smoke = wgpuDeviceCreateBindGroup(
      d, &(WGPUBindGroupDescriptor){.label      = STRVIEW("RP Smoke BG"),
                                    .layout     = state.bgl_smoke,
                                    .entryCount = 6,
                                    .entries    = e});
  }

  state.resources_dirty = false;
}

/* -------------------------------------------------------------------------- *
 * Pipeline init
 * -------------------------------------------------------------------------- */

static void rp_init_pipelines(wgpu_context_t* ctx)
{
  WGPUDevice d          = ctx->device;
  WGPUTextureFormat fmt = ctx->render_format;

  /* Pipeline layouts */
  {
    WGPUPipelineLayoutDescriptor pld;

    pld          = (WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
                                                  .bindGroupLayouts = &state.bgl_sky};
    state.pl_sky = wgpuDeviceCreatePipelineLayout(d, &pld);

    pld            = (WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
                                                    .bindGroupLayouts = &state.bgl_rocks};
    state.pl_rocks = wgpuDeviceCreatePipelineLayout(d, &pld);

    pld           = (WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
                                                   .bindGroupLayouts = &state.bgl_fern};
    state.pl_fern = wgpuDeviceCreatePipelineLayout(d, &pld);

    pld            = (WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
                                                    .bindGroupLayouts = &state.bgl_birds};
    state.pl_birds = wgpuDeviceCreatePipelineLayout(d, &pld);

    pld            = (WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
                                                    .bindGroupLayouts = &state.bgl_smoke};
    state.pl_smoke = wgpuDeviceCreatePipelineLayout(d, &pld);
  }

  WGPUDepthStencilState ds_write
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = ctx->depth_stencil_format, .depth_write_enabled = true});
  ds_write.depthCompare = WGPUCompareFunction_LessEqual;

  WGPUDepthStencilState ds_no_write
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = ctx->depth_stencil_format, .depth_write_enabled = false});
  ds_no_write.depthCompare = WGPUCompareFunction_LessEqual;

  /* Additive blend */
  WGPUBlendState blend_add = {
    .color = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One},
    .alpha = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One},
  };

  /* ---- SKY ---- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(d, rp_sky_shader_wgsl);
    WGPUVertexBufferLayout vbls[1] = {rp_vtx_layout20};

    WGPUDepthStencilState ds_sky = ds_write;
    ds_sky.depthCompare          = WGPUCompareFunction_LessEqual;
    /* Sky: render from inside sphere, use back-face cull (sphere has CCW
       winding viewed from inside) */
    state.sky_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Sky pipeline"),
        .layout   = state.pl_sky,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_Back,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_sky,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.sky_pipeline);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- ROCKS (vertex-lit + grass) ---- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(d, rp_rocks_shader_wgsl);
    WGPUVertexBufferLayout vbls[2] = {rp_vtx_layout32, rp_inst_layout};

    state.rocks_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Rocks pipeline"),
        .layout   = state.pl_rocks,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 2,
                     .buffers     = vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_Back,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_write,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.rocks_pipeline);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- FERN / TREES (alpha tested, instanced) ---- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(d, rp_fern_shader_wgsl);
    WGPUVertexBufferLayout vbls[2] = {rp_vtx_layout20, rp_inst_layout};

    state.fern_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Fern pipeline"),
        .layout   = state.pl_fern,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 2,
                     .buffers     = vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_write,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.fern_pipeline);

    /* Trees use same pipeline config as fern (alpha-tested instanced,
     * but bound with the pine_leaves texture via a different bind group). */
    state.trees_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Trees pipeline"),
        .layout   = state.pl_fern,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 2,
                     .buffers     = vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_write,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.trees_pipeline);

    wgpuShaderModuleRelease(sm);
  }

  /* ---- BIRDS (animated morph, instanced) ---- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(d, rp_birds_shader_wgsl);

    state.birds_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Birds pipeline"),
        .layout   = state.pl_birds,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 3,
                     .buffers     = rp_bird_vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_write,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.birds_pipeline);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- SMOKE (billboard, additive + depth fade) ---- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(d, rp_smoke_shader_wgsl);
    WGPUVertexBufferLayout vbls[1] = {rp_vtx_layout20};

    state.smoke_pipeline = wgpuDeviceCreateRenderPipeline(
      d,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("RP Smoke pipeline"),
        .layout   = state.pl_smoke,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = vbls},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .blend  = &blend_add,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_no_write,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF}});
    ASSERT(state.smoke_pipeline);
    wgpuShaderModuleRelease(sm);
  }
}

/* -------------------------------------------------------------------------- *
 * Billboard MVP computation
 * -------------------------------------------------------------------------- */

/* Compute view-proj matrix for a billboard quad (removes view rotation,
   keeps translation + projection). Billboard faces the camera. */
static void rp_compute_billboard_mvp(float tx, float ty, float tz, float scale,
                                     float rot_z, const mat4 view,
                                     const mat4 proj, mat4 out_mvp)
{
  mat4 model;
  glm_mat4_identity(model);
  glm_translate(model, (vec3){tx, ty, tz});
  glm_scale(model, (vec3){scale, scale, scale});
  glm_rotate_z(model, rot_z, model);

  mat4 mv;
  glm_mat4_mul(view, model, mv);

  /* Remove rotation part from MV: zero out upper 3x3 rotation, keep scale */
  float sx
    = sqrtf(mv[0][0] * mv[0][0] + mv[1][0] * mv[1][0] + mv[2][0] * mv[2][0]);
  float sy
    = sqrtf(mv[0][1] * mv[0][1] + mv[1][1] * mv[1][1] + mv[2][1] * mv[2][1]);
  float sz
    = sqrtf(mv[0][2] * mv[0][2] + mv[1][2] * mv[1][2] + mv[2][2] * mv[2][2]);
  mv[0][0] = sx;
  mv[1][0] = 0;
  mv[2][0] = 0;
  mv[0][1] = 0;
  mv[1][1] = sy;
  mv[2][1] = 0;
  mv[0][2] = 0;
  mv[1][2] = 0;
  mv[2][2] = sz;

  /* Re-apply rotation around Z */
  float c = cosf(rot_z), s = sinf(rot_z);
  mv[0][0] = c * sx;
  mv[1][0] = -s * sx;
  mv[2][0] = 0;
  mv[0][1] = s * sy;
  mv[1][1] = c * sy;
  mv[2][1] = 0;

  glm_mat4_mul(proj, mv, out_mvp);
}

/* -------------------------------------------------------------------------- *
 * Pre-compute UBOs before render pass
 * -------------------------------------------------------------------------- */

#define RP_UBO_SLOT_SKY 0
#define RP_UBO_SLOT_ROCKS1 1
#define RP_UBO_SLOT_ROCKS2 2
#define RP_UBO_SLOT_ROCKS3 3
#define RP_UBO_SLOT_ROCKS4 4
#define RP_UBO_SLOT_ROCKS5 5
#define RP_UBO_SLOT_GRASS1 6
#define RP_UBO_SLOT_GRASS2 7
#define RP_UBO_SLOT_GRASS3 8
#define RP_UBO_SLOT_GRASS4 9
#define RP_UBO_SLOT_GRASS5 10
#define RP_UBO_SLOT_TREES 11
#define RP_UBO_SLOT_BIRDS0 12 /* 6 bird path slots: 12..17 */
#define RP_UBO_SLOT_SMOKE 18  /* 40 smoke particles: 18..57 */

static uint8_t
  rp_ubo_staging[RP_UBO_DRAW_SLOTS * RP_UBO_STRIDE + 64 * RP_UBO_STRIDE];

static void rp_fill_ubo(rp_ubo_t* u, const mat4 view_proj, const mat4 view,
                        const rp_preset_t* p, float fog_start, float fog_dist,
                        float height_rand_mult, float height_fixed,
                        float scale_base, float scale_range, float morph,
                        float grass_amount)
{
  memcpy(u->view_proj, view_proj, 64);
  memcpy(u->view, view, 64);
  memcpy(u->color, p->color_ambient, 16);
  memcpy(u->color_sun, p->color_sun, 16);
  memcpy(u->light_dir, p->light_dir, 16);
  u->fog_params[0]       = fog_start;
  u->fog_params[1]       = fog_dist;
  u->fog_params[2]       = state.cfg.fog_height_offset;
  u->fog_params[3]       = state.cfg.fog_height_mult;
  u->placement_params[0] = height_rand_mult;
  u->placement_params[1] = height_fixed;
  u->placement_params[2] = scale_base;
  u->placement_params[3] = scale_range;
  u->misc_params[0]      = p->diffuse_exponent;
  u->misc_params[1]      = grass_amount;
  u->misc_params[2]      = morph;
  u->misc_params[3]      = state.timer_camera;
}

static size_t rp_ubo_total_slots;

static void rp_precompute_ubos(wgpu_context_t* ctx, const mat4 view_proj,
                               const mat4 view)
{
  const rp_preset_t* p = &rp_presets[state.cfg.preset];
  float fog_s          = state.cfg.fog_start * 0.666f
                + 0.333f * state.cfg.fog_start
                    * ((sinf(GLM_PI * 2.0f * state.timer_fog) + 1.0f) * 0.5f);
  float fog_d   = state.cfg.fog_distance;
  float ho      = state.cfg.height_offset;
  float tho     = state.cfg.trees_height_offset;
  float float_h = 6.0f * sinf(state.timer_rocks_move * GLM_PI * 2.0f);

  /* Count total UBO slots needed */
  /* sky(1) + rocks(5) + grass(5) + trees(1) + birds(6) + smoke(RP_SMOKE_COUNT)
   */
  size_t total       = 18 + RP_SMOKE_COUNT;
  rp_ubo_total_slots = total;

  size_t buf_sz = total * RP_UBO_STRIDE;
  if (buf_sz > sizeof(rp_ubo_staging))
    buf_sz = sizeof(rp_ubo_staging);
  memset(rp_ubo_staging, 0, buf_sz);

#define SLOT(n) ((rp_ubo_t*)(rp_ubo_staging + (n) * RP_UBO_STRIDE))

  /* Sky */
  rp_fill_ubo(SLOT(RP_UBO_SLOT_SKY), view_proj, view, p, fog_s, fog_d, 0, 0, 0,
              0, 0, 0);

  /* Rocks 1-5 */
  rp_fill_ubo(SLOT(RP_UBO_SLOT_ROCKS1), view_proj, view, p, fog_s, fog_d, ho,
              ho * 0.25f, 0.0055f, 0.004f, 0, state.cfg.grass_amount);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_ROCKS2), view_proj, view, p, fog_s, fog_d, ho,
              ho * 0.25f, 0.006f, 0.004f, 0, state.cfg.grass_amount);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_ROCKS3), view_proj, view, p, fog_s, fog_d, 0, 0,
              0.012f, 0.004f, 0, state.cfg.grass_amount);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_ROCKS4), view_proj, view, p, fog_s, fog_d,
              float_h, 34.0f, 0.0055f, 0.004f, 0, state.cfg.grass_amount);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_ROCKS5), view_proj, view, p, fog_s, fog_d, 0,
              -14.0f, 0.0055f, 0.004f, 0, state.cfg.grass_amount);

  /* Grass (fern) for each rock group */
  rp_fill_ubo(SLOT(RP_UBO_SLOT_GRASS1), view_proj, view, p, fog_s, fog_d, ho,
              ho * 0.25f, 0.0055f, 0.004f, 0, 0);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_GRASS2), view_proj, view, p, fog_s, fog_d, ho,
              ho * 0.25f, 0.006f, 0.004f, 0, 0);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_GRASS3), view_proj, view, p, fog_s, fog_d, 0, 0,
              0.012f, 0.004f, 0, 0);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_GRASS4), view_proj, view, p, fog_s, fog_d,
              float_h, 34.0f, 0.0055f, 0.004f, 0, 0);
  rp_fill_ubo(SLOT(RP_UBO_SLOT_GRASS5), view_proj, view, p, fog_s, fog_d, 0,
              -14.0f, 0.0055f, 0.004f, 0, 0);

  /* Trees */
  rp_fill_ubo(SLOT(RP_UBO_SLOT_TREES), view_proj, view, p, fog_s, fog_d, 0,
              -tho, 0.003f, 0.003f, 0, 0);

  /* Birds (6 paths × 1 UBO each — bird scatter is done in shader) */
  for (int i = 0; i < 6; i++) {
    float cx
      = state.bird_paths[i].end[0]
        + state.bird_paths[i].t
            * (state.bird_paths[i].start[0] - state.bird_paths[i].end[0]);
    float cy
      = state.bird_paths[i].end[1]
        + state.bird_paths[i].t
            * (state.bird_paths[i].start[1] - state.bird_paths[i].end[1]);
    float cz
      = state.bird_paths[i].end[2]
        + state.bird_paths[i].t
            * (state.bird_paths[i].start[2] - state.bird_paths[i].end[2]);
    /* height_rand_mult = cx, height_fixed = cy encode bird flock center */
    rp_ubo_t* bu = SLOT(RP_UBO_SLOT_BIRDS0 + i);
    rp_fill_ubo(bu, view_proj, view, p, fog_s, fog_d, cx, cy, cz,
                state.bird_paths[i].rotation, state.bird_morph, 0);
  }

  /* Smoke particles */
  float ratio_s
    = (ctx->height > 0) ? (float)ctx->width / (float)ctx->height : 1.0f;
  mat4 proj_only;
  glm_mat4_identity(proj_only);
  glm_perspective(glm_rad(RP_FOV_DEG), ratio_s, RP_Z_NEAR, RP_Z_FAR, proj_only);
  /* OpenGL [-1,1] → WebGPU [0,1]: new[2][2] = old[2][2]*0.5 + old[2][3]*0.5
   * where old[2][3] = -1 always → new[2][2] = old[2][2]*0.5 - 0.5 */
  proj_only[2][2] = proj_only[2][2] * 0.5f - 0.5f;
  proj_only[3][2] = proj_only[3][2] * 0.5f;

  for (int i = 0; i < RP_SMOKE_COUNT; i++) {
    float sx = state.smoke_positions[i * 3 + 0];
    float sy = state.smoke_positions[i * 3 + 1];
    float sz = state.smoke_positions[i * 3 + 2] + 12.0f;
    mat4 mvp;
    rp_compute_billboard_mvp(sx, sy, sz, state.cfg.clouds_scale, 0.0f, view,
                             proj_only, mvp);
    rp_ubo_t* su = SLOT(RP_UBO_SLOT_SMOKE + i);
    rp_fill_ubo(su, mvp, view, p, fog_s, fog_d, 0, 0, 1, 0, 0, 0);
    /* Override clouds_color into color_sun for fog sprite */
    memcpy(su->color_sun, p->clouds_color, 16);
    /* Store viewport inverse for soft-particle depth sampling */
    su->placement_params[0] = 1.0f / (float)ctx->width;
    su->placement_params[1] = 1.0f / (float)ctx->height;
    su->placement_params[2] = RP_SMOKE_SOFT;
    su->placement_params[3] = 0;
    su->misc_params[0]      = RP_Z_NEAR;
    su->misc_params[1]      = RP_Z_FAR;
  }
#undef SLOT

  wgpuQueueWriteBuffer(ctx->queue, state.ubo_buf.buffer, 0, rp_ubo_staging,
                       buf_sz);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void rp_draw_gui(void)
{
  if (!state.initialized)
    return;

  igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){260, 240}, ImGuiCond_FirstUseEver);
  igBegin("Rock Pillars", NULL, ImGuiWindowFlags_NoMove);

  /* Preset selector */
  igText("Preset");
  igSameLine(0, 4);
  for (int i = 0; i < RP_PRESET_COUNT; i++) {
    if (i)
      igSameLine(0, 4);
    if (igButton(rp_presets[i].name, (ImVec2){0, 0})) {
      if (i != state.cfg.preset && !state.cubemap_loading) {
        state.pending_preset = i;
      }
    }
  }

  igSeparator();
  igText("Fog");
  igSliderFloat("Start##fog", &state.cfg.fog_start, 10.0f, 200.0f, "%.0f",
                ImGuiSliderFlags_None);
  igSliderFloat("Distance##fog", &state.cfg.fog_distance, 10.0f, 200.0f, "%.0f",
                ImGuiSliderFlags_None);
  igSliderFloat("Height offs", &state.cfg.fog_height_offset, -10.0f, 20.0f,
                "%.1f", ImGuiSliderFlags_None);
  igSliderFloat("Height mult", &state.cfg.fog_height_mult, 0.01f, 0.5f, "%.3f",
                ImGuiSliderFlags_None);

  igSeparator();
  igText("Scene");
  igSliderFloat("Height ofs##scene", &state.cfg.height_offset, 0.0f, 30.0f,
                "%.1f", ImGuiSliderFlags_None);
  igSliderFloat("Grass amount", &state.cfg.grass_amount, 0.0f, 5.0f, "%.2f",
                ImGuiSliderFlags_None);
  igSliderFloat("Clouds scale", &state.cfg.clouds_scale, 0.1f, 2.0f, "%.2f",
                ImGuiSliderFlags_None);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Scene drawing (within render pass encoder)
 * -------------------------------------------------------------------------- */

static void rp_draw_instanced_model(WGPURenderPassEncoder rp, rp_model_t* model,
                                    WGPUBuffer inst_buf, int inst_count,
                                    WGPUBindGroup bg, uint32_t ubo_slot)
{
  if (!model->is_ready)
    return;
  uint32_t offset = ubo_slot * RP_UBO_STRIDE;
  wgpuRenderPassEncoderSetBindGroup(rp, 0, bg, 1, &offset);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, model->vtx.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 1, inst_buf, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rp, model->idx.buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rp, model->index_count, (uint32_t)inst_count,
                                   0, 0, 0);
}

static void rp_draw_model_simple(WGPURenderPassEncoder rp, rp_model_t* model,
                                 WGPUBindGroup bg, uint32_t ubo_slot)
{
  if (!model->is_ready)
    return;
  uint32_t offset = ubo_slot * RP_UBO_STRIDE;
  wgpuRenderPassEncoderSetBindGroup(rp, 0, bg, 1, &offset);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, model->vtx.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rp, model->idx.buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rp, model->index_count, 1, 0, 0, 0);
}

/* Draw bird flock (instanced animated morph) */
static void rp_draw_birds_flock(WGPURenderPassEncoder rp, int instances,
                                int bird_slot)
{
  if (!state.bird_model.is_ready)
    return;
  /* stride = 68 bytes (5 frames * 3 floats + 2 UV floats) */
  uint32_t uv_offset = RP_BIRD_FRAMES * 3 * 4; /* UV at byte offset 60 */
  uint64_t f1_off    = (uint64_t)state.bird_frame1 * 3 * 4;
  uint64_t f2_off    = (uint64_t)state.bird_frame2 * 3 * 4;

  uint32_t dynoff = (uint32_t)(RP_UBO_SLOT_BIRDS0 + bird_slot) * RP_UBO_STRIDE;
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bg_birds, 1, &dynoff);

  WGPUBuffer vb = state.bird_model.vtx.buffer;
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, vb, f1_off, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 1, vb, f2_off, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 2, vb, uv_offset, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rp, state.bird_model.idx.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rp, state.bird_model.index_count,
                                   (uint32_t)instances, 0, 0, 0);
}

/* Draw smoke particles (one quad each at precomputed MVP) */
static void rp_draw_smoke(WGPURenderPassEncoder rp)
{
  if (!state.smoke_model.is_ready)
    return;

  wgpuRenderPassEncoderSetPipeline(rp, state.smoke_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.smoke_model.vtx.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rp, state.smoke_model.idx.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  for (int i = 0; i < RP_SMOKE_COUNT; i++) {
    uint32_t off = (uint32_t)(RP_UBO_SLOT_SMOKE + i) * RP_UBO_STRIDE;
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bg_smoke, 1, &off);
    wgpuRenderPassEncoderDrawIndexed(rp, state.smoke_model.index_count, 1, 0, 0,
                                     0);
  }
}

/* -------------------------------------------------------------------------- *
 * Frame function
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* ctx)
{
  sfetch_dowork();

  /* Upload pending model/texture data */
  rp_upload_pending_models(ctx);
  rp_upload_pending_textures(ctx);

  /* Upload cubemap when all faces loaded */
  if (state.cubemap_dirty) {
    rp_upload_cubemap(ctx);
    /* Rebuild bind groups after cubemap change */
    rp_init_bind_groups(ctx);
  }

  /* Rebuild bind groups when textures changed */
  if (state.resources_dirty && !state.cubemap_dirty) {
    if (state.tex_cubemap.view) {
      rp_init_bind_groups(ctx);
    }
  }

  /* Handle preset switch */
  if (state.pending_preset >= 0 && !state.cubemap_loading) {
    state.cfg.preset     = state.pending_preset;
    state.pending_preset = -1;
    rp_load_cubemap_faces(rp_presets[state.cfg.preset].cubemap_dir);
  }

  /* Advance timers */
  uint64_t now_ticks = stm_now();
  float dt_sec       = (state.last_time > 0) ?
                         (float)stm_sec(stm_diff(now_ticks, state.last_time)) :
                         1.0f / 60.0f;
  state.last_time    = now_ticks;
  /* Clamp delta time to avoid large jumps after a stall (e.g. when the
   * window is first shown or unpaused after many seconds). */
  if (dt_sec > 0.1f)
    dt_sec = 0.1f;
  double t = stm_sec(now_ticks);
  state.timer_camera
    = fmodf((float)t * 1000.0f, RP_CAMERA_PERIOD) / RP_CAMERA_PERIOD;
  state.timer_fog = fmodf((float)t * 1000.0f, RP_FOG_PERIOD) / RP_FOG_PERIOD;
  state.timer_bird_anim
    = fmodf((float)t * 1000.0f, RP_BIRD_ANIM_PERIOD) / RP_BIRD_ANIM_PERIOD;
  state.timer_rocks_move
    = fmodf((float)t * 1000.0f, RP_ROCKS_MOVE_PERIOD) / RP_ROCKS_MOVE_PERIOD;

  /* Bird animation frames */
  {
    float ba          = state.timer_bird_anim;
    state.bird_frame1 = (int)(ba * RP_BIRD_FRAMES) % RP_BIRD_FRAMES;
    state.bird_frame2 = (state.bird_frame1 + 1) % RP_BIRD_FRAMES;
    state.bird_morph  = fmodf(ba * RP_BIRD_FRAMES, 1.0f);
  }

  /* Advance bird path timers using actual delta time so speed is
   * frame-rate independent (the TypeScript version uses elapsed ms). */
  for (int i = 0; i < 6; i++) {
    state.bird_paths[i].t += dt_sec * state.bird_paths[i].inv_dur;
    if (state.bird_paths[i].t > 1.0f) {
      state.bird_paths[i].t = 0.0f;
    }
  }

  /* Compute camera matrices */
  mat4 view, proj, view_proj;
  glm_mat4_identity(view);
  glm_mat4_identity(proj);
  glm_mat4_identity(view_proj);

  rp_position_camera(state.timer_camera, view);
  float ratio
    = (ctx->height > 0) ? (float)ctx->width / (float)ctx->height : 1.0f;

  /* WebGPU depth range [0..1] */
  glm_perspective(glm_rad(RP_FOV_DEG), ratio, RP_Z_NEAR, RP_Z_FAR, proj);
  /* OpenGL [-1,1] → WebGPU [0,1]: new[2][2] = old[2][2]*0.5 + old[2][3]*0.5
   * where old[2][3] = -1 always → new[2][2] = old[2][2]*0.5 - 0.5 */
  proj[2][2] = proj[2][2] * 0.5f - 0.5f;
  proj[3][2] = proj[3][2] * 0.5f;
  glm_mat4_mul(proj, view, view_proj);

  /* Pre-compute all per-draw UBOs */
  rp_precompute_ubos(ctx, view_proj, view);

  /* Render */
  WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(
    ctx->device, &(WGPUCommandEncoderDescriptor){.label = STRVIEW("RP cmd")});

  /* ---- Depth-only pass for soft particles ---- */
  /* Note: skipped in initial pass; smoke particles render without depth fade */

  /* ---- Main render pass ---- */
  {
    state.color_attach.view                       = ctx->swapchain_view;
    state.depth_attach.view                       = ctx->depth_stencil_view;
    state.render_pass_desc.colorAttachments       = &state.color_attach;
    state.render_pass_desc.depthStencilAttachment = &state.depth_attach;

    WGPURenderPassEncoder rp
      = wgpuCommandEncoderBeginRenderPass(cmd, &state.render_pass_desc);

    bool all_ready = state.models_loaded >= 10 && state.tex_cubemap.view != 0
                     && state.bg_sky != 0;

    if (all_ready) {
      /* --- Sky --- */
      wgpuRenderPassEncoderSetPipeline(rp, state.sky_pipeline);
      rp_draw_model_simple(rp, &state.sky_model, state.bg_sky, RP_UBO_SLOT_SKY);

      /* --- Rocks with vertex lighting --- */
      wgpuRenderPassEncoderSetPipeline(rp, state.rocks_pipeline);
      rp_draw_instanced_model(rp, &state.rock1_model, state.inst_rocks1,
                              RP_ROCKS1_COUNT, state.bg_rocks,
                              RP_UBO_SLOT_ROCKS1);
      rp_draw_instanced_model(rp, &state.rock2_model, state.inst_rocks2,
                              RP_ROCKS2_COUNT, state.bg_rocks,
                              RP_UBO_SLOT_ROCKS2);
      rp_draw_instanced_model(rp, &state.rock3_model, state.inst_rocks3,
                              RP_ROCKS3_COUNT, state.bg_rocks,
                              RP_UBO_SLOT_ROCKS3);
      rp_draw_instanced_model(rp, &state.rock1_model, state.inst_rocks4,
                              RP_ROCKS4_COUNT, state.bg_rocks,
                              RP_UBO_SLOT_ROCKS4);
      rp_draw_instanced_model(rp, &state.rock1_model, state.inst_rocks5,
                              RP_ROCKS5_COUNT, state.bg_rocks,
                              RP_UBO_SLOT_ROCKS5);

      /* --- Fern (alpha-tested) --- */
      wgpuRenderPassEncoderSetPipeline(rp, state.fern_pipeline);
      rp_draw_instanced_model(rp, &state.rock1g_model, state.inst_rocks1,
                              RP_ROCKS1_COUNT, state.bg_fern,
                              RP_UBO_SLOT_GRASS1);
      rp_draw_instanced_model(rp, &state.rock2g_model, state.inst_rocks2,
                              RP_ROCKS2_COUNT, state.bg_fern,
                              RP_UBO_SLOT_GRASS2);
      rp_draw_instanced_model(rp, &state.rock3g_model, state.inst_rocks3,
                              RP_ROCKS3_COUNT, state.bg_fern,
                              RP_UBO_SLOT_GRASS3);
      rp_draw_instanced_model(rp, &state.rock1g_model, state.inst_rocks4,
                              RP_ROCKS4_COUNT, state.bg_fern,
                              RP_UBO_SLOT_GRASS4);
      rp_draw_instanced_model(rp, &state.rock1g_model, state.inst_rocks5,
                              RP_ROCKS5_COUNT, state.bg_fern,
                              RP_UBO_SLOT_GRASS5);

      /* --- Trees (alpha-tested) --- */
      wgpuRenderPassEncoderSetPipeline(rp, state.trees_pipeline);
      rp_draw_instanced_model(rp, &state.tree_model, state.inst_trees,
                              RP_TREES_COUNT, state.bg_trees,
                              RP_UBO_SLOT_TREES);

      /* --- Birds --- */
      wgpuRenderPassEncoderSetPipeline(rp, state.birds_pipeline);
      for (int i = 0; i < 6; i++) {
        rp_draw_birds_flock(rp, 9, i);
      }

      /* --- Smoke (additive, depth fade) --- */
      rp_draw_smoke(rp);
    }

    wgpuRenderPassEncoderEnd(rp);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rp)
  }

  /* GUI: new frame, draw widgets, then submit + render */
  {
    static uint64_t last_t = 0;
    uint64_t now           = stm_now();
    float dt = last_t ? (float)stm_sec(stm_diff(now, last_t)) : 1.0f / 60.0f;
    last_t   = now;
    imgui_overlay_new_frame(ctx, dt);
    rp_draw_gui();
  }

  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(
    cmd, &(WGPUCommandBufferDescriptor){.label = STRVIEW("RP frame")});
  wgpuCommandEncoderRelease(cmd);
  wgpuQueueSubmit(ctx->queue, 1, &cb);
  wgpuCommandBufferRelease(cb);

  imgui_overlay_render(ctx);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* ctx)
{
  state.wgpu_context   = ctx;
  state.pending_preset = -1;

  /* Default config */
  state.cfg.fog_start           = 70.0f;
  state.cfg.fog_distance        = 50.0f;
  state.cfg.fog_height_offset   = 2.0f;
  state.cfg.fog_height_mult     = 0.08f;
  state.cfg.height_offset       = 10.0f;
  state.cfg.trees_height_offset = 5.2f;
  state.cfg.diffuse_exponent    = 0.025f;
  state.cfg.grass_amount        = 1.5f;
  state.cfg.clouds_scale        = 0.3f;
  state.cfg.preset              = 2; /* Sunset */

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 64,
    .num_channels = 4,
    .num_lanes    = 4,
#ifndef __WAJIC__
    .logger.func = slog_func,
#endif
  });
  stm_setup();
  imgui_overlay_init(ctx);

  /* UBO dynamic buffer */
  {
    size_t sz     = (18 + RP_SMOKE_COUNT) * RP_UBO_STRIDE + 256;
    state.ubo_buf = wgpu_create_buffer(
      ctx, &(wgpu_buffer_desc_t){
             .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
             .size  = sz,
           });
  }

  /* Samplers */
  state.sampler = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("RP sampler"),
                   .addressModeU  = WGPUAddressMode_Repeat,
                   .addressModeV  = WGPUAddressMode_Repeat,
                   .minFilter     = WGPUFilterMode_Linear,
                   .magFilter     = WGPUFilterMode_Linear,
                   .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                   .lodMaxClamp   = 4.0f,
                   .maxAnisotropy = 1,
                 });
  state.sampler_clamp = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("RP sampler clamp"),
                   .addressModeU  = WGPUAddressMode_ClampToEdge,
                   .addressModeV  = WGPUAddressMode_ClampToEdge,
                   .minFilter     = WGPUFilterMode_Linear,
                   .magFilter     = WGPUFilterMode_Linear,
                   .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                   .maxAnisotropy = 1,
                 });
  state.sampler_cube = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("RP sampler cube"),
                   .addressModeU  = WGPUAddressMode_ClampToEdge,
                   .addressModeV  = WGPUAddressMode_ClampToEdge,
                   .addressModeW  = WGPUAddressMode_ClampToEdge,
                   .minFilter     = WGPUFilterMode_Linear,
                   .magFilter     = WGPUFilterMode_Linear,
                   .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                   .maxAnisotropy = 1,
                 });
  state.sampler_depth = wgpuDeviceCreateSampler(
    ctx->device, &(WGPUSamplerDescriptor){
                   .label         = STRVIEW("RP sampler depth"),
                   .addressModeU  = WGPUAddressMode_ClampToEdge,
                   .addressModeV  = WGPUAddressMode_ClampToEdge,
                   .minFilter     = WGPUFilterMode_Nearest,
                   .magFilter     = WGPUFilterMode_Nearest,
                   .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                   .maxAnisotropy = 1,
                 });

  /* Offscreen depth */
  rp_init_offscreen_depth(ctx);

  /* Render pass descriptors */
  state.color_attach = (WGPURenderPassColorAttachment){
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  state.depth_attach = (WGPURenderPassDepthStencilAttachment){
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  };
  state.render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attach,
    .depthStencilAttachment = &state.depth_attach,
  };

  state.depth_only_attach = (WGPURenderPassDepthStencilAttachment){
    .view            = state.depth_offscreen_view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };
  state.depth_only_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &state.depth_only_attach,
  };

  /* Allocate GPU model buffers */
  const char* base = RP_ASSETS_BASE "/models";
  rp_alloc_model_buffers(ctx, &state.sky_model, 6 * 1024, 2 * 1024, "sky");
  rp_alloc_model_buffers(ctx, &state.rock1_model, 10 * 1024, 30 * 1024,
                         "rock1");
  rp_alloc_model_buffers(ctx, &state.rock2_model, 10 * 1024, 50 * 1024,
                         "rock2");
  rp_alloc_model_buffers(ctx, &state.rock3_model, 12 * 1024, 60 * 1024,
                         "rock3");
  rp_alloc_model_buffers(ctx, &state.rock1g_model, 12 * 1024, 45 * 1024,
                         "rock1g");
  rp_alloc_model_buffers(ctx, &state.rock2g_model, 18 * 1024, 70 * 1024,
                         "rock2g");
  rp_alloc_model_buffers(ctx, &state.rock3g_model, 20 * 1024, 70 * 1024,
                         "rock3g");
  rp_alloc_model_buffers(ctx, &state.tree_model, 4 * 1024, 10 * 1024, "tree");
  rp_alloc_model_buffers(ctx, &state.bird_model, 256, 2 * 1024, "bird");
  rp_alloc_model_buffers(ctx, &state.smoke_model, 512, 1024, "smoke");

  /* Load models */
  char path[256];
#define MODEL(m, base_path, stride, idx_sz, vtx_sz)                            \
  snprintf(path, sizeof(path), "%s/%s", base, base_path);                      \
  rp_load_model(&state.m, path, stride, idx_sz, vtx_sz)

  MODEL(sky_model, "sky", RP_STRIDE_F20, 6 * 1024, 1 * 1024);
  MODEL(rock1_model, "rock-8", RP_STRIDE_HF16, 10 * 1024, 15 * 1024);
  MODEL(rock2_model, "rock-11", RP_STRIDE_HF16, 10 * 1024, 25 * 1024);
  MODEL(rock3_model, "rock-6", RP_STRIDE_HF16, 12 * 1024, 30 * 1024);
  MODEL(rock1g_model, "rock-8-grass", RP_STRIDE_HF12, 12 * 1024, 25 * 1024);
  MODEL(rock2g_model, "rock-11-grass", RP_STRIDE_HF12, 18 * 1024, 40 * 1024);
  MODEL(rock3g_model, "rock-6-grass", RP_STRIDE_HF12, 20 * 1024, 40 * 1024);
  MODEL(tree_model, "pinetree", RP_STRIDE_HF12, 4 * 1024, 4 * 1024);
  MODEL(bird_model, "bird-anim-uv", RP_STRIDE_F68, 256, 1024);
  MODEL(smoke_model, "cloud", RP_STRIDE_F20, 512, 512);
#undef MODEL

  /* Load textures */
  int* tc = &state.textures_loaded;
#define TEX(field, file, sz)                                                   \
  do {                                                                         \
    snprintf(path, sizeof(path), "%s/textures/%s", RP_ASSETS_BASE, file);      \
    rp_load_texture(ctx, &state.field, path, sz, tc);                          \
  } while (0)

  TEX(tex_rocks, "rocks.png", 512 * 512 * 4);
  TEX(tex_trees, "pine_leaves.png", 128 * 128 * 4);
  TEX(tex_grass, "grass.png", 256 * 256 * 4);
  TEX(tex_fern, "fern.png", 256 * 128 * 4);
  TEX(tex_bird, "bird2.png", 128 * 128 * 4);
  TEX(tex_smoke, "smoke.png", 128 * 128 * 4);
  TEX(tex_white, "white.png", 256);
#undef TEX

  /* Load cubemap for initial preset */
  rp_load_cubemap_faces(rp_presets[state.cfg.preset].cubemap_dir);

  /* Generate instance data with deterministic seed */
  rp_rng_seed(12345ULL);
  {
    float* d;

    d = rp_gen_instances(RP_ROCKS1_COUNT, 0.7f, 23.0f, NULL, NULL);
    state.inst_rocks1
      = rp_create_instance_buffer(ctx, d, RP_ROCKS1_COUNT, "rocks1");
    free(d);

    d = rp_gen_instances(RP_ROCKS2_COUNT, 0.72f, 25.0f, NULL, NULL);
    state.inst_rocks2
      = rp_create_instance_buffer(ctx, d, RP_ROCKS2_COUNT, "rocks2");
    free(d);

    d = rp_gen_instances(RP_ROCKS3_COUNT, 0.75f, 60.0f, NULL, NULL);
    state.inst_rocks3
      = rp_create_instance_buffer(ctx, d, RP_ROCKS3_COUNT, "rocks3");
    free(d);

    d = rp_gen_instances(RP_ROCKS4_COUNT, 0.0f, 10.0f, NULL, NULL);
    state.inst_rocks4
      = rp_create_instance_buffer(ctx, d, RP_ROCKS4_COUNT, "rocks4");
    free(d);

    d = rp_gen_instances(RP_ROCKS5_COUNT, 0.0f, 300.0f, NULL, NULL);
    state.inst_rocks5
      = rp_create_instance_buffer(ctx, d, RP_ROCKS5_COUNT, "rocks5");
    free(d);

    d = rp_gen_instances(RP_TREES_COUNT, 0.0f, 60.0f, NULL, NULL);
    state.inst_trees
      = rp_create_instance_buffer(ctx, d, RP_TREES_COUNT, "trees");
    free(d);

    /* Smoke: generate with positions */
    d = rp_gen_instances(RP_SMOKE_COUNT, 0.0f, 40.0f, state.smoke_positions,
                         NULL);
    state.inst_smoke
      = rp_create_instance_buffer(ctx, d, RP_SMOKE_COUNT, "smoke");
    free(d);
  }

  /* Bird paths (from ObjectsPlacement.ts) */
  {
    static const float bp_start[6][3] = {
      {-21.162f, 257.884f, 45},  {-183.714f, 99.322f, 45},
      {-257.136f, 193.587f, 45}, {-167.416f, -272.670f, 44},
      {249.592f, 274.224f, 45},  {-293.574f, -56.970f, 44},
    };
    static const float bp_end[6][3] = {
      {13.270f, -567.828f, 47},   {181.346f, 29.236f, 51},
      {88.294f, -352.470f, 44},   {213.550f, 31.967f, 44},
      {-359.851f, -126.645f, 46}, {204.348f, -165.245f, 44},
    };
    static const float bp_dur[6]
      = {20000 / 1300.f, 30000 / 1300.f, 24000 / 1300.f,
         22000 / 1300.f, 27000 / 1300.f, 28000 / 1300.f};
    static const float bp_rot[6] = {0, 1.3f, 0.4f, 2.4f, 5.2f, 1.3f};

    for (int i = 0; i < 6; i++) {
      memcpy(state.bird_paths[i].start, bp_start[i], 12);
      memcpy(state.bird_paths[i].end, bp_end[i], 12);
      state.bird_paths[i].t        = 0.0f;
      state.bird_paths[i].inv_dur  = 1.0f / bp_dur[i];
      state.bird_paths[i].rotation = bp_rot[i];
      state.bird_paths[i].reverse  = true;
    }
  }

  /* Init bind group layouts and pipelines.
   * Bind groups are NOT created here — the cubemap texture (loaded
   * asynchronously) would be NULL, causing a validation error.
   * frame() builds them once rp_upload_cubemap() makes the view available. */
  rp_init_bind_group_layouts(ctx);
  rp_init_pipelines(ctx);
  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Event / resize
 * -------------------------------------------------------------------------- */

static void on_resize(wgpu_context_t* ctx)
{
  rp_init_offscreen_depth(ctx);
  /* Rebuild smoke bind groups since depth view changed */
  if (state.bg_smoke) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.bg_smoke)
    WGPUBindGroupEntry e[6] = {
      [0] = {.binding = 0,
             .buffer  = state.ubo_buf.buffer,
             .size    = sizeof(rp_ubo_t)},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.tex_smoke.view},
      [3] = {.binding = 3, .textureView = state.tex_cubemap.view},
      [4] = {.binding = 4, .textureView = state.depth_offscreen_view},
      [5] = {.binding = 5, .sampler = state.sampler_depth},
    };
    state.bg_smoke = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){.label  = STRVIEW("RP Smoke BG"),
                                              .layout = state.bgl_smoke,
                                              .entryCount = 6,
                                              .entries    = e});
    /* Update depth-only pass attachment */
    state.depth_only_attach.view = state.depth_offscreen_view;
  }
}

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    on_resize(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* ctx)
{
  (void)ctx;
  sfetch_shutdown();
  imgui_overlay_shutdown();

  /* Free pending pixels */
  for (int i = 0; i < 6; i++) {
    free(state.cubemap_pixels[i]);
  }

  /* Models */
  rp_model_t* models[] = {
    &state.sky_model,    &state.rock1_model,  &state.rock2_model,
    &state.rock3_model,  &state.rock1g_model, &state.rock2g_model,
    &state.rock3g_model, &state.tree_model,   &state.bird_model,
    &state.smoke_model,
  };
  for (int i = 0; i < 10; i++) {
    wgpu_destroy_buffer(&models[i]->vtx);
    wgpu_destroy_buffer(&models[i]->idx);
    free(models[i]->pending_vtx);
    free(models[i]->pending_idx);
  }

  /* Textures */
  wgpu_texture_t* texes[] = {
    &state.tex_rocks, &state.tex_trees, &state.tex_grass, &state.tex_fern,
    &state.tex_bird,  &state.tex_smoke, &state.tex_white, &state.tex_cubemap,
  };
  for (int i = 0; i < 8; i++)
    wgpu_destroy_texture(texes[i]);

  /* Instance buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_rocks1)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_rocks2)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_rocks3)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_rocks4)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_rocks5)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_trees)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_smoke)

  /* UBO */
  wgpu_destroy_buffer(&state.ubo_buf);

  /* Samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_clamp)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_cube)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_depth)

  /* Offscreen depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_offscreen_view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_offscreen_tex)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_sky)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_rocks)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_fern)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_trees)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_birds)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_smoke)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_sky)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_rocks)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_fern)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_birds)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bgl_smoke)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_sky)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_rocks)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_fern)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_birds)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pl_smoke)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.sky_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.rocks_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.fern_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.trees_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.birds_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.smoke_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  wgpu_start(&(wgpu_desc_t){
    .title          = "Floating Rock Pillars",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });
  return 0;
}

/* ============================================================================
 * WGSL Shaders
 * ============================================================================
 */

/* -------------------------------------------------------------------------- *
 * Sky shader — cubemap sphere
 * -------------------------------------------------------------------------- */
/* clang-format off */
static const char* rp_sky_shader_wgsl = CODE(
  struct Uniforms {
    view_proj: mat4x4f,
    view:      mat4x4f,
    /* remaining fields unused for sky */
  }

  @group(0) @binding(0) var<uniform> u:        Uniforms;
  @group(0) @binding(1) var          samp:     sampler;
  @group(0) @binding(2) var          skybox:   texture_cube<f32>;

  struct VIn  { @location(0) pos: vec3f, @location(1) uv: vec2f }
  struct VOut {
    @builtin(position) clip: vec4f,
    @location(0)       dir:  vec3f,
  }

  @vertex fn vertexMain(in: VIn) -> VOut {
    var out: VOut;
    out.clip = u.view_proj * vec4f(in.pos, 1.0);
    /* The cubemap direction is the world-space vertex position.
       The sky sphere is at the scene origin with scale 1, so vertex == world
       direction. We strip translation from view by using the 3x3 part. */
    let p = u.view * vec4f(in.pos, 1.0);
    /* inverse of 3x3 rotation = transpose (orthonormal matrix) */
    let r = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.dir = transpose(r) * p.xyz;
    return out;
  }

  @fragment fn fragmentMain(in: VOut) -> @location(0) vec4f {
    return textureSample(skybox, samp, in.dir);
  }
);

/* -------------------------------------------------------------------------- *
 * Rocks shader — instanced, vertex-lit, grass blend, height fog
 *
 * Vertex input:
 *   @location(0) pos:    vec3f  (stride 32, offset  0)
 *   @location(1) uv:     vec2f  (stride 32, offset 12)
 *   @location(2) normal: vec3f  (stride 32, offset 20)
 *   @location(3) inst:   vec4f  (tx, ty, scale_rand, angle) — per instance
 * -------------------------------------------------------------------------- */
static const char* rp_rocks_shader_wgsl = CODE(
  struct Uniforms {
    view_proj:         mat4x4f,
    view:              mat4x4f,
    color:             vec4f,     /* ambient */
    color_sun:         vec4f,     /* sun color */
    light_dir:         vec4f,
    fog_params:        vec4f,     /* x=fog_start, y=fog_dist, z=h_offset, w=h_mult */
    placement:         vec4f,     /* x=h_rand, y=h_fixed, z=scale_base, w=scale_range */
    misc:              vec4f,     /* x=diffuse_exp, y=grass_amount, z=morph, w=tc */
  }

  @group(0) @binding(0) var<uniform> u:       Uniforms;
  @group(0) @binding(1) var          samp:    sampler;
  @group(0) @binding(2) var          tex_rock:texture_2d<f32>;
  @group(0) @binding(3) var          tex_grass:texture_2d<f32>;
  @group(0) @binding(4) var          skybox:  texture_cube<f32>;

  struct VIn {
    @location(0) pos:    vec3f,
    @location(1) uv:     vec2f,
    @location(2) normal: vec3f,
    @location(3) inst:   vec4f,  /* tx, ty, scale_rand, angle */
  }
  struct VOut {
    @builtin(position) clip:   vec4f,
    @location(0)       uv:     vec2f,
    @location(1)       diffuse:vec4f,
    @location(2)       grass:  f32,
    @location(3)       fog_amt:f32,
    @location(4)       fog_z:  f32,
    @location(5)       fog_dir:vec3f,
  }

  @vertex fn vertexMain(in: VIn) -> VOut {
    var out: VOut;
    let tx = in.inst.x;
    let ty = in.inst.y;
    let scale = u.placement.z + in.inst.z * u.placement.w;
    let angle = in.inst.w;
    let s = sin(angle);
    let c = cos(angle);

    /* 2D rotation in XY, identity in Z */
    var pos = in.pos;
    let rx = c * pos.x - s * pos.y;
    let ry = s * pos.x + c * pos.y;
    pos = vec3f(rx * scale, ry * scale, pos.z * scale);
    pos.z += s * u.placement.x + u.placement.y;
    pos.x += tx;
    pos.y += ty;

    let world_pos = vec4f(pos, 1.0);
    out.clip = u.view_proj * world_pos;
    out.uv   = in.uv;

    /* Normal transform (same 2D rotation around Z, no scale) */
    let nrx = c * in.normal.x - s * in.normal.y;
    let nry = s * in.normal.x + c * in.normal.y;
    let world_normal = normalize(vec3f(nrx, nry, in.normal.z));

    /* Vertex lighting: diffuse = pow(max(0, dot(N, L)), exp) */
    let d = pow(max(0.0, dot(world_normal, u.light_dir.xyz)), u.misc.x);
    out.diffuse = mix(u.color_sun, u.color, d);

    /* Grass blending: based on normal.z (upward facing = more grass) */
    out.grass = smoothstep(0.2, 0.3, world_normal.z * u.misc.y);

    /* Fog */
    let fog_dist = clamp((length(out.clip) - u.fog_params.x) / u.fog_params.y, 0.0, 1.0);
    out.fog_amt = fog_dist;
    out.fog_z   = pos.z;

    /* Fog color direction: strip translation from view, use vertex world pos as direction */
    let view_pos = u.view * world_pos;
    let r = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.fog_dir = transpose(r) * view_pos.xyz;

    return out;
  }

  @fragment fn fragmentMain(in: VOut) -> @location(0) vec4f {
    /* Height fog */
    let h_fog = clamp(1.0 - (in.fog_z + u.fog_params.z) * u.fog_params.w, 0.0, 1.0);
    let fog_amount = clamp(in.fog_amt + h_fog, 0.0, 1.0);
    let fog_color  = textureSample(skybox, samp, in.fog_dir);

    let rock  = textureSample(tex_rock,  samp, in.uv);
    let grass = textureSample(tex_grass, samp, in.uv * 5.0);
    let mixed = mix(rock, grass, in.grass);
    let diffuse = mixed * in.diffuse;
    return mix(diffuse, fog_color, fog_amount);
  }
);

/* -------------------------------------------------------------------------- *
 * Fern / Trees shader — instanced, alpha-tested, fog
 * -------------------------------------------------------------------------- */
static const char* rp_fern_shader_wgsl = CODE(
  struct Uniforms {
    view_proj:  mat4x4f,
    view:       mat4x4f,
    color:      vec4f,
    color_sun:  vec4f,
    light_dir:  vec4f,
    fog_params: vec4f,
    placement:  vec4f,   /* x=h_rand, y=h_fixed, z=scale_base, w=scale_range */
    misc:       vec4f,
  }

  @group(0) @binding(0) var<uniform> u:    Uniforms;
  @group(0) @binding(1) var          samp: sampler;
  @group(0) @binding(2) var          tex:  texture_2d<f32>;
  @group(0) @binding(3) var          skybox: texture_cube<f32>;

  struct VIn {
    @location(0) pos:  vec3f,
    @location(1) uv:   vec2f,
    @location(3) inst: vec4f,
  }
  struct VOut {
    @builtin(position) clip:    vec4f,
    @location(0)       uv:      vec2f,
    @location(1)       fog_amt: f32,
    @location(2)       fog_z:   f32,
    @location(3)       fog_dir: vec3f,
  }

  @vertex fn vertexMain(in: VIn) -> VOut {
    var out: VOut;
    let tx = in.inst.x;
    let ty = in.inst.y;
    let scale = u.placement.z + in.inst.z * u.placement.w;
    let angle = in.inst.w;
    let s = sin(angle);
    let c = cos(angle);

    var pos = in.pos;
    let rx = c * pos.x - s * pos.y;
    let ry = s * pos.x + c * pos.y;
    pos = vec3f(rx * scale, ry * scale, pos.z * scale);
    pos.z += s * u.placement.x + u.placement.y;
    pos.x += tx;
    pos.y += ty;

    let wp = vec4f(pos, 1.0);
    out.clip = u.view_proj * wp;
    out.uv   = in.uv;

    let fog_dist = clamp((length(out.clip) - u.fog_params.x) / u.fog_params.y, 0.0, 1.0);
    out.fog_amt = fog_dist;
    out.fog_z   = pos.z;

    let vp = u.view * wp;
    let r  = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.fog_dir = transpose(r) * vp.xyz;

    return out;
  }

  @fragment fn fragmentMain(in: VOut) -> @location(0) vec4f {
    let h_fog    = clamp(1.0 - (in.fog_z + u.fog_params.z) * u.fog_params.w, 0.0, 1.0);
    let fog_amt  = clamp(in.fog_amt + h_fog, 0.0, 1.0);
    let fog_col  = textureSample(skybox, samp, in.fog_dir);

    let diffuse = textureSample(tex, samp, in.uv) * u.color;
    if diffuse.a < 0.5 { discard; }
    return mix(diffuse, fog_col, fog_amt);
  }
);

/* -------------------------------------------------------------------------- *
 * Birds shader — morph animation, instanced, fog
 *
 * Bird vertex buffer layout (stride 68 = 5×pos3 + uv2):
 *   slot 0: pos1 @location(0)  — offset = frame1 * 12
 *   slot 1: pos2 @location(1)  — offset = frame2 * 12
 *   slot 2: uv   @location(2)  — offset = 60 (always)
 *
 * The bird scatter is computed from gl_InstanceIndex and seed values
 * stored in placement.x (seed_x) and placement.y (seed_y).
 * The bird flock center is in placement (height_rand_mult=cx, height_fixed=cy,
 * scale_base=cz, scale_range=rotation).
 * -------------------------------------------------------------------------- */
static const char* rp_birds_shader_wgsl = CODE(
  struct Uniforms {
    view_proj:  mat4x4f,
    view:       mat4x4f,
    color:      vec4f,      /* sun color for birds */
    color_sun:  vec4f,
    light_dir:  vec4f,
    fog_params: vec4f,
    placement:  vec4f,      /* x=flock_cx, y=flock_cy, z=flock_cz, w=rotation */
    misc:       vec4f,      /* z=morph */
  }

  @group(0) @binding(0) var<uniform> u:    Uniforms;
  @group(0) @binding(1) var          samp: sampler;
  @group(0) @binding(2) var          tex:  texture_2d<f32>;
  @group(0) @binding(3) var          skybox: texture_cube<f32>;

  struct VIn {
    @location(0) pos1: vec3f,
    @location(1) pos2: vec3f,
    @location(2) uv:   vec2f,
    @builtin(instance_index) iid: u32,
  }
  struct VOut {
    @builtin(position) clip:    vec4f,
    @location(0)       uv:      vec2f,
    @location(1)       fog_amt: f32,
    @location(2)       fog_z:   f32,
    @location(3)       fog_dir: vec3f,
  }

  fn random_f(st: f32) -> f32 {
    return fract(sin(st) * 43758.5453123);
  }

  @vertex fn vertexMain(in: VIn) -> VOut {
    var out: VOut;
    let morph = u.misc.z;
    var pos = mix(in.pos1, in.pos2, morph);

    /* Random scatter: use FIXED seed constants (1.1, 2.2) matching the
     * TypeScript reference — seed must NOT depend on the flock position
     * (u.placement.x/y) or individual bird positions would flicker every
     * frame as the flock moves, making the birds appear to fly very fast. */
    let fi  = f32(in.iid);
    let r1  = random_f(fi + 1.1);
    let r2  = random_f(fi + 2.2);

    /* Correct transformation order (matches TypeScript MVP construction):
     *   1. Add scatter in model space
     *   2. Scale (0.4)
     *   3. Rotate around Z
     *   4. Translate to flock center (AFTER scale/rotate, not before)  */
    pos += vec3f(150.0 * r1, 300.0 * r2, 42.0 * r1);
    pos  *= 0.4;
    let rot = u.placement.w;
    let s   = sin(rot);
    let c   = cos(rot);
    let bx  = c * pos.x - s * pos.y;
    let by  = s * pos.x + c * pos.y;
    pos = vec3f(bx, by, pos.z);
    /* Translate to flock center (placement.xyz) */
    pos += vec3f(u.placement.x, u.placement.y, u.placement.z - 20.0);

    let wp = vec4f(pos, 1.0);
    out.clip = u.view_proj * wp;
    out.uv   = in.uv;

    let fog_dist = clamp((length(out.clip) - u.fog_params.x) / u.fog_params.y, 0.0, 1.0);
    out.fog_amt = fog_dist;
    out.fog_z   = pos.z;
    let vp = u.view * wp;
    let r3 = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.fog_dir = transpose(r3) * vp.xyz;

    return out;
  }

  @fragment fn fragmentMain(in: VOut) -> @location(0) vec4f {
    let h_fog   = clamp(1.0 - (in.fog_z + u.fog_params.z) * u.fog_params.w, 0.0, 1.0);
    let fog_amt = clamp(in.fog_amt + h_fog, 0.0, 1.0);
    let fog_col = textureSample(skybox, samp, in.fog_dir);
    let base    = textureSample(tex, samp, in.uv);
    if base.a < 0.95 { discard; }
    return mix(base * u.color, fog_col, fog_amt);
  }
);

/* -------------------------------------------------------------------------- *
 * Smoke / cloud sprite shader — billboard, additive, soft-particle depth fade
 *
 * UBO overload:
 *   view_proj  = full billboard MVP (pre-computed)
 *   color_sun  = clouds_color (fog sprite color)
 *   placement  = x=inv_w, y=inv_h, z=transition_size
 *   misc       = x=z_near, y=z_far
 * -------------------------------------------------------------------------- */
static const char* rp_smoke_shader_wgsl = CODE(
  struct Uniforms {
    view_proj:  mat4x4f,     /* billboard MVP */
    view:       mat4x4f,
    color:      vec4f,       /* ambient (unused for smoke) */
    color_sun:  vec4f,       /* clouds_color */
    light_dir:  vec4f,
    fog_params: vec4f,
    placement:  vec4f,       /* x=inv_vp_w, y=inv_vp_h, z=transition */
    misc:       vec4f,       /* x=z_near, y=z_far */
  }

  @group(0) @binding(0) var<uniform>    u:       Uniforms;
  @group(0) @binding(1) var             samp:    sampler;
  @group(0) @binding(2) var             tex:     texture_2d<f32>;
  @group(0) @binding(3) var             skybox:  texture_cube<f32>;
  @group(0) @binding(4) var             depth_tex:texture_depth_2d;
  @group(0) @binding(5) var             depth_samp:sampler;

  struct VIn  { @location(0) pos: vec3f, @location(1) uv: vec2f }
  struct VOut {
    @builtin(position) clip: vec4f,
    @location(0)       uv:   vec2f,
    @location(1)       fog_amt: f32,
    @location(2)       fog_z:   f32,
    @location(3)       fog_dir: vec3f,
  }

  fn linearize_depth(z: f32, near: f32, far: f32) -> f32 {
    return (2.0 * near) / (far + near - z * (far - near));
  }

  @vertex fn vertexMain(in: VIn) -> VOut {
    var out: VOut;
    out.clip = u.view_proj * vec4f(in.pos, 1.0);
    out.uv   = in.uv;
    let fog_dist = clamp((length(out.clip) - u.fog_params.x) / u.fog_params.y, 0.0, 1.0);
    out.fog_amt = fog_dist;
    out.fog_z   = in.pos.z;
    /* fog direction from world pos */
    let wp = u.view * vec4f(in.pos, 1.0);
    let r  = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    out.fog_dir = transpose(r) * wp.xyz;
    return out;
  }

  @fragment fn fragmentMain(in: VOut) -> @location(0) vec4f {
    /* Soft particle depth comparison */
    let coords   = vec2f(in.clip.x * u.placement.x, in.clip.y * u.placement.y);
    let scene_d  = textureLoad(depth_tex, vec2u(u32(in.clip.x), u32(in.clip.y)), 0);
    let geom_dep = linearize_depth(scene_d,   u.misc.x, u.misc.y);
    let part_dep = linearize_depth(in.clip.z, u.misc.x, u.misc.y);
    let a        = clamp(geom_dep - part_dep, 0.0, 1.0);
    let b        = smoothstep(0.0, u.placement.z, a);

    /* Height fog */
    let h_fog   = clamp(1.0 - (in.fog_z + u.fog_params.z) * u.fog_params.w, 0.0, 1.0);
    let fog_amt = clamp(in.fog_amt + h_fog, 0.0, 1.0);

    /* Smoke texture uses R channel as alpha */
    let diffuse = textureSample(tex, samp, in.uv).rrrr * u.color_sun;
    let near_fade = smoothstep(0.0, 0.03, part_dep);
    return diffuse * (1.0 - fog_amt) * b * near_fade;
  }
);
/* clang-format on */
