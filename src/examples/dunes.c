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

/* In WAjic, WGPU handles are uint32_t */
#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Dunes Rendering
 *
 * Demonstrates a dunes terrain scene using the WebGPU graphics API. Features
 * include: multi-textured dunes terrain with wind animation, palm trees,
 * animated birds, dust particle effects, a sky sphere, and a sun flare.
 * Four time-of-day presets (Night, Day, Sunset, Sunrise) are provided.
 * Camera rotates automatically in "Rotating" mode, or cycles through random
 * viewpoints in "Random" mode.
 *
 * Ported from the WebGL Dunes demo:
 * https://github.com/keaukraine/webgl-dunes
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* dunes_terrain_shader_wgsl;
static const char* dunes_diffuse_shader_wgsl;
static const char* dunes_diffuse_colored_shader_wgsl;
static const char* dunes_diffuse_alpha_shader_wgsl;
static const char* dunes_animated_colored_shader_wgsl;
static const char* dunes_soft_particle_shader_wgsl;
static const char* dunes_depth_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define DUNES_DEPTH_FORMAT (WGPUTextureFormat_Depth24PlusStencil8)

/* Camera */
#define DUNES_Z_NEAR (20.0f)
#define DUNES_Z_FAR (40000.0f)
#define DUNES_FOV_LANDSCAPE (70.0f)
#define DUNES_FOV_PORTRAIT (80.0f)

/* Terrain */
#define DUNES_TERRAIN_SCALE (100.0f)

/* Animation speeds */
#define DUNES_YAW_SPEED (200.0f)         /* ms per degree  */
#define DUNES_DUST_TIMER_SPEED (3000.0f) /* ms per 0..100  */
#define DUNES_DUST_ROTATION_SPEED (903333.0f)
#define DUNES_DUST_MOVEMENT_SPEED (2500.0f)
#define DUNES_BIRD_ANIM_PERIOD1 (2000.0f)
#define DUNES_BIRD_ANIM_PERIOD2 (800.0f)
#define DUNES_BIRD_FLIGHT_PERIOD (50000.0f)
#define DUNES_RANDOM_CAMERA_PERIOD (6000.0f)
#define DUNES_CAMERA_WOBBLE_PERIOD (40000.0f)

#define DUNES_BIRD_FLIGHT_RADIUS (2000.0f)
#define DUNES_BIRD_FRAMES (5)
#define DUNES_DUST_TRAVEL_X (-500.0f)
#define DUNES_DUST_TRAVEL_Z (-200.0f)
#define DUNES_DUST_MAX_SCALE (7.0f)
#define DUNES_SMOKE_SOFTNESS (0.012f)

/* Dynamic UBO: one large buffer per draw type, each instance at stride 256
 * bytes. wgpuQueueWriteBuffer is a queue op - all writes to the same buffer are
 * coalesced before the command buffer executes (last write wins). Using dynamic
 * offsets lets each draw call read from its own pre-written slot in the buffer.
 */
#define DUNES_UBO_STRIDE 256u  /* minUniformBufferOffsetAlignment */
#define DUNES_TERRAIN_SLOTS 9u /* 1 main + 8 skirts */
#define DUNES_COLORED_SLOTS 2u /* 0=palms, 1=sun */
#define DUNES_BIRD_SLOTS 2u    /* 0=bird1, 1=bird2 */

/* Dynamic UBO: one large buffer per draw type, each instance at stride 256
 * bytes (minUniformBufferOffsetAlignment = 256 bytes in WebGPU) */
#define DUNES_UBO_STRIDE 256u
#define DUNES_TERRAIN_SLOTS 9u /* 1 main + 8 skirts */
#define DUNES_COLORED_SLOTS 2u /* 0=palm, 1=sun */
#define DUNES_BIRD_SLOTS 2u    /* 0=bird1, 1=bird2 */

/* Number of preset textures */
#define DUNES_TEX_COUNT                                                        \
  (13u) /* night/day/sunset/sunrise sky + diffuse/dust/                        \
           detail/smoke/sun_flare/bird/palm_alpha/palm_diff                    \
           */

/* Number of particles caps */
#define DUNES_PARTICLE_GROUPS (1u)
#define DUNES_PARTICLES_PER_GROUP_MAX (148u)

/* Preset count */
#define DUNES_PRESET_COUNT (4)

/* Camera mode */
typedef enum {
  DUNES_CAMERA_ROTATING = 0,
  DUNES_CAMERA_RANDOM   = 1,
} dunes_camera_mode_t;

/* -------------------------------------------------------------------------- *
 * Preset data
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 sand_color;
  vec3 shadow_color;
  vec3 fog_color;
  vec3 deco_color;
  vec3 dust_color;
  vec3 waves_color;
  /* Sun transform */
  float sun_tx, sun_ty, sun_tz;
  float sun_sx, sun_sy, sun_sz;
  /* Fog distances */
  float fog_start_distance;
  float fog_distance;
  /* Shader variant: 0 = sunset (lightmap index 0), 1 = day/night (index 1) */
  int dunes_shader_variant;
  /* Sky texture index (0=night,1=day,2=sunset,3=sunrise) */
  int sky_tex_idx;
} dunes_preset_t;

/* clang-format off */
static const dunes_preset_t dunes_presets[DUNES_PRESET_COUNT] = {
  /* Night */
  {
    .sand_color   = {85/255.f*0.45f, 130/255.f*0.45f, 150/255.f*0.45f},
    .shadow_color = {-212/255.f*0.5f, -200/255.f*0.5f, -138/255.f*0.5f},
    .fog_color    = {6/255.f, 8/255.f, 13/255.f},
    .deco_color   = {100/255.f*0.18f, 110/255.f*0.18f, 170/255.f*0.18f},
    .dust_color   = {55/255.f*0.25f, 62/255.f*0.25f, 81/255.f*0.25f},
    .waves_color  = {143/255.f, 152/255.f, 182/255.f},
    .sun_tx = 0, .sun_ty = -22000, .sun_tz = 8500,
    .sun_sx = 0.0f, .sun_sy = 1, .sun_sz = 1,
    .fog_start_distance = 200, .fog_distance = 12000,
    .dunes_shader_variant = 1, .sky_tex_idx = 0,
  },
  /* Day */
  {
    .sand_color   = {246/255.f, 158/255.f, 59/255.f},
    .shadow_color = {87/255.f, 56/255.f, 33/255.f},
    .fog_color    = {147/255.f, 178/255.f, 205/255.f},
    .deco_color   = {150/255.f, 150/255.f, 90/255.f},
    .dust_color   = {235/255.f*0.20f, 162/255.f*0.20f, 48/255.f*0.20f},
    .waves_color  = {255/255.f*0.99f, 226/255.f*0.99f, 171/255.f*0.99f},
    .sun_tx = 0, .sun_ty = -22000, .sun_tz = 8500,
    .sun_sx = 300.0f, .sun_sy = 1, .sun_sz = 1,
    .fog_start_distance = 1000, .fog_distance = 45000,
    .dunes_shader_variant = 1, .sky_tex_idx = 1,
  },
  /* Sunset */
  {
    .sand_color   = {255/255.f, 131/255.f, 44/255.f},
    .shadow_color = {64/255.f, 52/255.f, 100/255.f},
    .fog_color    = {109/255.f, 113/255.f, 137/255.f},
    .deco_color   = {200/255.f*0.7f, 150/255.f*0.7f, 90/255.f*0.7f},
    .dust_color   = {235/255.f*0.20f, 162/255.f*0.20f, 48/255.f*0.20f},
    .waves_color  = {255/255.f*0.99f, 226/255.f*0.99f, 171/255.f*0.99f},
    .sun_tx = 0, .sun_ty = 22000, .sun_tz = 100,
    .sun_sx = 200.0f, .sun_sy = 1, .sun_sz = 1,
    .fog_start_distance = 1000, .fog_distance = 45000,
    .dunes_shader_variant = 0, .sky_tex_idx = 2,
  },
  /* Sunrise */
  {
    .sand_color   = {255/255.f, 142/255.f, 50/255.f},
    .shadow_color = {64/255.f, 62/255.f, 110/255.f},
    .fog_color    = {109/255.f, 113/255.f, 137/255.f},
    .deco_color   = {200/255.f*0.7f, 150/255.f*0.7f, 90/255.f*0.7f},
    .dust_color   = {235/255.f*0.20f, 162/255.f*0.20f, 48/255.f*0.20f},
    .waves_color  = {255/255.f*0.99f, 226/255.f*0.99f, 171/255.f*0.99f},
    .sun_tx = 0, .sun_ty = 22000, .sun_tz = 100,
    .sun_sx = 200.0f, .sun_sy = 1, .sun_sz = 1,
    .fog_start_distance = 1000, .fog_distance = 45000,
    .dunes_shader_variant = 0, .sky_tex_idx = 3,
  },
};
/* clang-format on */

/* -------------------------------------------------------------------------- *
 * Particle positions (dust caps on dune crests)
 * Ported from DuneCapsParticles.ts
 * -------------------------------------------------------------------------- */

/* clang-format off */
static const float dunes_particles[][3] = {
  {3.797847f,-6.286594f,-7.197602f},{3.257277f,-7.911734f,-6.909169f},
  {2.434960f,-9.520692f,-6.680535f},{1.537812f,-10.600622f,-6.578402f},
  {1.095975f,-11.738160f,-6.499978f},{0.327464f,-13.221872f,-6.359738f},
  {0.136057f,-15.155611f,-6.520223f},{0.402834f,-16.198668f,-6.532903f},
  {1.516392f,-17.189711f,-6.495209f},{-11.773868f,-11.695264f,-6.455437f},
  {-13.204258f,-10.992879f,-6.607471f},{-13.967643f,-9.917362f,-6.760038f},
  {-13.576998f,-8.467162f,-6.636061f},{-12.168928f,-7.106575f,-6.749729f},
  {-10.473310f,-5.867888f,-6.739241f},{-9.454960f,-4.724994f,-6.628104f},
  {-9.239940f,-0.640998f,-6.744101f},{-10.041563f,1.080194f,-5.970007f},
  {-9.797917f,2.903881f,-6.014874f},{-9.816783f,4.771972f,-5.855431f},
  {-10.719968f,6.510313f,-5.586894f},{-11.532514f,8.180022f,-5.538589f},
  {-12.306478f,10.042962f,-5.626741f},{-12.592357f,11.609372f,-5.999370f},
  {-11.636161f,13.976769f,-6.255528f},{-10.783170f,16.487112f,-6.895906f},
  {-11.892997f,18.238689f,-6.908795f},{-13.784470f,20.472527f,-7.353825f},
  {-16.344418f,22.429338f,-7.701272f},{-16.214247f,24.985479f,-7.531041f},
  {-15.582586f,26.783958f,-7.479846f},{-13.494443f,28.462156f,-7.465666f},
  {-11.091772f,30.380070f,-7.121482f},{-24.015993f,23.117214f,-7.778510f},
  {-23.426142f,21.482300f,-7.811589f},{-23.065290f,20.323423f,-7.828486f},
  {-24.772711f,18.825783f,-7.291061f},{-25.983522f,17.457598f,-6.931326f},
  {-27.532780f,17.045506f,-6.780737f},{-12.781576f,3.861169f,-5.955958f},
  {-14.767230f,5.113492f,-6.390795f},{-16.293585f,7.043332f,-6.518155f},
  {-18.581059f,9.066100f,-6.664069f},{-20.395561f,11.061415f,-6.407583f},
  {-22.181387f,12.940652f,-6.421689f},{-24.389889f,12.917274f,-6.228668f},
  {-25.524071f,11.112617f,-6.374214f},{-27.414509f,8.460627f,-6.282979f},
  {-28.574360f,5.777741f,-6.151673f},{-29.049454f,3.101688f,-6.188148f},
  {-28.735722f,0.503012f,-6.295793f},{-28.296024f,-1.992660f,-6.323021f},
  {-28.152142f,-4.369975f,-6.293451f},{-28.381042f,-6.603413f,-6.249017f},
  {-29.188606f,-8.625000f,-6.228706f},{-30.241766f,-10.400248f,-6.284088f},
  {-31.307451f,-11.900000f,-6.418659f},{-31.997452f,-13.288460f,-6.616817f},
  {-31.706684f,-14.764450f,-6.748453f},{-30.451832f,-15.957067f,-6.722595f},
  {-28.555950f,-16.737274f,-6.662701f},{-26.391378f,-17.098587f,-6.694832f},
  {-24.009874f,-17.277174f,-6.782200f},{-21.693974f,-17.569271f,-6.825508f},
  {-19.378075f,-17.861368f,-6.868816f},{-17.062176f,-18.153465f,-6.912124f},
  {-14.746277f,-18.445562f,-6.955432f},{-12.430378f,-18.737659f,-6.998740f},
  {-10.114478f,-19.029757f,-7.042048f},{-7.798579f,-19.321854f,-7.085356f},
  {-5.482680f,-19.613951f,-7.128665f},{-3.166781f,-19.906048f,-7.171973f},
  {-0.850881f,-20.198145f,-7.215281f},{1.465018f,-20.490243f,-7.258589f},
  {3.780917f,-20.782340f,-7.301897f},{6.096816f,-21.074437f,-7.345205f},
  {8.412715f,-21.366534f,-7.388513f},{10.728614f,-21.658631f,-7.431822f},
  {13.044513f,-21.950729f,-7.475130f},{15.360413f,-22.242826f,-7.518438f},
  {17.676312f,-22.534923f,-7.561746f},{19.992211f,-22.827020f,-7.605054f},
  {22.308110f,-23.119117f,-7.648362f},{24.624009f,-23.411215f,-7.691670f},
  {26.939908f,-23.703312f,-7.734978f},{29.255808f,-23.995409f,-7.778287f},
  {31.571707f,-24.287506f,-7.821595f},{33.887606f,-24.579603f,-7.864903f},
  {36.203505f,-24.871700f,-7.908211f},{38.519404f,-25.163798f,-7.951519f},
  {40.835303f,-25.455895f,-7.994827f},{43.151202f,-25.747992f,-8.038136f},
  {45.467102f,-26.040089f,-8.081444f},{47.783001f,-26.332186f,-8.124752f},
  {50.098900f,-26.624284f,-8.168060f},{52.414799f,-26.916381f,-8.211368f},
  {54.730698f,-27.208478f,-8.254676f},{57.046597f,-27.500575f,-8.297984f},
  {59.362496f,-27.792672f,-8.341293f},{61.678396f,-28.084770f,-8.384601f},
  {63.994295f,-28.376867f,-8.427909f},{66.310194f,-28.668964f,-8.471217f},
  {68.626093f,-28.961061f,-8.514525f},{70.941992f,-29.253158f,-8.557833f},
  {73.257892f,-29.545256f,-8.601141f},{75.573791f,-29.837353f,-8.644449f},
  {77.889690f,-30.129450f,-8.687758f},{80.205589f,-30.421547f,-8.731066f},
  {82.521488f,-30.713644f,-8.774374f},{84.837387f,-31.005741f,-8.817682f},
  {87.153287f,-31.297839f,-8.860990f},{89.469186f,-31.589936f,-8.904298f},
  {91.785085f,-31.882033f,-8.947607f},{94.100984f,-32.174130f,-8.990915f},
  {96.416883f,-32.466227f,-9.034223f},{98.732783f,-32.758324f,-9.077531f},
  {101.048682f,-33.050422f,-9.120839f},{103.364581f,-33.342519f,-9.164147f},
  {105.680480f,-33.634616f,-9.207455f},{107.996379f,-33.926713f,-9.250764f},
  {110.312279f,-34.218810f,-9.294072f},{112.628178f,-34.510908f,-9.337380f},
  {114.944077f,-34.803005f,-9.380688f},{117.259976f,-35.095102f,-9.423996f},
  {119.575875f,-35.387199f,-9.467304f},{121.891775f,-35.679296f,-9.510612f},
  {124.207674f,-35.971394f,-9.553921f},{126.523573f,-36.263491f,-9.597229f},
  {128.839472f,-36.555588f,-9.640537f},{131.155371f,-36.847685f,-9.683845f},
  {133.471271f,-37.139782f,-9.727153f},{135.787170f,-37.431879f,-9.770461f},
  {138.103069f,-37.723977f,-9.813769f},{140.418968f,-38.016074f,-9.857078f},
  {142.734867f,-38.308171f,-9.900386f},{145.050766f,-38.600268f,-9.943694f},
  {11.603891f,24.970663f,-6.206645f},{12.344122f,23.281786f,-6.440074f},
  {12.507367f,21.285116f,-6.593583f},
};
/* clang-format on */

#define DUNES_PARTICLE_COUNT                                                   \
  ((uint32_t)(sizeof(dunes_particles) / sizeof(dunes_particles[0])))

/* -------------------------------------------------------------------------- *
 * Random camera positions
 * -------------------------------------------------------------------------- */

/* clang-format off */
static const float dunes_random_positions[12][3] = {
  {-1165.3851f,  852.6129f, -465.5009f},
  {-1004.7345f,-1210.1492f, -496.7282f},
  {  -48.3319f,-1273.2711f, -544.5004f},
  {  705.2008f,-1375.7559f, -502.4748f},
  { 1627.4290f,  550.0985f, -581.4216f},
  {-3481.8162f, 1101.4440f, -487.1667f},
  { -211.8611f,   87.4571f, -675.6992f},
  {  460.9389f,  630.7889f, -590.1151f},
  {   -1.3695f, 3444.9368f, -630.3264f},
  { 1557.5774f, 1292.7252f, -613.5891f},
  { -651.7662f,  166.9022f, -295.8658f},
  {  129.6327f, 1876.6643f, -402.3247f},
};
/* clang-format on */

#define DUNES_RANDOM_CAM_COUNT                                                 \
  ((uint32_t)(sizeof(dunes_random_positions)                                   \
              / sizeof(dunes_random_positions[0])))

/* -------------------------------------------------------------------------- *
 * Uniform buffer types
 * -------------------------------------------------------------------------- */

/* Terrain dunes uniform buffer */
typedef struct {
  mat4 view_proj_matrix; /* offset   0, 64 bytes */
  vec4 color;            /* sand color            */
  vec4 fog_color;
  vec4 shadow_color;
  vec4 waves_color;
  float time;
  float dust_opacity;
  float fog_start_distance;
  float fog_distance;
  float detail_start_distance;
  float detail_distance;
  float lightmap_select; /* 0.0=use tex.r (sunset/sunrise), 1.0=use tex.g
                            (day/night) */
  float _pad[1];         /* total = 160 bytes */
} dunes_terrain_uniforms_t;

/* Diffuse (sky, plain texture) uniform buffer */
typedef struct {
  mat4 view_proj_matrix;
} dunes_diffuse_uniforms_t;

/* Diffuse colored (sun, palms) uniform buffer */
typedef struct {
  mat4 view_proj_matrix;
  vec4 color;
} dunes_diffuse_colored_uniforms_t;

/* Animated colored (bird) uniform buffer */
typedef struct {
  mat4 view_proj_matrix;
  vec4 color;
  float morph;   /* interpolation between frame1 and frame2 */
  float _pad[7]; /* pad to 112 bytes to match WGSL struct alignment */
} dunes_animated_uniforms_t;

/* Soft particle (dust) uniform buffer */
typedef struct {
  mat4 view_proj_matrix; /* offset   0, 64 bytes */
  vec4 color;            /* offset  64, 16 bytes */
  float camera_near;     /* offset  80,  4 bytes */
  float camera_far;      /* offset  84,  4 bytes */
  float inv_viewport_x;  /* offset  88,  4 bytes */
  float inv_viewport_y;  /* offset  92,  4 bytes */
  float transition_size; /* offset  96,  4 bytes */
  float _pad[7]; /* offset 100, 28 bytes → total 128 bytes (WGSL vec3f pad
                    aligns to 16) */
} dunes_soft_particle_uniforms_t;

/* Depth pass uniform buffer */
typedef struct {
  mat4 view_proj_matrix;
} dunes_depth_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Model (binary geometry)
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count; /* number of uint16 indices */
  bool is_ready;
  /* Pending data before GPU upload */
  uint8_t* pending_idx;
  size_t pending_idx_size;
  uint8_t* pending_vtx;
  size_t pending_vtx_size;
  bool idx_pending;
  bool vtx_pending;
} dunes_model_t;

/* -------------------------------------------------------------------------- *
 * Texture slots
 * -------------------------------------------------------------------------- */

typedef enum {
  DUNES_TEX_SKY_NIGHT   = 0,
  DUNES_TEX_SKY_DAY     = 1,
  DUNES_TEX_SKY_SUNSET  = 2,
  DUNES_TEX_SKY_SUNRISE = 3,
  DUNES_TEX_DIFFUSE     = 4, /* dunes-diffuse */
  DUNES_TEX_DUST        = 5, /* upwind */
  DUNES_TEX_DETAIL      = 6, /* detail */
  DUNES_TEX_SMOKE       = 7,
  DUNES_TEX_SUN_FLARE   = 8,
  DUNES_TEX_BIRD        = 9,
  DUNES_TEX_PALM_ALPHA  = 10,
  DUNES_TEX_PALM_DIFF   = 11,
  DUNES_TEX__COUNT      = 12,
} dunes_tex_id_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Context */
  wgpu_context_t* wgpu_context;

  /* Models */
  dunes_model_t dunes_model;
  dunes_model_t sky_model;
  dunes_model_t smoke_model;
  dunes_model_t sun_model;
  dunes_model_t bird_model;
  dunes_model_t palms_model;

  /* Textures */
  wgpu_texture_t textures[DUNES_TEX__COUNT];
  WGPUSampler sampler;
  WGPUSampler sampler_repeat;

  /* Depth texture (main pass) */
  wgpu_texture_t depth_texture;
  /* Off-screen depth for soft particles */
  wgpu_texture_t offscreen_depth;

  /* Uniform buffers */
  wgpu_buffer_t terrain_ubo;
  wgpu_buffer_t diffuse_ubo;
  wgpu_buffer_t colored_ubo;
  wgpu_buffer_t animated_ubo;
  wgpu_buffer_t particle_ubo;
  wgpu_buffer_t depth_ubo;

  /* Bind group layouts */
  WGPUBindGroupLayout terrain_bgl;
  WGPUBindGroupLayout diffuse_bgl;
  WGPUBindGroupLayout colored_bgl;
  WGPUBindGroupLayout animated_bgl;
  WGPUBindGroupLayout particle_bgl;
  WGPUBindGroupLayout depth_bgl;

  /* Bind groups */
  WGPUBindGroup terrain_bg;
  WGPUBindGroup sky_bg;
  WGPUBindGroup colored_bg;
  WGPUBindGroup animated_bg;
  WGPUBindGroup particle_bg;
  WGPUBindGroup depth_bg;
  WGPUBindGroup palms_bg;

  /* Pipelines */
  WGPUPipelineLayout terrain_pl;
  WGPUPipelineLayout diffuse_pl;
  WGPUPipelineLayout colored_pl;
  WGPUPipelineLayout animated_pl;
  WGPUPipelineLayout particle_pl;
  WGPUPipelineLayout depth_pl;

  WGPURenderPipeline terrain_pipeline;
  WGPURenderPipeline sky_pipeline;
  WGPURenderPipeline colored_pipeline;
  WGPURenderPipeline animated_pipeline;
  WGPURenderPipeline particle_pipeline;
  WGPURenderPipeline depth_pipeline;
  WGPURenderPipeline palms_pipeline;

  /* Render passes */
  WGPURenderPassColorAttachment color_attach;
  WGPURenderPassDepthStencilAttachment depth_attach;
  WGPURenderPassDescriptor render_pass_desc;

  /* Depth-only pass for soft particles */
  WGPURenderPassDepthStencilAttachment offscreen_depth_attach;
  WGPURenderPassDescriptor depth_pass_desc;

  /* Animation timers ([0..1) unless noted) */
  float angle_yaw;  /* [0..360) */
  float dust_timer; /* [0..100) */
  float timer_dust_rotation;
  float timer_dust_movement;
  float timer_bird1;
  float timer_bird2;
  float timer_birds_fly;
  float timer_camera_wobble;
  float timer_random_camera;
  uint64_t last_time;

  /* Camera */
  dunes_camera_mode_t camera_mode;
  uint32_t random_cam_idx;
  float random_lookat[3];
  float random_fov;
  mat4 view_matrix;
  mat4 proj_matrix;

  /* Per-frame bird data set in precompute_ubos, used in render_scene.
   * Uses 3 vertex buffer slots: pos1 (frame1 offset), pos2 (frame2 offset),
   * uv (fixed offset 60) each with stride 68 so each reads one attribute. */
  uint64_t bird_vtx_pos1[DUNES_BIRD_SLOTS]; /* frame1 base offset (pos1) */
  uint64_t bird_vtx_pos2[DUNES_BIRD_SLOTS]; /* frame2 base offset (pos2) */

  /* Preset */
  int current_preset;

  /* Loaded flags */
  uint32_t textures_loaded;
  uint32_t models_loaded;
  bool resources_dirty;
  bool initialized;

  /* Frame timing */
  uint64_t last_frame_time;
} state;

/* -------------------------------------------------------------------------- *
 * Helper: smootherstep
 * -------------------------------------------------------------------------- */

static float dunes_clampf(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

static float dunes_smootherstep(float edge0, float edge1, float x)
{
  x = dunes_clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture);
  WGPUTextureDescriptor desc = {
    .label         = STRVIEW("Dunes - Depth texture"),
    .dimension     = WGPUTextureDimension_2D,
    .format        = DUNES_DEPTH_FORMAT,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = {wgpu_context->width, wgpu_context->height, 1},
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &desc);
  ASSERT(state.depth_texture.handle != NULL);
  WGPUTextureViewDescriptor vd = {
    .label           = STRVIEW("Dunes - Depth texture view"),
    .format          = DUNES_DEPTH_FORMAT,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.handle, &vd);
  ASSERT(state.depth_texture.view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Offscreen depth texture (for soft particle depth sampling)
 * -------------------------------------------------------------------------- */

static void init_offscreen_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.offscreen_depth);
  WGPUTextureDescriptor desc = {
    .label         = STRVIEW("Dunes - Offscreen depth texture"),
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = {wgpu_context->width, wgpu_context->height, 1},
    /* Must be both RenderAttachment AND TextureBinding for depth sampling */
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  state.offscreen_depth.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &desc);
  ASSERT(state.offscreen_depth.handle != NULL);
  WGPUTextureViewDescriptor vd = {
    .label           = STRVIEW("Dunes - Offscreen depth view"),
    .format          = WGPUTextureFormat_Depth32Float,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_DepthOnly,
  };
  state.offscreen_depth.view
    = wgpuTextureCreateView(state.offscreen_depth.handle, &vd);
  ASSERT(state.offscreen_depth.view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Samplers
 * -------------------------------------------------------------------------- */

static void init_samplers(wgpu_context_t* wgpu_context)
{
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Dunes - Linear sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 8.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.sampler != NULL);

  state.sampler_repeat = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Dunes - Repeat sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 8.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.sampler_repeat != NULL);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Dynamic UBOs: large buffers with N pre-computed instances at UBO_STRIDE
   * intervals. Each draw call uses a different dynamic offset to select its
   * own slot, so all writes can be submitted BEFORE the render pass. */
  state.terrain_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Terrain UBO (dynamic)",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = DUNES_TERRAIN_SLOTS * DUNES_UBO_STRIDE,
                  });
  state.diffuse_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Diffuse UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(dunes_diffuse_uniforms_t),
                  });
  state.colored_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Colored UBO (dynamic)",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = DUNES_COLORED_SLOTS * DUNES_UBO_STRIDE,
                  });
  state.animated_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Animated UBO (dynamic)",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = DUNES_BIRD_SLOTS * DUNES_UBO_STRIDE,
                  });
  state.particle_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Particle UBO (dynamic)",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = DUNES_PARTICLE_COUNT * DUNES_UBO_STRIDE,
                  });
  state.depth_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Dunes - Depth UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(dunes_depth_uniforms_t),
                  });
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Terrain: UBO + repeat_sampler + 3 textures (diffuse, dust, detail) */
  {
    WGPUBindGroupLayoutEntry e[5] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(dunes_terrain_uniforms_t)}},
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
    state.terrain_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){.label = STRVIEW("Dunes - Terrain BGL"),
                                       .entryCount = 5,
                                       .entries    = e});
    ASSERT(state.terrain_bgl != NULL);
  }

  /* Diffuse: UBO + sampler + texture */
  {
    WGPUBindGroupLayoutEntry e[3] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(dunes_diffuse_uniforms_t)}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
    };
    state.diffuse_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){.label = STRVIEW("Dunes - Diffuse BGL"),
                                       .entryCount = 3,
                                       .entries    = e});
    ASSERT(state.diffuse_bgl != NULL);
  }

  /* Colored: UBO + sampler + texture (or alpha texture pair) */
  {
    WGPUBindGroupLayoutEntry e[4] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer
             = {.type             = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = true,
                .minBindingSize   = sizeof(dunes_diffuse_colored_uniforms_t)}},
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
    };
    state.colored_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){.label = STRVIEW("Dunes - Colored BGL"),
                                       .entryCount = 4,
                                       .entries    = e});
    ASSERT(state.colored_bgl != NULL);
  }

  /* Animated: UBO + sampler + texture */
  {
    WGPUBindGroupLayoutEntry e[3] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                            .hasDynamicOffset = true,
                            .minBindingSize   = sizeof(dunes_animated_uniforms_t)}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
    };
    state.animated_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){.label = STRVIEW("Dunes - Animated BGL"),
                                       .entryCount = 3,
                                       .entries    = e});
    ASSERT(state.animated_bgl != NULL);
  }

  /* Soft particle: UBO + sampler + color_tex + depth_tex(float) */
  {
    WGPUBindGroupLayoutEntry e[4] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer
             = {.type             = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = true,
                .minBindingSize = 128}}, /* 128 bytes (WGSL struct alignment) */
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [3] = {.binding    = 3,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Depth,
                            .viewDimension = WGPUTextureViewDimension_2D}},
    };
    state.particle_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){.label = STRVIEW("Dunes - Particle BGL"),
                                       .entryCount = 4,
                                       .entries    = e});
    ASSERT(state.particle_bgl != NULL);
  }

  /* Depth: UBO only */
  {
    WGPUBindGroupLayoutEntry e[1] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(dunes_depth_uniforms_t)}},
    };
    state.depth_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label = STRVIEW("Dunes - Depth BGL"), .entryCount = 1, .entries = e});
    ASSERT(state.depth_bgl != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups (rebuilt after textures load or preset changes)
 * -------------------------------------------------------------------------- */

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  const dunes_preset_t* p = &dunes_presets[state.current_preset];

  /* Terrain */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.terrain_bg)
    WGPUBindGroupEntry e[5] = {
      [0] = {.binding = 0,
             .buffer  = state.terrain_ubo.buffer,
             .size = DUNES_UBO_STRIDE}, /* per-slot size for dynamic offset */
      [1] = {.binding = 1, .sampler = state.sampler_repeat},
      [2]
      = {.binding = 2, .textureView = state.textures[DUNES_TEX_DIFFUSE].view},
      [3] = {.binding = 3, .textureView = state.textures[DUNES_TEX_DUST].view},
      [4]
      = {.binding = 4, .textureView = state.textures[DUNES_TEX_DETAIL].view},
    };
    state.terrain_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Terrain BG"),
                                 .layout     = state.terrain_bgl,
                                 .entryCount = 5,
                                 .entries    = e});
    ASSERT(state.terrain_bg != NULL);
  }

  /* Sky */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.sky_bg)
    WGPUBindGroupEntry e[3] = {
      [0] = {.binding = 0,
             .buffer  = state.diffuse_ubo.buffer,
             .size    = state.diffuse_ubo.size},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[p->sky_tex_idx].view},
    };
    state.sky_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Sky BG"),
                                 .layout     = state.diffuse_bgl,
                                 .entryCount = 3,
                                 .entries    = e});
    ASSERT(state.sky_bg != NULL);
  }

  /* Colored (sun flare + palms share colored_ubo) */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.colored_bg)
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.colored_ubo.buffer,
             .size = DUNES_UBO_STRIDE}, /* per-slot size for dynamic offset */
      [1] = {.binding = 1, .sampler = state.sampler},
      [2]
      = {.binding = 2, .textureView = state.textures[DUNES_TEX_SUN_FLARE].view},
      [3]
      = {.binding = 3, .textureView = state.textures[DUNES_TEX_SUN_FLARE].view},
    };
    state.colored_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Colored BG"),
                                 .layout     = state.colored_bgl,
                                 .entryCount = 4,
                                 .entries    = e});
    ASSERT(state.colored_bg != NULL);
  }

  /* Palms (2 textures: diffuse + alpha) */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.palms_bg)
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.colored_ubo.buffer,
             .size = DUNES_UBO_STRIDE}, /* per-slot size for dynamic offset */
      [1] = {.binding = 1, .sampler = state.sampler},
      [2]
      = {.binding = 2, .textureView = state.textures[DUNES_TEX_PALM_DIFF].view},
      [3] = {.binding     = 3,
             .textureView = state.textures[DUNES_TEX_PALM_ALPHA].view},
    };
    state.palms_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Palms BG"),
                                 .layout     = state.colored_bgl,
                                 .entryCount = 4,
                                 .entries    = e});
    ASSERT(state.palms_bg != NULL);
  }

  /* Animated (bird) */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.animated_bg)
    WGPUBindGroupEntry e[3] = {
      [0] = {.binding = 0,
             .buffer  = state.animated_ubo.buffer,
             .size = DUNES_UBO_STRIDE}, /* per-slot size for dynamic offset */
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[DUNES_TEX_BIRD].view},
    };
    state.animated_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Bird BG"),
                                 .layout     = state.animated_bgl,
                                 .entryCount = 3,
                                 .entries    = e});
    ASSERT(state.animated_bg != NULL);
  }

  /* Soft particles (smoke + offscreen depth) */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bg)
    WGPUBindGroupEntry e[4] = {
      [0] = {.binding = 0,
             .buffer  = state.particle_ubo.buffer,
             .size = DUNES_UBO_STRIDE}, /* per-slot size for dynamic offset */
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[DUNES_TEX_SMOKE].view},
      [3] = {.binding = 3, .textureView = state.offscreen_depth.view},
    };
    state.particle_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Particle BG"),
                                 .layout     = state.particle_bgl,
                                 .entryCount = 4,
                                 .entries    = e});
    ASSERT(state.particle_bg != NULL);
  }

  /* Depth pass */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.depth_bg)
    WGPUBindGroupEntry e[1] = {
      [0] = {.binding = 0,
             .buffer  = state.depth_ubo.buffer,
             .size    = state.depth_ubo.size},
    };
    state.depth_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){.label      = STRVIEW("Dunes - Depth BG"),
                                 .layout     = state.depth_bgl,
                                 .entryCount = 1,
                                 .entries    = e});
    ASSERT(state.depth_bg != NULL);
  }

  state.resources_dirty = false;
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat fmt = wgpu_context->render_format;

  /* Vertex layout: pos3 + uv2 + normal3 (stride=32) used by dunes/sky/palms */
  WGPU_VERTEX_BUFFER_LAYOUT(
    mesh32, 8 * sizeof(float),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 3 * sizeof(float)),
    WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3, 5 * sizeof(float)))

  /* Vertex layout: pos3 + uv2 (stride=20) used by sky and sun */
  WGPU_VERTEX_BUFFER_LAYOUT(
    mesh20, 5 * sizeof(float),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 3 * sizeof(float)))

  /* Bird vertex layout:
   * stride = (BIRD_FRAMES * 3 + 2) * 4 = 17 * 4 = 68 bytes
   * pos1: @location(0) at offset (frame1 * 3 * 4)
   * pos2: @location(1) at offset (frame2 * 3 * 4)
   * uv:   @location(2) at offset (BIRD_FRAMES * 3 * 4) = 60
   * Both positions use same stride but different offsets, so we use 3 vertex
   * buffers pointing to the same GPU buffer with different offsets. */
  /* We actually pass frame indices + morph via UBO and do offset calculations
   * in WGSL using @builtin(vertex_index), but the simplest approach is to bind
   * the same buffer twice as two separate vertex buffers at the correct
   * offsets. We use a single buffer at different base offsets per draw call. */

  WGPUDepthStencilState ds_write
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = DUNES_DEPTH_FORMAT, .depth_write_enabled = true});
  ds_write.depthCompare = WGPUCompareFunction_Less;

  WGPUDepthStencilState ds_lequal
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = DUNES_DEPTH_FORMAT, .depth_write_enabled = true});
  ds_lequal.depthCompare = WGPUCompareFunction_LessEqual;

  WGPUDepthStencilState ds_nowrite
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = DUNES_DEPTH_FORMAT, .depth_write_enabled = false});
  ds_nowrite.depthCompare = WGPUCompareFunction_Always;

  WGPUDepthStencilState ds_depth32_write
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format = WGPUTextureFormat_Depth32Float, .depth_write_enabled = true});
  ds_depth32_write.depthCompare = WGPUCompareFunction_Less;

  /* Additive blend (sun/particles) */
  WGPUBlendState blend_add = {
    .color = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One},
    .alpha = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_Zero,
              .dstFactor = WGPUBlendFactor_One},
  };

  /* Alpha blend (soft particles) */
  WGPUBlendState blend_alpha = {
    .color = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One},
    .alpha = {.operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_Zero,
              .dstFactor = WGPUBlendFactor_One},
  };

  /* --- Terrain pipeline --- */
  {
    state.terrain_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Terrain PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.terrain_bgl});
    WGPUShaderModule sm    = wgpu_create_shader_module(wgpu_context->device,
                                                       dunes_terrain_shader_wgsl);
    state.terrain_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Dunes - Terrain pipeline"),
        .layout   = state.terrain_pl,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = &mesh32_vertex_buffer_layout},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None, /* skirts use neg scale */
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_write,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.terrain_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Sky pipeline (diffuse, no cull, always depth) --- */
  {
    state.diffuse_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Diffuse PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.diffuse_bgl});
    WGPUShaderModule sm = wgpu_create_shader_module(wgpu_context->device,
                                                    dunes_diffuse_shader_wgsl);
    WGPUDepthStencilState ds_sky
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format = DUNES_DEPTH_FORMAT, .depth_write_enabled = true});
    ds_sky.depthCompare = WGPUCompareFunction_LessEqual;
    state.sky_pipeline  = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
         .label    = STRVIEW("Dunes - Sky pipeline"),
         .layout   = state.diffuse_pl,
         .vertex   = {.module      = sm,
                      .entryPoint  = STRVIEW("vertexMain"),
                      .bufferCount = 1,
                      .buffers     = &mesh20_vertex_buffer_layout},
         .fragment = &(
          WGPUFragmentState){.module      = sm,
                              .entryPoint  = STRVIEW("fragmentMain"),
                              .targetCount = 1,
                              .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                          .writeMask
                                                          = WGPUColorWriteMask_All}},
         .primitive
        = {.topology = WGPUPrimitiveTopology_TriangleList,
            .cullMode
            = WGPUCullMode_Back, /* sky is inverted sphere (CCW from inside) */
            .frontFace = WGPUFrontFace_CCW},
         .depthStencil = &ds_sky,
         .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.sky_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Sun/colored pipeline (additive blend, no depth write) --- */
  {
    state.colored_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Colored PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.colored_bgl});
    WGPUShaderModule sm = wgpu_create_shader_module(
      wgpu_context->device, dunes_diffuse_colored_shader_wgsl);
    state.colored_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Dunes - Colored pipeline"),
        .layout   = state.colored_pl,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = &mesh20_vertex_buffer_layout},
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
        .depthStencil = &ds_nowrite,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.colored_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Palms pipeline (alpha discard, no cull) --- */
  {
    WGPUShaderModule sm = wgpu_create_shader_module(
      wgpu_context->device, dunes_diffuse_alpha_shader_wgsl);
    state.palms_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Dunes - Palms pipeline"),
        .layout   = state.colored_pl,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = &mesh20_vertex_buffer_layout},
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
        .depthStencil = &ds_lequal,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.palms_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Bird / animated pipeline (alpha discard, no cull) --- */
  {
    state.animated_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Animated PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.animated_bgl});
    WGPUShaderModule sm = wgpu_create_shader_module(
      wgpu_context->device, dunes_animated_colored_shader_wgsl);
    /* Bird animation vertex buffer layout (Bug 2 fix):
     * The bird buffer has stride=68:
     * [f0_pos(12)][f1_pos(12)][f2_pos(12)][f3_pos(12)][f4_pos(12)][uv(8)] Using
     * a single buffer with base_offset=frame1*12 correctly addresses pos1 and
     * pos2, BUT it also shifts the UV reads (which must stay at byte 60 within
     * each vertex). Fix: use 3 separate vertex buffer slots, each with
     * stride=68 and ONE attribute at offset 0.
     * - Slot 0 (pos1): bound at base_offset = frame1 * 3 * 4
     * - Slot 1 (pos2): bound at base_offset = frame2 * 3 * 4
     * - Slot 2 (uv):   bound at fixed base_offset = 60 bytes into each vertex
     * start Each slot reads its attribute at offset 0 within stride 68, so:
     *   slot 0, vertex i: buffer[frame1*12 + i*68 + 0] = frame1 positions ✓
     *   slot 1, vertex i: buffer[frame2*12 + i*68 + 0] = frame2 positions ✓
     *   slot 2, vertex i: buffer[60 + i*68 + 0]        = UV data ✓ */
    WGPUVertexAttribute pos1_attr = {
      .format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0};
    WGPUVertexAttribute pos2_attr = {
      .format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 1};
    WGPUVertexAttribute uv_attr = {
      .format = WGPUVertexFormat_Float32x2, .offset = 0, .shaderLocation = 2};
    WGPUVertexBufferLayout bird_vbls[3] = {
      {.arrayStride    = 68,
       .stepMode       = WGPUVertexStepMode_Vertex,
       .attributeCount = 1,
       .attributes     = &pos1_attr},
      {.arrayStride    = 68,
       .stepMode       = WGPUVertexStepMode_Vertex,
       .attributeCount = 1,
       .attributes     = &pos2_attr},
      {.arrayStride    = 68,
       .stepMode       = WGPUVertexStepMode_Vertex,
       .attributeCount = 1,
       .attributes     = &uv_attr},
    };
    state.animated_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Dunes - Animated pipeline"),
        .layout   = state.animated_pl,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 3,
                     .buffers     = bird_vbls},
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
        .depthStencil = &ds_lequal,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.animated_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Soft particle pipeline (additive blend, no depth write) --- */
  {
    state.particle_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Particle PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.particle_bgl});
    WGPUShaderModule sm = wgpu_create_shader_module(
      wgpu_context->device, dunes_soft_particle_shader_wgsl);
    state.particle_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Dunes - Particle pipeline"),
        .layout   = state.particle_pl,
        .vertex   = {.module      = sm,
                     .entryPoint  = STRVIEW("vertexMain"),
                     .bufferCount = 1,
                     .buffers     = &mesh20_vertex_buffer_layout},
        .fragment = &(
          WGPUFragmentState){.module      = sm,
                             .entryPoint  = STRVIEW("fragmentMain"),
                             .targetCount = 1,
                             .targets     = &(
                               WGPUColorTargetState){.format = fmt,
                                                         .blend  = &blend_alpha,
                                                         .writeMask
                                                         = WGPUColorWriteMask_All}},
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_None,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_nowrite,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.particle_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* --- Depth-only pipeline (renders dunes to offscreen depth) --- */
  {
    state.depth_pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){.label = STRVIEW("Dunes - Depth PL"),
                                      .bindGroupLayoutCount = 1,
                                      .bindGroupLayouts = &state.depth_bgl});
    WGPUShaderModule sm  = wgpu_create_shader_module(wgpu_context->device,
                                                     dunes_depth_shader_wgsl);
    state.depth_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = STRVIEW("Dunes - Depth pipeline"),
        .layout       = state.depth_pl,
        .vertex       = {.module      = sm,
                         .entryPoint  = STRVIEW("vertexMain"),
                         .bufferCount = 1,
                         .buffers     = &mesh32_vertex_buffer_layout},
        .fragment     = NULL, /* depth-only, no color output */
        .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                         .cullMode  = WGPUCullMode_Back,
                         .frontFace = WGPUFrontFace_CCW},
        .depthStencil = &ds_depth32_write,
        .multisample  = {.count = 1, .mask = 0xffffffff}});
    ASSERT(state.depth_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }
}

/* -------------------------------------------------------------------------- *
 * Asset loading
 * -------------------------------------------------------------------------- */

/* Generic model fetch state */
typedef struct {
  dunes_model_t* model;
  bool is_indices; /* true=index buffer, false=vertex buffer */
} dunes_fetch_model_t;

static dunes_fetch_model_t fetch_states[12]; /* 6 models * 2 buffers each */
static int fetch_state_count = 0;

static void fetch_model_cb(const sfetch_response_t* response)
{
  dunes_fetch_model_t* fs = (dunes_fetch_model_t*)response->user_data;
  if (!response->fetched) {
    printf("[DUNES] Model fetch failed (error %d)\n", response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  /* Allocate padded to 4 bytes to allow 4-byte aligned GPU writes */
  size_t data_size   = response->data.size;
  size_t padded_size = (data_size + 3u) & ~3u;
  uint8_t* copy = (uint8_t*)calloc(1, padded_size); /* calloc zeros padding */
  if (!copy) {
    free((void*)response->buffer.ptr);
    return;
  }
  memcpy(copy, response->data.ptr, data_size);

  if (fs->is_indices) {
    fs->model->pending_idx      = copy;
    fs->model->pending_idx_size = padded_size;
    fs->model->index_count      = (uint32_t)(data_size / 2);
    fs->model->idx_pending      = true;
  }
  else {
    fs->model->pending_vtx      = copy;
    fs->model->pending_vtx_size = padded_size;
    fs->model->vtx_pending      = true;
  }
  free((void*)response->buffer.ptr);
}

typedef struct {
  wgpu_texture_t* tex;
  uint32_t* loaded_count;
} dunes_tex_fetch_t;

static dunes_tex_fetch_t tex_fetch_states[DUNES_TEX__COUNT];

static void fetch_texture_cb(const sfetch_response_t* response)
{
  dunes_tex_fetch_t* tf = (dunes_tex_fetch_t*)response->user_data;
  if (!response->fetched) {
    printf("[DUNES] Texture fetch failed (error %d)\n", response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  int w, h, channels;
  uint8_t* pixels
    = image_pixels_from_memory((const uint8_t*)response->data.ptr,
                               (int)response->data.size, &w, &h, &channels, 4);

  if (pixels) {
    tf->tex->desc = (wgpu_texture_desc_t){
      .extent = {(uint32_t)w, (uint32_t)h, 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {.ptr = pixels, .size = (size_t)(w * h * 4)},
    };
    tf->tex->desc.is_dirty = true;
    (*tf->loaded_count)++;
  }
  free((void*)response->buffer.ptr);
}

static void load_model(dunes_model_t* model, const char* base_path,
                       size_t idx_buf_size, size_t vtx_buf_size)
{
  char path[256];

  dunes_fetch_model_t* fi = &fetch_states[fetch_state_count++];
  fi->model               = model;
  fi->is_indices          = true;

  snprintf(path, sizeof(path), "%s-indices.bin", base_path);
  uint8_t* ib = (uint8_t*)malloc(idx_buf_size);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = fetch_model_cb,
    .buffer    = {.ptr = ib, .size = idx_buf_size},
    .user_data = {.ptr = fi, .size = sizeof(dunes_fetch_model_t)},
  });

  dunes_fetch_model_t* fv = &fetch_states[fetch_state_count++];
  fv->model               = model;
  fv->is_indices          = false;

  snprintf(path, sizeof(path), "%s-strides.bin", base_path);
  uint8_t* vb = (uint8_t*)malloc(vtx_buf_size);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = fetch_model_cb,
    .buffer    = {.ptr = vb, .size = vtx_buf_size},
    .user_data = {.ptr = fv, .size = sizeof(dunes_fetch_model_t)},
  });
}

static void load_texture(wgpu_context_t* wgpu_context, dunes_tex_id_t slot,
                         const char* path, size_t buf_size)
{
  state.textures[slot] = wgpu_create_color_bars_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
               | WGPUTextureUsage_RenderAttachment,
    });

  tex_fetch_states[slot].tex          = &state.textures[slot];
  tex_fetch_states[slot].loaded_count = &state.textures_loaded;

  uint8_t* buf = (uint8_t*)malloc(buf_size);
  sfetch_send(&(sfetch_request_t){
    .path     = path,
    .callback = fetch_texture_cb,
    .buffer   = {.ptr = buf, .size = buf_size},
    .user_data
    = {.ptr = &tex_fetch_states[slot], .size = sizeof(dunes_tex_fetch_t)},
  });
}

/* -------------------------------------------------------------------------- *
 * GPU buffer allocation and upload
 * -------------------------------------------------------------------------- */

static void alloc_model_buffers(wgpu_context_t* wgpu_context,
                                dunes_model_t* model, size_t idx_size,
                                size_t vtx_size, const char* label_idx,
                                const char* label_vtx)
{
  /* WebGPU requires buffer sizes to be a multiple of 4 bytes */
  size_t idx_size_aligned = (idx_size + 3u) & ~3u;
  size_t vtx_size_aligned = (vtx_size + 3u) & ~3u;
  model->index_buffer     = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){.label = label_idx,
                                            .usage = WGPUBufferUsage_CopyDst
                                                     | WGPUBufferUsage_Index,
                                            .size = idx_size_aligned});
  model->vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){.label = label_vtx,
                                        .usage = WGPUBufferUsage_CopyDst
                                                 | WGPUBufferUsage_Vertex,
                                        .size = vtx_size_aligned});
}

static void upload_model_pending(wgpu_context_t* wgpu_context,
                                 dunes_model_t* model)
{
  if (model->idx_pending && model->pending_idx) {
    /* WebGPU requires write size to be a multiple of 4 bytes */
    size_t write_size = (model->pending_idx_size + 3u) & ~3u;
    wgpuQueueWriteBuffer(wgpu_context->queue, model->index_buffer.buffer, 0,
                         model->pending_idx, write_size);
    free(model->pending_idx);
    model->pending_idx = NULL;
    model->idx_pending = false;
  }
  if (model->vtx_pending && model->pending_vtx) {
    size_t write_size = (model->pending_vtx_size + 3u) & ~3u;
    wgpuQueueWriteBuffer(wgpu_context->queue, model->vertex_buffer.buffer, 0,
                         model->pending_vtx, write_size);
    free(model->pending_vtx);
    model->pending_vtx = NULL;
    model->vtx_pending = false;
  }
  if (!model->is_ready && !model->idx_pending && !model->vtx_pending
      && model->index_count > 0) {
    model->is_ready = true;
    state.models_loaded++;
  }
}

static void upload_pending_textures(wgpu_context_t* wgpu_context)
{
  bool any = false;
  for (uint32_t i = 0; i < DUNES_TEX__COUNT; ++i) {
    if (state.textures[i].desc.is_dirty) {
      wgpu_recreate_texture(wgpu_context, &state.textures[i]);
      FREE_TEXTURE_PIXELS(state.textures[i]);
      any = true;
    }
  }
  if (any) {
    state.resources_dirty = true;
  }
}

static void upload_pending_models(wgpu_context_t* wgpu_context)
{
  upload_model_pending(wgpu_context, &state.dunes_model);
  upload_model_pending(wgpu_context, &state.sky_model);
  upload_model_pending(wgpu_context, &state.smoke_model);
  upload_model_pending(wgpu_context, &state.sun_model);
  upload_model_pending(wgpu_context, &state.bird_model);
  upload_model_pending(wgpu_context, &state.palms_model);
}

/* -------------------------------------------------------------------------- *
 * Camera helpers
 * -------------------------------------------------------------------------- */

static void dunes_randomize_camera(void)
{
  /* Advance by a random amount to avoid repeating */
  uint32_t skip        = 1 + (uint32_t)(rand() % (DUNES_RANDOM_CAM_COUNT - 1));
  state.random_cam_idx = (state.random_cam_idx + skip) % DUNES_RANDOM_CAM_COUNT;

  const float* pos = dunes_random_positions[state.random_cam_idx];

  /* Implement WebGL's minRandom(threshold): returns a value with magnitude >=
   * 0.3 WebGL: let r = Math.random()-0.5; if(r<0 && r>-t) r-=t; else if(r>0 &&
   * r<t) r+=t; Without this, eye≈center gives a near-zero look direction →
   * degenerate lookAt. */
  const float t = 0.3f;
  float rx      = (float)rand() / RAND_MAX - 0.5f;
  if (rx < 0.0f && rx > -t)
    rx -= t;
  else if (rx > 0.0f && rx < t)
    rx += t;
  float ry = (float)rand() / RAND_MAX - 0.5f;
  if (ry < 0.0f && ry > -t)
    ry -= t;
  else if (ry > 0.0f && ry < t)
    ry += t;
  float rz = (float)rand() / RAND_MAX - 0.5f;
  if (rz < 0.0f && rz > -t)
    rz -= t;
  else if (rz > 0.0f && rz < t)
    rz += t;

  state.random_lookat[0] = pos[0] + rx;
  state.random_lookat[1] = pos[1] + ry;
  state.random_lookat[2] = pos[2] + rz * 0.1f; /* small Z offset like WebGL */

  state.random_fov = (float)rand() / RAND_MAX * 30.0f;
}

/* Build view matrix based on current camera state */
static void position_camera(void)
{
  if (state.camera_mode == DUNES_CAMERA_RANDOM) {
    /* Wobble lookat */
    float lx
      = state.random_lookat[0]
        + sinf(state.timer_camera_wobble * 2.0f * (float)M_PI * 2.0f) * 0.2f;
    float ly = state.random_lookat[1];
    float lz
      = state.random_lookat[2]
        + cosf(state.timer_camera_wobble * 3.0f * (float)M_PI * 2.0f) * 0.05f;
    const float* eye = dunes_random_positions[state.random_cam_idx];
    glm_lookat((vec3){eye[0], eye[1], eye[2]}, (vec3){lx, ly, lz},
               (vec3){0.0f, 0.0f, 1.0f}, state.view_matrix);
  }
  else {
    /* Rotating mode */
    glm_lookat((vec3){0.0f, 0.0f, -400.0f}, (vec3){1000.0f, 0.0f, -600.0f},
               (vec3){0.0f, 0.1f, 1.0f}, state.view_matrix);
    /* Same unit bug as WebGL version: pass degrees directly to glm_rotate_z */
    glm_rotate_z(state.view_matrix,
                 (state.angle_yaw + 280.0f) / 160.0f * 6.2831852f,
                 state.view_matrix);
  }
}

/* Build projection matrix */
static void set_projection(wgpu_context_t* wgpu_context)
{
  float w      = (float)wgpu_context->width;
  float h      = (float)wgpu_context->height;
  float aspect = (h > 0.0f) ? (w / h) : 1.0f;
  float fov    = (w >= h) ? DUNES_FOV_LANDSCAPE : DUNES_FOV_PORTRAIT;

  if (state.camera_mode == DUNES_CAMERA_RANDOM) {
    fov -= state.timer_random_camera * state.random_fov;
  }

  glm_perspective(glm_rad(fov), aspect, DUNES_Z_NEAR, DUNES_Z_FAR,
                  state.proj_matrix);
}

/* Compute billboard MVP: preserves translation/scale but removes rotation
 * (faces camera), then applies Z rotation for particle spin. */
static void compute_billboard_mvp(float tx, float ty, float tz, float sx,
                                  float sz, float rotation, mat4 out)
{
  /* Model matrix: translate then scale */
  mat4 model;
  glm_mat4_identity(model);
  glm_translate(model, (vec3){tx, ty, tz});
  glm_scale(model, (vec3){sx, 1.0f, sz});

  /* MV = view * model */
  mat4 mv;
  glm_mat4_mul(state.view_matrix, model, mv);

  /* Zero out rotational part of MV to face camera */
  float d
    = sqrtf(mv[0][0] * mv[0][0] + mv[0][1] * mv[0][1] + mv[0][2] * mv[0][2]);
  mv[0][0] = d;
  mv[1][0] = 0;
  mv[2][0] = 0;
  mv[0][1] = 0;
  mv[1][1] = d;
  mv[2][1] = 0;
  mv[0][2] = 0;
  mv[1][2] = 0;
  mv[2][2] = d;

  /* Apply Z rotation after billboard */
  glm_rotate_z(mv, rotation, mv);

  /* MVP = proj * MV */
  glm_mat4_mul(state.proj_matrix, mv, out);
}

/* Compute standard MVP (no billboard) */
static void compute_mvp(float tx, float ty, float tz, float rx, float ry,
                        float rz, float sx, float sy, float sz, mat4 out)
{
  mat4 model;
  glm_mat4_identity(model);
  glm_translate(model, (vec3){tx, ty, tz});
  glm_scale(model, (vec3){sx, sy, sz});
  if (rx != 0.0f)
    glm_rotate_x(model, rx, model);
  if (ry != 0.0f)
    glm_rotate_y(model, ry, model);
  if (rz != 0.0f)
    glm_rotate_z(model, rz, model);

  mat4 vp;
  glm_mat4_mul(state.proj_matrix, state.view_matrix, vp);
  glm_mat4_mul(vp, model, out);
}

/* -------------------------------------------------------------------------- *
 * Uniform update helpers
 * -------------------------------------------------------------------------- */

/* Fill terrain UBO for one tile (no GPU write; caller writes to the buffer). */
static void fill_terrain_uniforms(dunes_terrain_uniforms_t* u, float tx,
                                  float ty, float tz, float sx, float sy,
                                  float sz)
{
  const dunes_preset_t* p = &dunes_presets[state.current_preset];
  memset(u, 0, sizeof(*u));
  compute_mvp(tx, ty, tz, 0, 0, 0, sx, sy, sz, u->view_proj_matrix);
  glm_vec4_copy(
    (float[]){p->sand_color[0], p->sand_color[1], p->sand_color[2], 1.0f},
    u->color);
  glm_vec4_copy(
    (float[]){p->fog_color[0], p->fog_color[1], p->fog_color[2], 1.0f},
    u->fog_color);
  glm_vec4_copy(
    (float[]){p->shadow_color[0], p->shadow_color[1], p->shadow_color[2], 1.0f},
    u->shadow_color);
  glm_vec4_copy(
    (float[]){p->waves_color[0], p->waves_color[1], p->waves_color[2], 1.0f},
    u->waves_color);
  u->time                  = state.dust_timer;
  u->dust_opacity          = 0.075f;
  u->fog_start_distance    = p->fog_start_distance;
  u->fog_distance          = p->fog_distance;
  u->detail_start_distance = 400.0f;
  u->detail_distance       = 1200.0f;
  /* dunes_shader_variant 0 = sunset/sunrise, use tex.r; 1 = day/night, use
   * tex.g */
  u->lightmap_select = (float)p->dunes_shader_variant;
}

/* Update diffuse UBO (sky sphere, drawn once per frame). */
static void update_diffuse_uniforms(wgpu_context_t* wgpu_context, float tx,
                                    float ty, float tz, float sx, float sy,
                                    float sz)
{
  dunes_diffuse_uniforms_t u;
  compute_mvp(tx, ty, tz, 0, 0, 0, sx, sy, sz, u.view_proj_matrix);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.diffuse_ubo.buffer, 0, &u,
                       sizeof(u));
}

/* Update depth UBO (terrain depth pass, drawn once per frame). */
static void update_depth_uniforms(wgpu_context_t* wgpu_context)
{
  dunes_depth_uniforms_t u;
  compute_mvp(0, 0, 0, 0, 0, 0, DUNES_TERRAIN_SCALE, DUNES_TERRAIN_SCALE,
              DUNES_TERRAIN_SCALE, u.view_proj_matrix);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.depth_ubo.buffer, 0, &u,
                       sizeof(u));
}

/* Pre-compute ALL dynamic UBO data for the current frame and write them to
 * their respective dynamic buffers BEFORE the render pass starts.
 * Each instance is placed at a 256-byte-aligned offset within its buffer so
 * that per-draw dynamic offsets select the correct slot. */
static void precompute_ubos(wgpu_context_t* wgpu_context)
{
  const dunes_preset_t* p  = &dunes_presets[state.current_preset];
  float ts                 = DUNES_TERRAIN_SCALE;
  const float SKIRT_SCALE  = 1.5f;
  const float SKIRT_OFFSET = 9030.0f / 2.0f + (9030.0f / 2.0f * SKIRT_SCALE);

  /* ---- Terrain: 9 tiles ----
   * Tile layout matches WebGL source:
   *   [0] main:              tx=0,          ty=0,          sx=+ts,    sy=+ts
   *   [1] +X skirt:          tx=+SKIRT_OFF, ty=0,          sx=-ts*1.5, sy=+ts
   *   [2] -X skirt:          tx=-SKIRT_OFF, ty=0,          sx=-ts*1.5, sy=+ts
   *   [3] +Y skirt:          tx=0,          ty=+SKIRT_OFF, sx=+ts, sy=-ts*1.5
   *   [4] -Y skirt:          tx=0,          ty=-SKIRT_OFF, sx=+ts, sy=-ts*1.5
   *   [5..8] corners:  both negative scale
   */
  static const float tile_tx[9] = {0, 1, -1, 0, 0, 1, -1, 1, -1};
  static const float tile_ty[9] = {0, 0, 0, 1, -1, 1, -1, -1, 1};
  /* positive scale factor: 1.0 = normal, -1.5 = mirrored skirt */
  static const float tile_sfx[9]
    = {1.0f, -1.5f, -1.5f, 1.0f, 1.0f, -1.5f, -1.5f, -1.5f, -1.5f};
  static const float tile_sfy[9]
    = {1.0f, 1.0f, 1.0f, -1.5f, -1.5f, -1.5f, -1.5f, -1.5f, -1.5f};

  static uint8_t terrain_staging[DUNES_TERRAIN_SLOTS * DUNES_UBO_STRIDE];
  memset(terrain_staging, 0, sizeof(terrain_staging));
  for (uint32_t i = 0; i < DUNES_TERRAIN_SLOTS; i++) {
    dunes_terrain_uniforms_t* u
      = (dunes_terrain_uniforms_t*)(terrain_staging + i * DUNES_UBO_STRIDE);
    fill_terrain_uniforms(u, tile_tx[i] * SKIRT_OFFSET,
                          tile_ty[i] * SKIRT_OFFSET, 0.0f, tile_sfx[i] * ts,
                          tile_sfy[i] * ts, ts);
  }
  wgpuQueueWriteBuffer(wgpu_context->queue, state.terrain_ubo.buffer, 0,
                       terrain_staging, sizeof(terrain_staging));

  /* ---- Colored: slot 0=palms, slot 1=sun ---- */
  static uint8_t colored_staging[DUNES_COLORED_SLOTS * DUNES_UBO_STRIDE];
  memset(colored_staging, 0, sizeof(colored_staging));
  {
    /* Palms at slot 0 */
    dunes_diffuse_colored_uniforms_t* u0
      = (dunes_diffuse_colored_uniforms_t*)(colored_staging
                                            + 0 * DUNES_UBO_STRIDE);
    mat4 mvp;
    compute_mvp(0, 0, 0, 0, 0, 0, ts, ts, ts, mvp);
    glm_mat4_copy(mvp, u0->view_proj_matrix);
    glm_vec4_copy(
      (float[]){p->deco_color[0], p->deco_color[1], p->deco_color[2], 1.0f},
      u0->color);

    /* Sun at slot 1 */
    dunes_diffuse_colored_uniforms_t* u1
      = (dunes_diffuse_colored_uniforms_t*)(colored_staging
                                            + 1 * DUNES_UBO_STRIDE);
    float sun_rot = state.timer_dust_rotation * (float)M_PI * 2.0f;
    compute_billboard_mvp(p->sun_tx, p->sun_ty, p->sun_tz, p->sun_sx, p->sun_sz,
                          sun_rot, mvp);
    glm_mat4_copy(mvp, u1->view_proj_matrix);
    glm_vec4_copy((float[]){255.0f / 255.0f * 0.6f, 229.0f / 255.0f * 0.6f,
                            159.0f / 255.0f * 0.6f, 1.0f},
                  u1->color);
  }
  wgpuQueueWriteBuffer(wgpu_context->queue, state.colored_ubo.buffer, 0,
                       colored_staging, sizeof(colored_staging));

  /* ---- Birds: slot 0=bird1, slot 1=bird2 ---- */
  static uint8_t animated_staging[DUNES_BIRD_SLOTS * DUNES_UBO_STRIDE];
  memset(animated_staging, 0, sizeof(animated_staging));
  {
    float angle = state.timer_birds_fly * (float)M_PI * 2.0f;
    float bx1   = sinf(angle) * DUNES_BIRD_FLIGHT_RADIUS;
    float by1   = cosf(angle) * DUNES_BIRD_FLIGHT_RADIUS;
    float bx2   = sinf(-angle - (float)M_PI) * DUNES_BIRD_FLIGHT_RADIUS;
    float by2   = cosf(-angle - (float)M_PI) * DUNES_BIRD_FLIGHT_RADIUS;

    float timers[2] = {state.timer_bird1, state.timer_bird2};
    float bx[2]     = {bx1 + 1300.0f, bx2 - 1300.0f};
    float by_arr[2] = {by1, by2};
    float rz[2]     = {-angle - (float)M_PI * 0.5f, angle + (float)M_PI * 1.5f};
    float scales[2] = {6.0f, 5.0f};

    for (int b = 0; b < 2; b++) {
      int f1      = (int)(timers[b] * DUNES_BIRD_FRAMES) % DUNES_BIRD_FRAMES;
      int f2      = (f1 + 1) % DUNES_BIRD_FRAMES;
      float morph = (timers[b] * DUNES_BIRD_FRAMES)
                    - (int)(timers[b] * DUNES_BIRD_FRAMES);

      state.bird_vtx_pos1[b] = (uint64_t)(f1 * 3 * 4);
      state.bird_vtx_pos2[b] = (uint64_t)(f2 * 3 * 4);

      mat4 mvp;
      compute_mvp(bx[b], by_arr[b], 0, 0, 0, rz[b], scales[b], scales[b],
                  scales[b], mvp);

      dunes_animated_uniforms_t* u
        = (dunes_animated_uniforms_t*)(animated_staging + b * DUNES_UBO_STRIDE);
      glm_mat4_copy(mvp, u->view_proj_matrix);
      glm_vec4_copy(
        (float[]){p->deco_color[0], p->deco_color[1], p->deco_color[2], 1.0f},
        u->color);
      u->morph = morph;
    }
  }
  wgpuQueueWriteBuffer(wgpu_context->queue, state.animated_ubo.buffer, 0,
                       animated_staging, sizeof(animated_staging));

  /* ---- Particles: 147 slots ---- */
  static uint8_t particle_staging[DUNES_PARTICLE_COUNT * DUNES_UBO_STRIDE];
  memset(particle_staging, 0, sizeof(particle_staging));
  {
    float inv_w = 1.0f / (float)wgpu_context->width;
    float inv_h = 1.0f / (float)wgpu_context->height;

    for (uint32_t i = 0; i < DUNES_PARTICLE_COUNT; i++) {
      float timer = fmodf(state.timer_dust_movement + (float)i * 13.37f, 1.0f);
      const float* c = dunes_particles[i];
      float rotation
        = (float)i * 35.0f
          + state.timer_dust_rotation * (float)(i % 2 == 0 ? 360 : -360);
      float x       = c[0] * DUNES_TERRAIN_SCALE + timer * DUNES_DUST_TRAVEL_X;
      float y       = c[1] * DUNES_TERRAIN_SCALE;
      float z       = c[2] * DUNES_TERRAIN_SCALE + timer * DUNES_DUST_TRAVEL_Z;
      float scale   = timer * DUNES_DUST_MAX_SCALE;
      float opacity = dunes_smootherstep(0.01f, 0.1f, timer)
                      * (1.0f - dunes_smootherstep(0.7f, 0.99f, timer));
      float rot_rad
        = rotation; /* WebGL passes raw to mat4.rotateZ (radians expected
                     * but WebGL source passes degree-scale values directly,
                     * matching fast-spin visual from the original demo) */

      mat4 mvp;
      compute_billboard_mvp(x, y, z, scale, 1.0f, rot_rad, mvp);

      dunes_soft_particle_uniforms_t* u
        = (dunes_soft_particle_uniforms_t*)(particle_staging
                                            + i * DUNES_UBO_STRIDE);
      glm_mat4_copy(mvp, u->view_proj_matrix);
      glm_vec4_copy((float[]){p->dust_color[0] * opacity,
                              p->dust_color[1] * opacity,
                              p->dust_color[2] * opacity, 1.0f},
                    u->color);
      u->camera_near     = DUNES_Z_NEAR;
      u->camera_far      = DUNES_Z_FAR;
      u->inv_viewport_x  = inv_w;
      u->inv_viewport_y  = inv_h;
      u->transition_size = DUNES_SMOKE_SOFTNESS;
    }
  }
  wgpuQueueWriteBuffer(wgpu_context->queue, state.particle_ubo.buffer, 0,
                       particle_staging, sizeof(particle_staging));
}

/* -------------------------------------------------------------------------- *
 * Animation
 * -------------------------------------------------------------------------- */

static void animate(void)
{
  uint64_t now = stm_now();
  if (state.last_time == 0) {
    state.last_time = now;
    return;
  }

  float elapsed_ms = (float)stm_ms(stm_diff(now, state.last_time));
  state.last_time  = now;
  float now_ms     = (float)stm_ms(now);

  state.angle_yaw += elapsed_ms / DUNES_YAW_SPEED;
  if (state.angle_yaw >= 360.0f)
    state.angle_yaw -= 360.0f;

  state.dust_timer += elapsed_ms / DUNES_DUST_TIMER_SPEED;
  if (state.dust_timer >= 100.0f)
    state.dust_timer -= 100.0f;

  state.timer_dust_rotation
    = fmodf(now_ms, DUNES_DUST_ROTATION_SPEED) / DUNES_DUST_ROTATION_SPEED;
  state.timer_dust_movement
    = fmodf(now_ms, DUNES_DUST_MOVEMENT_SPEED) / DUNES_DUST_MOVEMENT_SPEED;
  state.timer_bird1
    = fmodf(now_ms, DUNES_BIRD_ANIM_PERIOD1) / DUNES_BIRD_ANIM_PERIOD1;
  state.timer_bird2
    = fmodf(now_ms, DUNES_BIRD_ANIM_PERIOD2) / DUNES_BIRD_ANIM_PERIOD2;
  state.timer_birds_fly
    = fmodf(now_ms, DUNES_BIRD_FLIGHT_PERIOD) / DUNES_BIRD_FLIGHT_PERIOD;
  state.timer_camera_wobble
    = fmodf(now_ms, DUNES_CAMERA_WOBBLE_PERIOD) / DUNES_CAMERA_WOBBLE_PERIOD;

  float prev_random = state.timer_random_camera;
  state.timer_random_camera
    = fmodf(now_ms, DUNES_RANDOM_CAMERA_PERIOD) / DUNES_RANDOM_CAMERA_PERIOD;

  if (state.camera_mode == DUNES_CAMERA_RANDOM
      && state.timer_random_camera < prev_random) {
    dunes_randomize_camera();
  }
}

/* -------------------------------------------------------------------------- *
 * Draw calls
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder rp, dunes_model_t* model)
{
  if (!model->is_ready)
    return;
  wgpuRenderPassEncoderSetVertexBuffer(rp, 0, model->vertex_buffer.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rp, model->index_buffer.buffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rp, model->index_count, 1, 0, 0, 0);
}

/* Draw one bird using pre-computed UBO slot and vertex offsets.
 * Uses 3 vertex buffer slots to correctly address pos1, pos2, and UV
 * without UV offset contamination (Bug 2 fix). */
static void draw_bird_slot(WGPURenderPassEncoder rp, uint32_t slot)
{
  if (!state.bird_model.is_ready)
    return;
  uint32_t offset  = slot * DUNES_UBO_STRIDE;
  uint64_t uv_base = 60; /* UV starts at byte 60 within each 68-byte vertex */
  wgpuRenderPassEncoderSetPipeline(rp, state.animated_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.animated_bg, 1, &offset);
  /* Slot 0: pos1 at frame1 offset */
  wgpuRenderPassEncoderSetVertexBuffer(
    rp, 0, state.bird_model.vertex_buffer.buffer, state.bird_vtx_pos1[slot],
    WGPU_WHOLE_SIZE);
  /* Slot 1: pos2 at frame2 offset */
  wgpuRenderPassEncoderSetVertexBuffer(
    rp, 1, state.bird_model.vertex_buffer.buffer, state.bird_vtx_pos2[slot],
    WGPU_WHOLE_SIZE);
  /* Slot 2: UV at fixed offset 60 bytes from each vertex start */
  wgpuRenderPassEncoderSetVertexBuffer(
    rp, 2, state.bird_model.vertex_buffer.buffer, uv_base, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rp, state.bird_model.index_buffer.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rp, state.bird_model.index_count, 1, 0, 0,
                                   0);
}

/* -------------------------------------------------------------------------- *
 * Scene rendering
 * -------------------------------------------------------------------------- */

static void render_depth_pass(wgpu_context_t* wgpu_context,
                              WGPUCommandEncoder cmd_enc)
{
  /* Depth-only pass: render dunes terrain to offscreen depth for soft
   * particles */
  WGPURenderPassEncoder rp
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.depth_pass_desc);

  update_depth_uniforms(wgpu_context);
  wgpuRenderPassEncoderSetPipeline(rp, state.depth_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.depth_bg, 0, NULL);
  draw_model(rp, &state.dunes_model);

  wgpuRenderPassEncoderEnd(rp);
  wgpuRenderPassEncoderRelease(rp);
}

static void render_scene(wgpu_context_t* wgpu_context, WGPURenderPassEncoder rp)
{
  /* All UBO data was pre-computed in precompute_ubos() before the render pass.
   * Dynamic offsets select the correct per-draw slot from each buffer. */
  UNUSED_VAR(wgpu_context);

  /* ---- Terrain: 9 tiles with dynamic offsets ---- */
  wgpuRenderPassEncoderSetPipeline(rp, state.terrain_pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(
    rp, 0, state.dunes_model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(rp, state.dunes_model.index_buffer.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);
  for (uint32_t i = 0; i < DUNES_TERRAIN_SLOTS; i++) {
    uint32_t offset = i * DUNES_UBO_STRIDE;
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.terrain_bg, 1, &offset);
    wgpuRenderPassEncoderDrawIndexed(rp, state.dunes_model.index_count, 1, 0, 0,
                                     0);
  }

  /* ---- Birds: 2 birds with dynamic offsets ---- */
  draw_bird_slot(rp, 0); /* bird1 */
  draw_bird_slot(rp, 1); /* bird2 */

  /* ---- Palm trees: colored slot 0 ---- */
  if (state.palms_model.is_ready) {
    uint32_t offset = 0 * DUNES_UBO_STRIDE; /* slot 0 = palms */
    wgpuRenderPassEncoderSetPipeline(rp, state.palms_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.palms_bg, 1, &offset);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 0, state.palms_model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      rp, state.palms_model.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
      WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, state.palms_model.index_count, 1, 0, 0,
                                     0);
  }

  /* ---- Sky sphere ---- */
  update_diffuse_uniforms(state.wgpu_context, 0, 0, -1200, 150, 150, 150);
  wgpuRenderPassEncoderSetPipeline(rp, state.sky_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.sky_bg, 0, NULL);
  draw_model(rp, &state.sky_model);

  /* ---- Dust particles: 147 slots ---- */
  if (state.smoke_model.is_ready) {
    wgpuRenderPassEncoderSetPipeline(rp, state.particle_pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 0, state.smoke_model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      rp, state.smoke_model.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
      WGPU_WHOLE_SIZE);
    for (uint32_t i = 0; i < DUNES_PARTICLE_COUNT; i++) {
      uint32_t offset = i * DUNES_UBO_STRIDE;
      wgpuRenderPassEncoderSetBindGroup(rp, 0, state.particle_bg, 1, &offset);
      wgpuRenderPassEncoderDrawIndexed(rp, state.smoke_model.index_count, 1, 0,
                                       0, 0);
    }
  }

  /* ---- Sun flare: colored slot 1 ---- */
  if (state.sun_model.is_ready) {
    uint32_t offset = 1 * DUNES_UBO_STRIDE; /* slot 1 = sun */
    wgpuRenderPassEncoderSetPipeline(rp, state.colored_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.colored_bg, 1, &offset);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 0, state.sun_model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, state.sun_model.index_buffer.buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, state.sun_model.index_count, 1, 0, 0,
                                     0);
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static const char* dunes_preset_names[DUNES_PRESET_COUNT]
  = {"Night", "Day", "Sunset", "Sunrise"};

static const char* dunes_camera_mode_names[2] = {"Rotating", "Random"};

static void render_gui(void)
{
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_Once, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){230.0f, 0.0f}, ImGuiCond_Always);
  igBegin("Dunes", NULL,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

  igText("Time of Day");
  if (igButton("Change Time of Day", (ImVec2){0, 0})) {
    state.current_preset  = (state.current_preset + 1) % DUNES_PRESET_COUNT;
    state.resources_dirty = true;
  }
  igText("Current: %s", dunes_preset_names[state.current_preset]);

  igSeparator();
  igText("Camera Mode");
  for (int i = 0; i < 2; ++i) {
    if (igRadioButton_Bool(dunes_camera_mode_names[i],
                           state.camera_mode == (dunes_camera_mode_t)i)) {
      state.camera_mode = (dunes_camera_mode_t)i;
      if (state.camera_mode == DUNES_CAMERA_RANDOM) {
        dunes_randomize_camera();
      }
    }
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  state.wgpu_context   = wgpu_context;
  state.current_preset = 1;                /* Start with Day */
  state.camera_mode = DUNES_CAMERA_RANDOM; /* Random terrain view on startup */

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 24,
    .num_channels = 4,
    .num_lanes    = 4,
#ifndef __WAJIC__
    .logger.func = slog_func,
#endif
  });

  stm_setup();
  srand(42);

  imgui_overlay_init(wgpu_context);

  init_depth_texture(wgpu_context);
  init_offscreen_depth_texture(wgpu_context);
  init_samplers(wgpu_context);
  init_uniform_buffers(wgpu_context);

  /* Allocate GPU model buffers (data filled by fetch callbacks) */
  alloc_model_buffers(wgpu_context, &state.dunes_model, 200 * 1024, 520 * 1024,
                      "Dunes - Terrain idx", "Dunes - Terrain vtx");
  alloc_model_buffers(wgpu_context, &state.sky_model, 24 * 1024, 40 * 1024,
                      "Dunes - Sky idx", "Dunes - Sky vtx");
  alloc_model_buffers(wgpu_context, &state.smoke_model, 512, 512,
                      "Dunes - Smoke idx", "Dunes - Smoke vtx");
  alloc_model_buffers(wgpu_context, &state.sun_model, 512, 2 * 1024,
                      "Dunes - Sun idx", "Dunes - Sun vtx");
  alloc_model_buffers(wgpu_context, &state.bird_model, 256, 2 * 1024,
                      "Dunes - Bird idx", "Dunes - Bird vtx");
  alloc_model_buffers(wgpu_context, &state.palms_model, 10 * 1024, 32 * 1024,
                      "Dunes - Palms idx", "Dunes - Palms vtx");

  /* Setup render pass descriptors */
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

  /* Depth-only pass (no color attachment) */
  state.offscreen_depth_attach = (WGPURenderPassDepthStencilAttachment){
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  };
  state.depth_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &state.offscreen_depth_attach,
  };

  init_bind_group_layouts(wgpu_context);

  /* Load assets */
  const char* base = "assets/models/Dunes";
  char path[128];

#define LOAD_TEX(slot, file, sz)                                               \
  snprintf(path, sizeof(path), "%s/textures/%s", base, file);                  \
  load_texture(wgpu_context, slot, path, sz);

  LOAD_TEX(DUNES_TEX_SKY_NIGHT, "night_1.jpg", 640 * 1024)
  LOAD_TEX(DUNES_TEX_SKY_DAY, "day_1.jpg", 512 * 1024)
  LOAD_TEX(DUNES_TEX_SKY_SUNSET, "sunset_1.jpg", 512 * 1024)
  LOAD_TEX(DUNES_TEX_SKY_SUNRISE, "sunrise_1.jpg", 640 * 1024)
  LOAD_TEX(DUNES_TEX_DIFFUSE, "dunes-diffuse.jpg", 3 * 1024 * 1024)
  LOAD_TEX(DUNES_TEX_DUST, "upwind.png", 16 * 1024)
  LOAD_TEX(DUNES_TEX_DETAIL, "detail.png", 64 * 1024)
  LOAD_TEX(DUNES_TEX_SMOKE, "smoke.png", 16 * 1024)
  LOAD_TEX(DUNES_TEX_SUN_FLARE, "sun_flare.png", 16 * 1024)
  LOAD_TEX(DUNES_TEX_BIRD, "bird2.png", 16 * 1024)
  LOAD_TEX(DUNES_TEX_PALM_ALPHA, "palm-alpha.png", 8 * 1024)
  LOAD_TEX(DUNES_TEX_PALM_DIFF, "palm-diffuse.png", 8 * 1024)
#undef LOAD_TEX

  snprintf(path, sizeof(path), "%s/models/dunes", base);
  load_model(&state.dunes_model, path, 200 * 1024, 520 * 1024);
  snprintf(path, sizeof(path), "%s/models/sky", base);
  load_model(&state.sky_model, path, 24 * 1024, 40 * 1024);
  snprintf(path, sizeof(path), "%s/models/smoke100", base);
  load_model(&state.smoke_model, path, 512, 512);
  snprintf(path, sizeof(path), "%s/models/sun_flare", base);
  load_model(&state.sun_model, path, 512, 2 * 1024);
  snprintf(path, sizeof(path), "%s/models/bird-anim-uv", base);
  load_model(&state.bird_model, path, 256, 2 * 1024);
  snprintf(path, sizeof(path), "%s/models/palms", base);
  load_model(&state.palms_model, path, 10 * 1024, 32 * 1024);

  /* Create initial bind groups with placeholder textures */
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  dunes_randomize_camera();

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  sfetch_dowork();

  /* Upload any pending fetched data */
  upload_pending_textures(wgpu_context);
  upload_pending_models(wgpu_context);

  /* Rebuild bind groups when textures/preset changed */
  if (state.resources_dirty) {
    init_bind_groups(wgpu_context);
  }

  /* Animate */
  animate();
  position_camera();
  set_projection(wgpu_context);

  /* Pre-compute all UBO data for this frame BEFORE any render passes.
   * This ensures all writes to dynamic UBO buffers are complete before the
   * command buffer executes (avoiding the "last write wins" issue). */
  if (state.models_loaded >= 6) {
    precompute_ubos(wgpu_context);
  }

  /* --- Depth pre-pass --- */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Update depth pass attachments */
  state.offscreen_depth_attach.view = state.offscreen_depth.view;
  if (state.dunes_model.is_ready) {
    render_depth_pass(wgpu_context, cmd_enc);
  }

  /* --- Main color pass --- */
  state.color_attach.view = wgpu_context->swapchain_view;
  state.depth_attach.view = state.depth_texture.view;

  WGPURenderPassEncoder rp
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_desc);

  if (state.dunes_model.is_ready && state.models_loaded >= 6) {
    render_scene(wgpu_context, rp);
  }

  /* GUI: build draw list while render pass is open */
  {
    uint64_t now = stm_now();
    if (state.last_frame_time == 0) {
      state.last_frame_time = now;
    }
    float dt = (float)stm_sec(stm_diff(now, state.last_frame_time));
    state.last_frame_time = now;
    imgui_overlay_new_frame(wgpu_context, dt);
    render_gui();
  }

  wgpuRenderPassEncoderEnd(rp);

  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);

  wgpuRenderPassEncoderRelease(rp);
  wgpuCommandBufferRelease(cmd_buf);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui renders with its own internal command encoder, after submit */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Input event
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    init_offscreen_depth_texture(wgpu_context);
    /* Rebuild bind groups to pick up new offscreen depth view */
    if (state.resources_dirty || true) {
      init_bind_groups(wgpu_context);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  sfetch_shutdown();

  /* Models */
  dunes_model_t* models[]
    = {&state.dunes_model, &state.sky_model,  &state.smoke_model,
       &state.sun_model,   &state.bird_model, &state.palms_model};
  for (int i = 0; i < 6; ++i) {
    wgpu_destroy_buffer(&models[i]->vertex_buffer);
    wgpu_destroy_buffer(&models[i]->index_buffer);
    free(models[i]->pending_idx);
    free(models[i]->pending_vtx);
  }

  /* Textures */
  for (uint32_t i = 0; i < DUNES_TEX__COUNT; ++i) {
    wgpu_destroy_texture(&state.textures[i]);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler_repeat)

  wgpu_destroy_texture(&state.depth_texture);
  wgpu_destroy_texture(&state.offscreen_depth);

  /* UBOs */
  wgpu_destroy_buffer(&state.terrain_ubo);
  wgpu_destroy_buffer(&state.diffuse_ubo);
  wgpu_destroy_buffer(&state.colored_ubo);
  wgpu_destroy_buffer(&state.animated_ubo);
  wgpu_destroy_buffer(&state.particle_ubo);
  wgpu_destroy_buffer(&state.depth_ubo);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.terrain_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.sky_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.colored_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.animated_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.depth_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.palms_bg)

  /* BGLs */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.terrain_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.diffuse_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.colored_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.animated_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.particle_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.depth_bgl)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.terrain_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.sky_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.colored_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.animated_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.particle_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.depth_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.palms_pipeline)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.terrain_pl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.diffuse_pl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.colored_pl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.animated_pl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.particle_pl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.depth_pl)
}

/* -------------------------------------------------------------------------- *
 * main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Dunes Rendering",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* ========================================================================== *
 * WGSL Shaders
 * ========================================================================== */

/* -------------------------------------------------------------------------- *
 * Dunes terrain shader
 *
 * Multi-textured terrain with wind animation, fog, slope-based coloring.
 * Vertex layout: @location(0) pos:vec3, @location(1) uv:vec2,
 *                @location(2) normal:vec3
 * -------------------------------------------------------------------------- */
// clang-format off
static const char* dunes_terrain_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix:      mat4x4f,
    color:                 vec4f,
    fog_color:             vec4f,
    shadow_color:          vec4f,
    waves_color:           vec4f,
    time:                  f32,
    dust_opacity:          f32,
    fog_start_distance:    f32,
    fog_distance:          f32,
    detail_start_distance: f32,
    detail_distance:       f32,
    lightmap_select:       f32,   /* 0.0=use tex.r (sunset), 1.0=use tex.g (day/night) */
    _pad:                  f32,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
    @location(2) normal:   vec3f,
  }

  struct VertexOutput {
    @builtin(position) position:        vec4f,
    @location(0)       uv:              vec2f,
    @location(1)       upwind_uv:       vec2f,
    @location(2)       leeward_uv:      vec2f,
    @location(3)       wind_spots_uv:   vec2f,
    @location(4)       detail_uv:       vec2f,
    @location(5)       normal:          vec3f,
    @location(6)       slope_coeff:     f32,
    @location(7)       slope_coeff2:    f32,
    @location(8)       fog_amount:      f32,
    @location(9)       detail_fade:     f32,
  }

  @group(0) @binding(0) var<uniform> u:        Uniforms;
  @group(0) @binding(1) var          tex_samp: sampler;
  @group(0) @binding(2) var          diffuse:  texture_2d<f32>;
  @group(0) @binding(3) var          dust:     texture_2d<f32>;
  @group(0) @binding(4) var          detail:   texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos4      = vec4f(in.position, 1.0);
    out.position  = u.view_proj_matrix * pos4;
    out.uv        = in.uv;
    out.normal    = in.normal;
    out.slope_coeff  = clamp(4.0 * dot(in.normal,
                        normalize(vec3f(1.0, 0.0, 0.13))), 0.0, 1.0);
    out.slope_coeff2 = clamp(14.0 * dot(in.normal,
                        normalize(vec3f(-1.0, 0.0, -0.2))), 0.0, 1.0);
    out.upwind_uv    = in.uv * vec2f(100.0, 10.0);
    out.upwind_uv.y += u.time;
    out.leeward_uv    = in.uv * vec2f(20.0, 30.0);
    out.leeward_uv.y += u.time;
    out.wind_spots_uv    = in.uv * vec2f(1.5, 1.5);
    out.wind_spots_uv.x += u.time * 0.1;
    out.detail_uv = in.uv * vec2f(100.0, 100.0);
    let dist      = length(out.position);
    out.fog_amount    = clamp((dist - u.fog_start_distance) / u.fog_distance,
                              0.0, 1.0);
    out.detail_fade   = 1.0 - clamp(
        (dist - u.detail_start_distance) / u.detail_distance, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    let windward     = textureSample(dust, tex_samp, in.upwind_uv);
    let leeward2     = textureSample(dust, tex_samp, in.leeward_uv);
    let detail_color = textureSample(detail, tex_samp, in.detail_uv);
    var detail1      = (detail_color.g - 0.5) * in.detail_fade;
    var detail2      = (detail_color.r - 0.5) * in.detail_fade;
    let tex_data     = textureSample(diffuse, tex_samp, in.uv);

    var col    = tex_data.r * u.color;
    var waves  = windward * u.dust_opacity * in.slope_coeff;
    waves     += leeward2 * u.dust_opacity * in.slope_coeff2;
    let wind_spots = textureSample(dust, tex_samp, in.wind_spots_uv);
    waves         *= 1.0 - clamp(wind_spots.r * 5.0, 0.0, 1.0);
    col           += waves * u.waves_color;
    col.x += mix(detail2, detail1, in.slope_coeff2);
    col.y += mix(detail2, detail1, in.slope_coeff2);
    col.z += mix(detail2, detail1, in.slope_coeff2);
    /* Shadow color: lightmap channel selected by uniform (r=sunset, g=day/night) */
    let lightmap = mix(tex_data.r, tex_data.g, u.lightmap_select);
    col *= mix(u.shadow_color, vec4f(1.0, 1.0, 1.0, 1.0), lightmap);
    col  = mix(col, u.fog_color, in.fog_amount);
    return vec4f(col.rgb, 1.0); /* force alpha=1 to avoid compositor darkening */
  }
);

/* -------------------------------------------------------------------------- *
 * Diffuse shader (sky sphere)
 * -------------------------------------------------------------------------- */
static const char* dunes_diffuse_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> u:        Uniforms;
  @group(0) @binding(1) var          tex_samp: sampler;
  @group(0) @binding(2) var          tex:      texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.view_proj_matrix * vec4f(in.position, 1.0);
    out.uv       = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex, tex_samp, in.uv);
  }
);

/* -------------------------------------------------------------------------- *
 * Diffuse colored shader (sun flare - additive blend)
 * -------------------------------------------------------------------------- */
static const char* dunes_diffuse_colored_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
    color:            vec4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> u:        Uniforms;
  @group(0) @binding(1) var          tex_samp: sampler;
  @group(0) @binding(2) var          tex:      texture_2d<f32>;
  @group(0) @binding(3) var          tex2:     texture_2d<f32>; /* unused slot */

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.view_proj_matrix * vec4f(in.position, 1.0);
    out.uv       = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex, tex_samp, in.uv) * u.color;
  }
);

/* -------------------------------------------------------------------------- *
 * Diffuse alpha-tested shader (palm trees)
 * -------------------------------------------------------------------------- */
static const char* dunes_diffuse_alpha_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
    color:            vec4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> u:          Uniforms;
  @group(0) @binding(1) var          tex_samp:   sampler;
  @group(0) @binding(2) var          diffuse:    texture_2d<f32>;
  @group(0) @binding(3) var          alpha_tex:  texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.view_proj_matrix * vec4f(in.position, 1.0);
    out.uv       = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    let alpha = textureSample(alpha_tex, tex_samp, in.uv);
    if (alpha.r < 0.5) { discard; }
    return textureSample(diffuse, tex_samp, in.uv) * u.color;
  }
);

/* -------------------------------------------------------------------------- *
 * Animated colored shader (bird with frame morphing)
 *
 * Vertex layout stride = 68 bytes:
 *   @location(0) pos1 (current frame) : vec3 at offset 0 (+ frame1*12)
 *   @location(1) pos2 (next frame)    : vec3 at offset 12 (= pos1 + 12)
 *   @location(2) uv                   : vec2 at offset 60
 *
 * The buffer is bound with base_offset = frame1 * 3 * 4 bytes.
 * So pos1 is at the frame1 position data, pos2 at frame1+1 data.
 * -------------------------------------------------------------------------- */
static const char* dunes_animated_colored_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
    color:            vec4f,
    morph:            f32,
    _pad:             vec3f,
  }

  struct VertexInput {
    @location(0) pos1: vec3f,
    @location(1) pos2: vec3f,
    @location(2) uv:   vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> u:        Uniforms;
  @group(0) @binding(1) var          tex_samp: sampler;
  @group(0) @binding(2) var          tex:      texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos      = mix(in.pos1, in.pos2, u.morph);
    out.position = u.view_proj_matrix * vec4f(pos, 1.0);
    out.uv       = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    let base = textureSample(tex, tex_samp, in.uv);
    if (base.a < 0.95) { discard; }
    return base * u.color;
  }
);

/* -------------------------------------------------------------------------- *
 * Soft diffuse particle shader
 *
 * Samples offscreen depth to create soft edges where particles intersect
 * scene geometry.
 * -------------------------------------------------------------------------- */
static const char* dunes_soft_particle_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
    color:            vec4f,
    camera_near:      f32,
    camera_far:       f32,
    inv_viewport_x:   f32,
    inv_viewport_y:   f32,
    transition_size:  f32,
    _pad:             vec3f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> u:         Uniforms;
  @group(0) @binding(1) var          tex_samp:  sampler;
  @group(0) @binding(2) var          color_tex: texture_2d<f32>;
  @group(0) @binding(3) var          depth_tex: texture_depth_2d;

  fn linearize_depth(z: f32) -> f32 {
    return (2.0 * u.camera_near)
           / (u.camera_far + u.camera_near
              - z * (u.camera_far - u.camera_near));
  }

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.view_proj_matrix * vec4f(in.position, 1.0);
    out.uv       = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput)
    -> @location(0) vec4f
  {
    let diffuse    = textureSample(color_tex, tex_samp, in.uv) * u.color;
    let coords_u32 = vec2u(u32(in.position.x), u32(in.position.y));
    let scene_z    = textureLoad(depth_tex, coords_u32, 0);
    let geom_depth = linearize_depth(scene_z);
    let part_depth = linearize_depth(in.position.z);
    let a          = clamp(geom_depth - part_depth, 0.0, 1.0);
    let b_smooth   = smoothstep(0.0, u.transition_size, a);
    var result     = diffuse * b_smooth;
    result        *= pow(1.0 - in.position.z, 0.3);
    return result;
  }
);

/* -------------------------------------------------------------------------- *
 * Depth-only shader
 * -------------------------------------------------------------------------- */
static const char* dunes_depth_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
    @location(2) normal:   vec3f,
  }

  @group(0) @binding(0) var<uniform> u: Uniforms;

  @vertex
  fn vertexMain(in: VertexInput) -> @builtin(position) vec4f {
    return u.view_proj_matrix * vec4f(in.position, 1.0);
  }
);
// clang-format on
