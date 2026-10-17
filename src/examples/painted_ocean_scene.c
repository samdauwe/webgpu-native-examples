#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"

#include <cglm/cglm.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

/* In WAjic, WGPU handles are uint32_t; redefine NULL to 0 so that handle
 * comparisons like `ASSERT(handle != NULL)` compile without warnings. */
#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#ifndef CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Painted Ocean Scene
 *
 * A wide-angle, panoramic view over a coral reef beneath clear turquoise
 * water and a sky of ray-marched clouds. The whole picture is painted by
 * fragment shaders on a single screen-covering triangle: no meshes, no depth
 * buffer.
 *
 *  - parallax-style shading of the seabed, refraction and caustics
 *  - GPU flocking (Craig Reynolds' boids) for the fish and the birds
 *  - branching coral marched as signed distance fields
 *  - ray-marched clouds with their own shadow map, a sky and horizon clouds
 *  - feathered birds and ACES tone mapping
 *
 * Passes per frame: boids compute, seabed cache fills (when required), cloud
 * shadow map, clouds + cloud reflections, sea, final composition, and a blit
 * that scales the scene to the window.
 *
 * Ref:
 * https://github.com/shovon/webgpu-painted-ocean-scene
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

/* The scene shader is far longer than ISO C99 allows for one string literal,
 * so it is stored as a list of parts that is joined once at start-up. */
#define MAIN_WGSL_MAX_PARTS 64u
#define SIM_WGSL_MAX_PARTS 16u
static const char* painted_ocean_wgsl_part[MAIN_WGSL_MAX_PARTS];
static const char* painted_ocean_sim_wgsl_part[SIM_WGSL_MAX_PARTS];
static const char* painted_ocean_blit_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Flocks. Fish and birds share one simulation shader compiled twice. */
#define FLOCK_WORLD 96u /* metres per side of the wrapped simulated square */
#define FISH_COUNT 8192u
#define FISH_GRID 192u
#define FISH_PER_CELL 16u
#define BIRD_COUNT 144u
#define BIRD_GRID 12u
#define BIRD_PER_CELL 144u
#define BIRD_DEPTH 8.75f  /* half height of the layer the birds fly in */
#define BIRD_HEIGHT 7.25f /* world height of the middle of that layer */
#define LURE_MAX 1024u
#define PREDATOR_MAX 64u
#define BOID_FLOATS 12u
#define BOID_BYTES (BOID_FLOATS * sizeof(float))
#define SIM_WORKGROUP_SIZE 64u
#define SIM_SETTINGS_FLOATS 32u

/* Seabed cache (fine, around the camera) and the wide rim-distance map. */
#define BED_SIZE 2048u
#define BED_TEXEL 0.06f
#define FAR_SIZE 2048u
#define FAR_TEXEL 0.5f
#define BED_REFILL_DISTANCE 10.0f
#define FAR_REFILL_DISTANCE 100.0f
#define BED_KEY_COUNT 14u

/* Scene. */
#define WATER_Y (-2.5f)
#define ORBIT_Y (-0.5f) /* usual height of the point the camera orbits */
#define HOME_X (-25.63f)
#define HOME_Z (-36.12f)
#define HOME_YAW (-26.0397f)
#define HOME_PITCH 0.1647f
#define HOME_DIST 4.2f
#define MIN_DIST 4.2f
#define MIN_FOLLOW_DIST 1.0f
#define MAX_DIST 150.0f
#define MIN_WATER_CLEARANCE 1.1f
#define MAX_PITCH 1.2f
#define ORBIT_SPEED 0.006f /* radians per pixel of mouse drag */
#define ZOOM_SPEED 0.1f
#define PLANET_EARTH_EXPONENT 6.8f
#define PLANET_EARTH_RADIUS 6371000.0f
#define MAX_FRAME_DELTA 0.1f

/* Waves: the same constants as in the shader. */
#define WAVE_TRAIN_COUNT 28u
#define WIND_ANGLE 0.35
#define SWELL_ANGLE 2.3

/* Uniform buffer: 23 vec4 of settings, then 2 vec4 per wave train. */
#define UNIFORM_SETTINGS_VEC4 23u
#define UNIFORM_FLOAT_COUNT (UNIFORM_SETTINGS_VEC4 * 4u + WAVE_TRAIN_COUNT * 8u)

#define TARGET_FORMAT_SCENE WGPUTextureFormat_RGBA8Unorm
#define TARGET_FORMAT_COLOR WGPUTextureFormat_RGBA16Float
#define TARGET_FORMAT_SCALAR WGPUTextureFormat_R16Float
#define MAX_TARGET_DIMENSION 8192u

/* -------------------------------------------------------------------------- *
 * GUI parameters
 *
 * One X-macro list per GUI section. Each list expands to the fields of the
 * global state, and to the table the GUI and the reset button walk over.
 * X(field, label, min, max, default, printf format)
 * -------------------------------------------------------------------------- */

#define LIGHT_SLIDERS(X)                                                       \
  X(fov, "Field of view", 15.0f, 110.0f, 35.0f, "%.0f deg")                    \
  X(sun_height, "Sun height", 2.0f, 80.0f, 34.0f, "%.0f deg")                  \
  X(sun_direction, "Sun direction", 0.0f, 360.0f, 205.0f, "%.0f deg")          \
  X(animation_speed, "Animation speed", 0.0f, 2.0f, 0.5f, "%.2fx")

#define SKY_SLIDERS(X)                                                         \
  X(horizon_clouds, "Horizon clouds", 0.0f, 1.0f, 0.5f, "%.2f")                \
  X(coverage, "Coverage", 0.0f, 1.0f, 0.5f, "%.2f")                            \
  X(density, "Density", 1.0f, 16.0f, 5.0f, "%.1f")                             \
  X(fluff, "Fluff", 0.0f, 1.5f, 0.72f, "%.2f")                                 \
  X(planet_exponent, "Planet radius (log10 m)", 2.0f, 6.8f, 6.8f, "%.2f")      \
  X(cloud_size, "Cloud size", 6.0f, 30.0f, 10.0f, "%.0f m")                    \
  X(cloud_spacing, "Cloud grid spacing", 20.0f, 150.0f, 20.0f, "%.0f m")       \
  X(cloud_skew, "Cloud grid skew", 60.0f, 90.0f, 60.0f, "%.0f deg")            \
  X(grid_turn, "Cloud grid rotation", 0.0f, 90.0f, 20.0f, "%.0f deg")          \
  X(cloud_detail, "Cloud detail range", 20.0f, 600.0f, 120.0f, "%.0f m")       \
  X(cloud_scatter, "Cloud scatter", 0.0f, 1.0f, 0.8f, "%.2f")                  \
  X(cloud_drift, "Cloud drift", 0.0f, 1.0f, 0.3f, "%.2f")                      \
  X(drift_direction, "Drift direction", 0.0f, 360.0f, 20.0f, "%.0f deg")       \
  X(cloud_wander, "Cloud wander", 0.0f, 1.0f, 0.4f, "%.2f")                    \
  X(cloud_turnover, "Cloud turnover", 0.0f, 1.0f, 0.5f, "%.2f")                \
  X(cloud_churn, "Cloud churn", 0.0f, 1.0f, 0.25f, "%.2f")

#define SEA_SLIDERS(X)                                                         \
  X(wave_height, "Wave height", 0.0f, 1.0f, 0.22f, "%.2f")                     \
  X(wave_length, "Longest wave", 0.5f, 20.0f, 4.0f, "%.1f m")                  \
  X(wave_random, "Wave randomness", 0.0f, 1.0f, 0.7f, "%.2f")

#define SEABED_SLIDERS(X)                                                      \
  X(shallow_depth, "Shallowest depth", 0.4f, 20.0f, 0.4f, "%.1f m")            \
  X(ocean_depth, "Open-ocean depth", 1.0f, 80.0f, 29.0f, "%.0f m")             \
  X(shoal_size, "Shoal size", 10.0f, 200.0f, 60.0f, "%.0f m")                  \
  X(shoal_cover, "Shoal cover", 0.0f, 1.0f, 0.64f, "%.2f")                     \
  X(lagoon_depth, "Lagoon depth inside shoals", 0.0f, 10.0f, 1.1f, "%.1f m")   \
  X(ring_gap, "Inner shoal spacing", 4.0f, 80.0f, 19.0f, "%.0f m")             \
  X(dips, "Broad dips in the seabed", 0.0f, 6.0f, 1.2f, "%.1f m")              \
  X(dip_size, "Size of the broad dips", 50.0f, 1000.0f, 250.0f, "%.0f m")      \
  X(relief, "Seabed relief", 0.0f, 1.0f, 1.0f, "%.2f")                         \
  X(sand_waves, "Sand waves", 0.0f, 1.0f, 1.0f, "%.2f")

#define CORAL_SLIDERS(X)                                                       \
  X(coral, "Coral", 0.0f, 1.0f, 0.84f, "%.2f")                                 \
  X(coral_top, "Coral starts at depth", 0.0f, 15.0f, 0.8f, "%.1f m")           \
  X(coral_range, "Coral depth range", 0.2f, 15.0f, 15.0f, "%.1f m")            \
  X(coral_dots, "Coral reaches inside the rim", 0.0f, 12.0f, 12.0f, "%.2f m")  \
  X(coral_sway, "Coral sway", 0.0f, 1.0f, 1.0f, "%.2f")                        \
  X(coral_spikes, "Coral branches", 0.0f, 1.0f, 1.0f, "%.2f")

#define LIFE_SLIDERS(X)                                                        \
  X(fish_depth, "Fish swim depth", 0.2f, 6.0f, 6.0f, "%.1f m")                 \
  X(fish_spread, "Fish spread up from that depth", 0.0f, 1.0f, 1.0f, "%.2f")

#define QUALITY_SLIDERS(X)                                                     \
  X(march_steps, "March steps", 16.0f, 128.0f, 28.0f, "%.0f")                  \
  X(scene_res, "Scene resolution", 0.25f, 2.0f, 1.0f, "%.2f")                  \
  X(sea_res, "Sea resolution", 0.25f, 1.0f, 0.7f, "%.2f")                      \
  X(cloud_res, "Cloud resolution", 0.1f, 1.0f, 0.5f, "%.2f")

#define FISH_SLIDERS(X)                                                        \
  X(fish_homes, "Homes per side", 1.0f, 12.0f, 4.0f, "%.0f")                   \
  X(fish_walls, "Wall avoidance", 0.0f, 10.0f, 0.0f, "%.1f")                   \
  X(fish_lures, "Lures", 0.0f, 1024.0f, 124.0f, "%.0f")                        \
  X(fish_lure_pull, "Lure pull", 0.0f, 8.0f, 8.0f, "%.1f")                     \
  X(fish_lure_range, "Lure range", 1.0f, 12.0f, 12.0f, "%.1f m")               \
  X(fish_lure_life, "Lure lifetime", 2.0f, 120.0f, 50.0f, "%.0f s")            \
  X(fish_predators, "Predators", 0.0f, 64.0f, 52.0f, "%.0f")                   \
  X(fish_fear, "Fear strength", 0.0f, 15.0f, 15.0f, "%.1f")                    \
  X(fish_fear_range, "Fear range", 1.0f, 10.0f, 5.5f, "%.1f m")                \
  X(fish_predator_life, "Predator lifetime", 5.0f, 120.0f, 120.0f, "%.0f s")   \
  X(fish_predator_room, "Predators keep apart", 0.0f, 30.0f, 12.0f, "%.0f m")  \
  X(fish_predator_speed, "Predator speed", 0.2f, 2.0f, 1.55f, "%.2f m/s")      \
  X(fish_caution, "Caution near walls", 0.0f, 1.0f, 0.4f, "%.2f")              \
  X(fish_space, "Personal space", 0.15f, 1.0f, 0.25f, "%.2f m")                \
  X(fish_separate, "Separation strength", 0.0f, 15.0f, 4.0f, "%.1f")           \
  X(fish_align, "Alignment strength", 0.0f, 4.0f, 3.25f, "%.2f")               \
  X(fish_align_range, "Alignment range", 0.3f, 3.0f, 2.4f, "%.1f m")           \
  X(fish_cohere, "Cohesion strength", 0.0f, 2.5f, 2.0f, "%.2f")                \
  X(fish_cohere_range, "Cohesion range", 0.5f, 4.0f, 3.4f, "%.1f m")           \
  X(fish_turn, "Turning limit", 0.3f, 5.0f, 0.3f, "%.1f")                      \
  X(fish_cruise, "Cruising speed", 0.2f, 1.2f, 1.2f, "%.2f m/s")               \
  X(fish_vary, "Speed variation", 0.0f, 0.6f, 0.6f, "%.2f")                    \
  X(fish_blind, "Blind spot behind", 0.0f, 120.0f, 0.0f, "%.0f deg")

#define BIRD_SLIDERS(X)                                                        \
  X(bird_speed, "Bird speed", 1.0f, 20.0f, 9.0f, "%.1f m/s")                   \
  X(bird_swoop, "Swooping", 0.0f, 1.0f, 0.6f, "%.2f")                          \
  X(bird_restless, "Restlessness", 0.0f, 10.0f, 4.9f, "%.1f")                  \
  X(bird_space, "Personal space", 0.5f, 15.0f, 11.5f, "%.1f m")                \
  X(bird_separate, "Separation strength", 0.0f, 10.0f, 8.0f, "%.1f")           \
  X(bird_align, "Alignment strength", 0.0f, 3.0f, 2.31f, "%.2f")               \
  X(bird_align_range, "Alignment range", 2.0f, 40.0f, 40.0f, "%.0f m")         \
  X(bird_cohere, "Cohesion strength", 0.0f, 1.0f, 0.53f, "%.2f")               \
  X(bird_cohere_range, "Cohesion range", 4.0f, 40.0f, 34.0f, "%.0f m")         \
  X(bird_lures, "Lures", 0.0f, 200.0f, 10.0f, "%.0f")                          \
  X(bird_lure_pull, "Lure pull", 0.0f, 6.0f, 6.0f, "%.1f")                     \
  X(bird_lure_range, "Lure range", 5.0f, 40.0f, 40.0f, "%.0f m")               \
  X(bird_lure_life, "Lure lifetime", 2.0f, 40.0f, 40.0f, "%.0f s")             \
  X(bird_blind, "Blind spot behind", 0.0f, 120.0f, 0.0f, "%.0f deg")

/* X(field, label, default) */
#define SKY_TOGGLES(X)                                                         \
  X(is_clouds_on, "Clouds", true)                                              \
  X(is_cloud_grid, "Clouds on a grid", false)                                  \
  X(is_mirror, "Reflect clouds in water", true)

#define SEA_TOGGLES(X)                                                         \
  X(is_waves_on, "Waves", true)                                                \
  X(is_wave_march, "Waves have real height", true)                             \
  X(is_caustics, "Caustics on the seabed", true)                               \
  X(is_whitecaps, "Whitecaps", true)

#define SEABED_TOGGLES(X)                                                      \
  X(is_bed_on, "Seabed shape", true)                                           \
  X(is_sand_on, "Sand waves enabled", true)                                    \
  X(is_bed_cache, "Seabed cache", true)

#define CORAL_TOGGLES(X)                                                       \
  X(is_coral_on, "Coral enabled", true)                                        \
  X(is_coral_focus, "Coral detail where you look", true)

#define LIFE_TOGGLES(X)                                                        \
  X(is_fish_on, "Fish", true)                                                  \
  X(is_birds_on, "Birds", true)                                                \
  X(is_follow_bird, "Follow a bird", false)                                    \
  X(is_bird_shadows, "Bird shadows", true)                                     \
  X(is_feather_sheen, "Feather sheen", true)

#define QUALITY_TOGGLES(X)                                                     \
  X(is_hold_fps, "Hold 60 fps by lowering resolution", true)

#define DEBUG_TOGGLES(X)                                                       \
  X(is_show_walls, "Show home walls", false)                                   \
  X(is_show_lures, "Show lures", false)                                        \
  X(is_show_predators, "Show predators", false)

#define SLIDER_FIELD(name, label, lo, hi, def, fmt) float name;
#define TOGGLE_FIELD(name, label, def) bool name;
#define SLIDER_ENTRY(name, label, lo, hi, def, fmt)                            \
  {label, &state.name, lo, hi, def, fmt},
#define TOGGLE_ENTRY(name, label, def) {label, &state.name, def},

typedef struct {
  const char* label;
  float* value;
  float lo;
  float hi;
  float def;
  const char* format;
} slider_param_t;

typedef struct {
  const char* label;
  bool* value;
  bool def;
} toggle_param_t;

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUTexture texture;
  WGPUTextureView view;
} target_texture_t;

typedef struct {
  uint32_t count;
  uint32_t grid;
  uint32_t per_cell;
  float depth; /* 0 for a flat flock, the half height of a flying layer */
} flock_config_t;

typedef struct {
  float speed;
  float size_lo;
  float size_hi;
  float lift_lo;
  float lift_hi;
  float clump_radius;
} flock_start_t;

typedef struct {
  flock_config_t config;
  WGPUBuffer members[2];
  WGPUBuffer cells;
  WGPUBuffer lures;
  WGPUBuffer lure_cells;
  WGPUBuffer predators;
  WGPUBuffer settings_buffer;
  WGPUBindGroup bind_groups[2];
  WGPUComputePipeline clear_pipeline;
  WGPUComputePipeline bin_pipeline;
  WGPUComputePipeline step_pipeline;
  WGPUComputePipeline lure_pipeline;
  WGPUComputePipeline predator_pipeline;
  float settings[SIM_SETTINGS_FLOATS];
} flock_t;

typedef struct {
  int32_t settle;
  bool probing;
  bool climbing;
  double last_probe;
  double probe_wait;
} governor_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* GUI controlled parameters */
  LIGHT_SLIDERS(SLIDER_FIELD)
  SKY_SLIDERS(SLIDER_FIELD)
  SEA_SLIDERS(SLIDER_FIELD)
  SEABED_SLIDERS(SLIDER_FIELD)
  CORAL_SLIDERS(SLIDER_FIELD)
  LIFE_SLIDERS(SLIDER_FIELD)
  QUALITY_SLIDERS(SLIDER_FIELD)
  FISH_SLIDERS(SLIDER_FIELD)
  BIRD_SLIDERS(SLIDER_FIELD)
  SKY_TOGGLES(TOGGLE_FIELD)
  SEA_TOGGLES(TOGGLE_FIELD)
  SEABED_TOGGLES(TOGGLE_FIELD)
  CORAL_TOGGLES(TOGGLE_FIELD)
  LIFE_TOGGLES(TOGGLE_FIELD)
  QUALITY_TOGGLES(TOGGLE_FIELD)
  DEBUG_TOGGLES(TOGGLE_FIELD)
  bool is_gui_visible;
  bool is_prev_follow_bird;
  float prev_scene_res;
  char camera_text[160];

  /* Camera: orbit camera that looks at a point just above the sea */
  camera_t camera;
  float pan_x;
  float pan_z;
  float orbit_y;
  uint8_t keys_down[KEY_NUM];

  /* Animation clocks */
  double now_ms;
  double last_ms;
  double fps_ms;
  uint32_t frames;
  bool is_stalled;
  float cloud_flow;
  float wave_clock;
  float cloud_drift_x;
  float cloud_drift_z;
  float wander_clock;
  float turnover_clock;

  /* Seabed cache and wide rim map */
  float bed_at_x;
  float bed_at_z;
  float far_at_x;
  float far_at_z;
  float bed_key[BED_KEY_COUNT];
  float far_key[BED_KEY_COUNT];
  bool has_bed_key;
  bool has_far_key;

  /* Dynamic resolution */
  governor_t governor;
  float scene_now;
  uint32_t window_width;
  uint32_t window_height;
  uint32_t scene_width;
  uint32_t scene_height;
  uint32_t low_width;
  uint32_t low_height;
  uint32_t sea_width;
  uint32_t sea_height;
  uint32_t shadow_size;
  float target_cloud_res;
  float target_sea_res;
  char stats[192];

  /* Following a bird: its record is read back from the GPU */
  struct {
    int32_t id;
    bool is_busy;
    bool is_pending;
    bool is_all_birds;
    bool has_at;
    uint32_t bytes;
    float near_x;
    float near_z;
    double time;
    float x;
    float z;
    float vx;
    float vz;
    float y;
    float climb;
    WGPUBuffer buffer;
  } follow;

  /* GPU objects */
  WGPUSampler linear_sampler;
  WGPUBuffer uniform_buffer;
  WGPUBindGroup uniform_bind_group;
  WGPUBindGroupLayout uniform_bgl;
  WGPUBindGroupLayout cloud_bgl;
  WGPUBindGroupLayout final_bgl;
  WGPUBindGroupLayout sea_bgl;
  WGPUBindGroupLayout bed_sand_bgl;
  WGPUBindGroupLayout far_rim_bgl;
  WGPUBindGroupLayout sim_bgl;
  WGPUBindGroupLayout blit_bgl;
  WGPUPipelineLayout sim_pipeline_layout;
  WGPURenderPipeline shadow_pipeline;
  WGPURenderPipeline bed_sand_pipeline;
  WGPURenderPipeline bed_pipeline;
  WGPURenderPipeline far_sand_pipeline;
  WGPURenderPipeline far_rim_pipeline;
  WGPURenderPipeline cloud_pipeline;
  WGPURenderPipeline sea_pipeline;
  WGPURenderPipeline final_pipeline;
  WGPURenderPipeline blit_pipeline;
  target_texture_t bed_sand;
  target_texture_t bed;
  target_texture_t far_sand;
  target_texture_t far_rim;
  target_texture_t shadow;
  target_texture_t cloud;
  target_texture_t cloud_near;
  target_texture_t sea;
  target_texture_t scene;
  WGPUBindGroup bed_sand_bind_group;
  WGPUBindGroup far_rim_bind_group;
  WGPUBindGroup cloud_bind_groups[2];
  WGPUBindGroup final_bind_groups[2];
  WGPUBindGroup sea_bind_groups[2];
  WGPUBindGroup blit_bind_group;
  flock_t fish;
  flock_t birds;
  uint32_t flock_now; /* which copy of the flocks is current this frame */
  float uniform_data[UNIFORM_FLOAT_COUNT];
  bool is_initialized;
} state = {0};

static const flock_config_t fish_config = {
  .count    = FISH_COUNT,
  .grid     = FISH_GRID,
  .per_cell = FISH_PER_CELL,
  .depth    = 0.0f,
};

static const flock_config_t bird_config = {
  .count    = BIRD_COUNT,
  .grid     = BIRD_GRID,
  .per_cell = BIRD_PER_CELL,
  .depth    = BIRD_DEPTH,
};

/* Fish: 0.6 m/s, in a band from their sheet to 45 cm above it. */
static const flock_start_t fish_start = {0.6f, 0.6f, 0.85f, 0.0f, 0.45f, 5.0f};
/* Birds: 8 m/s, spread through their layer and widely scattered. */
static const flock_start_t bird_start = {8.0f, 0.8f, 1.2f, 0.0f, 7.0f, 30.0f};

static const slider_param_t light_sliders[]   = {LIGHT_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t sky_sliders[]     = {SKY_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t sea_sliders[]     = {SEA_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t seabed_sliders[]  = {SEABED_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t coral_sliders[]   = {CORAL_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t life_sliders[]    = {LIFE_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t quality_sliders[] = {QUALITY_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t fish_sliders[]    = {FISH_SLIDERS(SLIDER_ENTRY)};
static const slider_param_t bird_sliders[]    = {BIRD_SLIDERS(SLIDER_ENTRY)};
static const toggle_param_t sky_toggles[]     = {SKY_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t sea_toggles[]     = {SEA_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t seabed_toggles[]  = {SEABED_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t coral_toggles[]   = {CORAL_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t life_toggles[]    = {LIFE_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t quality_toggles[] = {QUALITY_TOGGLES(TOGGLE_ENTRY)};
static const toggle_param_t debug_toggles[]   = {DEBUG_TOGGLES(TOGGLE_ENTRY)};

/* -------------------------------------------------------------------------- *
 * Math helpers
 * -------------------------------------------------------------------------- */

static float round_half_up(float x)
{
  return floorf(x + 0.5f);
}

static float planet_radius(void)
{
  return (state.planet_exponent >= PLANET_EARTH_EXPONENT) ?
           PLANET_EARTH_RADIUS :
           powf(10.0f, state.planet_exponent);
}

/* The Cloud drift slider is cubed, which gives fine control at the slow end:
 * 0.3 is 8 cm/s, 1 is 3 m/s. */
static float cloud_drift_speed(void)
{
  return 3.0f * state.cloud_drift * state.cloud_drift * state.cloud_drift;
}

static float fish_home_size(void)
{
  return (float)FLOCK_WORLD / fmaxf(round_half_up(state.fish_homes), 1.0f);
}

static float camera_yaw(void)
{
  return -glm_rad(state.camera.rotation[1]);
}

static float camera_pitch(void)
{
  return glm_rad(state.camera.rotation[0]);
}

static float camera_dist(void)
{
  return -state.camera.position[2];
}

static float clamp_dist(float dist)
{
  const float min_dist = state.is_follow_bird ? MIN_FOLLOW_DIST : MIN_DIST;
  return fmaxf(min_dist, fminf(MAX_DIST, dist));
}

/* Lets the camera drop to about a metre above the water, clear of the tallest
 * crest, but never lower. */
static float clamp_pitch(float pitch, float dist)
{
  const float lowest = asinf(
    fmaxf(-1.0f, (WATER_Y + MIN_WATER_CLEARANCE - state.orbit_y) / dist));
  return fmaxf(lowest, fminf(MAX_PITCH, pitch));
}

static void set_camera(float yaw, float pitch, float dist)
{
  camera_set_rotation(&state.camera,
                      (vec3){glm_deg(pitch), -glm_deg(yaw), 0.0f});
  camera_set_translation(&state.camera, (vec3){0.0f, 0.0f, -dist});
}

static void reset_camera(void)
{
  state.pan_x = HOME_X;
  state.pan_z = HOME_Z;
  set_camera(HOME_YAW, HOME_PITCH, HOME_DIST);
}

/* -------------------------------------------------------------------------- *
 * Parameters
 * -------------------------------------------------------------------------- */

static void reset_sliders(const slider_param_t* items, uint32_t count)
{
  for (uint32_t i = 0; i < count; ++i) {
    *items[i].value = items[i].def;
  }
}

static void reset_toggles(const toggle_param_t* items, uint32_t count)
{
  for (uint32_t i = 0; i < count; ++i) {
    *items[i].value = items[i].def;
  }
}

static void reset_parameters(void)
{
  reset_sliders(light_sliders, ARRAY_SIZE(light_sliders));
  reset_sliders(sky_sliders, ARRAY_SIZE(sky_sliders));
  reset_sliders(sea_sliders, ARRAY_SIZE(sea_sliders));
  reset_sliders(seabed_sliders, ARRAY_SIZE(seabed_sliders));
  reset_sliders(coral_sliders, ARRAY_SIZE(coral_sliders));
  reset_sliders(life_sliders, ARRAY_SIZE(life_sliders));
  reset_sliders(quality_sliders, ARRAY_SIZE(quality_sliders));
  reset_sliders(fish_sliders, ARRAY_SIZE(fish_sliders));
  reset_sliders(bird_sliders, ARRAY_SIZE(bird_sliders));
  reset_toggles(sky_toggles, ARRAY_SIZE(sky_toggles));
  reset_toggles(sea_toggles, ARRAY_SIZE(sea_toggles));
  reset_toggles(seabed_toggles, ARRAY_SIZE(seabed_toggles));
  reset_toggles(coral_toggles, ARRAY_SIZE(coral_toggles));
  reset_toggles(life_toggles, ARRAY_SIZE(life_toggles));
  reset_toggles(quality_toggles, ARRAY_SIZE(quality_toggles));
  reset_toggles(debug_toggles, ARRAY_SIZE(debug_toggles));
}

/* -------------------------------------------------------------------------- *
 * Shader sources
 * -------------------------------------------------------------------------- */

/* The sizes of the two flocks are written at the top of the shaders, so that
 * the simulation and the drawing shader always agree. */
static int write_flock_header(char* out, size_t capacity,
                              const flock_config_t* config)
{
  return snprintf(
    out, capacity,
    "const FISH_COUNT : u32 = %uu;\n"
    "const FISH_WORLD : f32 = %.1f;\n"
    "const FISH_GRID : i32 = %u;\n"
    "const FISH_CELL_SIZE : f32 = %.4f;\n"
    "const FISH_PER_CELL : u32 = %uu;\n"
    "const FLOCK_DEPTH : f32 = %.2f;\n"
    "const LURE_MAX : u32 = %uu;\n"
    "const LURE_GRID : u32 = %uu;\n"
    "const PREDATOR_MAX : u32 = %uu;\n"
    "const BED_SIZE : f32 = %.1f;\n"
    "const FAR_SIZE : f32 = %.1f;\n",
    (unsigned)config->count, (double)FLOCK_WORLD, (unsigned)config->grid,
    (double)FLOCK_WORLD / (double)config->grid, (unsigned)config->per_cell,
    (double)config->depth, (unsigned)LURE_MAX, (unsigned)FLOCK_WORLD,
    (unsigned)PREDATOR_MAX, (double)BED_SIZE, (double)FAR_SIZE);
}

static void write_draw_header(char* out, size_t capacity)
{
  const int used = write_flock_header(out, capacity, &fish_config);
  snprintf(out + used, capacity - (size_t)used,
           "const BIRD_COUNT : u32 = %uu;\n"
           "const BIRD_GRID : i32 = %u;\n"
           "const BIRD_CELL_SIZE : f32 = %.4f;\n"
           "const BIRD_PER_CELL : u32 = %uu;\n"
           "const BIRD_DEPTH : f32 = %.2f;\n"
           "const BIRD_HEIGHT : f32 = %.2f;\n",
           (unsigned)BIRD_COUNT, (unsigned)BIRD_GRID,
           (double)FLOCK_WORLD / (double)BIRD_GRID, (unsigned)BIRD_PER_CELL,
           (double)BIRD_DEPTH, (double)BIRD_HEIGHT);
}

/* Joins the header and the parts into one heap string. Called only while the
 * shader modules are built; the caller frees the result. */
static char* join_shader_source(const char* header, const char* const* parts,
                                uint32_t max_parts)
{
  /* Each piece is followed by a newline; one more byte for the terminator. */
  size_t total = strlen(header) + 2u;
  for (uint32_t i = 0; i < max_parts && parts[i]; ++i) {
    total += strlen(parts[i]) + 1u;
  }
  char* source = (char*)malloc(total);
  if (!source) {
    return NULL;
  }
  char* cursor  = source;
  size_t length = strlen(header);
  memcpy(cursor, header, length);
  cursor += length;
  *cursor++ = '\n';
  for (uint32_t i = 0; i < max_parts && parts[i]; ++i) {
    length = strlen(parts[i]);
    memcpy(cursor, parts[i], length);
    cursor += length;
    *cursor++ = '\n';
  }
  *cursor = '\0';
  return source;
}

static WGPUShaderModule create_module(WGPUDevice device, const char* header,
                                      const char* const* parts,
                                      uint32_t max_parts)
{
  char* source = join_shader_source(header, parts, max_parts);
  if (!source) {
    return NULL;
  }
  WGPUShaderModule module = wgpu_create_shader_module(device, source);
  free(source);
  return module;
}

/* -------------------------------------------------------------------------- *
 * Textures
 * -------------------------------------------------------------------------- */

static void create_target(wgpu_context_t* wgpu_context,
                          target_texture_t* target, WGPUStringView label,
                          uint32_t width, uint32_t height,
                          WGPUTextureFormat format)
{
  target->texture = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label = label,
                            .usage = WGPUTextureUsage_RenderAttachment
                                     | WGPUTextureUsage_TextureBinding,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {width, height, 1},
                            .format        = format,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });
  target->view = wgpuTextureCreateView(
    target->texture, &(WGPUTextureViewDescriptor){
                       .format          = format,
                       .dimension       = WGPUTextureViewDimension_2D,
                       .baseMipLevel    = 0,
                       .mipLevelCount   = 1,
                       .baseArrayLayer  = 0,
                       .arrayLayerCount = 1,
                       .aspect          = WGPUTextureAspect_All,
                     });
}

static void destroy_target(target_texture_t* target)
{
  WGPU_RELEASE_RESOURCE(TextureView, target->view)
  if (target->texture) {
    wgpuTextureDestroy(target->texture);
    wgpuTextureRelease(target->texture);
    target->texture = NULL;
  }
}

/* The seabed cache and the wide rim map never change size. */
static void init_cache_textures(wgpu_context_t* wgpu_context)
{
  create_target(wgpu_context, &state.bed_sand, STRVIEW("Bed sand - Texture"),
                BED_SIZE, BED_SIZE, TARGET_FORMAT_COLOR);
  create_target(wgpu_context, &state.bed, STRVIEW("Bed cache - Texture"),
                BED_SIZE, BED_SIZE, TARGET_FORMAT_COLOR);
  create_target(wgpu_context, &state.far_sand, STRVIEW("Far sand - Texture"),
                FAR_SIZE, FAR_SIZE, TARGET_FORMAT_COLOR);
  create_target(wgpu_context, &state.far_rim, STRVIEW("Far rim - Texture"),
                FAR_SIZE, FAR_SIZE, TARGET_FORMAT_COLOR);
}

static void init_sampler(wgpu_context_t* wgpu_context)
{
  state.linear_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Linear - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

/* -------------------------------------------------------------------------- *
 * Uniforms
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Uniforms - Buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.uniform_data),
    });
}

static void set_uniform(uint32_t slot, float x, float y, float z, float w)
{
  float* row = &state.uniform_data[slot * 4u];
  row[0]     = x;
  row[1]     = y;
  row[2]     = z;
  row[3]     = w;
}

static double wave_random(uint32_t i, double a, double b, double c)
{
  const double v = sin((double)i * a + b) * c;
  return v - floor(v);
}

/* The wave trains. Each has a direction, a wavelength and a few fixed random
 * numbers, none of which depend on the pixel, so they are worked out here
 * once per frame and handed to the shader as a table. The longest is set by
 * the Longest wave slider; each is 0.87 of the last. Most run within about a
 * radian of the wind, the shortest in any direction, and three of the long
 * ones come from a separate swell direction. */
static void update_wave_trains(float* table)
{
  double wavelength = (double)state.wave_length;
  for (uint32_t i = 0; i < WAVE_TRAIN_COUNT; ++i) {
    const double frac  = (double)i / (double)(WAVE_TRAIN_COUNT - 1u);
    const double rnd_a = wave_random(i, 12.9898, 4.1, 43758.5453);
    const double rnd_b = wave_random(i, 78.233, 1.3, 12543.123);
    const double rnd_c = wave_random(i, 39.346, 2.7, 24634.6345);
    double angle
      = WIND_ANGLE + (rnd_a - 0.5) * 2.0 * (1.0 + (GLM_PI - 1.0) * frac);
    if (i % 4u == 1u && i < 12u) {
      angle = SWELL_ANGLE + (rnd_a - 0.5) * 0.5;
    }
    const double length = wavelength * (0.85 + 0.3 * rnd_c);
    const double k      = 2.0 * GLM_PI / length;
    float* train        = table + i * 8u;
    train[0]            = (float)cos(angle);
    train[1]            = (float)sin(angle);
    train[2]            = (float)k;
    train[3]            = (float)sqrt(9.81 * k);
    train[4]            = (float)(0.6 + 0.8 * rnd_b);
    train[5]            = (float)(rnd_b * 2.0 * GLM_PI);
    train[6]            = (float)(rnd_c * 2.0 * GLM_PI);
    train[7]            = (float)length;
    wavelength *= 0.87;
  }
}

/* The settings that shape the seabed. When any changes, the seabed cache and
 * the wide rim map are filled again. */
static void bed_settings_key(float key[BED_KEY_COUNT])
{
  key[0]  = state.shallow_depth;
  key[1]  = state.ocean_depth;
  key[2]  = state.shoal_size;
  key[3]  = state.shoal_cover;
  key[4]  = state.lagoon_depth;
  key[5]  = state.ring_gap;
  key[6]  = state.dips;
  key[7]  = state.dip_size;
  key[8]  = state.relief;
  key[9]  = state.is_sand_on ? state.sand_waves : 0.0f;
  key[10] = state.is_coral_on ? state.coral : 0.0f;
  key[11] = state.coral_top;
  key[12] = state.coral_range;
  key[13] = state.coral_dots;
}

typedef struct {
  float x;
  float y;
  float z;
} camera_pose_t;

static camera_pose_t camera_position(void)
{
  const float dist  = camera_dist();
  const float yaw   = camera_yaw();
  const float pitch = camera_pitch();
  const float cp    = cosf(pitch);
  return (camera_pose_t){
    .x = dist * cp * sinf(yaw),
    .y = dist * sinf(pitch) + state.orbit_y,
    .z = dist * cp * cosf(yaw),
  };
}

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  const camera_pose_t cam = camera_position();
  const float el          = glm_rad(state.sun_height);
  const float az          = glm_rad(state.sun_direction);
  /* Field of view is the vertical angle the screen spans; the shader wants
   * the matching focal length. */
  const float focal = 0.5f / tanf(glm_rad(state.fov) * 0.5f);

  /* The cloud grid: two edges of equal length, the second turned from the
   * first by the skew angle, and the whole thing rotated. The shader gets the
   * edges and the inverse of the 2x2 matrix they form. */
  const float turn = glm_rad(state.grid_turn);
  const float skew = glm_rad(state.cloud_skew);
  const float e1x  = state.cloud_spacing * cosf(turn);
  const float e1z  = state.cloud_spacing * sinf(turn);
  const float e2x  = state.cloud_spacing * cosf(turn + skew);
  const float e2z  = state.cloud_spacing * sinf(turn + skew);
  const float det  = e1x * e2z - e2x * e1z;

  const float steps = round_half_up(state.march_steps);
  const bool far_ok = state.is_bed_on && state.has_far_key;

  update_wave_trains(&state.uniform_data[UNIFORM_SETTINGS_VEC4 * 4u]);

  set_uniform(0, (float)state.scene_width, (float)state.scene_height,
              state.cloud_flow, steps);
  set_uniform(1, cam.x, cam.y, cam.z, focal);
  set_uniform(2, cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az),
              state.coverage);
  set_uniform(3, state.density, state.fluff, state.pan_x, state.pan_z);
  set_uniform(4, state.is_coral_on ? state.coral : 0.0f, (float)state.low_width,
              (float)state.low_height, state.coral_sway);
  set_uniform(5, (float)state.shadow_size, state.shallow_depth, state.relief,
              state.ocean_depth);
  set_uniform(6, WATER_Y, state.is_mirror ? 1.0f : 0.0f, state.shoal_size,
              state.wave_clock);
  set_uniform(7, state.is_waves_on ? state.wave_height : 0.0f,
              state.is_wave_march ? 1.0f : 0.0f,
              state.is_caustics ? 1.0f : 0.0f,
              state.is_whitecaps ? 1.0f : 0.0f);
  set_uniform(8, state.horizon_clouds, planet_radius(), state.cloud_spacing,
              state.is_sand_on ? state.sand_waves : 0.0f);
  set_uniform(9, e1x, e1z, e2x, e2z);
  set_uniform(10, e2z / det, -e2x / det, -e1z / det, e1x / det);
  set_uniform(11, state.dips, state.is_fish_on ? 1.0f : 0.0f, state.fish_depth,
              state.is_show_lures ? 1.0f : 0.0f);
  set_uniform(12, state.is_show_predators ? 1.0f : 0.0f, state.fish_predators,
              state.fish_fear_range, state.cloud_scatter);
  set_uniform(13, state.cloud_drift_x, state.cloud_drift_z, state.wander_clock,
              state.turnover_clock);
  set_uniform(14, state.cloud_turnover, state.is_birds_on ? 1.0f : 0.0f,
              state.wave_length, state.wave_random);
  set_uniform(15, state.is_clouds_on ? 1.0f : 0.0f,
              state.is_bed_on ? 1.0f : 0.0f, state.is_cloud_grid ? 1.0f : 0.0f,
              state.cloud_size);
  set_uniform(16, state.bed_at_x, state.bed_at_z, BED_TEXEL,
              state.is_bed_cache ? 1.0f : 0.0f);
  set_uniform(17, state.cloud_detail >= 600.0f ? 1e9f : state.cloud_detail,
              state.is_show_walls ? 1.0f : 0.0f, fish_home_size(),
              state.coral_spikes);
  set_uniform(18, state.coral_top, state.coral_range,
              state.is_bird_shadows ? 1.0f : 0.0f,
              state.is_feather_sheen ? 1.0f : 0.0f);
  set_uniform(19, state.orbit_y, (float)state.sea_width,
              (float)state.sea_height, state.is_coral_focus ? 1.0f : 0.0f);
  set_uniform(20, state.coral_dots, state.shoal_cover, state.lagoon_depth,
              state.ring_gap);
  set_uniform(21, state.dip_size, state.fish_spread, 0.0f, 0.0f);
  set_uniform(22, state.far_at_x, state.far_at_z, FAR_TEXEL,
              far_ok ? 1.0f : 0.0f);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       state.uniform_data, sizeof(state.uniform_data));
}

/* -------------------------------------------------------------------------- *
 * Flocks
 * -------------------------------------------------------------------------- */

static float random_unit(void)
{
  return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* Starts in loose clumps, each clump heading roughly one way, so groups form
 * quickly. Twelve numbers per member: position x z, velocity x z, beat phase,
 * turning rate, size, height, vertical speed, effort and two unused. */
static void fill_flock_start(float* start, uint32_t count, float world,
                             const flock_start_t* desc)
{
  const uint32_t clumps
    = MAX(4u, (uint32_t)round_half_up((float)count / 136.0f));
  const uint32_t per_clump = (count + clumps - 1u) / clumps;
  float cx = 0.0f, cz = 0.0f, heading = 0.0f;
  for (uint32_t i = 0; i < count; ++i) {
    if (i % per_clump == 0u) {
      cx      = random_unit() * world;
      cz      = random_unit() * world;
      heading = random_unit() * 2.0f * GLM_PIf;
    }
    const float spread = desc->clump_radius * sqrtf(random_unit());
    const float around = random_unit() * 2.0f * GLM_PIf;
    const float dir    = heading + (random_unit() - 0.5f) * 0.8f;
    float* boid        = start + i * BOID_FLOATS;
    boid[0]            = fmodf(cx + spread * cosf(around) + world, world);
    boid[1]            = fmodf(cz + spread * sinf(around) + world, world);
    boid[2]            = desc->speed * cosf(dir);
    boid[3]            = desc->speed * sinf(dir);
    boid[4]            = random_unit() * 2.0f * GLM_PIf;
    boid[5]            = 0.0f;
    boid[6] = desc->size_lo + (desc->size_hi - desc->size_lo) * random_unit();
    boid[7] = desc->lift_lo + (desc->lift_hi - desc->lift_lo) * random_unit();
    boid[8] = boid[9] = boid[10] = boid[11] = 0.0f;
  }
}

static WGPUBuffer create_storage_buffer(wgpu_context_t* wgpu_context,
                                        WGPUStringView label, uint64_t size,
                                        WGPUBufferUsage extra_usage)
{
  return wgpuDeviceCreateBuffer(
    wgpu_context->device, &(WGPUBufferDescriptor){
                            .label = label,
                            .usage = WGPUBufferUsage_Storage | extra_usage,
                            .size  = size,
                          });
}

static WGPUComputePipeline create_compute_pipeline(WGPUDevice device,
                                                   WGPUPipelineLayout layout,
                                                   WGPUShaderModule module,
                                                   WGPUStringView entry)
{
  return wgpuDeviceCreateComputePipeline(
    device, &(WGPUComputePipelineDescriptor){
              .label   = STRVIEW("Flock - Compute pipeline"),
              .layout  = layout,
              .compute = {.module = module, .entryPoint = entry},
            });
}

/* Two copies of the members, so a step reads one and writes the other, plus
 * the grid they are filed into, the lures and the predators. All of it stays
 * on the GPU; the CPU only supplies the starting positions. */
static void init_flock(wgpu_context_t* wgpu_context, flock_t* flock,
                       const flock_config_t* config,
                       const flock_start_t* start_desc, WGPUShaderModule module)
{
  WGPUDevice device = wgpu_context->device;
  flock->config     = *config;

  const uint64_t members_size = (uint64_t)config->count * BOID_BYTES;
  const WGPUBufferUsage copy
    = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
  flock->members[0] = create_storage_buffer(
    wgpu_context, STRVIEW("Flock members A - Buffer"), members_size, copy);
  flock->members[1] = create_storage_buffer(
    wgpu_context, STRVIEW("Flock members B - Buffer"), members_size, copy);
  flock->cells
    = create_storage_buffer(wgpu_context, STRVIEW("Flock grid - Buffer"),
                            (uint64_t)config->grid * config->grid
                              * (config->per_cell + 1u) * sizeof(uint32_t),
                            0);
  flock->lures = create_storage_buffer(
    wgpu_context, STRVIEW("Flock lures - Buffer"), LURE_MAX * 16u, 0);
  flock->lure_cells = create_storage_buffer(
    wgpu_context, STRVIEW("Flock lure cells - Buffer"),
    (uint64_t)FLOCK_WORLD * FLOCK_WORLD * sizeof(uint32_t), 0);
  flock->predators = create_storage_buffer(
    wgpu_context, STRVIEW("Flock predators - Buffer"), PREDATOR_MAX * 32u, 0);
  flock->settings_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Flock settings - Buffer"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(flock->settings),
            });

  float* start = (float*)malloc((size_t)members_size);
  if (start) {
    fill_flock_start(start, config->count, (float)FLOCK_WORLD, start_desc);
    wgpuQueueWriteBuffer(wgpu_context->queue, flock->members[0], 0, start,
                         (size_t)members_size);
    wgpuQueueWriteBuffer(wgpu_context->queue, flock->members[1], 0, start,
                         (size_t)members_size);
    free(start);
  }

  flock->clear_pipeline = create_compute_pipeline(
    device, state.sim_pipeline_layout, module, STRVIEW("clearCells"));
  flock->bin_pipeline = create_compute_pipeline(
    device, state.sim_pipeline_layout, module, STRVIEW("binBoids"));
  flock->step_pipeline = create_compute_pipeline(
    device, state.sim_pipeline_layout, module, STRVIEW("stepBoids"));
  flock->lure_pipeline = create_compute_pipeline(
    device, state.sim_pipeline_layout, module, STRVIEW("updateLures"));
  flock->predator_pipeline = create_compute_pipeline(
    device, state.sim_pipeline_layout, module, STRVIEW("updatePredators"));

  /* bind_groups[n] reads copy n and writes the other one */
  for (uint32_t n = 0; n < 2u; ++n) {
    WGPUBindGroupEntry entries[8] = {
      {.binding = 0,
       .buffer  = flock->settings_buffer,
       .size    = sizeof(flock->settings)},
      {.binding = 1, .buffer = flock->members[n], .size = members_size},
      {.binding = 2, .buffer = flock->members[1u - n], .size = members_size},
      {.binding = 3,
       .buffer  = flock->cells,
       .size = (uint64_t)config->grid * config->grid * (config->per_cell + 1u)
               * sizeof(uint32_t)},
      {.binding = 4, .buffer = flock->lures, .size = LURE_MAX * 16u},
      {.binding = 5,
       .buffer  = flock->lure_cells,
       .size    = (uint64_t)FLOCK_WORLD * FLOCK_WORLD * sizeof(uint32_t)},
      {.binding = 6, .buffer = flock->predators, .size = PREDATOR_MAX * 32u},
      {.binding = 7, .textureView = state.bed.view},
    };
    flock->bind_groups[n] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Flock - Bind group"),
                .layout     = state.sim_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

static void destroy_flock(flock_t* flock)
{
  for (uint32_t n = 0; n < 2u; ++n) {
    WGPU_RELEASE_RESOURCE(BindGroup, flock->bind_groups[n])
    WGPU_RELEASE_RESOURCE(Buffer, flock->members[n])
  }
  WGPU_RELEASE_RESOURCE(Buffer, flock->cells)
  WGPU_RELEASE_RESOURCE(Buffer, flock->lures)
  WGPU_RELEASE_RESOURCE(Buffer, flock->lure_cells)
  WGPU_RELEASE_RESOURCE(Buffer, flock->predators)
  WGPU_RELEASE_RESOURCE(Buffer, flock->settings_buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, flock->clear_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, flock->bin_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, flock->step_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, flock->lure_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, flock->predator_pipeline)
}

static uint32_t group_count(uint32_t items)
{
  return (items + SIM_WORKGROUP_SIZE - 1u) / SIM_WORKGROUP_SIZE;
}

/* Files the flock as it is now into its grid, applies the flocking rules to
 * produce the next copy, then lets its lures and predators update themselves.
 * `now` says which copy is current. */
static void run_flock(wgpu_context_t* wgpu_context, WGPUComputePassEncoder pass,
                      flock_t* flock, uint32_t now)
{
  wgpuQueueWriteBuffer(wgpu_context->queue, flock->settings_buffer, 0,
                       flock->settings, sizeof(flock->settings));
  wgpuComputePassEncoderSetBindGroup(pass, 0, flock->bind_groups[now], 0, NULL);
  wgpuComputePassEncoderSetPipeline(pass, flock->clear_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass, group_count(flock->config.grid * flock->config.grid), 1, 1);
  wgpuComputePassEncoderSetPipeline(pass, flock->bin_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass, group_count(flock->config.count), 1, 1);
  wgpuComputePassEncoderSetPipeline(pass, flock->step_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    pass, group_count(flock->config.count), 1, 1);
  wgpuComputePassEncoderSetPipeline(pass, flock->lure_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(pass, group_count(LURE_MAX), 1, 1);
  wgpuComputePassEncoderSetPipeline(pass, flock->predator_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(pass, group_count(PREDATOR_MAX), 1,
                                           1);
}

/* One row per vec4 in the compute shader's Sim struct. */
static void update_flock_settings(float dt)
{
  const float fish_dt                       = state.is_fish_on ? dt : 0.0f;
  float* f                                  = state.fish.settings;
  const float fish_row[SIM_SETTINGS_FLOATS] = {
    fish_dt,
    state.wave_clock,
    state.fish_space,
    state.fish_separate,
    state.fish_align,
    state.fish_align_range,
    state.fish_cohere,
    state.fish_cohere_range,
    state.fish_turn,
    state.fish_cruise,
    state.fish_vary,
    -cosf(glm_rad(state.fish_blind)),
    fish_home_size(),
    state.fish_walls,
    0.6f,
    state.fish_predator_room,
    state.fish_lures,
    state.fish_lure_pull,
    state.fish_lure_range,
    state.fish_lure_life,
    state.fish_predators,
    state.fish_fear,
    state.fish_fear_range,
    state.fish_predator_life,
    state.fish_predator_speed,
    state.fish_depth,
    0.0f,
    state.fish_caution,
    state.bed_at_x,
    state.bed_at_z,
    BED_TEXEL,
    (state.is_bed_cache && state.is_bed_on && state.has_bed_key) ? 1.0f : 0.0f,
  };
  memcpy(f, fish_row, sizeof(fish_row));

  /* A faster bird needs to turn harder to hold the same size of turn, so its
   * turning limit follows its speed. Birds have no walls and no predators. */
  const float bird_dt                       = state.is_birds_on ? dt : 0.0f;
  const float bird_row[SIM_SETTINGS_FLOATS] = {
    bird_dt,
    state.wave_clock,
    state.bird_space,
    state.bird_separate,
    state.bird_align,
    state.bird_align_range,
    state.bird_cohere,
    state.bird_cohere_range,
    state.bird_speed * 1.4f,
    state.bird_speed,
    0.25f,
    -cosf(glm_rad(state.bird_blind)),
    (float)FLOCK_WORLD,
    0.0f,
    state.bird_restless,
    state.bird_swoop,
    state.bird_lures,
    state.bird_lure_pull,
    state.bird_lure_range,
    state.bird_lure_life,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
  };
  memcpy(state.birds.settings, bird_row, sizeof(bird_row));
}

/* -------------------------------------------------------------------------- *
 * Following a bird
 *
 * The birds live on the GPU, so to follow one the example asks for a copy:
 * each frame one bird's 48 bytes are copied into a small buffer that can be
 * read, and are read when they arrive, a frame or two later. In between, the
 * bird's last known position is carried forward by its velocity, so the camera
 * does not lag. When following starts, all the birds are copied once, to pick
 * the one nearest the middle of the view.
 * -------------------------------------------------------------------------- */

static void follow_map_cb(WGPUMapAsyncStatus status, WGPUStringView message,
                          void* userdata1, void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);

  state.follow.is_busy = false;
  if (status != WGPUMapAsyncStatus_Success) {
    return;
  }

  float record[BIRD_COUNT * BOID_FLOATS];
  const void* mapped
    = wgpuBufferGetConstMappedRange(state.follow.buffer, 0, state.follow.bytes);
  if (mapped) {
    memcpy(record, mapped, state.follow.bytes);
  }
  wgpuBufferUnmap(state.follow.buffer);
  if (!mapped || !state.is_follow_bird) {
    return;
  }

  uint32_t k = 0;
  if (state.follow.is_all_birds) {
    /* Pick the bird nearest the point the camera was orbiting, counting
     * across the world's wrap. */
    float best        = 1e9f;
    const float world = (float)FLOCK_WORLD;
    for (uint32_t i = 0; i < BIRD_COUNT; ++i) {
      const float dx = record[i * BOID_FLOATS] - state.follow.near_x;
      const float dz = record[i * BOID_FLOATS + 1u] - state.follow.near_z;
      const float d  = hypotf(dx - world * roundf(dx / world),
                              dz - world * roundf(dz / world));
      if (d < best) {
        best            = d;
        state.follow.id = (int32_t)i;
      }
    }
    k = (uint32_t)state.follow.id * BOID_FLOATS;
  }
  /* A bird's record: position x, z; velocity x, z; ...; height within the
   * layer at [7]; rate of climb at [8]. */
  state.follow.x      = record[k];
  state.follow.z      = record[k + 1u];
  state.follow.vx     = record[k + 2u];
  state.follow.vz     = record[k + 3u];
  state.follow.y      = BIRD_HEIGHT + record[k + 7u];
  state.follow.climb  = record[k + 8u];
  state.follow.has_at = true;
}

static void init_follow_buffer(wgpu_context_t* wgpu_context)
{
  state.follow.id     = -1;
  state.follow.buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Follow bird - Buffer"),
      .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
      .size  = BIRD_COUNT * BOID_BYTES,
    });
}

static void record_follow_copy(WGPUCommandEncoder encoder)
{
  if (!state.is_follow_bird || !state.is_birds_on || state.follow.is_busy) {
    return;
  }
  const bool all = state.follow.id < 0;
  const uint32_t bytes
    = all ? (uint32_t)(BIRD_COUNT * BOID_BYTES) : (uint32_t)BOID_BYTES;
  const uint64_t from
    = all ? 0u : (uint64_t)state.follow.id * (uint64_t)BOID_BYTES;
  wgpuCommandEncoderCopyBufferToBuffer(encoder,
                                       state.birds.members[state.flock_now],
                                       from, state.follow.buffer, 0, bytes);
  state.follow.is_pending   = true;
  state.follow.is_all_birds = all;
  state.follow.bytes        = bytes;
  state.follow.time         = state.now_ms;
  state.follow.near_x       = state.pan_x;
  state.follow.near_z       = state.pan_z;
}

static void request_follow_map(void)
{
  if (!state.follow.is_pending) {
    return;
  }
  state.follow.is_pending = false;
  state.follow.is_busy    = true;
  wgpuBufferMapAsync(state.follow.buffer, WGPUMapMode_Read, 0,
                     state.follow.bytes,
                     (WGPUBufferMapCallbackInfo){
                       .mode     = WGPUCallbackMode_AllowProcessEvents,
                       .callback = follow_map_cb,
                     });
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static WGPUBindGroupLayoutEntry bgl_texture(uint32_t binding,
                                            WGPUShaderStage visibility)
{
  return (WGPUBindGroupLayoutEntry){
    .binding    = binding,
    .visibility = visibility,
    .texture    = {
      .sampleType    = WGPUTextureSampleType_Float,
      .viewDimension = WGPUTextureViewDimension_2D,
    },
  };
}

static WGPUBindGroupLayoutEntry bgl_sampler(uint32_t binding)
{
  return (WGPUBindGroupLayoutEntry){
    .binding    = binding,
    .visibility = WGPUShaderStage_Fragment,
    .sampler    = {.type = WGPUSamplerBindingType_Filtering},
  };
}

static WGPUBindGroupLayoutEntry bgl_read_only_storage(uint32_t binding)
{
  return (WGPUBindGroupLayoutEntry){
    .binding    = binding,
    .visibility = WGPUShaderStage_Fragment,
    .buffer     = {.type = WGPUBufferBindingType_ReadOnlyStorage},
  };
}

static WGPUBindGroupLayoutEntry bgl_compute_buffer(uint32_t binding,
                                                   WGPUBufferBindingType type)
{
  return (WGPUBindGroupLayoutEntry){
    .binding    = binding,
    .visibility = WGPUShaderStage_Compute,
    .buffer     = {.type = type},
  };
}

static WGPUBindGroupLayout
create_bind_group_layout(WGPUDevice device, WGPUStringView label,
                         const WGPUBindGroupLayoutEntry* entries,
                         uint32_t entry_count)
{
  return wgpuDeviceCreateBindGroupLayout(device,
                                         &(WGPUBindGroupLayoutDescriptor){
                                           .label      = label,
                                           .entryCount = entry_count,
                                           .entries    = entries,
                                         });
}

/* Bind group 0 holds the uniforms, shared by every render pipeline. Bind group
 * 1 holds whichever textures and buffers that pass reads. Binding numbers match
 * the shader. */
static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device          = wgpu_context->device;
  const WGPUShaderStage frag = WGPUShaderStage_Fragment;

  {
    const WGPUBindGroupLayoutEntry entries[] = {{
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.uniform_data),
      },
    }};
    state.uniform_bgl = create_bind_group_layout(
      device, STRVIEW("Uniforms - Bind group layout"), entries,
      ARRAY_SIZE(entries));
  }

  /* Clouds: shadow map, the fish and birds (for their shadows), the seabed
   * cache and the wide rim map. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_texture(0, frag),     bgl_sampler(1),
      bgl_read_only_storage(3), bgl_read_only_storage(4),
      bgl_read_only_storage(8), bgl_read_only_storage(9),
      bgl_texture(11, frag),    bgl_texture(14, frag),
    };
    state.cloud_bgl
      = create_bind_group_layout(device, STRVIEW("Cloud - Bind group layout"),
                                 entries, ARRAY_SIZE(entries));
  }

  /* Final: everything the sea needs, plus the clouds, how near their front
   * face is and the sea buffer. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_texture(0, frag),     bgl_sampler(1),
      bgl_texture(2, frag),     bgl_read_only_storage(3),
      bgl_read_only_storage(4), bgl_read_only_storage(5),
      bgl_read_only_storage(6), bgl_read_only_storage(7),
      bgl_read_only_storage(8), bgl_read_only_storage(9),
      bgl_texture(10, frag),    bgl_texture(11, frag),
      bgl_texture(12, frag),    bgl_texture(14, frag),
    };
    state.final_bgl
      = create_bind_group_layout(device, STRVIEW("Final - Bind group layout"),
                                 entries, ARRAY_SIZE(entries));
  }

  /* The sea pass reads everything the final pass does except its own output. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_texture(0, frag),     bgl_sampler(1),
      bgl_read_only_storage(3), bgl_read_only_storage(4),
      bgl_read_only_storage(5), bgl_read_only_storage(6),
      bgl_read_only_storage(7), bgl_read_only_storage(8),
      bgl_read_only_storage(9), bgl_texture(11, frag),
      bgl_texture(14, frag),
    };
    state.sea_bgl = create_bind_group_layout(
      device, STRVIEW("Sea - Bind group layout"), entries, ARRAY_SIZE(entries));
  }

  /* The seabed cache is filled in two passes: the bare sand first, then the
   * coral on it, which reads the first. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_sampler(1),
      bgl_texture(13, frag),
      bgl_texture(14, frag),
    };
    state.bed_sand_bgl = create_bind_group_layout(
      device, STRVIEW("Bed sand - Bind group layout"), entries,
      ARRAY_SIZE(entries));
  }
  {
    const WGPUBindGroupLayoutEntry entries[] = {bgl_texture(13, frag)};
    state.far_rim_bgl
      = create_bind_group_layout(device, STRVIEW("Far rim - Bind group layout"),
                                 entries, ARRAY_SIZE(entries));
  }

  /* The flocks: settings, members before and after, grid, lures, lure cells,
   * predators and the seabed cache. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_compute_buffer(0, WGPUBufferBindingType_Uniform),
      bgl_compute_buffer(1, WGPUBufferBindingType_ReadOnlyStorage),
      bgl_compute_buffer(2, WGPUBufferBindingType_Storage),
      bgl_compute_buffer(3, WGPUBufferBindingType_Storage),
      bgl_compute_buffer(4, WGPUBufferBindingType_Storage),
      bgl_compute_buffer(5, WGPUBufferBindingType_Storage),
      bgl_compute_buffer(6, WGPUBufferBindingType_Storage),
      bgl_texture(7, WGPUShaderStage_Compute),
    };
    state.sim_bgl
      = create_bind_group_layout(device, STRVIEW("Flock - Bind group layout"),
                                 entries, ARRAY_SIZE(entries));
    state.sim_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device, &(WGPUPipelineLayoutDescriptor){
                .label                = STRVIEW("Flock - Pipeline layout"),
                .bindGroupLayoutCount = 1,
                .bindGroupLayouts     = &state.sim_bgl,
              });
  }

  /* Blit: sampler and the scene. */
  {
    const WGPUBindGroupLayoutEntry entries[] = {
      bgl_texture(0, frag),
      bgl_sampler(1),
    };
    state.blit_bgl
      = create_bind_group_layout(device, STRVIEW("Blit - Bind group layout"),
                                 entries, ARRAY_SIZE(entries));
  }
}

/* -------------------------------------------------------------------------- *
 * Render targets and the bind groups that read them
 *
 * These depend on the window size and on the resolution sliders, so they are
 * rebuilt when either changes. Pipelines and static buffers are not.
 * -------------------------------------------------------------------------- */

static WGPUBindGroupEntry bg_texture(uint32_t binding, WGPUTextureView view)
{
  return (WGPUBindGroupEntry){.binding = binding, .textureView = view};
}

static WGPUBindGroupEntry bg_sampler(uint32_t binding, WGPUSampler sampler)
{
  return (WGPUBindGroupEntry){.binding = binding, .sampler = sampler};
}

static WGPUBindGroupEntry bg_buffer(uint32_t binding, WGPUBuffer buffer,
                                    uint64_t size)
{
  return (WGPUBindGroupEntry){
    .binding = binding, .buffer = buffer, .offset = 0, .size = size};
}

static uint64_t flock_cells_size(const flock_t* flock)
{
  return (uint64_t)flock->config.grid * flock->config.grid
         * (flock->config.per_cell + 1u) * sizeof(uint32_t);
}

static void destroy_render_targets(void)
{
  for (uint32_t n = 0; n < 2u; ++n) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.cloud_bind_groups[n])
    WGPU_RELEASE_RESOURCE(BindGroup, state.final_bind_groups[n])
    WGPU_RELEASE_RESOURCE(BindGroup, state.sea_bind_groups[n])
  }
  WGPU_RELEASE_RESOURCE(BindGroup, state.blit_bind_group)
  destroy_target(&state.shadow);
  destroy_target(&state.cloud);
  destroy_target(&state.cloud_near);
  destroy_target(&state.sea);
  destroy_target(&state.scene);
}

static uint32_t scaled_size(uint32_t size, float scale)
{
  const uint32_t scaled = (uint32_t)round_half_up((float)size * scale);
  return MAX(1u, MIN(MAX_TARGET_DIMENSION, scaled));
}

static void init_render_targets(wgpu_context_t* wgpu_context)
{
  destroy_render_targets();

  state.window_width  = (uint32_t)MAX(1, wgpu_context->width);
  state.window_height = (uint32_t)MAX(1, wgpu_context->height);
  state.scene_width   = scaled_size(state.window_width, state.scene_now);
  state.scene_height  = scaled_size(state.window_height, state.scene_now);
  state.low_width     = scaled_size(state.scene_width, state.cloud_res);
  state.low_height    = scaled_size(state.scene_height, state.cloud_res);
  /* At 100% the sea buffer is not used: the final pass works the sea out
   * directly. */
  const bool sea_direct = state.sea_res >= 1.0f;
  state.sea_width       = sea_direct ? state.scene_width :
                                       scaled_size(state.scene_width, state.sea_res);
  state.sea_height      = sea_direct ?
                            state.scene_height :
                            scaled_size(state.scene_height, state.sea_res);
  state.shadow_size
    = MAX(64u, (uint32_t)round_half_up(512.0f * state.cloud_res));
  state.target_cloud_res = state.cloud_res;
  state.target_sea_res   = state.sea_res;

  create_target(wgpu_context, &state.scene, STRVIEW("Scene - Texture"),
                state.scene_width, state.scene_height, TARGET_FORMAT_SCENE);
  create_target(wgpu_context, &state.cloud, STRVIEW("Cloud - Texture"),
                state.low_width, state.low_height, TARGET_FORMAT_COLOR);
  create_target(wgpu_context, &state.cloud_near,
                STRVIEW("Cloud front - Texture"), state.low_width,
                state.low_height, TARGET_FORMAT_SCALAR);
  create_target(wgpu_context, &state.shadow, STRVIEW("Cloud shadow - Texture"),
                state.shadow_size, state.shadow_size, TARGET_FORMAT_SCALAR);
  create_target(wgpu_context, &state.sea, STRVIEW("Sea - Texture"),
                sea_direct ? 1u : state.sea_width,
                sea_direct ? 1u : state.sea_height, TARGET_FORMAT_COLOR);

  WGPUDevice device = wgpu_context->device;
  /* One set of bind groups per copy of the flocks, so drawing can follow
   * whichever is current. */
  for (uint32_t n = 0; n < 2u; ++n) {
    const uint64_t members_fish  = (uint64_t)FISH_COUNT * BOID_BYTES;
    const uint64_t members_birds = (uint64_t)BIRD_COUNT * BOID_BYTES;
    const uint64_t lures_size    = LURE_MAX * 16u;
    const uint64_t lure_cells    = (uint64_t)FLOCK_WORLD * FLOCK_WORLD * 4u;
    const uint64_t predators     = PREDATOR_MAX * 32u;

    const WGPUBindGroupEntry cloud_entries[] = {
      bg_texture(0, state.shadow.view),
      bg_sampler(1, state.linear_sampler),
      bg_buffer(3, state.fish.members[n], members_fish),
      bg_buffer(4, state.fish.cells, flock_cells_size(&state.fish)),
      bg_buffer(8, state.birds.members[n], members_birds),
      bg_buffer(9, state.birds.cells, flock_cells_size(&state.birds)),
      bg_texture(11, state.bed.view),
      bg_texture(14, state.far_rim.view),
    };
    state.cloud_bind_groups[n] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Cloud - Bind group"),
                .layout     = state.cloud_bgl,
                .entryCount = ARRAY_SIZE(cloud_entries),
                .entries    = cloud_entries,
              });

    const WGPUBindGroupEntry final_entries[] = {
      bg_texture(0, state.shadow.view),
      bg_sampler(1, state.linear_sampler),
      bg_texture(2, state.cloud.view),
      bg_buffer(3, state.fish.members[n], members_fish),
      bg_buffer(4, state.fish.cells, flock_cells_size(&state.fish)),
      bg_buffer(5, state.fish.lures, lures_size),
      bg_buffer(6, state.fish.lure_cells, lure_cells),
      bg_buffer(7, state.fish.predators, predators),
      bg_buffer(8, state.birds.members[n], members_birds),
      bg_buffer(9, state.birds.cells, flock_cells_size(&state.birds)),
      bg_texture(10, state.cloud_near.view),
      bg_texture(11, state.bed.view),
      bg_texture(12, state.sea.view),
      bg_texture(14, state.far_rim.view),
    };
    state.final_bind_groups[n] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Final - Bind group"),
                .layout     = state.final_bgl,
                .entryCount = ARRAY_SIZE(final_entries),
                .entries    = final_entries,
              });

    const WGPUBindGroupEntry sea_entries[] = {
      bg_texture(0, state.shadow.view),
      bg_sampler(1, state.linear_sampler),
      bg_buffer(3, state.fish.members[n], members_fish),
      bg_buffer(4, state.fish.cells, flock_cells_size(&state.fish)),
      bg_buffer(5, state.fish.lures, lures_size),
      bg_buffer(6, state.fish.lure_cells, lure_cells),
      bg_buffer(7, state.fish.predators, predators),
      bg_buffer(8, state.birds.members[n], members_birds),
      bg_buffer(9, state.birds.cells, flock_cells_size(&state.birds)),
      bg_texture(11, state.bed.view),
      bg_texture(14, state.far_rim.view),
    };
    state.sea_bind_groups[n] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Sea - Bind group"),
                .layout     = state.sea_bgl,
                .entryCount = ARRAY_SIZE(sea_entries),
                .entries    = sea_entries,
              });
  }

  const WGPUBindGroupEntry blit_entries[] = {
    bg_texture(0, state.scene.view),
    bg_sampler(1, state.linear_sampler),
  };
  state.blit_bind_group = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .label      = STRVIEW("Blit - Bind group"),
              .layout     = state.blit_bgl,
              .entryCount = ARRAY_SIZE(blit_entries),
              .entries    = blit_entries,
            });
}

/* The two passes that fill the seabed cache read these. */
static void init_cache_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  const WGPUBindGroupEntry sand_entries[] = {
    bg_sampler(1, state.linear_sampler),
    bg_texture(13, state.bed_sand.view),
    bg_texture(14, state.far_rim.view),
  };
  state.bed_sand_bind_group = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .label      = STRVIEW("Bed sand - Bind group"),
              .layout     = state.bed_sand_bgl,
              .entryCount = ARRAY_SIZE(sand_entries),
              .entries    = sand_entries,
            });

  const WGPUBindGroupEntry rim_entries[] = {
    bg_texture(13, state.far_sand.view),
  };
  state.far_rim_bind_group = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .label      = STRVIEW("Far rim - Bind group"),
              .layout     = state.far_rim_bgl,
              .entryCount = ARRAY_SIZE(rim_entries),
              .entries    = rim_entries,
            });
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 *
 * There is no mesh, vertex buffer or depth buffer. Every pass draws one
 * screen-covering triangle whose corners are generated in the vertex shader,
 * and all the work happens per pixel.
 * -------------------------------------------------------------------------- */

static WGPURenderPipeline create_fullscreen_pipeline(
  WGPUDevice device, WGPUShaderModule module, WGPUStringView label,
  WGPUStringView vertex_entry, WGPUStringView fragment_entry,
  WGPUBindGroupLayout* layouts, uint32_t layout_count,
  const WGPUTextureFormat* formats, uint32_t format_count)
{
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = label,
              .bindGroupLayoutCount = layout_count,
              .bindGroupLayouts     = layouts,
            });

  WGPUColorTargetState targets[2];
  for (uint32_t i = 0; i < format_count; ++i) {
    targets[i] = (WGPUColorTargetState){
      .format    = formats[i],
      .writeMask = WGPUColorWriteMask_All,
    };
  }

  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(
    device,
    &(WGPURenderPipelineDescriptor){
      .label  = label,
      .layout = pipeline_layout,
      .vertex = {
        .module     = module,
        .entryPoint = vertex_entry,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff,
      },
      .fragment = &(WGPUFragmentState){
        .module      = module,
        .entryPoint  = fragment_entry,
        .targetCount = format_count,
        .targets     = targets,
      },
    });

  wgpuPipelineLayoutRelease(pipeline_layout);
  return pipeline;
}

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  char header[1536];
  write_draw_header(header, sizeof(header));
  WGPUShaderModule module = create_module(
    device, header, painted_ocean_wgsl_part, MAIN_WGSL_MAX_PARTS);
  WGPUShaderModule blit_module
    = wgpu_create_shader_module(device, painted_ocean_blit_wgsl);

  const WGPUStringView vs          = STRVIEW("vs_full");
  const WGPUTextureFormat color[]  = {TARGET_FORMAT_COLOR};
  const WGPUTextureFormat scalar[] = {TARGET_FORMAT_SCALAR};
  const WGPUTextureFormat scene[]  = {TARGET_FORMAT_SCENE};
  const WGPUTextureFormat cloud[] = {TARGET_FORMAT_COLOR, TARGET_FORMAT_SCALAR};
  WGPUBindGroupLayout uniform_only[]  = {state.uniform_bgl};
  WGPUBindGroupLayout bed_layouts[]   = {state.uniform_bgl, state.bed_sand_bgl};
  WGPUBindGroupLayout rim_layouts[]   = {state.uniform_bgl, state.far_rim_bgl};
  WGPUBindGroupLayout cloud_layouts[] = {state.uniform_bgl, state.cloud_bgl};
  WGPUBindGroupLayout sea_layouts[]   = {state.uniform_bgl, state.sea_bgl};
  WGPUBindGroupLayout final_layouts[] = {state.uniform_bgl, state.final_bgl};

  state.shadow_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Cloud shadow - Render pipeline"), vs,
    STRVIEW("fs_shadow"), uniform_only, 1, scalar, 1);
  state.bed_sand_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Bed sand - Render pipeline"), vs,
    STRVIEW("fs_bedSand"), uniform_only, 1, color, 1);
  state.bed_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Bed - Render pipeline"), vs, STRVIEW("fs_bed"),
    bed_layouts, 2, color, 1);
  state.far_sand_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Far sand - Render pipeline"), vs,
    STRVIEW("fs_farSand"), uniform_only, 1, color, 1);
  state.far_rim_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Far rim - Render pipeline"), vs,
    STRVIEW("fs_farRim"), rim_layouts, 2, color, 1);
  /* The cloud pass writes two things at once: the clouds, and how near their
   * front face is. */
  state.cloud_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Cloud - Render pipeline"), vs, STRVIEW("fs_cloud"),
    cloud_layouts, 2, cloud, 2);
  state.sea_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Sea - Render pipeline"), vs, STRVIEW("fs_sea"),
    sea_layouts, 2, color, 1);
  state.final_pipeline = create_fullscreen_pipeline(
    device, module, STRVIEW("Final - Render pipeline"), vs, STRVIEW("fs_final"),
    final_layouts, 2, scene, 1);

  WGPUBindGroupLayout blit_layouts[]  = {state.blit_bgl};
  const WGPUTextureFormat swapchain[] = {wgpu_context->render_format};
  state.blit_pipeline                 = create_fullscreen_pipeline(
    device, blit_module, STRVIEW("Blit - Render pipeline"), STRVIEW("vs_main"),
    STRVIEW("fs_main"), blit_layouts, 1, swapchain, 1);

  wgpuShaderModuleRelease(module);
  wgpuShaderModuleRelease(blit_module);

  /* The two flocks are one shader compiled with different sizes. */
  const flock_config_t* configs[2] = {&fish_config, &bird_config};
  flock_t* flocks[2]               = {&state.fish, &state.birds};
  const flock_start_t* starts[2]   = {&fish_start, &bird_start};
  for (uint32_t i = 0; i < 2u; ++i) {
    char sim_header[1024];
    write_flock_header(sim_header, sizeof(sim_header), configs[i]);
    WGPUShaderModule sim_module = create_module(
      device, sim_header, painted_ocean_sim_wgsl_part, SIM_WGSL_MAX_PARTS);
    init_flock(wgpu_context, flocks[i], configs[i], starts[i], sim_module);
    wgpuShaderModuleRelease(sim_module);
  }
}

/* -------------------------------------------------------------------------- *
 * Dynamic resolution
 *
 * Holding 60 frames a second. The Scene resolution slider sets the ceiling;
 * scene_now is the resolution actually in use, which the governor lowers when
 * frames come too slowly and raises again when there is room. Work per frame
 * is roughly proportional to the number of pixels, which goes as the square of
 * the resolution, so the correction uses a square root. It starts low and
 * climbs, so every machine opens smoothly and a fast one sharpens to full
 * within a second or two.
 * -------------------------------------------------------------------------- */

static void init_governor(void)
{
  state.scene_now = state.scene_res;
  if (state.is_hold_fps) {
    state.scene_now
      = fmaxf(0.25f, fminf(state.scene_res, state.scene_res * 0.5f));
  }
  state.governor = (governor_t){
    .settle     = 2,
    .probing    = false,
    .climbing   = state.scene_now < state.scene_res,
    .last_probe = 0.0,
    .probe_wait = 2000.0,
  };
  state.prev_scene_res = state.scene_res;
}

/* Called with the measured frame rate: four times a second during the opening
 * climb, twice a second after it. Returns the resolution to use. */
static float govern(float fps, double now, float current, float ceiling)
{
  governor_t* g = &state.governor;
  if (g->settle > 0) {
    g->settle--; /* ignore the reading right after a change */
    return current;
  }
  float next = current;
  if (g->climbing) {
    /* While the frame rate holds, take big steps up. The first reading that
     * falls short ends the climb: step back down by what that reading says,
     * and carry on cautiously from there. */
    if (fps < 57.0f) {
      g->climbing   = false;
      g->last_probe = now;
      g->probe_wait = 4000.0;
      next          = current * fmaxf(0.6f, sqrtf(fps / 60.0f)) * 0.95f;
    }
    else if (current >= ceiling) {
      g->climbing = false;
    }
    else {
      next = current * 1.3f;
    }
  }
  else if (fps < 57.0f) {
    /* Too slow. If this follows a trial increase, wait twice as long before
     * trying one again. */
    if (g->probing) {
      g->probe_wait = fmin(g->probe_wait * 2.0, 30000.0);
    }
    g->probing    = false;
    g->last_probe = now;
    next          = current * fmaxf(0.6f, sqrtf(fps / 60.0f)) * 0.97f;
  }
  else if (fps > 72.0f) {
    /* Clearly faster than needed (a display that refreshes above 60 Hz). */
    g->probing = false;
    next       = current * fminf(1.12f, sqrtf(fps / 64.0f));
  }
  else if (current < ceiling && now - g->last_probe > g->probe_wait) {
    /* Holding 60 says nothing about spare capacity. Try a little more. */
    g->probing    = true;
    g->last_probe = now;
    next          = current * 1.05f;
  }
  else if (g->probing && now - g->last_probe > 1500.0) {
    /* The trial increase held, so trials can come a little sooner again. */
    g->probing    = false;
    g->probe_wait = fmax(2000.0, g->probe_wait * 0.8);
  }
  next = fmaxf(0.25f, fminf(ceiling, next));
  if (fabsf(next - current) < 0.02f * current) {
    return current; /* not worth rebuilding the buffers for */
  }
  g->settle = 1;
  return next;
}

static void update_stats(float fps)
{
  const bool sea_direct = state.sea_res >= 1.0f;
  char sea[40]          = "";
  char automatic[40]    = "";
  if (!sea_direct) {
    snprintf(sea, sizeof(sea), " | sea %ux%u", (unsigned)state.sea_width,
             (unsigned)state.sea_height);
  }
  if (state.is_hold_fps && state.scene_now < state.scene_res - 0.005f) {
    snprintf(automatic, sizeof(automatic), " | auto %d%%",
             (int)round_half_up(state.scene_now * 100.0f));
  }
  snprintf(state.stats, sizeof(state.stats),
           "%d fps | scene %ux%u%s\nclouds %ux%u%s", (int)round_half_up(fps),
           (unsigned)state.scene_width, (unsigned)state.scene_height, sea,
           (unsigned)state.low_width, (unsigned)state.low_height, automatic);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void gui_sliders(const slider_param_t* items, uint32_t count)
{
  for (uint32_t i = 0; i < count; ++i) {
    imgui_overlay_slider_float(items[i].label, items[i].value, items[i].lo,
                               items[i].hi, items[i].format);
  }
}

static void gui_toggles(const toggle_param_t* items, uint32_t count)
{
  for (uint32_t i = 0; i < count; ++i) {
    imgui_overlay_checkbox(items[i].label, items[i].value);
  }
}

static void gui_copy_camera(void)
{
  snprintf(state.camera_text, sizeof(state.camera_text),
           "yaw %.4f, pitch %.4f, dist %.2f, x %.2f, z %.2f, fov %.0f",
           (double)camera_yaw(), (double)camera_pitch(), (double)camera_dist(),
           (double)state.pan_x, (double)state.pan_z, (double)state.fov);
  igSetClipboardText(state.camera_text);
  printf("Camera: %s\n", state.camera_text);
}

static void render_gui(void)
{
  if (!state.is_gui_visible) {
    return;
  }

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){360.0f, 620.0f}, ImGuiCond_FirstUseEver);
  if (igBegin("Painted Ocean Scene", NULL, 0)) {
    imgui_overlay_text("%s", state.stats);
    imgui_overlay_text("Drag: orbit | Wheel: zoom | WASD/arrows: travel");
    imgui_overlay_text("Shift: faster | Home: reset view | H: hide GUI");
    if (imgui_overlay_button("Reset view")) {
      reset_camera();
    }
    igSameLine(0.0f, 8.0f);
    if (imgui_overlay_button("Copy camera position")) {
      gui_copy_camera();
    }
    igSameLine(0.0f, 8.0f);
    if (imgui_overlay_button("Reset all")) {
      reset_parameters();
      init_governor();
    }

    if (igCollapsingHeader_BoolPtr("Camera and light", NULL,
                                   ImGuiTreeNodeFlags_DefaultOpen)) {
      gui_sliders(light_sliders, ARRAY_SIZE(light_sliders));
    }
    if (igCollapsingHeader_BoolPtr("Sky and clouds", NULL, 0)) {
      gui_toggles(sky_toggles, ARRAY_SIZE(sky_toggles));
      gui_sliders(sky_sliders, ARRAY_SIZE(sky_sliders));
      const float radius = planet_radius();
      if (radius >= PLANET_EARTH_RADIUS) {
        imgui_overlay_text("Planet radius: Earth (6,371 km)");
      }
      else if (radius >= 1000.0f) {
        imgui_overlay_text("Planet radius: %.1f km",
                           (double)(radius / 1000.0f));
      }
      else {
        imgui_overlay_text("Planet radius: %.0f m", (double)radius);
      }
      imgui_overlay_text("Cloud drift speed: %.2f m/s",
                         (double)cloud_drift_speed());
    }
    if (igCollapsingHeader_BoolPtr("Sea", NULL, 0)) {
      gui_toggles(sea_toggles, ARRAY_SIZE(sea_toggles));
      gui_sliders(sea_sliders, ARRAY_SIZE(sea_sliders));
    }
    if (igCollapsingHeader_BoolPtr("Seabed", NULL, 0)) {
      gui_toggles(seabed_toggles, ARRAY_SIZE(seabed_toggles));
      gui_sliders(seabed_sliders, ARRAY_SIZE(seabed_sliders));
    }
    if (igCollapsingHeader_BoolPtr("Coral", NULL, 0)) {
      gui_toggles(coral_toggles, ARRAY_SIZE(coral_toggles));
      gui_sliders(coral_sliders, ARRAY_SIZE(coral_sliders));
    }
    if (igCollapsingHeader_BoolPtr("Fish and birds", NULL, 0)) {
      gui_toggles(life_toggles, ARRAY_SIZE(life_toggles));
      gui_sliders(life_sliders, ARRAY_SIZE(life_sliders));
      gui_toggles(debug_toggles, ARRAY_SIZE(debug_toggles));
    }
    if (igCollapsingHeader_BoolPtr("Quality", NULL, 0)) {
      gui_toggles(quality_toggles, ARRAY_SIZE(quality_toggles));
      gui_sliders(quality_sliders, ARRAY_SIZE(quality_sliders));
    }
  }
  igEnd();

  igSetNextWindowPos((ImVec2){390.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){330.0f, 560.0f}, ImGuiCond_FirstUseEver);
  igSetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
  if (igBegin("Fish behaviour", NULL, 0)) {
    gui_sliders(fish_sliders, ARRAY_SIZE(fish_sliders));
    imgui_overlay_text("Home size: %.1f m", (double)fish_home_size());
    imgui_overlay_text("Homes divide the fish world into square pens with");
    imgui_overlay_text("walls. Set Wall avoidance to 0 for open water.");
  }
  igEnd();

  igSetNextWindowPos((ImVec2){730.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){330.0f, 460.0f}, ImGuiCond_FirstUseEver);
  igSetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
  if (igBegin("Bird behaviour", NULL, 0)) {
    gui_sliders(bird_sliders, ARRAY_SIZE(bird_sliders));
    imgui_overlay_text("Gulls: wide personal space, weak alignment");
    imgui_overlay_text("and cohesion. Starlings: personal space about 1 m,");
    imgui_overlay_text("alignment 1.5 over 5 m, cohesion 0.25 over 9 m,");
    imgui_overlay_text("restlessness under 1.");
  }
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Per-frame updates
 * -------------------------------------------------------------------------- */

static bool key_held(keycode_t key)
{
  return state.keys_down[key] != 0u;
}

/* Cloud motion is accumulated frame by frame, so changing a speed never makes
 * the clouds jump. `dt` is animation time: real time scaled by the Animation
 * speed slider. */
static void update_clocks(float dt)
{
  state.cloud_flow += dt * state.cloud_churn * 0.6f;
  state.wave_clock += dt;
  const float way   = glm_rad(state.drift_direction);
  const float speed = cloud_drift_speed();
  state.cloud_drift_x += dt * speed * cosf(way);
  state.cloud_drift_z += dt * speed * sinf(way);
  state.wander_clock += dt * state.cloud_wander * 0.15f;
  state.turnover_clock += dt * 0.12f;
}

/* Travel along the direction the camera faces (dolly) and across it (truck),
 * level with the sea. Speed scales with zoom distance so it feels the same
 * close up and far out. While a bird is followed the camera tracks it. */
static void update_camera(float real_dt)
{
  const float yaw = camera_yaw();
  float dist      = camera_dist();
  const bool keys = !imgui_overlay_want_capture_keyboard();
  const float ahead
    = keys ? (float)((key_held(KEY_W) || key_held(KEY_UP)) ? 1 : 0)
               - (float)((key_held(KEY_S) || key_held(KEY_DOWN)) ? 1 : 0) :
             0.0f;
  const float aside
    = keys ? (float)((key_held(KEY_D) || key_held(KEY_RIGHT)) ? 1 : 0)
               - (float)((key_held(KEY_A) || key_held(KEY_LEFT)) ? 1 : 0) :
             0.0f;
  const bool fast = key_held(KEY_LEFT_SHIFT) || key_held(KEY_RIGHT_SHIFT);

  const bool following = state.is_follow_bird && state.follow.has_at;
  if (following) {
    /* Where the followed bird is now: its last reading, carried forward by its
     * velocity for the time since (at most a fifth of a second, in case a
     * reading is late). */
    const float since
      = fminf((float)(state.now_ms - state.follow.time) / 1000.0f, 0.2f)
        * state.animation_speed;
    const float world = (float)FLOCK_WORLD;
    const float bx    = state.follow.x + state.follow.vx * since;
    const float bz    = state.follow.z + state.follow.vz * since;
    /* The birds' world repeats across the sea; stay with the copy of the bird
     * the camera is already at. */
    state.pan_x   = bx + world * roundf((state.pan_x - bx) / world);
    state.pan_z   = bz + world * roundf((state.pan_z - bz) / world);
    state.orbit_y = state.follow.y + state.follow.climb * since;
  }
  else {
    state.orbit_y += (ORBIT_Y - state.orbit_y) * fminf(1.0f, real_dt * 4.0f);
  }
  if ((ahead != 0.0f || aside != 0.0f) && !following) {
    const float speed = fmaxf(dist, 6.0f) * (fast ? 3.0f : 1.0f) * real_dt
                        / hypotf(ahead, aside);
    state.pan_x += (-sinf(yaw) * ahead + cosf(yaw) * aside) * speed;
    state.pan_z += (-cosf(yaw) * ahead - sinf(yaw) * aside) * speed;
  }

  /* Keep the camera above the water and inside the zoom range. */
  dist              = clamp_dist(dist);
  const float pitch = clamp_pitch(camera_pitch(), dist);
  set_camera(yaw, pitch, dist);
}

/* The seabed cache is centred a little toward the camera from the point it
 * orbits, so the water nearest the viewer is inside it. It is refilled when
 * that spot has moved 10 m, or when any setting that shapes the bed has
 * changed. Its middle is kept on a whole number of samples, so that the
 * samples land on the same spots after a move and the seabed does not
 * shimmer. */
static void update_cache_state(bool* refill_bed, bool* refill_far)
{
  const float dist   = camera_dist();
  const float cp     = cosf(camera_pitch());
  const float yaw    = camera_yaw();
  const float want_x = state.pan_x + dist * cp * sinf(yaw) * 0.3f;
  const float want_z = state.pan_z + dist * cp * cosf(yaw) * 0.3f;

  float key[BED_KEY_COUNT];
  bed_settings_key(key);

  const bool bed_changed
    = !state.has_bed_key || memcmp(key, state.bed_key, sizeof(key)) != 0;
  *refill_bed
    = state.is_bed_cache && state.is_bed_on
      && (bed_changed || fabsf(want_x - state.bed_at_x) > BED_REFILL_DISTANCE
          || fabsf(want_z - state.bed_at_z) > BED_REFILL_DISTANCE);
  if (*refill_bed) {
    state.bed_at_x = round_half_up(want_x / BED_TEXEL) * BED_TEXEL;
    state.bed_at_z = round_half_up(want_z / BED_TEXEL) * BED_TEXEL;
    memcpy(state.bed_key, key, sizeof(key));
    state.has_bed_key = true;
  }

  /* The wide map is refilled when the camera has travelled 100 m, or a
   * setting that shapes the bed changes. */
  const bool far_changed
    = !state.has_far_key || memcmp(key, state.far_key, sizeof(key)) != 0;
  *refill_far
    = state.is_bed_on
      && (far_changed
          || fabsf(state.pan_x - state.far_at_x) > FAR_REFILL_DISTANCE
          || fabsf(state.pan_z - state.far_at_z) > FAR_REFILL_DISTANCE);
  if (*refill_far) {
    state.far_at_x = round_half_up(state.pan_x / FAR_TEXEL) * FAR_TEXEL;
    state.far_at_z = round_half_up(state.pan_z / FAR_TEXEL) * FAR_TEXEL;
    memcpy(state.far_key, key, sizeof(key));
    state.has_far_key = true;
  }
}

/* -------------------------------------------------------------------------- *
 * Rendering
 * -------------------------------------------------------------------------- */

static void draw_fullscreen(WGPUCommandEncoder encoder, WGPUStringView label,
                            const WGPUTextureView* views, uint32_t view_count,
                            WGPURenderPipeline pipeline, WGPUBindGroup group_0,
                            WGPUBindGroup group_1)
{
  WGPURenderPassColorAttachment attachments[2];
  for (uint32_t i = 0; i < view_count; ++i) {
    attachments[i] = (WGPURenderPassColorAttachment){
      .view       = views[i],
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
  }
  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder, &(WGPURenderPassDescriptor){
               .label                = label,
               .colorAttachmentCount = view_count,
               .colorAttachments     = attachments,
             });
  wgpuRenderPassEncoderSetPipeline(pass, pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, group_0, 0, NULL);
  if (group_1) {
    wgpuRenderPassEncoderSetBindGroup(pass, 1, group_1, 0, NULL);
  }
  wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(pass);
  wgpuRenderPassEncoderRelease(pass);
}

static void record_frame(wgpu_context_t* wgpu_context, bool refill_bed,
                         bool refill_far)
{
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Frame - Command encoder"),
                          });
  const uint32_t now     = state.flock_now;
  WGPUBindGroup uniforms = state.uniform_bind_group;

  /* Flocking, entirely on the GPU. Drawing below uses the current copy of
   * each flock and the grid just built from it; the new copies become
   * current next frame. */
  WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(
    encoder, &(WGPUComputePassDescriptor){
               .label = STRVIEW("Flocking - Compute pass"),
             });
  run_flock(wgpu_context, cpass, &state.fish, now);
  run_flock(wgpu_context, cpass, &state.birds, now);
  wgpuComputePassEncoderEnd(cpass);
  wgpuComputePassEncoderRelease(cpass);

  record_follow_copy(encoder);

  /* The wide map first: the cache's fill reads it. */
  if (refill_far) {
    draw_fullscreen(encoder, STRVIEW("Far sand - Render pass"),
                    &state.far_sand.view, 1, state.far_sand_pipeline, uniforms,
                    NULL);
    draw_fullscreen(encoder, STRVIEW("Far rim - Render pass"),
                    &state.far_rim.view, 1, state.far_rim_pipeline, uniforms,
                    state.far_rim_bind_group);
  }
  if (refill_bed) {
    draw_fullscreen(encoder, STRVIEW("Bed sand - Render pass"),
                    &state.bed_sand.view, 1, state.bed_sand_pipeline, uniforms,
                    NULL);
    draw_fullscreen(encoder, STRVIEW("Bed cache - Render pass"),
                    &state.bed.view, 1, state.bed_pipeline, uniforms,
                    state.bed_sand_bind_group);
  }

  /* Pass 1: the centre cloud's shadow map, low resolution, looking down the
   * sun direction. */
  draw_fullscreen(encoder, STRVIEW("Cloud shadow - Render pass"),
                  &state.shadow.view, 1, state.shadow_pipeline, uniforms, NULL);

  /* Pass 2: ray-march the clouds and their reflections into the
   * low-resolution cloud buffer. */
  const WGPUTextureView cloud_views[2]
    = {state.cloud.view, state.cloud_near.view};
  draw_fullscreen(encoder, STRVIEW("Cloud - Render pass"), cloud_views, 2,
                  state.cloud_pipeline, uniforms, state.cloud_bind_groups[now]);

  /* Pass 3: the sea at its own resolution, then sea and sky at full scene
   * resolution with the clouds and birds laid over them. */
  if (state.sea_res < 1.0f) {
    draw_fullscreen(encoder, STRVIEW("Sea - Render pass"), &state.sea.view, 1,
                    state.sea_pipeline, uniforms, state.sea_bind_groups[now]);
  }
  draw_fullscreen(encoder, STRVIEW("Final - Render pass"), &state.scene.view, 1,
                  state.final_pipeline, uniforms, state.final_bind_groups[now]);

  /* Pass 4: scale the scene to the window. */
  draw_fullscreen(encoder, STRVIEW("Blit - Render pass"),
                  &wgpu_context->swapchain_view, 1, state.blit_pipeline,
                  state.blit_bind_group, NULL);

  WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &commands);
  wgpuCommandBufferRelease(commands);
  wgpuCommandEncoderRelease(encoder);

  request_follow_map();
  state.flock_now = 1u - now;
}

/* -------------------------------------------------------------------------- *
 * Input
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  memcpy(state.keys_down, input_event->keys_down, sizeof(state.keys_down));

  switch (input_event->type) {
    case INPUT_EVENT_TYPE_MOUSE_MOVE:
      if (!imgui_overlay_want_capture_mouse() && input_event->mouse_btn_pressed
          && input_event->mouse_button == BUTTON_LEFT) {
        camera_on_input_event(&state.camera, input_event);
      }
      break;
    case INPUT_EVENT_TYPE_MOUSE_SCROLL:
      if (!imgui_overlay_want_capture_mouse()) {
        set_camera(camera_yaw(), camera_pitch(),
                   clamp_dist(camera_dist()
                              * expf(-input_event->scroll_y * ZOOM_SPEED)));
      }
      break;
    case INPUT_EVENT_TYPE_KEY_DOWN:
      if (imgui_overlay_want_capture_keyboard()) {
        break;
      }
      if (input_event->key_code == KEY_H) {
        state.is_gui_visible = !state.is_gui_visible;
      }
      else if (input_event->key_code == KEY_HOME) {
        reset_camera();
      }
      break;
    default:
      break;
  }
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static void update_time(float* real_dt, float* anim_dt)
{
  state.now_ms = (double)wgpu_now_ns() * 1e-6;
  if (state.last_ms == 0.0) {
    state.last_ms = state.now_ms;
    state.fps_ms  = state.now_ms;
  }
  if (state.now_ms - state.last_ms > 250.0) {
    state.is_stalled = true; /* the window was hidden or the app stalled */
  }
  *real_dt
    = fminf((float)(state.now_ms - state.last_ms) / 1000.0f, MAX_FRAME_DELTA);
  *anim_dt      = *real_dt * state.animation_speed;
  state.last_ms = state.now_ms;
}

/* Measures the frame rate twice a second (four times during the opening
 * climb) and lets the governor adjust the scene resolution. */
static void update_governor(void)
{
  state.frames++;
  const double interval
    = (state.governor.climbing && state.is_hold_fps) ? 250.0 : 500.0;
  if (state.now_ms - state.fps_ms <= interval) {
    return;
  }
  const float fps
    = (float)((double)state.frames * 1000.0 / (state.now_ms - state.fps_ms));
  state.frames = 0;
  state.fps_ms = state.now_ms;
  update_stats(fps);
  /* A long gap means the window was hidden or stalled, which says nothing
   * about speed. */
  if (!state.is_hold_fps) {
    state.scene_now = state.scene_res;
  }
  else if (!state.is_stalled) {
    state.scene_now
      = govern(fps, state.now_ms, state.scene_now, state.scene_res);
  }
  state.is_stalled = false;
}

/* Moving the slider restarts the automatic adjustment from the new ceiling. */
static void update_resolution_controls(void)
{
  if (state.scene_res != state.prev_scene_res) {
    state.prev_scene_res    = state.scene_res;
    state.scene_now         = state.scene_res;
    state.governor.climbing = false;
  }
  if (state.is_follow_bird != state.is_prev_follow_bird) {
    state.is_prev_follow_bird = state.is_follow_bird;
    state.follow.id           = -1;
    state.follow.has_at       = false;
    set_camera(camera_yaw(), camera_pitch(), clamp_dist(camera_dist()));
  }
}

static bool targets_out_of_date(const wgpu_context_t* wgpu_context)
{
  return (uint32_t)MAX(1, wgpu_context->width) != state.window_width
         || (uint32_t)MAX(1, wgpu_context->height) != state.window_height
         || state.scene_width
              != scaled_size(state.window_width, state.scene_now)
         || state.cloud_res != state.target_cloud_res
         || state.sea_res != state.target_sea_res;
}

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  srand(wgpu_random_seed());
  reset_parameters();
  state.is_gui_visible = true;
  init_governor();

  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  state.camera.rotation_speed = glm_deg(ORBIT_SPEED);
  state.orbit_y               = ORBIT_Y;
  reset_camera();

  init_sampler(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_cache_textures(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_cache_bind_groups(wgpu_context);

  /* The flocks are built before the render targets: their bind groups point
   * at the flock buffers. */
  init_pipelines(wgpu_context);
  init_render_targets(wgpu_context);
  init_follow_buffer(wgpu_context);

  const WGPUBindGroupEntry uniform_entries[] = {
    bg_buffer(0, state.uniform_buffer, sizeof(state.uniform_data)),
  };
  state.uniform_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Uniforms - Bind group"),
                            .layout     = state.uniform_bgl,
                            .entryCount = ARRAY_SIZE(uniform_entries),
                            .entries    = uniform_entries,
                          });

  imgui_overlay_init(wgpu_context);
  update_stats(0.0f);
  state.is_initialized = true;
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.is_initialized) {
    return EXIT_FAILURE;
  }

  float real_dt = 0.0f;
  float anim_dt = 0.0f;
  update_time(&real_dt, &anim_dt);

  /* GUI first, so a slider moved this frame takes effect this frame. */
  imgui_overlay_new_frame(wgpu_context, real_dt);
  render_gui();
  update_resolution_controls();
  update_governor();

  /* Recreate only what depends on the window size and the resolution. */
  if (targets_out_of_date(wgpu_context)) {
    init_render_targets(wgpu_context);
  }

  update_clocks(anim_dt);
  update_camera(real_dt);
  bool refill_bed = false;
  bool refill_far = false;
  update_cache_state(&refill_bed, &refill_far);
  update_flock_settings(anim_dt);
  update_uniforms(wgpu_context);

  record_frame(wgpu_context, refill_bed, refill_far);

  /* Render the GUI on top of the blitted scene. */
  imgui_overlay_render(wgpu_context);
  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  destroy_render_targets();
  destroy_target(&state.bed_sand);
  destroy_target(&state.bed);
  destroy_target(&state.far_sand);
  destroy_target(&state.far_rim);
  destroy_flock(&state.fish);
  destroy_flock(&state.birds);

  WGPU_RELEASE_RESOURCE(Buffer, state.follow.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.linear_sampler)
  WGPU_RELEASE_RESOURCE(BindGroup, state.uniform_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bed_sand_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.far_rim_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.uniform_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cloud_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.final_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.sea_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bed_sand_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.far_rim_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.sim_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.blit_bgl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.sim_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.bed_sand_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.bed_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.far_sand_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.far_rim_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.cloud_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.sea_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.final_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blit_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title           = "Painted Ocean Scene",
    .no_depth_buffer = true,
    .init_cb         = init,
    .frame_cb        = frame,
    .shutdown_cb     = shutdown,
    .input_event_cb  = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 *
 * Comments are left out of the shader text: the C preprocessor strips them
 * before CODE() can turn the tokens into a string.
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* painted_ocean_blit_wgsl = CODE(
  @group(0) @binding(0) var sceneTex : texture_2d<f32>;
  @group(0) @binding(1) var sceneSampler : sampler;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var corner = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    let p = corner[vertexIndex];
    return VertexOutput(vec4f(p, 0.0, 1.0), vec2f(0.5 * (p.x + 1.0), 0.5 * (1.0 - p.y)));
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return vec4f(textureSampleLevel(sceneTex, sceneSampler, in.uv, 0.0).rgb, 1.0);
  }
);

/* Flocking simulation: Craig Reynolds' boids, run on the GPU. The same shader
 * is compiled twice with different sizes: once for the fish, once for the
 * birds. Each member steers by separation, alignment and cohesion, plus lures
 * that attract and predators that frighten. Members are filed into a grid
 * every frame so a member only reads the cells within its range of vision. */
static const char* painted_ocean_sim_wgsl_part[SIM_WGSL_MAX_PARTS] = {
  CODE(
    struct Boid {
      pos : vec2f,
      vel : vec2f,
      phase : f32,
      turn : f32,
      size : f32,
      lift : f32,
      climb : f32,
      effort : f32,
      pad1 : f32,
      pad2 : f32,
    }
    struct Sim {
      a : vec4f,
      b : vec4f,
      c : vec4f,
      d : vec4f,
      e : vec4f,
      f : vec4f,
      g : vec4f,
      h : vec4f,
    }
    @group(0) @binding(0) var<uniform> sim : Sim;
    @group(0) @binding(1) var<storage, read> boidsIn : array<Boid>;
    @group(0) @binding(2) var<storage, read_write> boidsOut : array<Boid>;
    @group(0) @binding(3) var<storage, read_write> cells : array<atomic<u32>>;
    const CELL_STRIDE : u32 = FISH_PER_CELL + 1u;
    struct Lure {
      pos : vec2f,
      age : f32,
      life : f32,
    }
    @group(0) @binding(4) var<storage, read_write> lures : array<Lure>;
    @group(0) @binding(5) var<storage, read_write> lureCells : array<u32>;
    const CLAIM_DISTANCE : f32 = 0.4;
    struct Predator {
      pos : vec2f,
      vel : vec2f,
      age : f32,
      life : f32,
      pad0 : f32,
      pad1 : f32,
    }
    @group(0) @binding(6) var<storage, read_write> predators : array<Predator>;
    @group(0) @binding(7) var bedTex : texture_2d<f32>;
    const PREDATOR_SENSE : f32 = 10.0;
    @compute @workgroup_size(64)
    fn clearCells(@builtin(global_invocation_id) id : vec3u) {
      if (id.x < u32(FISH_GRID * FISH_GRID)) { atomicStore(&cells[id.x * CELL_STRIDE], 0u); }
      if (id.x < LURE_GRID * LURE_GRID) { lureCells[id.x] = 0u; }
    }
    @compute @workgroup_size(64)
    fn binBoids(@builtin(global_invocation_id) id : vec3u) {
      if (id.x >= FISH_COUNT) { return; }
      let c = clamp(vec2i(floor(boidsIn[id.x].pos / FISH_CELL_SIZE)), vec2i(0), vec2i(FISH_GRID - 1));
      let cell = u32(c.y * FISH_GRID + c.x);
      let slot = atomicAdd(&cells[cell * CELL_STRIDE], 1u);
      if (slot < FISH_PER_CELL) { atomicStore(&cells[cell * CELL_STRIDE + 1u + slot], id.x); }
    }
  ),
  CODE(
    @compute @workgroup_size(64)
    fn stepBoids(@builtin(global_invocation_id) id : vec3u) {
      if (id.x >= FISH_COUNT) { return; }
      let me = boidsIn[id.x];
      let deep = FLOCK_DEPTH > 0.0;
      let myPos = vec3f(me.pos, select(0.0, me.lift, deep));
      let myVel = vec3f(me.vel, select(0.0, me.climb, deep));
      let heading = myVel / max(length(myVel), 1e-4);
      let flatHeading = me.vel / max(length(me.vel), 1e-4);
      let home = clamp(vec2i(floor(me.pos / FISH_CELL_SIZE)), vec2i(0), vec2i(FISH_GRID - 1));
      let dt = sim.a.x;
      let personalSpace = sim.a.z;
      let alignRange = sim.b.y;
      let cohesionRange = sim.b.w;
      let farthest = max(max(alignRange, cohesionRange), personalSpace);
      let reach = i32(ceil(farthest / FISH_CELL_SIZE));
      let homeSize = sim.d.x;
      let walls = sim.d.y;
      let pen = floor(me.pos / homeSize);
      var push = vec3f(0.0);
      var sumVel = vec3f(0.0);
      var aligned = 0.0;
      var sumOff = vec3f(0.0);
      var drawn = 0.0;
      for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
          let c = (home + vec2i(dx, dy) + FISH_GRID) % FISH_GRID;
          let cell = u32(c.y * FISH_GRID + c.x);
          let count = min(atomicLoad(&cells[cell * CELL_STRIDE]), FISH_PER_CELL);
          for (var k = 0u; k < count; k++) {
            let other = atomicLoad(&cells[cell * CELL_STRIDE + 1u + k]);
            if (other == id.x) { continue; }
            let them = boidsIn[other];
            if (walls > 0.0 && any(floor(them.pos / homeSize) != pen)) { continue; }
            var flat = them.pos - me.pos;
            flat -= FISH_WORLD * round(flat / FISH_WORLD);
            let d = vec3f(flat, select(0.0, them.lift - me.lift, deep));
            let dist = length(d);
            if (dist > farthest || dist < 1e-4) { continue; }
            if (dot(d, heading) < sim.c.w * dist) { continue; }
            if (dist < personalSpace) { push -= d / dist * (personalSpace - dist) / personalSpace; }
            if (dist < alignRange) { sumVel += vec3f(them.vel, select(0.0, them.climb, deep)); aligned += 1.0; }
            if (dist < cohesionRange) { sumOff += d; drawn += 1.0; }
          }
        }
      }
  ),
  CODE(
      var lurePull = vec3f(0.0);
      let lureCount = min(u32(sim.e.x), LURE_MAX);
      let lureRange = sim.e.z;
      for (var n = 0u; n < lureCount; n++) {
        let lurePos = lures[n].pos;
        if (walls > 0.0 && any(floor(lurePos / homeSize) != pen)) { continue; }
        var flat = lurePos - me.pos;
        flat -= FISH_WORLD * round(flat / FISH_WORLD);
        let pick = fract(sin(f32(n) * 91.17) * 5471.3);
        let lureHeight = select(mix(0.05, 0.8, (pick - 0.3) / 0.7), mix(-0.92, -0.6, pick / 0.3), pick < 0.3) * FLOCK_DEPTH;
        let d = vec3f(flat, select(0.0, lureHeight - me.lift, deep));
        let dist = length(d);
        if (dist > lureRange || dist < 1e-3) { continue; }
        lurePull += d / dist * min(lureRange * lureRange / (dist * dist) - 1.0, 60.0);
      }
      var flee = vec3f(0.0);
      let predatorCount = min(u32(sim.f.x), PREDATOR_MAX);
      let fearRange = sim.f.z;
      for (var n = 0u; n < predatorCount; n++) {
        var flat = me.pos - predators[n].pos;
        flat -= FISH_WORLD * round(flat / FISH_WORLD);
        let dist = length(flat);
        if (dist > fearRange || dist < 1e-3) { continue; }
        flee += vec3f(flat / dist, 0.0) * min(fearRange * fearRange / (dist * dist) - 1.0, 80.0);
      }
      let fright = clamp(length(flee) * 0.5, 0.0, 1.0);
      var steer = push * sim.a.w;
      steer += lurePull * (sim.e.y / 3.0) * (1.0 - fright);
      if (aligned > 0.0) { steer += (sumVel / aligned - myVel) * sim.b.x; }
      if (drawn > 0.0) { steer += (sumOff / drawn) * sim.b.z; }
      let whim = fract(sin(f32(id.x) * 12.9898 + floor(sim.a.y * 2.0) * 78.233) * 43758.5453) - 0.5;
      steer += vec3f(-flatHeading.y, flatHeading.x, 0.0) * whim * sim.d.z;
      if (deep) {
        let rise = fract(sin(f32(id.x) * 4.1414 + floor(sim.a.y * 1.3) * 33.71) * 9631.7) - 0.5;
        steer.z += rise * 0.6;
      }
      let turnLimit = sim.c.x * (1.0 + 2.0 * fright);
      let hard = length(steer);
      if (hard > turnLimit) { steer *= turnLimit / hard; }
      var bolt = flee * sim.f.y * 0.25;
      let panic = length(bolt);
      if (panic > 40.0) { bolt *= 40.0 / panic; }
      steer += bolt;
  ),
  CODE(
      let feelWalls = walls > 0.0;
      if (feelWalls) {
        let reachAhead = 0.4 + mix(0.6, 2.2, sim.g.w) * max(length(me.vel), 0.2);
        let left = vec2f(-flatHeading.y, flatHeading.x);
        let wayL = normalize(flatHeading + left * 0.7);
        let wayR = normalize(flatHeading - left * 0.7);
        let wallFrom = min(mix(0.5, 3.0, sim.g.w), homeSize / 3.0);
        let penLow = pen * homeSize;
        var ahead = 0.0;
        var onLeft = 0.0;
        var onRight = 0.0;
        for (var n = 0; n < 2; n++) {
          let far = select(0.45, 1.0, n == 1) * reachAhead;
          let weight = select(2.0, 1.0, n == 1) / 3.0;
          let spots = array<vec2f, 3>(me.pos + flatHeading * far, me.pos + wayL * far, me.pos + wayR * far);
          var bad = vec3f(0.0);
          for (var s = 0; s < 3; s++) {
            var here = 0.0;
            if (feelWalls) {
              let inside = min(spots[s] - penLow, penLow + homeSize - spots[s]);
              here = max(here, clamp(1.0 - min(inside.x, inside.y) / wallFrom, 0.0, 1.0) * min(walls / 4.0, 2.0));
            }
            bad[s] = here;
          }
          ahead += bad.x * weight;
          onLeft += bad.y * weight;
          onRight += bad.z * weight;
        }
        var clearer = select(-1.0, 1.0, onLeft < onRight);
        if (abs(onLeft - onRight) < 0.03) { clearer = select(-1.0, 1.0, (id.x & 1u) == 0u); }
        steer += vec3f(left * ((onRight - onLeft) * 3.0 + clearer * ahead * 3.5), 0.0);
      }
      if (deep) {
        let t = sim.a.y;
        let own = f32(id.x) * 2.399;
        let pattern = 0.5 + 0.25 * sin(0.075 * me.pos.x + 0.21 * t + 0.9 * sin(own)) + 0.25 * sin(0.095 * me.pos.y - 0.17 * t + 1.7 + 0.9 * cos(own));
        let lowFrom = mix(1.0, 0.3, sim.d.w);
        let preferred = FLOCK_DEPTH * mix(0.45, -0.9, smoothstep(lowFrom, lowFrom + 0.22, pattern));
        steer.z += clamp((preferred - me.lift) * 1.2, -7.0, 7.0);
      }
      if (deep) {
        let edge = max(abs(me.lift) / FLOCK_DEPTH - 0.88, 0.0) / 0.12;
        steer.z -= sign(me.lift) * edge * edge * 14.0;
      }
  ),
  CODE(
      var vel = myVel;
      var effort = 0.0;
      if (deep) {
        let gravity = 9.81;
        let maxThrust = 5.0;
        let cruise = sim.c.y;
        let drag = gravity / 10.0 / (cruise * cruise);
        let speed = length(myVel);
        let across = steer - heading * dot(steer, heading);
        var way = normalize(myVel + across * dt);
        let steepest = 0.57 * clamp((speed / cruise - 0.55) / 0.3, 0.15, 1.0);
        way.z = clamp(way.z, -0.57, steepest);
        way = vec3f(normalize(way.xy) * sqrt(1.0 - way.z * way.z), way.z);
        let wanted = cruise * (1.0 + 0.8 * fright) + clamp(dot(steer, heading) * 0.5, -0.25 * cruise, 0.25 * cruise);
        let resisting = gravity * way.z + drag * speed * speed;
        effort = clamp((resisting + (wanted - speed) * 1.5) / maxThrust, 0.0, 1.0);
        let newSpeed = clamp(speed + (effort * maxThrust - resisting) * dt, 0.5 * cruise, 2.5 * cruise);
        vel = way * newSpeed;
      } else {
        let cruise = sim.c.y * (1.0 + 0.8 * fright);
        steer += heading * (cruise - length(myVel)) * 0.8;
        vel = myVel + steer * dt;
        let speed = length(vel);
        vel = select(heading, vel / max(speed, 1e-4), speed > 1e-4) * clamp(speed, sim.c.y * (1.0 - sim.c.z), cruise * (1.0 + sim.c.z));
      }
      var pos = myPos + vel * dt;
      if (walls > 0.0) {
        let lo = pen * homeSize + 0.02;
        let hi = (pen + 1.0) * homeSize - 0.02;
        let held = clamp(pos.xy, lo, hi);
        let slid = select(vel.xy, -0.25 * vel.xy, held != pos.xy);
        vel = vec3f(slid, vel.z);
        pos = vec3f(held, pos.z);
      }
      pos = vec3f(pos.xy - FISH_WORLD * floor(pos.xy / FISH_WORLD), clamp(pos.z, -FLOCK_DEPTH, FLOCK_DEPTH));
      let newSpeed = length(vel);
      let bodyLength = 0.525 * me.size;
      let beats = select((newSpeed / bodyLength + 1.0) / 0.75, mix(2.4, 4.2, effort), deep);
      let phase = fract((me.phase + 6.2832 * beats * dt) / 6.2832) * 6.2832;
      var turn = me.turn;
      if (dt > 1e-4) {
        let newHeading = vel.xy / max(length(vel.xy), 1e-4);
        let swing = (flatHeading.x * newHeading.y - flatHeading.y * newHeading.x) / dt;
        turn = mix(me.turn, swing, clamp(dt * 6.0, 0.0, 1.0));
      }
      let shown = mix(me.effort, effort, clamp(dt * 4.0, 0.0, 1.0));
      boidsOut[id.x] = Boid(pos.xy, vel.xy, phase, turn, me.size, select(me.lift, pos.z, deep), select(0.0, vel.z, deep), shown, 0.0, 0.0);
    }
  ),
  CODE(
    @compute @workgroup_size(64)
    fn updatePredators(@builtin(global_invocation_id) id : vec3u) {
      if (id.x >= min(u32(sim.f.x), PREDATOR_MAX)) { return; }
      var hunter = predators[id.x];
      let dt = sim.a.x;
      let speed = sim.g.x;
      hunter.age += dt;
      if (hunter.age >= hunter.life) {
        let seed = f32(id.x) * 37.719 + sim.a.y * 51.337 + 9.1;
        let r = fract(sin(vec4f(seed, seed + 1.618, seed + 2.718, seed + 3.142)) * vec4f(43758.5453, 22578.1459, 19642.3490, 31415.9265));
        let way = r.z * 6.2832;
        hunter = Predator(r.xy * FISH_WORLD, speed * vec2f(cos(way), sin(way)), 0.0, sim.f.w * mix(0.6, 1.4, r.w), 0.0, 0.0);
      } else {
        var toward = vec2f(0.0);
        let home = vec2i(floor(hunter.pos / FISH_CELL_SIZE));
        let reach = i32(ceil(PREDATOR_SENSE / FISH_CELL_SIZE));
        for (var dy = -reach; dy <= reach; dy++) {
          for (var dx = -reach; dx <= reach; dx++) {
            let c = ((home + vec2i(dx, dy)) % FISH_GRID + FISH_GRID) % FISH_GRID;
            let cell = u32(c.y * FISH_GRID + c.x);
            let count = min(atomicLoad(&cells[cell * CELL_STRIDE]), FISH_PER_CELL);
            for (var k = 0u; k < count; k++) {
              var d = boidsIn[atomicLoad(&cells[cell * CELL_STRIDE + 1u + k])].pos - hunter.pos;
              d -= FISH_WORLD * round(d / FISH_WORLD);
              let dist = length(d);
              if (dist < PREDATOR_SENSE) { toward += d / (1.0 + dist * dist); }
            }
          }
        }
        var apart = vec2f(0.0);
        let room = sim.d.w;
        if (room > 0.0) {
          let others = min(u32(sim.f.x), PREDATOR_MAX);
          for (var k = 0u; k < others; k++) {
            if (k == id.x) { continue; }
            var d = predators[k].pos - hunter.pos;
            d -= FISH_WORLD * round(d / FISH_WORLD);
            let dist = length(d);
            if (dist < room && dist > 1e-3) {
              let near = 1.0 - dist / room;
              apart -= d / dist * (near * near);
            }
          }
        }
        var want = apart * 3.0;
        if (dot(toward, toward) > 1e-8) { want += normalize(toward); }
        if (dot(want, want) > 1e-8) {
          hunter.vel += (normalize(want) * speed - hunter.vel) * clamp(dt * 1.5, 0.0, 1.0);
        }
        hunter.vel = hunter.vel / max(length(hunter.vel), 1e-4) * speed;
        hunter.pos += hunter.vel * dt;
        hunter.pos -= FISH_WORLD * floor(hunter.pos / FISH_WORLD);
      }
      predators[id.x] = hunter;
    }
    @compute @workgroup_size(64)
    fn updateLures(@builtin(global_invocation_id) id : vec3u) {
      if (id.x >= min(u32(sim.e.x), LURE_MAX)) { return; }
      var lure = lures[id.x];
      lure.age += sim.a.x;
      var claimed = false;
      let home = vec2i(floor(lure.pos / FISH_CELL_SIZE));
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let c = (home + vec2i(dx, dy) + FISH_GRID) % FISH_GRID;
          let cell = u32(c.y * FISH_GRID + c.x);
          let count = min(atomicLoad(&cells[cell * CELL_STRIDE]), FISH_PER_CELL);
          for (var k = 0u; k < count; k++) {
            var d = boidsIn[atomicLoad(&cells[cell * CELL_STRIDE + 1u + k])].pos - lure.pos;
            d -= FISH_WORLD * round(d / FISH_WORLD);
            if (dot(d, d) < CLAIM_DISTANCE * CLAIM_DISTANCE) { claimed = true; }
          }
        }
      }
      if (claimed || lure.age >= lure.life) {
        let seed = f32(id.x) * 12.9898 + sim.a.y * 78.233;
        let r = fract(sin(vec3f(seed, seed + 1.618, seed + 2.718)) * vec3f(43758.5453, 22578.1459, 19642.3490));
        lure = Lure(r.xy * FISH_WORLD, 0.0, sim.e.w * mix(0.6, 1.4, r.z));
      }
      lures[id.x] = lure;
      let lc = clamp(vec2u(floor(lure.pos)), vec2u(0u), vec2u(LURE_GRID - 1u));
      lureCells[lc.y * LURE_GRID + lc.x] = id.x + 1u;
    }
  ),
  NULL
};

/* The scene shader. One module holds every pass: the cloud shadow map, the
 * seabed cache fills, the clouds, the sea, and the final composition. The
 * uniform struct has one vec4 per row of the example's update_uniforms(). */
static const char* painted_ocean_wgsl_part[MAIN_WGSL_MAX_PARTS] = {
  CODE(
    struct Uniforms {
      a : vec4f,
      b : vec4f,
      c : vec4f,
      d : vec4f,
      e : vec4f,
      f : vec4f,
      g : vec4f,
      h : vec4f,
      i : vec4f,
      j : vec4f,
      k : vec4f,
      l : vec4f,
      m : vec4f,
      n : vec4f,
      o : vec4f,
      p : vec4f,
      q : vec4f,
      r : vec4f,
      s : vec4f,
      v : vec4f,
      w : vec4f,
      y : vec4f,
      x : vec4f,
      trains : array<vec4f, 56>,
    }
    @group(0) @binding(0) var<uniform> u : Uniforms;
    @group(1) @binding(0) var shadowTex : texture_2d<f32>;
    @group(1) @binding(1) var linearSamp : sampler;
    @group(1) @binding(2) var cloudTex : texture_2d<f32>;
    struct Boid {
      pos : vec2f,
      vel : vec2f,
      phase : f32,
      turn : f32,
      size : f32,
      lift : f32,
      climb : f32,
      effort : f32,
      pad1 : f32,
      pad2 : f32,
    }
    @group(1) @binding(3) var<storage, read> boids : array<Boid>;
    @group(1) @binding(4) var<storage, read> fishCells : array<u32>;
    const FISH_STRIDE : u32 = FISH_PER_CELL + 1u;
    struct Lure {
      pos : vec2f,
      age : f32,
      life : f32,
    }
    @group(1) @binding(5) var<storage, read> lures : array<Lure>;
    @group(1) @binding(6) var<storage, read> lureCells : array<u32>;
    struct Predator {
      pos : vec2f,
      vel : vec2f,
      age : f32,
      life : f32,
      pad0 : f32,
      pad1 : f32,
    }
    @group(1) @binding(7) var<storage, read> predators : array<Predator>;
    @group(1) @binding(8) var<storage, read> birds : array<Boid>;
    @group(1) @binding(9) var<storage, read> birdCells : array<u32>;
    @group(1) @binding(10) var cloudNearTex : texture_2d<f32>;
    @group(1) @binding(11) var bedTex : texture_2d<f32>;
    @group(1) @binding(13) var bedSandTex : texture_2d<f32>;
    @group(1) @binding(14) var farRimTex : texture_2d<f32>;
    @group(1) @binding(12) var seaTex : texture_2d<f32>;
    var<private> pixelScale : f32 = 1.0;
    var<private> viewFocus : f32 = 0.0;
    var<private> pixelAcross : f32 = 1.0;
    const BIRD_STRIDE : u32 = BIRD_PER_CELL + 1u;
    const BOUND_R : f32 = 3.0;
    const LIGHT_STEPS : i32 = 6;
    const LIGHT_STEP : f32 = 0.25;
    const HAZE : f32 = 0.004;
    const HAZE_LIMIT : f32 = 600.0;
  ),
  CODE(
    fn hash13(p : vec3f) -> f32 {
      var q = fract(p * 0.3183099 + vec3f(0.1, 0.17, 0.13));
      q = q * 17.0;
      return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
    }
    fn vnoiseSlope(x : vec3f) -> vec3f {
      let i = floor(x);
      let f = fract(x);
      let w = f * f * (3.0 - 2.0 * f);
      let dw = 6.0 * f * (1.0 - f);
      let a = hash13(i);
      let b = hash13(i + vec3f(1.0, 0.0, 0.0));
      let c = hash13(i + vec3f(0.0, 1.0, 0.0));
      let d = hash13(i + vec3f(1.0, 1.0, 0.0));
      let e = hash13(i + vec3f(0.0, 0.0, 1.0));
      let g = hash13(i + vec3f(1.0, 0.0, 1.0));
      let j = hash13(i + vec3f(0.0, 1.0, 1.0));
      let k = hash13(i + vec3f(1.0, 1.0, 1.0));
      let x00 = mix(a, b, w.x);
      let x10 = mix(c, d, w.x);
      let x01 = mix(e, g, w.x);
      let x11 = mix(j, k, w.x);
      return vec3f(mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z),
                   mix(mix(b - a, d - c, w.y), mix(g - e, k - j, w.y), w.z) * dw.x,
                   mix(x10 - x00, x11 - x01, w.z) * dw.y);
    }
    fn vnoise(x : vec3f) -> f32 {
      let i = floor(x);
      let f = fract(x);
      let w = f * f * (3.0 - 2.0 * f);
      return mix(
        mix(mix(hash13(i + vec3f(0.0, 0.0, 0.0)), hash13(i + vec3f(1.0, 0.0, 0.0)), w.x),
            mix(hash13(i + vec3f(0.0, 1.0, 0.0)), hash13(i + vec3f(1.0, 1.0, 0.0)), w.x), w.y),
        mix(mix(hash13(i + vec3f(0.0, 0.0, 1.0)), hash13(i + vec3f(1.0, 0.0, 1.0)), w.x),
            mix(hash13(i + vec3f(0.0, 1.0, 1.0)), hash13(i + vec3f(1.0, 1.0, 1.0)), w.x), w.y),
        w.z);
    }
    fn fbm(p : vec3f, octaves : i32) -> f32 {
      var q = p;
      var amp = 0.5;
      var sum = 0.0;
      for (var i = 0; i < octaves; i++) {
        sum += amp * vnoise(q);
        q = q * 2.02 + vec3f(7.1, 3.3, 5.7);
        amp *= 0.5;
      }
      return sum;
    }
  ),
  CODE(
    fn cloudSpacing() -> f32 {
      return select(u.p.w, u.i.z, u.p.z > 0.5);
    }
    fn cloudCell(xz : vec2f) -> vec2f {
      let moved = xz - u.n.xy;
      if (u.p.z > 0.5) { return floor(vec2f(dot(u.k.xy, moved), dot(u.k.zw, moved))); }
      let v = moved / u.p.w;
      let b = v.y / 0.8660254;
      return floor(vec2f(v.x - 0.5 * b, b));
    }
    fn cloudHalfCell() -> vec2f {
      return u.p.w * vec2f(0.75, 0.4330127);
    }
    fn cloudAt(c : vec2f) -> CloudSite {
      if (u.p.z > 0.5) { return cloudSite(vec4f(u.j.xy * c.x + u.j.zw * c.y, c)); }
      return cloudSite(vec4f(u.p.w * (c.x + 0.5 * c.y), u.p.w * 0.8660254 * c.y, c));
    }
    fn cloudEdge(spot : vec2f) -> f32 {
      let q = spot / (4.0 * u.p.w);
      let morph = u.n.w * 0.2;
      let n = 0.62 * vnoise(vec3f(q.x, morph, q.y)) + 0.38 * vnoise(vec3f(q.x * 2.3 + 7.0, morph * 1.3 + 3.0, q.y * 2.3));
      return (n - mix(0.8, 0.42, u.c.w)) / 0.12;
    }
    const CLUSTER_BASE : f32 = 1.2;
    fn siteRandom(c : vec2f) -> vec3f {
      return vec3f(hash13(vec3f(c.x, c.y, 1.7)), hash13(vec3f(c.y, 4.3, c.x)), hash13(vec3f(9.1, c.x, c.y)));
    }
    struct CloudSite {
      centre : vec3f,
      scale : f32,
      seed : f32,
      cover : f32,
      present : bool,
      spin : f32,
    }
    const CLOUD_MAX_RADIUS : f32 = 4.05;
  ),
  CODE(
    fn cloudSite(site : vec4f) -> CloudSite {
      let at = site.xy + u.n.xy;
      if (u.p.z < 0.5) {
        let spacing = u.p.w;
        let r = siteRandom(site.zw + 517.0);
        let q = siteRandom(site.zw + 91.0);
        let bearing = q.y * 6.2832;
        let offset = 0.4 * spacing * sqrt(q.x) * vec2f(cos(bearing), sin(bearing));
        let edge = cloudEdge(site.xy + offset);
        let size = clamp(edge / 1.4, 0.0, 1.0);
        let scale = spacing * (0.06 + 0.223 * r.y * r.y * (0.35 + 0.65 * size));
        let formed = smoothstep(0.0, 0.6, edge);
        return CloudSite(vec3f(at.x + offset.x, CLUSTER_BASE + 0.7 * scale, at.y + offset.y), scale,
                         13.0 + r.x * 97.0 + r.y * 31.0, mix(-1.9, (size - 0.6) * 0.2, formed), edge > 0.0, q.z * 6.2832);
      }
      if (all(site.zw == vec2f(0.0))) {
        return CloudSite(vec3f(at.x, 0.0, at.y), 1.0, 0.0, 0.0, true, 0.0);
      }
      let r = siteRandom(site.zw);
      let q = siteRandom(site.zw + 91.0);
      let slack = max(0.5 * u.i.z - CLOUD_MAX_RADIUS, 0.0);
      let bearing = q.y * 6.2832 + u.n.z * (q.z - 0.5) * 2.0;
      let out = sqrt(q.x) * (0.75 + 0.25 * sin(u.n.z * 0.7 * (0.5 + q.x) + q.z * 6.2832));
      let offset = u.m.w * slack * out * vec2f(cos(bearing), sin(bearing));
      let swell = u.o.x * 0.35 * sin(u.n.w * (0.6 + 0.8 * r.x) + r.y * 6.2832);
      return CloudSite(vec3f(at.x + offset.x, mix(-0.5, 3.0, r.z), at.y + offset.y), mix(0.75, 1.35, r.y),
                       13.0 + r.x * 97.0 + r.y * 31.0, (r.z - 0.5) * 0.25 + swell, r.x > 0.17, q.z * 6.2832);
    }
    const CLOUD_LAYER_HIGH : f32 = 7.3;
    fn cloudDensityAt(pIn : vec3f, seed : f32, cover : f32, octaves : i32, spin : f32) -> f32 {
      let r = length(pIn);
      if (r > BOUND_R) { return 0.0; }
      let p = vec3f(pIn.x * cos(spin) - pIn.z * sin(spin), pIn.y, pIn.x * sin(spin) + pIn.z * cos(spin));
      var shape = 1.0 - length(p * vec3f(0.55, 1.2, 0.7));
      shape -= (1.0 - smoothstep(-0.8, -0.25, p.y)) * 0.9;
      let bias = shape + (u.c.w + cover - 0.5);
      if (bias + 1.2 * u.d.y < 0.0) { return 0.0; }
      let drift = vec3f(u.a.z, -0.2 * u.a.z, 0.35 * u.a.z);
      let n = fbm(p * 1.3 + drift + vec3f(seed), octaves);
      let d = bias + (n - 0.5) * 2.4 * u.d.y;
      let edge = 1.0 - smoothstep(BOUND_R - 0.5, BOUND_R, r);
      return clamp(d * 1.5, 0.0, 1.0) * edge;
    }
  ),
  CODE(
    fn boundHit(ro : vec3f, rd : vec3f) -> vec2f {
      let bq = dot(ro, rd);
      let cq = dot(ro, ro) - BOUND_R * BOUND_R;
      let disc = bq * bq - cq;
      if (disc <= 0.0) { return vec2f(1.0, -1.0); }
      let sq = sqrt(disc);
      return vec2f(max(-bq - sq, 0.0), -bq + sq);
    }
    fn cloudShadow(p : vec3f, jitter : f32) -> f32 {
      let sunDir = u.c.xyz;
      let hit = boundHit(p, sunDir);
      if (hit.y <= hit.x) { return 1.0; }
      let n = 18;
      let dt = (hit.y - hit.x) / f32(n);
      var depth = 0.0;
      for (var i = 0; i < n; i++) {
        depth += cloudDensityAt(p + sunDir * (hit.x + dt * (f32(i) + jitter)), 0.0, 0.0, 3, 0.0);
      }
      return exp(-depth * dt * u.d.x * 0.6);
    }
    fn sunRight() -> vec3f {
      let s = u.c.xyz;
      let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(s.y) > 0.9);
      return normalize(cross(s, helper));
    }
    fn sunUp() -> vec3f {
      return cross(u.c.xyz, sunRight());
    }
    struct ShadeSites {
      site : array<CloudSite, 9>,
      count : i32,
    }
    fn cloudShadowSites(p : vec3f) -> ShadeSites {
      var found : ShadeSites;
      found.count = 0;
      let sunDir = u.c.xyz;
      let scattered = u.p.z < 0.5;
      let cloudsAt = select(0.0, CLUSTER_BASE + 0.8, scattered);
      let up = p + sunDir * (max(cloudsAt - p.y, 0.0) / max(sunDir.y, 0.05));
      let base = cloudCell(up.xz + select(vec2f(0.0), cloudHalfCell(), scattered));
      let first = select(0, -1, scattered);
      for (var j = first; j < 2; j++) {
        for (var i = first; i < 2; i++) {
          let site = cloudAt(base + vec2f(f32(i), f32(j)));
          if (!site.present) { continue; }
          found.site[found.count] = site;
          found.count++;
        }
      }
      return found;
    }
    fn cloudShadowFrom(found : ShadeSites, p : vec3f) -> f32 {
      var light = 1.0;
      for (var n = 0; n < found.count; n++) {
        let site = found.site[n];
        let local = (p - site.centre) / site.scale;
        let uv = vec2f(dot(local, sunRight()), dot(local, sunUp())) / BOUND_R * 0.5 + 0.5;
        if (any(abs(uv - 0.5) > vec2f(0.5))) { continue; }
        light *= mix(1.0, textureSampleLevel(shadowTex, linearSamp, uv, 0.0).r, smoothstep(-0.4, -0.1, site.cover));
      }
      return light;
    }
    fn cloudShadowFromMap(p : vec3f) -> f32 {
      if (u.p.x < 0.5) { return 1.0; }
      return cloudShadowFrom(cloudShadowSites(p), p);
    }
    var<private> seaShadowAtSurface : f32 = 1.0;
  ),
  CODE(
    fn hg(mu : f32, g : f32) -> f32 {
      let g2 = g * g;
      return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5);
    }
    fn sunColor() -> vec3f {
      let h = smoothstep(0.0, 0.5, u.c.y);
      return mix(vec3f(1.0, 0.42, 0.16), vec3f(1.0, 0.95, 0.86), h) * 2.4;
    }
    fn skyBase(rd : vec3f) -> vec3f {
      let h = smoothstep(0.0, 0.5, u.c.y);
      let zenith = mix(vec3f(0.05, 0.08, 0.22), vec3f(0.05, 0.16, 0.48), h);
      let horizon = mix(vec3f(0.85, 0.38, 0.16), vec3f(0.32, 0.50, 0.78), h);
      return mix(horizon, zenith, pow(max(rd.y, 0.0), 0.55));
    }
    fn skyGlow(rd : vec3f) -> vec3f {
      let mu = max(dot(rd, u.c.xyz), 0.0);
      return skyBase(rd) + sunColor() * pow(mu, 24.0) * 0.18;
    }
    const HORIZON_WALL : f32 = 900.0;
    const CIRRUS_HEIGHT : f32 = 500.0;
    fn horizonClouds(dir : vec3f, dip : f32, sunDir : vec3f, viewerXZ : vec2f) -> vec4f {
      let amount = u.i.x;
      let el = asin(clamp(dir.y, -1.0, 1.0)) + dip;
      if (amount <= 0.0 || el <= 0.0) { return vec4f(0.0); }
      let aim = normalize(dir.xz + vec2f(1e-6, 0.0));
      let rd = vec3f(aim.x * cos(el), sin(min(el, 1.5707)), aim.y * cos(el));
      let drift = u.g.w * 0.0015;
      let c = viewerXZ * min(1.0, 0.9 * HORIZON_WALL / max(length(viewerXZ), 1e-3));
      let b = dot(c, aim);
      let reach = (-b + sqrt(b * b - dot(c, c) + HORIZON_WALL * HORIZON_WALL)) / HORIZON_WALL;
      let ring = c / HORIZON_WALL + aim * reach;
      let up = tan(min(el, 1.4)) * reach;
      let band = smoothstep(0.012, 0.03, up) * (1.0 - smoothstep(0.05, 0.17, up));
      let pc = vec3f(ring.x * 14.0 + drift, up * 40.0, ring.y * 14.0);
      let n = fbm(pc, 4);
      let cover = clamp((n - (1.0 - 0.48 * band * (0.6 + 0.8 * amount))) * 5.0, 0.0, 1.0);
      let sunRing = normalize(sunDir.xz + vec2f(1e-4, 0.0));
      let toward = sunRing - ring * dot(sunRing, ring);
      let n2 = fbm(pc + vec3f(toward.x * 0.3, 0.3, toward.y * 0.3), 3);
      let lit = clamp(0.65 + (n - n2) * 2.5, 0.3, 1.0);
      let cumulus = mix(sunColor() * 0.42 * lit + skyAmbient() * 0.9, skyBase(rd), 0.5);
      let a1 = cover * 0.9;
      let sheet = rd.xz / (rd.y + 0.08) + viewerXZ / CIRRUS_HEIGHT;
      let streak = fbm(vec3f(sheet.x * 0.35 + drift * 3.0, 4.0, sheet.y * 1.1), 4);
      let cirrusBand = smoothstep(0.06, 0.2, el) * (1.0 - smoothstep(0.35, 0.7, el));
      let a2 = smoothstep(0.5, 0.78, streak) * 0.3 * cirrusBand * min(amount * 1.5, 1.0);
      let cirrus = sunColor() * 0.4 + skyAmbient();
      return vec4f(cumulus * a1 + cirrus * a2 * (1.0 - a1), a1 + a2 * (1.0 - a1));
    }
  ),
  CODE(
    fn ign(px : vec2f) -> f32 {
      return fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
    }
    fn tonemap(x : vec3f) -> vec3f {
      let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
      return pow(clamp(y, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
    }
    fn camForward() -> vec3f {
      return normalize(vec3f(0.0, u.v.x, 0.0) - u.b.xyz);
    }
    fn viewRay(px : vec2f) -> vec3f {
      let res = u.a.xy;
      var uv = (px - 0.5 * res) / res.y;
      uv.y = -uv.y;
      let fw = camForward();
      let rt = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)));
      let up = cross(rt, fw);
      return normalize(fw * u.b.w + rt * uv.x + up * uv.y);
    }
    @fragment
    fn fs_shadow(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      if (u.p.x < 0.5) { return vec4f(1.0, 0.0, 0.0, 1.0); }
      let uv = frag.xy / u.f.x * 2.0 - 1.0;
      let origin = (sunRight() * uv.x + sunUp() * uv.y) * BOUND_R - u.c.xyz * BOUND_R;
      return vec4f(cloudShadow(origin, ign(frag.xy)), 0.0, 0.0, 1.0);
    }
    @vertex
    fn vs_full(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
      var tri = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
      return vec4f(tri[vi], 0.0, 1.0);
    }
    struct WaterHit {
      t : f32,
      surf : vec3f,
      rd : vec3f,
      world : vec3f,
      up : vec3f,
    }
    fn toLocalUp(v : vec3f, n : vec3f) -> vec3f {
      let w = vec3f(-n.z, 0.0, n.x);
      return v + cross(w, v) + cross(w, cross(w, v)) / max(1.0 + n.y, 1e-3);
    }
    fn fromLocalUp(v : vec3f, n : vec3f) -> vec3f {
      let w = vec3f(n.z, 0.0, -n.x);
      return v + cross(w, v) + cross(w, cross(w, v)) / max(1.0 + n.y, 1e-3);
    }
  ),
  CODE(
    fn worldToMap(p : vec3f) -> vec3f {
      let radius = u.i.y;
      let h = p.y - u.g.x;
      let chord = length(p.xz);
      let arc = radius * atan2(chord, h + radius);
      let q = chord * chord + h * h + 2.0 * radius * h;
      let altitude = q / (sqrt(radius * radius + q) + radius);
      return vec3f(p.x * arc / max(chord, 1e-6) + u.d.z, u.g.x + altitude, p.z * arc / max(chord, 1e-6) + u.d.w);
    }
    struct SiteFrame {
      centre : vec3f,
      up : vec3f,
    }
    fn siteFrame(mapPos : vec3f) -> SiteFrame {
      let radius = u.i.y;
      let fromTop = mapPos.xz - u.d.zw;
      let along = length(fromTop);
      let bearing = fromTop / max(along, 1e-6);
      let theta = along / radius;
      let altitude = mapPos.y - u.g.x;
      let up = vec3f(sin(theta) * bearing.x, cos(theta), sin(theta) * bearing.y);
      let half = sin(0.5 * theta);
      return SiteFrame(vec3f(up.x * (radius + altitude), u.g.x + altitude * up.y - 2.0 * radius * half * half, up.z * (radius + altitude)), up);
    }
    fn waterHit(ro : vec3f, rd : vec3f) -> WaterHit {
      let radius = u.i.y;
      let h = ro.y - u.g.x;
      let b = ro.x * rd.x + ro.z * rd.z + (h + radius) * rd.y;
      let c = ro.x * ro.x + ro.z * ro.z + h * h + 2.0 * radius * h;
      let disc = b * b - c;
      if (b >= 0.0 || disc < 0.0 || c <= 0.0) { return WaterHit(1e9, vec3f(0.0), rd, vec3f(0.0), vec3f(0.0, 1.0, 0.0)); }
      let t = c / (-b + sqrt(disc));
      let p = ro + rd * t;
      let n = vec3f(p.x / radius, 1.0 + (p.y - u.g.x) / radius, p.z / radius);
      let local = normalize(toLocalUp(rd, n));
      let m = worldToMap(p);
      return WaterHit(t, vec3f(m.x, u.g.x, m.z), vec3f(local.x, min(local.y, -1e-4), local.z), p, n);
    }
    struct SkyView {
      dir : vec3f,
      dip : f32,
      color : vec3f,
    }
    fn skyView(ro : vec3f, rd : vec3f) -> SkyView {
      let radius = u.i.y;
      let upHere = normalize(vec3f(ro.x, ro.y - u.g.x + radius, ro.z));
      let h = max(worldToMap(ro).y - u.g.x, 0.0);
      let tangentSq = h * (2.0 * radius + h);
      let edge = min(sqrt(tangentSq), 1.517 / HAZE);
      let dip = asin(clamp((tangentSq + edge * edge) / (2.0 * (radius + h) * max(edge, 1e-3)), 0.0, 1.0));
      let dir = toLocalUp(rd, upHere);
      let el = clamp(asin(clamp(dir.y, -1.0, 1.0)) + dip, 0.0, 1.5708);
      let mu = max(dot(rd, u.c.xyz), 0.0);
      return SkyView(dir, dip, skyBase(vec3f(0.0, sin(el), 0.0)) + sunColor() * pow(mu, 24.0) * 0.18);
    }
    @fragment
    fn fs_sea(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let fullPx = frag.xy / u.v.yz * u.a.xy;
      pixelScale = u.a.y / u.v.z;
      viewFocus = u.v.w * (1.0 - smoothstep(0.08, 0.32, abs(fullPx.y / u.a.y - 0.5)));
      let ro = u.b.xyz;
      let rd = viewRay(fullPx);
      let hit = waterHit(ro, rd);
      if (hit.t >= 1e8) { return vec4f(0.0); }
      return vec4f(shadeWater(hit.surf - hit.rd * hit.t, hit.rd, hit.t, skyView(ro, rd).color), 1.0);
    }
  ),
  CODE(
    fn seaColor(px : vec2f, ro : vec3f, rd : vec3f, t : f32, hazeCol : vec3f) -> vec3f {
      if (u.v.y >= u.a.x - 0.5) { return shadeWater(ro, rd, t, hazeCol); }
      let at = px / u.a.xy * u.v.yz - 0.5;
      let base = floor(at);
      let f = at - base;
      let last = vec2i(u.v.yz) - 1;
      var total = vec4f(0.0);
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
          let texel = textureLoad(seaTex, clamp(vec2i(base) + vec2i(i, j), vec2i(0), last), 0);
          let weight = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1) * texel.a;
          total += vec4f(texel.rgb * weight, weight);
        }
      }
      if (total.a < 1e-3) { return shadeWater(ro, rd, t, hazeCol); }
      return total.rgb / total.a;
    }
    fn backdrop(px : vec2f, ro : vec3f, rd : vec3f) -> vec3f {
      let view = skyView(ro, rd);
      let far = horizonClouds(view.dir, view.dip, toLocalUp(u.c.xyz, normalize(vec3f(ro.x, ro.y - u.g.x + u.i.y, ro.z))), ro.xz);
      let hit = waterHit(ro, rd);
      if (hit.t < 1e8) {
        let sea = seaColor(px, hit.surf - hit.rd * hit.t, hit.rd, hit.t, view.color);
        let f = min(hit.t, HAZE_LIMIT) * HAZE;
        let hazed = 1.0 - exp(-f * f);
        return sea * (1.0 - far.a * hazed) + far.rgb * hazed;
      }
      let mu = max(dot(rd, u.c.xyz), 0.0);
      let sky = view.color + sunColor() * pow(mu, 3000.0) * 8.0;
      return sky * (1.0 - far.a) + far.rgb;
    }
    const PI : f32 = 3.14159265;
    const WATER_ETA : f32 = 0.752;
    const WATER_ABSORB : vec3f = vec3f(0.35, 0.05, 0.02);
    const WATER_SCATTER : vec3f = vec3f(0.004, 0.010, 0.016);
    const SAND_ALBEDO : vec3f = vec3f(0.85, 0.80, 0.65);
    fn skyAmbient() -> vec3f {
      let skyTop = skyBase(vec3f(0.0, 1.0, 0.0));
      return mix(skyTop, vec3f(dot(skyTop, vec3f(0.2126, 0.7152, 0.0722))), 0.5) * 0.7;
    }
    const WAVE_COUNT : i32 = 28;
    const WIND_ANGLE : f32 = 0.35;
    const SWELL_ANGLE : f32 = 2.3;
    const HEIGHT_WAVES : i32 = 8;
    struct Wave {
      slope : vec2f,
      lost : f32,
      height : f32,
      curve : vec3f,
      peak : f32,
    }
  ),
  CODE(
    fn waveField(xz : vec2f, footprint : f32, count : i32, wantCurve : bool) -> Wave {
      if (u.h.x <= 0.0) { return Wave(vec2f(0.0), 0.0, 0.0, vec3f(0.0), 0.0); }
      let time = u.g.w;
      let randomness = u.o.w;
      let steep = 0.06 * u.h.x * (1.0 - 0.5 * randomness);
      let windDir = vec2f(cos(WIND_ANGLE), sin(WIND_ANGLE));
      let adv = xz - windDir * time * 1.5;
      let gust = 0.6 * vnoise(vec3f(adv * 0.06, time * 0.05)) + 0.4 * vnoise(vec3f(adv * 0.017 + 9.0, time * 0.03));
      let ripple = mix(0.25, 1.7, smoothstep(0.3, 0.7, gust));
      let sets = mix(0.55, 1.45, vnoise(vec3f(xz * 0.02 + 31.0, time * 0.08)));
      let sizeScale = u.o.z / 14.0;
      let warp = (vec2f(vnoise(vec3f(xz * 0.035 / sizeScale, 3.0)), vnoise(vec3f(xz * 0.035 / sizeScale, 11.0))) - 0.5) * 5.0 * sizeScale;
      let q = xz + warp;
      let fine = (vec2f(vnoise(vec3f(xz * 0.7 / sizeScale, time * 0.25)), vnoise(vec3f(xz * 0.7 / sizeScale + 17.0, time * 0.25 + 5.0))) - 0.5) * 0.6 * sizeScale;
      var slope = vec2f(0.0);
      var lost = 0.0;
      var height = 0.0;
      var curve = vec3f(0.0);
      var peak = 0.0;
      for (var i = 0; i < count; i++) {
        let a = u.trains[2 * i];
        let b = u.trains[2 * i + 1];
        let frac = f32(i) / f32(WAVE_COUNT - 1);
        let back = mix(0.25, 0.6, frac);
        let keep = clamp(b.w / (footprint * 4.0) - 0.5, 0.0, 1.0);
        let amp = steep * b.x * mix(sets, ripple, smoothstep(0.2, 0.6, frac));
        if (keep <= 0.0) { lost += 0.5 * amp * amp * (1.0 + back * back); continue; }
        let dir = a.xy;
        let k = a.z;
        let phase = k * dot(dir, q + fine * smoothstep(0.3, 0.8, frac)) + b.y;
        let fwd = phase - a.w * time;
        let rev = phase + a.w * time + b.z;
        let crest = sin(fwd) + back * sin(rev);
        slope += dir * amp * keep * (cos(fwd) + back * cos(rev));
        height += amp / k * keep * crest;
        curve -= vec3f(dir.x * dir.x, dir.y * dir.y, dir.x * dir.y) * (amp * k * keep * crest);
        peak += amp * crest * (1.0 - smoothstep(0.35, 0.7, frac));
        lost += 0.5 * amp * amp * (1.0 + back * back) * (1.0 - keep * keep);
      }
      if (count == WAVE_COUNT && randomness > 0.0) {
        var size = 0.45 * u.o.z;
        var chop = 0.06 * u.h.x * 2.6 * randomness * ripple;
        for (var layer = 0; layer < 3; layer++) {
          let keep = clamp(size / (footprint * 3.0) - 0.5, 0.0, 1.0);
          if (keep > 0.0) {
            let turned = WIND_ANGLE + 0.65 * f32(layer + 1);
            let ca = cos(turned);
            let sa = sin(turned);
            let moved = xz - windDir * time * 1.25 * sqrt(size);
            let p = vec3f((moved.x * ca + moved.y * sa) / size, (moved.y * ca - moved.x * sa) / size, time * 0.4 + 7.0 * f32(layer));
            let n = vnoiseSlope(p);
            let g = n.yz * 0.8;
            slope += vec2f(g.x * ca - g.y * sa, g.x * sa + g.y * ca) * (chop * keep);
            if (wantCurve) {
              let e = 0.2;
              let c = vec2f(vnoiseSlope(p + vec3f(e, 0.0, 0.0)).y - g.x, vnoiseSlope(p + vec3f(0.0, e, 0.0)).z - g.y) / e;
              curve += vec3f(c.x * ca * ca + c.y * sa * sa, c.x * sa * sa + c.y * ca * ca, (c.x - c.y) * ca * sa) * (chop * keep / size);
            }
          }
          lost += 0.4 * chop * chop * (1.0 - keep * keep);
          size *= 0.47;
          chop *= 0.85;
        }
      }
      return Wave(slope, lost, height, curve, peak);
    }
    fn waveSlopes(xz : vec2f, footprint : f32) -> vec3f {
      let w = waveField(xz, footprint, WAVE_COUNT, false);
      return vec3f(w.slope, w.lost);
    }
    fn caustic(w : Wave, sunPath : f32) -> f32 {
      let reach = 0.25 * sunPath;
      let area = (1.0 + reach * w.curve.x) * (1.0 + reach * w.curve.y) - reach * reach * w.curve.z * w.curve.z;
      let focus = 1.0 / sqrt(area * area + 0.04);
      let wash = reach * u.h.x * 14.0 / u.o.z;
      return mix(1.0, focus, 1.0 / (1.0 + wash * wash));
    }
  ),
  CODE(
    struct WaveSetting {
      warp : vec2f,
      sets : f32,
      ripple : f32,
    }
    fn waveSetting(xz : vec2f) -> WaveSetting {
      let time = u.g.w;
      let windDir = vec2f(cos(WIND_ANGLE), sin(WIND_ANGLE));
      let adv = xz - windDir * time * 1.5;
      let gust = 0.6 * vnoise(vec3f(adv * 0.06, time * 0.05)) + 0.4 * vnoise(vec3f(adv * 0.017 + 9.0, time * 0.03));
      let sizeScale = u.o.z / 14.0;
      return WaveSetting(
        (vec2f(vnoise(vec3f(xz * 0.035 / sizeScale, 3.0)), vnoise(vec3f(xz * 0.035 / sizeScale, 11.0))) - 0.5) * 5.0 * sizeScale,
        mix(0.55, 1.45, vnoise(vec3f(xz * 0.02 + 31.0, time * 0.08))),
        mix(0.25, 1.7, smoothstep(0.3, 0.7, gust)));
    }
    fn waveHeightQuick(xz : vec2f, footprint : f32, setting : WaveSetting) -> f32 {
      let time = u.g.w;
      let steep = 0.06 * u.h.x * (1.0 - 0.5 * u.o.w);
      let q = xz + setting.warp;
      var height = 0.0;
      for (var i = 0; i < HEIGHT_WAVES; i++) {
        let a = u.trains[2 * i];
        let b = u.trains[2 * i + 1];
        let frac = f32(i) / f32(WAVE_COUNT - 1);
        let keep = clamp(b.w / (footprint * 4.0) - 0.5, 0.0, 1.0);
        let amp = steep * b.x * mix(setting.sets, setting.ripple, smoothstep(0.2, 0.6, frac));
        let phase = a.z * dot(a.xy, q) + b.y;
        height += amp / a.z * keep * (sin(phase - a.w * time) + mix(0.25, 0.6, frac) * sin(phase + a.w * time + b.z));
      }
      return height;
    }
    fn aboveWaves(p : vec3f, footprint : f32, setting : WaveSetting) -> f32 {
      return p.y - u.g.x - waveHeightQuick(p.xz, footprint, setting);
    }
    fn waveHit(ro : vec3f, rd : vec3f, tPlane : f32, footprint : f32) -> f32 {
      let band = 0.9 * u.h.x * u.o.z / 14.0;
      let reach = band / max(-rd.y, 1e-3);
      if (reach < footprint) { return tPlane; }
      let setting = waveSetting((ro + rd * tPlane).xz);
      let t0 = max(tPlane - reach, 0.0);
      let t1 = tPlane + reach;
      var ta = t0;
      var ga = aboveWaves(ro + rd * t0, footprint, setting);
      if (ga <= 0.0) { return t0; }
      var tb = t1;
      var gb = min(aboveWaves(ro + rd * t1, footprint, setting), -1e-4);
      for (var i = 1; i < 8; i++) {
        let t = mix(t0, t1, f32(i) / 8.0);
        let g = aboveWaves(ro + rd * t, footprint, setting);
        if (g < 0.0) { tb = t; gb = g; break; }
        ta = t;
        ga = g;
      }
      for (var i = 0; i < 2; i++) {
        let t = mix(ta, tb, ga / (ga - gb));
        let g = aboveWaves(ro + rd * t, footprint, setting);
        if (g < 0.0) { tb = t; gb = g; } else { ta = t; ga = g; }
      }
      return mix(ta, tb, ga / (ga - gb));
    }
    fn waterNormal(slopes : vec3f, v : vec3f) -> vec3f {
      let n = normalize(vec3f(-slopes.x, 1.0, -slopes.y));
      return normalize(n + v * max(0.03 - dot(n, v), 0.0));
    }
    fn waterRough(lost : f32) -> f32 {
      let base = mix(0.18, 0.07, clamp(u.h.x * 5.0, 0.0, 1.0));
      return min(sqrt(sqrt(base * base * base * base + 2.0 * lost)), 0.6);
    }
  ),
  CODE(
    fn sandDepthAt(xz : vec2f) -> f32 {
      if (u.p.y < 0.5) { return 0.5 * (u.f.y + u.f.w); }
      let ocean = u.f.w;
      let top = min(u.f.y, ocean);
      let at = vec3f(xz.x * 0.5 / u.g.z, 5.3, xz.y * 0.5 / u.g.z);
      let broad = fbm(at, 5);
      let climb = (broad - mix(0.66, 0.38, u.w.y)) / 0.12;
      let up = clamp(climb, 0.0, 1.0);
      var base = mix(top, ocean, (1.0 - up) * (1.0 - up));
      if (climb > 1.0 && u.w.z > 0.0) {
        let h = 0.04;
        let here = fbm(at, 2);
        let slope = vec2f(fbm(at + vec3f(h, 0.0, 0.0), 2), fbm(at + vec3f(0.0, 0.0, h), 2)) - here;
        let rate = sqrt(dot(slope, slope) / (h * h) + 0.04) * 0.5 / u.g.z;
        let inward = (climb - 1.0) * 0.12 / rate;
        let t = fract(inward / u.w.w);
        base += u.w.z * smoothstep(0.0, 0.8, t) * (1.0 - smoothstep(0.8, 1.0, t));
      }
      if (u.l.x > 0.0 && up > 0.0) {
        let slow = fbm(vec3f(xz.x / u.y.x, 11.9, xz.y / u.y.x), 2) / 0.75;
        base += u.l.x * up * smoothstep(0.55, 0.9, slow);
      }
      let n = fbm(vec3f(xz.x * 0.07, 1.7, xz.y * 0.07), 5);
      let shape = mix(0.45, -0.45, smoothstep(0.3, 0.7, n));
      let bumpy = base + u.f.z * shape * min(base - top, 5.0);
      if (u.i.w <= 0.0) { return bumpy; }
      let across = vec2f(cos(WIND_ANGLE), sin(WIND_ANGLE));
      let phase = dot(across, xz) * 1.25 + 7.0 * vnoise(vec3f(xz * 0.11, 8.0));
      let patches = smoothstep(0.3, 0.6, vnoise(vec3f(xz * 0.045, 20.0)));
      let ridge = sin(phase) + 0.3 * sin(2.0 * phase + 1.57);
      let tall = min(u.i.w * 0.2 * exp(-base / SAND_WAVE_FADE), 0.07 * base);
      return bumpy - tall * patches * ridge;
    }
    const SAND_WAVE_FADE : f32 = 5.0;
  ),
  CODE(
    fn coralDeepest() -> f32 { return u.s.x + u.s.y; }
    fn coralInto(xz : vec2f, sandDepth : f32, fromRim : f32) -> f32 {
      let soft = clamp(0.2 * u.s.y, 0.03, 0.6);
      let band = smoothstep(u.s.x, u.s.x + soft, sandDepth) * (1.0 - smoothstep(coralDeepest() - soft, coralDeepest(), sandDepth));
      let n = fbm(vec3f(xz.x * 0.34, 23.7, xz.y * 0.34), 4);
      let ring = ((n - mix(0.75, 0.3, u.e.x)) / 0.16 + 0.4) * band - 0.2 * (1.0 - band);
      if (u.w.x <= 0.0 || sandDepth >= u.s.x + soft || fromRim >= u.w.x) { return ring; }
      let fade = clamp(1.0 - fromRim / u.w.x, 0.0, 1.0);
      return max(ring, coralClumps(xz, fade));
    }
    fn coralClumps(xz : vec2f, fade : f32) -> f32 {
      let g = xz / CLUMP_GRID;
      let cell = floor(g);
      var best = -0.2;
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = cell + vec2f(f32(i), f32(j));
          let r = siteRandom(c + 310.0);
          let radius = CLUMP_GRID * 0.55 * clamp(fade * 1.25 - r.z, 0.0, 1.0);
          if (radius < 0.06) { continue; }
          let d = length(g - (c + 0.15 + 0.7 * r.xy)) * CLUMP_GRID;
          best = max(best, (radius - d) / 0.12);
        }
      }
      return best;
    }
    const CLUMP_GRID : f32 = 0.8;
    fn coralAt(xz : vec2f, waterDepth : f32, footprint : f32, fromRim : f32) -> vec3f {
      if (u.e.x <= 0.0 || waterDepth >= coralDeepest() || (waterDepth <= u.s.x && fromRim >= u.w.x)) { return vec3f(0.0); }
      let core = clamp(coralInto(xz, waterDepth, fromRim), 0.0, 1.0);
      if (core <= 0.0) { return vec3f(0.0); }
      let middle = 0.0;
      let rise = min(core * 3.0, 1.0);
      let detail = clamp(0.12 / footprint - 0.3, 0.0, 1.0);
      var heads = 0.6;
      if (detail > 0.0) { heads = mix(0.6, coralDomes(xz, BALL_SIZE, 300.0, 0.0), detail); }
      let height = core * CORAL_BODY + rise * CORAL_HEAD * heads;
      let room = max(0.5 * (waterDepth - (WATER_OVER_CORAL + 0.08)), 0.05);
      return vec3f(core, room * (1.0 - exp(-height / room)), middle);
    }
    const CORAL_BODY : f32 = 0.45;
    const CORAL_HEAD : f32 = 0.2;
    const BALL_SIZE : f32 = 0.34;
    fn coralDomes(xz : vec2f, size : f32, seed : f32, uneven : f32) -> f32 {
      let g = xz / size;
      let cell = floor(g);
      var top = 0.0;
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = cell + vec2f(f32(i), f32(j));
          let r = siteRandom(c + seed);
          let d = length(g - (c + 0.5 + (r.xy - 0.5) * 0.7)) / mix(0.55, 0.85, r.z);
          let tall = mix(1.0, mix(0.4, 1.3, fract(r.z * 7.31)), uneven);
          top = max(top, sqrt(max(1.0 - d * d, 0.0)) * tall);
        }
      }
      return top;
    }
  ),
  CODE(
    fn coralBumps(xz : vec2f) -> vec2f {
      let size = 0.05;
      let g = xz / size;
      let cell = floor(g);
      var top = 0.0;
      var slope = vec2f(0.0);
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = cell + vec2f(f32(i), f32(j));
          let r = siteRandom(c + 900.0);
          let radius = mix(0.38, 1.05, r.z * r.z);
          let d = (g - (c + 0.5 + (r.xy - 0.5) * 0.8)) / radius;
          let dome = sqrt(max(1.0 - dot(d, d), 0.0));
          if (dome * radius > top) {
            top = dome * radius;
            slope = -d / max(dome, 0.3);
          }
        }
      }
      return slope * 0.22;
    }
    const BRANCH_SIZE : f32 = 0.5;
    const BRANCH_CELL : f32 = 0.24;
    const BRANCH_TOP : f32 = 0.46;
    const WATER_OVER_CORAL : f32 = 0.04;
    fn branchRoom(headDepth : f32) -> f32 {
      return clamp(headDepth - 0.08 - WATER_OVER_CORAL, 0.04, BRANCH_TOP);
    }
    fn branchHash(p0 : vec2f) -> f32 {
      var p = fract(p0 * vec2f(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }
    fn branchTree(p0 : vec3f, fork : vec2f, turn : vec2f, thick : f32, bushy : bool) -> f32 {
      var p = vec3f(turn.x * p0.x - turn.y * p0.z, p0.y, turn.y * p0.x + turn.x * p0.z);
      var d = 1e5;
      var s = 1.0;
      for (var i = 0; i < 6; i++) {
        let tipRadius = select(0.02, 0.0, i == 5) * thick;
        let along = clamp(p.y / 0.3, 0.0, 1.0);
        d = min(d, (length(p - vec3f(0.0, along * 0.3, 0.0)) - mix(0.028 * thick, tipRadius, along)) * s);
        p.y -= 0.3;
        let twisted = vec2f(-0.3233 * p.x - 0.9463 * p.z, 0.9463 * p.x - 0.3233 * p.z);
        p = vec3f(abs(twisted.x), p.y, twisted.y);
        let tilted = vec2f(fork.x * p.x - fork.y * p.y, fork.y * p.x + fork.x * p.y);
        p = vec3f(tilted.x, tilted.y, p.z);
        if (bushy || i % 2 == 0) {
          let lean = normalize(vec2f(fork.x, 0.6 * fork.y));
          let leaned = vec2f(lean.x * abs(p.z) - lean.y * p.y, lean.y * abs(p.z) + lean.x * p.y);
          p = vec3f(p.x, leaned.y, leaned.x);
        }
        p *= 1.4;
        s /= 1.4;
      }
      return d;
    }
  ),
  CODE(
    fn coralUnder(xz : vec2f) -> f32 {
      let at = (xz - u.q.xy) / u.q.z + 0.5 * BED_SIZE;
      let inside = min(min(at.x, at.y), BED_SIZE - max(at.x, at.y));
      if (u.q.w < 0.5 || inside < 34.0) { return -2.0; }
      return textureSampleLevel(bedTex, linearSamp, at / BED_SIZE, 0.0).a;
    }
    var<private> branchKind : i32 = 0;
    var<private> branchNarrow : f32 = 1.0;
    var<private> branchShade : f32 = 0.5;
    fn branchScene(p : vec3f, share : f32) -> f32 {
      let forkOf = array<f32, 6>(0.45, 0.72, 0.58, 0.85, 0.34, 0.66);
      let thickOf = array<f32, 6>(1.0, 0.9, 1.6, 0.85, 0.7, 1.25);
      let sizeOf = array<f32, 6>(1.0, 0.75, 0.82, 0.62, 1.08, 0.8);
      let bushyOf = array<bool, 6>(false, true, false, true, false, true);
      let swayOf = array<f32, 6>(0.25, 0.8, 0.25, 0.5, 1.0, 1.0);
      let lean = u.e.w * 0.07 * (p.y / BRANCH_TOP) * (p.y / BRANCH_TOP);
      let beat = u.g.w * 1.1 + dot(p.xz, vec2f(0.35, 0.21));
      let slide = lean * vec2f(0.94 * sin(beat), 0.34 * sin(beat * 0.73 + 1.3));
      let base = floor(p.xz / BRANCH_CELL - 0.5);
      var d = 1e5;
      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          let c = base + vec2f(f32(i), f32(j));
          let r1 = branchHash(c);
          let r2 = branchHash(c + 17.3);
          if (branchHash(c + 41.7) > share) { continue; }
          let centre = (c + 0.5 + (vec2f(r1, r2) - 0.5) * 0.3) * BRANCH_CELL;
          let kind = coralKind(branchHash(c + 71.9));
          let size = BRANCH_SIZE * (0.7 + 0.3 * r2) * sizeOf[kind];
          let at = (p.xz - centre - slide * swayOf[kind]) / branchNarrow;
          let local = vec3f(at.x, p.y, at.y) / size;
          var dt = length(local - vec3f(0.0, 0.43, 0.0)) - 0.7;
          if (dt <= 0.25) {
            var under = coralUnder(centre);
            if (under > -1.5 && under < 0.02) {
              under = max(coralUnder(centre + vec2f(0.09, 0.0)), max(coralUnder(centre + vec2f(-0.045, 0.078)), coralUnder(centre + vec2f(-0.045, -0.078))));
              if (under < 0.02) { continue; }
            }
            let fork = forkOf[kind] * (0.85 + 0.3 * r1);
            let turn = r1 * 6.2832;
            dt = branchTree(local, vec2f(cos(fork), sin(fork)), vec2f(cos(turn), sin(turn)), thickOf[kind], bushyOf[kind]);
          }
          if (dt * size * branchNarrow < d) {
            d = dt * size * branchNarrow;
            branchKind = kind;
            branchShade = r2;
          }
        }
      }
      return d;
    }
  ),
  CODE(
    fn depthAt(xz : vec2f, footprint : f32) -> f32 {
      return bedLookup(xz, footprint).depth;
    }
    struct Bed {
      depth : f32,
      core : f32,
      head : f32,
      middle : f32,
    }
    fn bedFromFormulas(xz : vec2f, footprint : f32) -> Bed {
      let sand = sandDepthAt(xz);
      var fromRim = 0.0;
      if (sand < u.s.x) { fromRim = select(1e9, farRim(xz), u.e.x > 0.0 && u.w.x > 0.0); }
      let coral = coralAt(xz, sand, footprint, fromRim);
      let rise = min(coral.x * 3.0, 1.0) * CORAL_HEAD;
      return Bed(max(sand - coral.y, 0.05), coral.x, clamp((coral.y - coral.x * CORAL_BODY) / (rise + 1e-4), 0.0, 1.0), coral.z);
    }
    fn bedLookup(xz : vec2f, footprint : f32) -> Bed {
      if (u.p.y < 0.5) { return Bed(0.5 * (u.f.y + u.f.w), 0.0, 0.6, 0.0); }
      let at = (xz - u.q.xy) / u.q.z + 0.5 * BED_SIZE;
      let inside = min(min(at.x, at.y), BED_SIZE - max(at.x, at.y));
      let trust = clamp((inside - 2.0) / 32.0, 0.0, 1.0) * step(0.5, u.q.w);
      var cached = Bed(0.0, 0.0, 0.6, 0.0);
      if (trust > 0.0) {
        let s = textureSampleLevel(bedTex, linearSamp, at / BED_SIZE, 0.0);
        let detail = clamp(0.12 / footprint - 0.3, 0.0, 1.0);
        let core = clamp(s.a, 0.0, 1.0);
        let rise = min(core * 3.0, 1.0) * CORAL_HEAD;
        var head = 0.6;
        if (detail > 0.0 && rise > 0.0) { head = mix(0.6, coralDomes(xz, BALL_SIZE, 300.0, 0.0), detail); }
        cached = Bed(max(s.r + s.g - rise * (head - 0.6), 0.05), core, head, 0.0);
      }
      if (trust >= 1.0) { return cached; }
      let exact = bedFromFormulas(xz, footprint);
      return Bed(mix(exact.depth, cached.depth, trust), mix(exact.core, cached.core, trust), mix(exact.head, cached.head, trust), mix(exact.middle, cached.middle, trust));
    }
  ),
  CODE(
    fn rockHash(p : vec3f) -> vec3f {
      var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
      q += dot(q, q.yxz + 33.33);
      return fract((q.xxy + q.yxx) * q.zyx);
    }
    fn rockNoise(p : vec3f) -> f32 {
      let i = floor(p);
      let f = fract(p);
      let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
      let a = dot(rockHash(i) * 2.0 - 1.0, f);
      let b = dot(rockHash(i + vec3f(1.0, 0.0, 0.0)) * 2.0 - 1.0, f - vec3f(1.0, 0.0, 0.0));
      let c = dot(rockHash(i + vec3f(0.0, 1.0, 0.0)) * 2.0 - 1.0, f - vec3f(0.0, 1.0, 0.0));
      let d = dot(rockHash(i + vec3f(1.0, 1.0, 0.0)) * 2.0 - 1.0, f - vec3f(1.0, 1.0, 0.0));
      let e = dot(rockHash(i + vec3f(0.0, 0.0, 1.0)) * 2.0 - 1.0, f - vec3f(0.0, 0.0, 1.0));
      let g = dot(rockHash(i + vec3f(1.0, 0.0, 1.0)) * 2.0 - 1.0, f - vec3f(1.0, 0.0, 1.0));
      let h = dot(rockHash(i + vec3f(0.0, 1.0, 1.0)) * 2.0 - 1.0, f - vec3f(0.0, 1.0, 1.0));
      let k = dot(rockHash(i + vec3f(1.0, 1.0, 1.0)) * 2.0 - 1.0, f - vec3f(1.0, 1.0, 1.0));
      return mix(mix(mix(a, b, w.x), mix(c, d, w.x), w.y), mix(mix(e, g, w.x), mix(h, k, w.x), w.y), w.z);
    }
    fn rockCells(p : vec3f) -> vec2f {
      let i = floor(p);
      let f = fract(p);
      var d = vec2f(8.0);
      for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
          for (var z = -1; z <= 1; z++) {
            let o = vec3f(f32(x), f32(y), f32(z));
            let r = o + rockHash(i + o) - f;
            let dd = dot(r, r);
            if (dd < d.x) { d.y = d.x; d.x = dd; } else if (dd < d.y) { d.y = dd; }
          }
        }
      }
      return sqrt(d);
    }
    const ROCK_UNIT : f32 = 2.0;
    const ROCK_FREQ : f32 = 8.0;
    const ROCK_HURST : f32 = 1.0;
    const ROCK_AMP : f32 = 0.118;
    const ROCK_RIDGE : f32 = 0.1;
    const ROCK_CELLS : f32 = 0.12;
    const ROCK_CRACKS : f32 = 0.0;
    const ROCK_CELL_FREQ : f32 = 8.0;
    const ROCK_WARP : f32 = 0.5;
  ),
  CODE(
    fn rockWarp(p : vec3f) -> vec3f {
      let q = p * 0.9;
      return p + ROCK_WARP * vec3f(rockNoise(q), rockNoise(q + vec3f(31.4, 5.2, 11.7)), rockNoise(q + vec3f(8.3, 47.1, 23.9)));
    }
    fn rockHeight(p : vec3f, layers : i32, fine : bool) -> f32 {
      let gain = exp2(-ROCK_HURST);
      var q = p * ROCK_FREQ;
      var a = 1.0;
      var sum = vec2f(0.0);
      for (var i = 0; i < layers; i++) {
        let n = rockNoise(q);
        sum += a * vec2f(n, 0.5 - 2.0 * abs(n));
        q = q * 2.03 + vec3f(17.1, 3.7, 9.2);
        a *= gain;
      }
      sum *= 1.0 - gain;
      var h = ROCK_AMP * sum.x + ROCK_RIDGE * sum.y;
      if (fine) {
        let w = rockCells(p * ROCK_CELL_FREQ);
        h += ROCK_CELLS * (0.5 - w.x);
        if (ROCK_CRACKS > 0.0) { h -= ROCK_CRACKS * (1.0 - smoothstep(0.0, 0.07, w.y - w.x)); }
      }
      return h;
    }
    fn bedSteep(xz : vec2f) -> f32 {
      if (u.p.y < 0.5) { return 0.0; }
      let at = (xz - u.q.xy) / u.q.z + 0.5 * BED_SIZE;
      let inside = min(min(at.x, at.y), BED_SIZE - max(at.x, at.y));
      if (u.q.w > 0.5 && inside >= 34.0) { return textureSampleLevel(bedTex, linearSamp, at / BED_SIZE, 0.0).b; }
      let here = sandDepthAt(xz);
      return length(vec2f(sandDepthAt(xz + vec2f(1.0, 0.0)) - here, sandDepthAt(xz + vec2f(0.0, 1.0)) - here));
    }
    @fragment
    fn fs_bedSand(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let sand = sandDepthAt(u.q.xy + (frag.xy - 0.5 * BED_SIZE) * u.q.z);
      let coarse = floor(sand * 4.0) / 4.0;
      return vec4f(coarse, sand - coarse, 0.0, 1.0);
    }
    fn rimDistance(at : vec2f, texel : f32, size : f32) -> f32 {
      let reach = u.w.x / texel;
      var nearest = 1e9;
      for (var k = 0; k < 16; k++) {
        let a = (f32(k) + 0.5 * fract(0.5 * (at.x + at.y))) * 0.3927;
        let dir = vec2f(cos(a), sin(a));
        var r = 1.0;
        for (var n = 0; n < 48; n++) {
          if (r > reach || r >= nearest) { break; }
          let q = at + dir * r;
          if (any(q < vec2f(0.0)) || any(q >= vec2f(size))) { break; }
          let t = textureLoad(bedSandTex, vec2i(q), 0);
          if (t.r + t.g >= u.s.x) { nearest = r; break; }
          r = r * 1.15 + 0.5;
        }
      }
      return nearest * texel;
    }
  ),
  CODE(
    @fragment
    fn fs_farSand(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let sand = sandDepthAt(u.x.xy + (frag.xy - 0.5 * FAR_SIZE) * u.x.z);
      let coarse = floor(sand * 4.0) / 4.0;
      return vec4f(coarse, sand - coarse, 0.0, 1.0);
    }
    @fragment
    fn fs_farRim(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let own = textureLoad(bedSandTex, vec2i(frag.xy), 0);
      var d = 0.0;
      if (own.r + own.g < u.s.x) { d = min(rimDistance(frag.xy, u.x.z, FAR_SIZE), 500.0); }
      return vec4f(d, fishCeiling(frag.xy, own.r + own.g), 0.0, 1.0);
    }
    const FISH_RAMP : f32 = 0.7;
    fn fishCeiling(at : vec2f, ownSand : f32) -> f32 {
      let deepest = u.f.y + FISH_RAMP * 16.0 * u.x.z;
      var best = min(ownSand, deepest);
      if (best <= u.f.y + 0.01) { return best; }
      let radii = array<f32, 10>(1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
      for (var k = 0; k < 12; k++) {
        let a = f32(k) * 0.5236;
        let dir = vec2f(cos(a), sin(a));
        for (var n = 0; n < 10; n++) {
          let ramp = FISH_RAMP * radii[n] * u.x.z;
          if (u.f.y + ramp >= best) { break; }
          let q = clamp(at + dir * radii[n], vec2f(0.0), vec2f(FAR_SIZE - 1.0));
          let t = textureLoad(bedSandTex, vec2i(q), 0);
          best = min(best, t.r + t.g + ramp);
        }
      }
      return best;
    }
    fn fishLayer(xz : vec2f) -> f32 {
      var ceiling = u.f.y;
      let at = (xz - u.x.xy) / u.x.z + 0.5 * FAR_SIZE;
      if (u.x.w > 0.5 && all(at > vec2f(1.0)) && all(at < vec2f(FAR_SIZE - 1.0))) {
        ceiling = textureSampleLevel(farRimTex, linearSamp, at / FAR_SIZE, 0.0).g;
      }
      return max(min(u.l.z, ceiling - 0.14), 0.12);
    }
    fn farRim(xz : vec2f) -> f32 {
      if (u.x.w < 0.5) { return 1e9; }
      let at = (xz - u.x.xy) / u.x.z + 0.5 * FAR_SIZE;
      let edge = max(abs(at.x - 0.5 * FAR_SIZE), abs(at.y - 0.5 * FAR_SIZE)) / (0.5 * FAR_SIZE);
      if (edge >= 1.0) { return 1e9; }
      return textureSampleLevel(farRimTex, linearSamp, at / FAR_SIZE, 0.0).r + u.w.x * smoothstep(0.8, 0.97, edge);
    }
  ),
  CODE(
    @fragment
    fn fs_bed(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let xz = u.q.xy + (frag.xy - 0.5 * BED_SIZE) * u.q.z;
      let own = textureLoad(bedSandTex, vec2i(frag.xy), 0);
      let sand = own.r + own.g;
      var fromRim = 0.0;
      if (sand < u.s.x) {
        fromRim = 1e9;
        if (u.e.x > 0.0 && u.w.x > 0.0) { fromRim = min(rimDistance(frag.xy, u.q.z, BED_SIZE), farRim(xz)); }
      }
      let coral = coralAt(xz, sand, 1e3, fromRim);
      let smoothed = max(sand - coral.y, 0.05);
      let coarse = floor(smoothed * 4.0) / 4.0;
      var patchDepth = 0.0;
      if (u.e.x > 0.0) { patchDepth = clamp(coralInto(xz, sand, fromRim), -1.0, 1.0); }
      let reach = 0.5 / u.q.z;
      let lim = vec2f(BED_SIZE - 1.0);
      let ex = textureLoad(bedSandTex, vec2i(clamp(frag.xy + vec2f(reach, 0.0), vec2f(0.0), lim)), 0) - textureLoad(bedSandTex, vec2i(clamp(frag.xy - vec2f(reach, 0.0), vec2f(0.0), lim)), 0);
      let ez = textureLoad(bedSandTex, vec2i(clamp(frag.xy + vec2f(0.0, reach), vec2f(0.0), lim)), 0) - textureLoad(bedSandTex, vec2i(clamp(frag.xy - vec2f(0.0, reach), vec2f(0.0), lim)), 0);
      let steep = length(vec2f(ex.r + ex.g, ez.r + ez.g));
      return vec4f(coarse, smoothed - coarse, steep, patchDepth);
    }
    fn coralKind(pick : f32) -> i32 {
      let upTo = array<f32, 6>(0.33, 0.57, 0.75, 0.89, 0.95, 1.0);
      var kind = 5;
      for (var n = 4; n >= 0; n--) {
        if (pick < upTo[n]) { kind = n; }
      }
      return kind;
    }
    fn coralColor(kind : i32, shade : f32) -> vec3f {
      let colors = array<vec3f, 6>(vec3f(0.25, 0.165, 0.095), vec3f(0.33, 0.215, 0.075), vec3f(0.37, 0.185, 0.06),
                                   vec3f(0.41, 0.08, 0.065), vec3f(0.28, 0.12, 0.29), vec3f(0.32, 0.155, 0.165));
      return colors[kind] * mix(0.8, 1.25, shade);
    }
    const COLONY_SIZE : f32 = 0.3;
    fn coralAlbedo(xz : vec2f, core : f32) -> vec3f {
      let bent = xz + (vec2f(vnoise(vec3f(xz * 4.0, 61.0)), vnoise(vec3f(xz * 4.0, 83.0))) - 0.5) * 0.14;
      let g = bent / COLONY_SIZE;
      let cell = floor(g);
      var nearest = 1e9;
      var who = vec3f(0.0);
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = cell + vec2f(f32(i), f32(j));
          let r = siteRandom(c + 1200.0);
          let d = length(g - (c + 0.5 + (r.xy - 0.5) * 0.8));
          if (d < nearest) { nearest = d; who = r; }
        }
      }
      let hue = coralColor(coralKind(who.z), fract(who.x * 7.7));
      return mix(hue * 1.3, hue * 0.85, core);
    }
    fn keepHue(depth : f32, t : vec3f, ts : vec3f) -> vec3f {
      let path = depth * (1.0 / max(-t.y, 0.2) + 1.0 / max(-ts.y, 0.2));
      let undo = min(exp(0.9 * WATER_ABSORB * path), vec3f(45.0));
      return undo / min(undo.x, min(undo.y, undo.z));
    }
  ),
  CODE(
    struct FishLook {
      skyLit : vec3f,
      sunLit : vec3f,
      cover : f32,
      s : f32,
    }
    struct EllipsoidHit {
      normal : vec3f,
      s : f32,
      near : f32,
    }
    fn hitEllipsoid(o : vec3f, d : vec3f, radii : vec3f) -> EllipsoidHit {
      let os = o / radii;
      let ds = d / radii;
      let a = dot(ds, ds);
      let b = dot(os, ds);
      let closest = os - ds * (b / a);
      let near = length(closest);
      let disc = b * b - a * (dot(os, os) - 1.0);
      if (disc > 0.0) {
        let s = (-b - sqrt(disc)) / a;
        return EllipsoidHit(normalize((os + ds * s) / radii), s, near);
      }
      return EllipsoidHit(normalize(closest / radii), -b / a, near);
    }
    fn spine(x : f32, size : f32, phase : f32, curvature : f32, style : f32) -> vec2f {
      let len = 0.525 * size;
      let along = clamp((0.2 * size - x) / len, 0.0, 1.0);
      let c = mix(vec3f(0.02, -0.0825, 0.1625), vec3f(0.02, -0.04, 0.12), style);
      let amp = c.x + c.y * along + c.z * along * along;
      let ampSlope = c.y + 2.0 * c.z * along;
      let wave = 6.2832 * along / 0.95 - phase;
      return vec2f(len * amp * sin(wave) + 0.5 * curvature * x * x,
                   -(ampSlope * sin(wave) + amp * cos(wave) * 6.2832 / 0.95) + curvature * x);
    }
    fn hitBentEllipsoid(o : vec3f, d : vec3f, centre : vec3f, radii : vec3f, x0 : f32, bend : vec2f) -> EllipsoidHit {
      let oo = vec3f(o.x, o.y - bend.x - bend.y * (o.x - x0), o.z) - centre;
      let dd = vec3f(d.x, d.y - bend.y * d.x, d.z);
      let hit = hitEllipsoid(oo, dd, radii);
      return EllipsoidHit(normalize(vec3f(hit.normal.x - bend.y * hit.normal.y, hit.normal.y, hit.normal.z)), hit.s, hit.near);
    }
    fn fishSpread(layer : f32) -> f32 {
      let room = max(layer - 0.1, 0.0);
      return min(room, max(0.45, u.y.y * room));
    }
  ),
  CODE(
    fn fishAt(xz : vec2f, footprint : f32, ray : vec3f, sunRay : vec3f, sheetDepth : f32, over : f32, limit : f32) -> FishLook {
      var look = FishLook(vec3f(0.0), vec3f(0.0), 0.0, 0.0);
      let sharp = clamp(0.14 / footprint - 0.5, 0.0, 1.0);
      let smudge = (1.0 - sharp) * clamp(2.5 / footprint - 1.0, 0.0, 1.0);
      if (sharp + smudge <= 0.0) { return look; }
      let p = xz - FISH_WORLD * floor(xz / FISH_WORLD);
      let middle = vec2i(floor(p / FISH_CELL_SIZE));
      var nearest = 1e9;
      var crowd = 0.0;
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = (middle + vec2i(i, j) + FISH_GRID) % FISH_GRID;
          let cell = u32(c.y * FISH_GRID + c.x);
          let count = min(fishCells[cell * FISH_STRIDE], FISH_PER_CELL);
          crowd += f32(count);
          if (sharp <= 0.0) { continue; }
          for (var k = 0u; k < count; k++) {
            let id = fishCells[cell * FISH_STRIDE + 1u + k];
            let fish = boids[id];
            let r = siteRandom(vec2f(f32(id % 1024u), f32(id / 1024u)) + 700.0);
            let size = fish.size;
            var d = p - fish.pos;
            d -= FISH_WORLD * round(d / FISH_WORLD);
            if (dot(d, d) > 0.36) { continue; }
            let speed = length(fish.vel);
            let ahead = fish.vel / max(speed, 1e-4);
            let side = vec2f(-ahead.y, ahead.x);
            let o = vec3f(dot(d, ahead), dot(d, side), over - fish.lift / 0.45 * fishSpread(sheetDepth));
            let dir = vec3f(dot(ray.xz, ahead), dot(ray.xz, side), ray.y);
            let curvature = clamp(fish.turn / max(speed, 0.2), -2.5, 2.5) * 0.5;
            let bodyRadii = size * vec3f(0.2, 0.034, 0.068);
            let across = max(dir.y * dir.y + dir.z * dir.z, 1e-5);
            var x0 = clamp(o.x - dir.x * (o.y * dir.y + o.z * dir.z) / across, -0.2 * size, 0.2 * size);
            var body = hitBentEllipsoid(o, dir, vec3f(0.0), bodyRadii, x0, spine(x0, size, fish.phase, curvature, r.y));
            x0 = clamp(o.x + dir.x * body.s, -0.2 * size, 0.2 * size);
            body = hitBentEllipsoid(o, dir, vec3f(0.0), bodyRadii, x0, spine(x0, size, fish.phase, curvature, r.y));
            let tailX = -0.25 * size;
            let tail = hitBentEllipsoid(o, dir, vec3f(tailX, 0.0, 0.0), size * vec3f(0.075, 0.008, 0.07), tailX,
                                        spine(tailX, size, fish.phase, curvature, r.y));
            let soft = clamp(footprint / (0.05 * size), 0.05, 0.6);
            let bodyCover = 1.0 - smoothstep(1.0 - soft, 1.0 + soft, body.near);
            let tailCover = 1.0 - smoothstep(1.0 - soft, 1.0 + soft * 2.0, tail.near);
            var hit = body;
            var cover = bodyCover;
            if (tailCover > 0.0 && (bodyCover <= 0.0 || tail.s < body.s)) { hit = tail; cover = tailCover; }
            if (cover <= 0.0 || (look.cover > 0.5 && hit.s > nearest)) { continue; }
            if (hit.s > limit) { continue; }
            nearest = hit.s;
            let yellow = r.z > 0.9;
            let back = select(vec3f(0.09, 0.10, 0.13), vec3f(0.50, 0.38, 0.05), yellow) * mix(0.75, 1.35, r.x);
            let flank = select(vec3f(0.42, 0.47, 0.52), vec3f(0.75, 0.62, 0.15), yellow);
            var albedo = mix(flank, back, smoothstep(0.1, 0.7, hit.normal.z));
            albedo = mix(albedo, vec3f(0.8, 0.82, 0.84), clamp(-hit.normal.z - 0.3, 0.0, 1.0));
            let n = vec3f(ahead.x * hit.normal.x + side.x * hit.normal.y, hit.normal.z, ahead.y * hit.normal.x + side.y * hit.normal.y);
            let sunFacing = clamp((dot(n, -sunRay) + 0.3) / 1.3, 0.0, 1.0);
            let glint = pow(max(dot(n, normalize(-sunRay - ray)), 0.0), 40.0) * (1.0 - abs(hit.normal.z)) * 1.5;
            look = FishLook(albedo, albedo * sunFacing + vec3f(glint), cover * sharp, hit.s);
          }
        }
      }
      if (smudge > 0.0) {
        let cover = clamp(crowd * 0.022, 0.0, 0.45) * smudge;
        if (cover > look.cover) { look = FishLook(vec3f(0.09, 0.10, 0.13), vec3f(0.09, 0.10, 0.13) * max(-sunRay.y, 0.0), cover, 0.0); }
      }
      return look;
    }
  ),
  CODE(
    fn lureMarker(xz : vec2f) -> f32 {
      let p = xz - FISH_WORLD * floor(xz / FISH_WORLD);
      let middle = vec2i(floor(p));
      var mark = 0.0;
      for (var j = -1; j < 2; j++) {
        for (var i = -1; i < 2; i++) {
          let c = (middle + vec2i(i, j) + i32(LURE_GRID)) % i32(LURE_GRID);
          let entry = lureCells[u32(c.y) * LURE_GRID + u32(c.x)];
          if (entry == 0u) { continue; }
          let lure = lures[entry - 1u];
          var d = p - lure.pos;
          d -= FISH_WORLD * round(d / FISH_WORLD);
          let dist = length(d);
          let ring = mix(0.6, 0.2, clamp(lure.age / max(lure.life, 1e-3), 0.0, 1.0));
          mark = max(mark, max(1.0 - smoothstep(0.1, 0.14, dist), 1.0 - smoothstep(0.02, 0.05, abs(dist - ring))));
        }
      }
      return mark;
    }
    fn predatorMarker(xz : vec2f) -> f32 {
      let p = xz - FISH_WORLD * floor(xz / FISH_WORLD);
      var mark = 0.0;
      let count = min(u32(u.m.y), PREDATOR_MAX);
      for (var n = 0u; n < count; n++) {
        let hunter = predators[n];
        var d = p - hunter.pos;
        d -= FISH_WORLD * round(d / FISH_WORLD);
        let dist = length(d);
        if (dist > u.m.z + 0.2) { continue; }
        let ahead = hunter.vel / max(length(hunter.vel), 1e-4);
        let q = vec2f(dot(d, ahead), dot(d, vec2f(-ahead.y, ahead.x)));
        let arrow = max(abs(q.y) - 0.3 * (0.5 - q.x), -0.5 - q.x);
        mark = max(mark, max(1.0 - smoothstep(-0.03, 0.03, arrow), (1.0 - smoothstep(0.02, 0.06, abs(dist - u.m.z))) * 0.6));
      }
      return mark;
    }
  ),
  CODE(
    fn waterBody(surf : vec3f, t : vec3f, shadowed : bool, footprint : f32) -> vec3f {
      let up = vec3f(0.0, 1.0, 0.0);
      let sunDir = u.c.xyz;
      let extinction = WATER_ABSORB + WATER_SCATTER;
      let ts = refract(-sunDir, up, WATER_ETA);
      let sunFresnel = 0.02 + 0.98 * pow(1.0 - max(sunDir.y, 0.0), 5.0);
      let beamSpread = max(sunDir.y, 0.0) / max(-ts.y, 1e-3);
      let sunIn = sunColor() * 0.8 * PI * (1.0 - sunFresnel) * beamSpread;
      let skyIn = skyAmbient();
      var depth = u.f.y;
      var bedNormal = up;
      var bedAlbedo = SAND_ALBEDO;
      var bedShade = 1.0;
      var wrap = 0.0;
      if (shadowed) {
        let slide = t.xz / max(-t.y, 1e-3);
        var d = 0.0;
        var gap = depthAt(surf.xz, 1e3);
        var above = 0.0;
        var gapAbove = gap;
        if (u.p.y < 0.5) { d = gap; gap = 0.0; }
        for (var i = 0; i < 14; i++) {
          if (gap < 0.02) { break; }
          above = d;
          gapAbove = gap;
          d += gap * 0.45;
          gap = depthAt(surf.xz + slide * d, select(footprint, 1e3, gapAbove > 0.7)) - d;
        }
        if (gap < 0.0) {
          var below = d;
          var gapBelow = gap;
          for (var i = 0; i < 4; i++) {
            let mid = 0.5 * (above + below);
            let g = depthAt(surf.xz + slide * mid, footprint) - mid;
            if (g < 0.0) { below = mid; gapBelow = g; } else { above = mid; gapAbove = g; }
          }
          d = mix(above, below, clamp(gapAbove / max(gapAbove - gapBelow, 1e-5), 0.0, 1.0));
        }
        let bxz = surf.xz + slide * d;
        let bed = bedLookup(bxz, footprint);
        depth = bed.depth;
        let coral = vec2f(bed.core, 0.0);
        let cover = smoothstep(0.0, 0.25, coral.x) * (1.0 - bed.middle);
        let e = select(0.3, 0.02, coral.x > 0.0);
        bedNormal = normalize(vec3f(depthAt(bxz + vec2f(e, 0.0), footprint) - depth, e, depthAt(bxz + vec2f(0.0, e), footprint) - depth));
        if (coral.x > 0.0) {
          let head = bed.head;
          bedAlbedo = mix(SAND_ALBEDO, coralAlbedo(bxz, coral.x) * keepHue(depth, t, ts) * mix(0.4, 1.2, head), cover);
          let bump = coralBumps(bxz) * (cover * clamp(0.014 / footprint - 0.3, 0.0, 1.0));
          bedNormal = normalize(bedNormal - vec3f(bump.x, 0.0, bump.y));
        }
        let rippleShow = u.i.w * clamp(0.15 / footprint - 0.5, 0.0, 1.0) * (1.0 - cover) * exp(-depth / SAND_WAVE_FADE);
        if (rippleShow > 0.0) {
          let rippleDir = vec2f(cos(WIND_ANGLE + 0.5), sin(WIND_ANGLE + 0.5));
          let ripplePhase = dot(rippleDir, bxz) * 11.0 + 5.0 * vnoise(vec3f(bxz * 0.6, 2.0));
          let rippleSlope = 0.45 * rippleShow * cos(ripplePhase);
          bedNormal = normalize(bedNormal - vec3f(rippleDir.x, 0.0, rippleDir.y) * rippleSlope);
        }
  ),
  CODE(
        let seen = u.r.w * max(clamp(0.028 / footprint - 0.6, 0.0, 1.0), viewFocus * clamp(0.05 / pixelAcross - 0.6, 0.0, 1.0));
        let rocky = select(0.0, smoothstep(0.45, 0.85, bedSteep(bxz)) * (1.0 - cover), cover < 1.0);
        let moundy = cover * seen;
        if (rocky + moundy > 0.0) {
          let rp = rockWarp(vec3f(bxz.x, -depth, bxz.y) / ROCK_UNIT);
          let lump = ROCK_UNIT / ROCK_FREQ;
          let layers = i32(clamp(5.5 - log2(max(footprint * 6.0 / lump, 1.0)), 1.0, 5.0));
          let fine = footprint < 0.5 * ROCK_UNIT / ROCK_CELL_FREQ;
          let e = max(0.012, 0.7 * footprint);
          let h = rockHeight(rp, layers, fine);
          let tilt = vec2f(rockHeight(rp + vec3f(e / ROCK_UNIT, 0.0, 0.0), layers, fine), rockHeight(rp + vec3f(0.0, 0.0, e / ROCK_UNIT), layers, fine)) - h;
          let bumps = clamp(1.0 - footprint / 1.2, 0.0, 1.0);
          let amount = min(rocky + moundy, 1.0);
          bedNormal = normalize(bedNormal - vec3f(tilt.x, 0.0, tilt.y) * (ROCK_UNIT / e) * amount * bumps);
          let hollow = smoothstep(-1.0, 0.5, h / (ROCK_AMP + ROCK_RIDGE + ROCK_CELLS + ROCK_CRACKS));
          let mottle = 0.5 + 0.5 * rockNoise(rp * 3.0);
          let marks = (0.9 + 0.25 * rockNoise(rp * 70.0) * step(footprint, 0.02)) * mix(1.0, mix(0.5, 1.05, hollow), bumps);
          let stone = mix(vec3f(0.21, 0.195, 0.18), vec3f(0.42, 0.39, 0.35), mottle) * marks;
          let reefRock = mix(vec3f(0.17, 0.105, 0.065), vec3f(0.19, 0.06, 0.065), smoothstep(0.35, 0.65, mottle)) * marks * mix(0.55, 1.1, bed.head) * keepHue(depth, t, ts);
          bedAlbedo = mix(mix(bedAlbedo, stone, rocky), reefRock, moundy);
          bedShade *= mix(1.0, mix(0.6, 1.0, hollow), amount * bumps);
        }
        if (u.e.x > 0.0 && depth < coralDeepest() + 0.7 && u.p.y > 0.5) {
          let back = -ts.xz / max(-ts.y, 1e-3);
          var blocked = 0.0;
          for (var i = 0; i < 4; i++) {
            let rise = select(0.09 + 0.16 * f32(i - 1) * (1.0 + 0.35 * f32(i - 1)), 0.025, i == 0);
            let over = (depth - rise) - depthAt(bxz + back * rise, footprint);
            blocked = max(blocked, clamp(over / 0.12, 0.0, 1.0));
          }
          bedShade = 1.0 - 0.85 * blocked;
        }
        let branchNear = max(clamp(0.028 / footprint - 0.6, 0.0, 1.0), viewFocus * clamp(0.05 / pixelAcross - 0.6, 0.0, 1.0));
        var nearCoral = bed.core;
        if (nearCoral < 0.02 && u.r.w * branchNear > 0.0) {
          let across = 0.2 * normalize(vec2f(-slide.y, slide.x) + vec2f(1e-5, 0.0));
          nearCoral = max(max(coralUnder(bxz - slide * 0.25), coralUnder(bxz - slide * 0.55)),
                          max(max(coralUnder(bxz + across), coralUnder(bxz - across)), coralUnder(bxz + normalize(slide + vec2f(1e-5, 0.0)) * 0.2)));
        }
        let branchShow = u.r.w * branchNear * step(0.02, nearCoral) * (1.0 - smoothstep(0.25, 0.7, bed.middle));
  ),
  CODE(
        if (branchShow > 0.0) {
          let share = 0.95 * u.r.w;
          let drop = max(-t.y, 1e-3);
          var dv = max(depth - (BRANCH_TOP + 0.25), 0.0);
          var touched = false;
          var at = vec3f(0.0);
          let steps = i32(mix(16.0, 40.0, branchNear));
          let headShift = depthAt(bxz, 1e3) - depth;
          var inLayer = false;
          var touchedSquash = 1.0;
          for (var i = 0; i < steps; i++) {
            let q = surf.xz + slide * dv;
            let heads = depthAt(q, 1e3);
            if (heads - headShift - dv <= 0.0) { break; }
            let squash = BRANCH_TOP / branchRoom(heads);
            branchNarrow = clamp(pow(1.0 / squash, 0.33), 0.5, 1.0);
            let height = (heads - headShift - dv) * squash;
            if (!inLayer) {
              let over = height - BRANCH_TOP - 0.1;
              if (over > 0.0) {
                dv += max(over / squash * 0.7, 0.01) * drop;
                if (dv >= depth) { break; }
                continue;
              }
              inLayer = true;
            }
            var clear = height - BRANCH_TOP;
            if (clear < 0.02) {
              clear = branchScene(vec3f(q.x, height, q.y), share);
              if (clear < 0.0015 + 0.5 * pixelAcross) { touched = true; at = vec3f(q.x, height, q.y); touchedSquash = squash; break; }
            }
            dv += min(clear, 0.3 * BRANCH_CELL) / squash * 0.85 * drop;
            if (dv >= depth) { break; }
          }
          let kindTouched = branchKind;
          let shadeTouched = branchShade;
          let toSun = -ts;
          var origin = vec3f(bxz.x, 0.004, bxz.y);
          var faces = bedNormal;
          if (touched) {
            let e = 0.0015 + 0.7 * pixelAcross;
            let here = branchScene(at, share);
            faces = normalize(vec3f(branchScene(at + vec3f(e, 0.0, 0.0), share), branchScene(at + vec3f(0.0, e, 0.0), share),
                                    branchScene(at + vec3f(0.0, 0.0, e), share)) - here + 1e-6);
            origin = at + faces * 0.004;
          }
          var open = 1.0;
          var s = 0.012;
          let walk = select(i32(mix(3.0, 7.0, branchNear)), 0, touched && dot(faces, toSun) <= 0.0);
          if (walk == 0) { open = 0.0; }
          for (var j = 0; j < walk; j++) {
            let q = origin + toSun * s;
            if (q.y > BRANCH_TOP) { break; }
            let d = branchScene(q, share);
            open = min(open, 10.0 * d / s);
            s += clamp(d, 0.008, 0.3 * BRANCH_CELL);
          }
          open = clamp(open, 0.0, 1.0);
          let hemmed = 0.7 + 0.3 * clamp(branchScene(origin + faces * 0.03, share) / 0.03, 0.0, 1.0);
          if (touched) {
            let tint = coralColor(kindTouched, shadeTouched) * keepHue(depth - at.y / touchedSquash, t, ts);
            let bloom = clamp(at.y / (BRANCH_TOP * 0.8), 0.0, 1.0);
            let branchAlbedo = mix(mix(vec3f(0.16, 0.13, 0.10), tint, 0.4), tint * 1.6 + 0.02, bloom * bloom) * hemmed;
            bedAlbedo = mix(bedAlbedo, branchAlbedo, branchShow);
            wrap = 0.6 * branchShow;
            bedNormal = normalize(mix(bedNormal, faces, branchShow));
            bedShade = mix(bedShade, open, branchShow);
            depth = mix(depth, depth - at.y / touchedSquash, branchShow);
          } else {
            bedAlbedo *= mix(1.0, hemmed, branchShow);
            bedShade *= mix(1.0, open, branchShow);
          }
        }
      }
  ),
  CODE(
      let viewPath = depth / max(-t.y, 1e-3);
      let sunPath = depth / max(-ts.y, 1e-3);
      var sandLight = 1.0;
      var surfLight = 1.0;
      var ripple = 1.0;
      if (shadowed) {
        let bed = surf + t * viewPath;
        let flatEntry = bed - ts * sunPath;
        let wave = waveField(flatEntry.xz, footprint, WAVE_COUNT, u.h.z > 0.5);
        let tsWave = refract(-sunDir, waterNormal(vec3f(wave.slope, 0.0), sunDir), WATER_ETA);
        let entry = bed - tsWave * (depth / max(-tsWave.y, 1e-3));
        if (u.h.z > 0.5) { ripple = caustic(wave, sunPath); }
        sandLight = 1.0;
        surfLight = 1.0;
        if (u.p.x > 0.5) {
          let clouds = cloudShadowSites(entry);
          sandLight = cloudShadowFrom(clouds, entry);
          surfLight = cloudShadowFrom(clouds, surf);
        }
        seaShadowAtSurface = surfLight;
        if (depth < 12.0) { sandLight *= birdShadow(entry.xz, entry.y); }
      }
      var fish = FishLook(vec3f(0.0), vec3f(0.0), 0.0, 0.0);
      var fishDepth = 0.0;
      var fishLight = surfLight;
      if (shadowed && u.l.y > 0.5 && footprint < 2.5) {
        let slide = t.xz / max(-t.y, 1e-3);
        var dPrev = 0.0;
        var gPrev = fishLayer(surf.xz);
        var layer = gPrev;
        fishDepth = gPrev;
        for (var i = 0; i < 5; i++) {
          layer = fishLayer(surf.xz + slide * fishDepth);
          let g = layer - fishDepth;
          if (abs(g) < 0.01 || abs(g - gPrev) < 1e-5) { break; }
          let next = fishDepth - g * (fishDepth - dPrev) / (g - gPrev);
          dPrev = fishDepth;
          gPrev = g;
          fishDepth = clamp(next, 0.0, 8.0);
        }
        let reached = min(fishDepth, depth);
        let column = fishSpread(layer);
        let downRate = max(-t.y, 1e-3);
        let tries = i32(clamp(ceil(column * length(slide) / 0.9) + 1.0, 1.0, 8.0));
        var nearestFish = 1e9;
        var seenAt = reached;
        for (var k = 0; k < tries; k++) {
          let at = max(reached - column * f32(k) / f32(max(tries - 1, 1)), 0.0);
          let found = fishAt(surf.xz + slide * at, footprint, t, ts, layer, fishDepth - at, select(1e9, (reached - at) / downRate, depth < fishDepth));
          let along = at / downRate + found.s;
          if (found.cover > 0.0 && (fish.cover <= 0.5 || along < nearestFish)) {
            fish = found;
            nearestFish = along;
            seenAt = clamp(along * downRate, 0.05, reached);
          }
          if (footprint > 0.28) { break; }
        }
        fishDepth = seenAt;
        let bedXZ = surf.xz + t.xz * viewPath;
        let layerOverBed = fishLayer(bedXZ);
        let above = depth - layerOverBed;
        fishLight = mix(surfLight, sandLight * ripple, exp(-max(above, 0.0)));
        if (above > 0.0 && above < 7.0) {
          let overhead = fishAt(bedXZ - ts.xz * (above / max(-ts.y, 1e-3)), footprint, ts, ts, layerOverBed, 0.0, 1e9);
          sandLight *= 1.0 - 0.7 * overhead.cover * exp(-above / 3.0);
        }
      }
      let facing = dot(bedNormal, -ts);
      let sunFacing = mix(max(facing, 0.0), clamp((facing + 0.6) / 1.6, 0.0, 1.0), wrap);
      let lit = bedAlbedo;
      let bedSun = lit * (sunFacing * sandLight * bedShade * ripple);
      var result = waterColumn(bedSun, lit, depth, t, ts, sunIn, skyIn, surfLight);
      if (fish.cover > 0.0) {
        result = mix(result, waterColumn(fish.sunLit * fishLight, fish.skyLit, fishDepth, t, ts, sunIn, skyIn, surfLight), fish.cover);
      }
      return result;
    }
  ),
  CODE(
    fn waterColumn(sunLit : vec3f, skyLit : vec3f, depth : f32, t : vec3f, ts : vec3f, sunIn : vec3f, skyIn : vec3f, surfLight : f32) -> vec3f {
      let extinction = WATER_ABSORB + WATER_SCATTER;
      let viewPath = depth / max(-t.y, 1e-3);
      let sunPath = depth / max(-ts.y, 1e-3);
      var bottom = sunLit / PI * sunIn * exp(-extinction * sunPath) + skyLit * skyIn * exp(-extinction * depth);
      bottom *= exp(-extinction * viewPath);
      let kSun = 1.0 + (-t.y) / max(-ts.y, 1e-3);
      let kSky = 1.0 + (-t.y);
      let scatterSun = WATER_SCATTER * sunIn / (4.0 * PI) * surfLight
                     * (1.0 - exp(-extinction * kSun * viewPath)) / (extinction * kSun);
      let scatterSky = WATER_SCATTER * skyIn * 0.5
                     * (1.0 - exp(-extinction * kSky * viewPath)) / (extinction * kSky);
      return bottom + scatterSun + scatterSky;
    }
    fn waterBounce() -> vec3f {
      return waterBody(vec3f(0.0), vec3f(0.0, -1.0, 0.0), false, 0.0);
    }
    const WATER_F0 : f32 = 0.02;
    fn mirrorSky(r : vec3f) -> vec3f {
      let sky = skyGlow(vec3f(r.x, max(r.y, 0.0), r.z));
      return mix(waterBounce(), sky, smoothstep(-0.2, 0.0, r.y));
    }
    fn mirrorWeight(nv : f32, rough : f32, f0 : f32) -> f32 {
      return f0 + (max(1.0 - rough, f0) - f0) * pow(1.0 - nv, 5.0);
    }
    fn surfaceLight(n : vec3f, v : vec3f, albedo : vec3f, rough : f32, f0 : f32, vis : f32, mirror : vec3f) -> vec3f {
      let sunDir = u.c.xyz;
      let h = normalize(v + sunDir);
      let nl = max(dot(n, sunDir), 0.0);
      let nv = max(dot(n, v), 1e-3);
      let nh = max(dot(n, h), 0.0);
      let vh = max(dot(v, h), 0.0);
      let a = rough * rough;
      let a2 = a * a;
      let dd = nh * nh * (a2 - 1.0) + 1.0;
      let ggxD = a2 / (PI * dd * dd);
      let smithV = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-5);
      let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
      let sunIrradiance = sunColor() * 0.8 * PI;
      let direct = ((1.0 - fresnel) * albedo / PI + ggxD * smithV * fresnel) * sunIrradiance * nl * vis;
      let skyGrey = skyAmbient();
      let diffuseAmbient = mix(waterBounce() * 0.5 + skyGrey * 0.25, skyGrey, n.y * 0.5 + 0.5);
      let reflection = mix(mirror, diffuseAmbient, rough);
      let weight = mirrorWeight(nv, rough, f0);
      return direct + (1.0 - weight) * albedo * diffuseAmbient + weight * reflection;
    }
    fn reflectionBackdrop(r : vec3f, mapXZ : vec2f) -> vec3f {
      let far = horizonClouds(r, 0.0, u.c.xyz, mapXZ - u.d.zw);
      return mirrorSky(r) * (1.0 - far.a) + far.rgb;
    }
  ),
  CODE(
    fn shadeWater(ro : vec3f, rd : vec3f, tPlane : f32, hazeCol : vec3f) -> vec3f {
      if (tPlane > HAZE_LIMIT) { return hazeCol; }
      let sunDir = u.c.xyz;
      let up = vec3f(0.0, 1.0, 0.0);
      let v = -rd;
      let footprint = tPlane / (u.b.w * u.a.y) / max(-rd.y, 0.02) * pixelScale;
      pixelAcross = tPlane / (u.b.w * u.a.y) * pixelScale;
      var dist = tPlane;
      if (u.h.y > 0.5 && u.h.x > 0.0) {
        dist = waveHit(ro, rd, tPlane, footprint);
      }
      let surf = ro + rd * dist;
      seaShadowAtSurface = 1.0;
      let field = waveField(surf.xz, footprint, WAVE_COUNT, false);
      let wave = vec3f(field.slope, field.lost);
      let wn = waterNormal(wave, v);
      let viewFresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - max(dot(wn, v), 0.0), 5.0);
      let body = (1.0 - viewFresnel) * waterBody(surf, refract(rd, wn, WATER_ETA), true, footprint);
      let vis = seaShadowAtSurface;
      let mirror = reflectionBackdrop(reflect(rd, wn), surf.xz);
      var col = surfaceLight(wn, v, vec3f(0.0), waterRough(wave.z), WATER_F0, vis, mirror) + body;
      if (u.h.w > 0.5 && u.h.x > 0.0) {
        let detail = clamp(0.1 / footprint, 0.0, 1.0);
        let fray = (fbm(vec3f(surf.xz * 2.5, u.g.w * 0.3), 3) - 0.5) * 0.2 * detail;
        let lace = smoothstep(0.3, 0.6, fbm(vec3f(surf.xz * 8.0, u.g.w * 0.5), 3));
        let foam = smoothstep(0.16, 0.24, field.peak + fray) * mix(1.0, mix(0.45, 1.0, lace), detail);
        let foamNormal = normalize(wn + up);
        let foamColor = 0.85 * (sunColor() * 0.8 * max(dot(foamNormal, sunDir), 0.0) * vis + skyAmbient());
        col = mix(col, foamColor, foam);
      }
      if (u.l.w > 0.5) {
        col = mix(col, vec3f(6.0, 0.15, 3.0), lureMarker(surf.xz));
      }
      if (u.m.x > 0.5) {
        col = mix(col, vec3f(6.0, 0.9, 0.0), predatorMarker(surf.xz));
      }
      if (u.r.y > 0.5) {
        let p = surf.xz - FISH_WORLD * floor(surf.xz / FISH_WORLD);
        let toWall = (0.5 - abs(fract(p / u.r.z) - 0.5)) * u.r.z;
        let width = max(0.04, 1.2 * footprint);
        let onWall = 1.0 - smoothstep(width * 0.6, width, min(toWall.x, toWall.y));
        col = mix(col, vec3f(0.1, 5.0, 5.5), onWall * clamp(1.5 / footprint, 0.0, 1.0));
      }
      let f = dist * HAZE;
      return mix(hazeCol, col, exp(-f * f));
    }
  ),
  CODE(
    var<private> cloudFront : f32 = 1e9;
    fn cloudImpostor(ro : vec3f, rd : vec3f, tMax : f32, site : CloudSite, gathered : vec4f, sunDir : vec3f, hazeCol : vec3f) -> vec4f {
      let cs = cos(site.spin);
      let sn = sin(site.spin);
      let o = vec3f(ro.x * cs - ro.z * sn, ro.y, ro.x * sn + ro.z * cs) / site.scale;
      let d = vec3f(rd.x * cs - rd.z * sn, rd.y, rd.x * sn + rd.z * cs);
      let sunL = vec3f(sunDir.x * cs - sunDir.z * sn, sunDir.y, sunDir.x * sn + sunDir.z * cs);
      let nearest = o - d * dot(o, d);
      let reach = 0.5 + u.c.w + site.cover + (fbm(nearest * 1.1 + vec3f(site.seed), 3) - 0.5) * 1.6 * u.d.y;
      if (reach <= 0.05) { return gathered; }
      let k = vec3f(0.55, 1.2, 0.7) / reach;
      let os = o * k;
      let ds = d * k;
      let a = dot(ds, ds);
      let b = dot(os, ds);
      let disc = b * b - a * (dot(os, os) - 1.0);
      if (disc <= 0.0) { return gathered; }
      var s0 = (-b - sqrt(disc)) / a;
      var s1 = (-b + sqrt(disc)) / a;
      if (abs(d.y) > 1e-4) {
        let sb = (-0.5 - o.y) / d.y;
        if (d.y > 0.0) { s0 = max(s0, sb); } else { s1 = min(s1, sb); }
      } else if (o.y < -0.5) {
        return gathered;
      }
      s0 = max(s0, 0.0);
      s1 = min(s1, tMax / site.scale);
      if (s1 <= s0) { return gathered; }
      let tNear = s0 * site.scale;
      let thin = 1.0 - smoothstep(0.7 * HAZE_LIMIT, HAZE_LIMIT, tNear);
      let alpha = (1.0 - exp(-(s1 - s0) * site.scale * u.d.x * 0.45)) * thin;
      let enter = o + d * s0;
      let facing = dot(normalize(enter * k * k), sunL) * 0.5 + 0.5;
      let sunCol = sunColor() * smoothstep(-0.1, 0.1, sunDir.y);
      let skyTop = skyBase(vec3f(0.0, 1.0, 0.0));
      let skyLow = skyBase(normalize(vec3f(rd.x, 0.05, rd.z)));
      let mu = dot(rd, sunDir);
      let phase = 0.65 * hg(mu, 0.55) + 0.35 * hg(mu, -0.2);
      let ambient = mix(skyLow * 0.3, skyTop * 0.75, clamp(enter.y * 0.6 + 0.5, 0.0, 1.0));
      let hazeAmount = 1.0 - exp(-(tNear * HAZE) * (tNear * HAZE));
      let lum = mix(sunCol * mix(0.28, 1.0, facing * facing) * phase + ambient, hazeCol, hazeAmount);
      if (gathered.a * (1.0 - alpha) < 0.67 && cloudFront > 1e8) { cloudFront = tNear; }
      return vec4f(gathered.rgb + gathered.a * lum * alpha, gathered.a * (1.0 - alpha));
    }
  ),
  CODE(
    fn marchSite(ro : vec3f, rd : vec3f, tMax : f32, site : CloudSite, steps : i32, dither : f32, gathered : vec4f, sunDir : vec3f, hazeCol : vec3f) -> vec4f {
      let radius = BOUND_R * site.scale;
      let oc = ro;
      let bq = dot(oc, rd);
      let disc = bq * bq - (dot(oc, oc) - radius * radius);
      if (disc <= 0.0) { return gathered; }
      let sq = sqrt(disc);
      let tNear = max(-bq - sq, 0.0);
      let tFar = min(-bq + sq, tMax);
      if (tFar <= tNear) { return gathered; }
      let sunCol = sunColor() * smoothstep(-0.1, 0.1, sunDir.y);
      let skyTop = skyBase(vec3f(0.0, 1.0, 0.0));
      let skyLow = skyBase(normalize(vec3f(rd.x, 0.05, rd.z)));
      let count = clamp(i32(f32(steps) * 40.0 / max(tNear, 40.0)), min(steps, 10), steps);
      let octaves = select(5, 4, tNear > 120.0);
      let hazeAmount = 1.0 - exp(-(tNear * HAZE) * (tNear * HAZE));
      let thin = 1.0 - smoothstep(0.7 * HAZE_LIMIT, HAZE_LIMIT, tNear);
      var col = gathered.rgb;
      var trans = gathered.a;
      let dt = (tFar - tNear) / f32(count);
      let mu = dot(rd, sunDir);
      let phase = 0.65 * hg(mu, 0.55) + 0.35 * hg(mu, -0.2);
      let sigma = u.d.x;
      var t = tNear + dt * dither;
      for (var i = 0; i < count; i++) {
        let here = ro + rd * t;
        let p = here / site.scale;
        let dens = cloudDensityAt(p, site.seed, site.cover, octaves, site.spin) * thin;
        if (dens > 0.002) {
          var depthSum = 0.0;
          for (var j = 0; j < LIGHT_STEPS; j++) {
            depthSum += cloudDensityAt(p + sunDir * LIGHT_STEP * (f32(j) + 0.5), site.seed, site.cover, 3, site.spin);
          }
          let tau = depthSum * LIGHT_STEP * site.scale * sigma;
          let sunT = max(exp(-tau), 0.35 * exp(-0.3 * tau));
          let ambient = mix(skyLow * 0.3, skyTop * 0.75, clamp(p.y * 0.6 + 0.5, 0.0, 1.0));
          let lum = mix(sunCol * sunT * phase + ambient, hazeCol, hazeAmount);
          let stepT = exp(-dens * sigma * dt);
          col += trans * lum * (1.0 - stepT);
          trans *= stepT;
          if (trans < 0.67 && cloudFront > 1e8) { cloudFront = t; }
          if (trans < 0.01) { break; }
        }
        t += dt;
      }
      return vec4f(col, trans);
    }
  ),
  CODE(
    fn marchCloud(ro : vec3f, rd : vec3f, tMax : f32, steps : i32, dither : f32, hazeCol : vec3f) -> vec4f {
      var gathered = vec4f(0.0, 0.0, 0.0, 1.0);
      let radius = u.i.y;
      let roof = select(CLUSTER_BASE + 0.55 * u.p.w + 0.5, CLOUD_LAYER_HIGH, u.p.z > 0.5);
      let top = roof - u.g.x;
      let h = ro.y - u.g.x;
      let b = ro.x * rd.x + ro.z * rd.z + (h + radius) * rd.y;
      let c = ro.x * ro.x + ro.z * ro.z + (h - top) * (h + top) + 2.0 * radius * (h - top);
      let disc = b * b - c;
      if (disc < 0.0) { return gathered; }
      let sq = sqrt(disc);
      var ta = 0.0;
      var tb = 0.0;
      if (c > 0.0) {
        if (b >= 0.0) { return gathered; }
        ta = c / (-b + sq);
        tb = -b + sq;
      } else if (b < 0.0) {
        tb = -b + sq;
      } else {
        tb = -c / (b + sq);
      }
      tb = min(tb, min(tMax, HAZE_LIMIT));
      if (tb <= ta) { return gathered; }
      let stride = select(0.45, 0.65, u.p.z < 0.5) * cloudSpacing();
      let strides = min(i32(ceil((tb - ta) / stride)) + 1, 96);
      let sun = u.c.xyz;
      var lastBase = vec2f(1e9);
      let scattered = u.p.z < 0.5;
      let first = select(0, -1, scattered);
      let last = 2;
      for (var i = 0; i < strides; i++) {
        let t = min(ta + stride * f32(i), tb);
        let base = cloudCell(worldToMap(ro + rd * t).xz + select(vec2f(0.0), cloudHalfCell(), scattered));
        var found : array<vec2f, 9>;
        var foundT : array<f32, 9>;
        var foundCount = 0;
        for (var cj = first; cj < last; cj++) {
          for (var ci = first; ci < last; ci++) {
            let c = base + vec2f(f32(ci), f32(cj));
            if (all(c >= lastBase + f32(first)) && all(c <= lastBase + f32(last - 1))) { continue; }
            let site = cloudAt(c);
            if (!site.present) { continue; }
            let frame = siteFrame(site.centre);
            let o = ro - frame.centre;
            let bq = dot(o, rd);
            let radius = BOUND_R * site.scale;
            let disc = bq * bq - (dot(o, o) - radius * radius);
            if (disc <= 0.0) { continue; }
            let enter = max(-bq - sqrt(disc), 0.0);
            if (enter >= tMax || -bq + sqrt(disc) <= 0.0) { continue; }
            found[foundCount] = c;
            foundT[foundCount] = enter;
            foundCount++;
          }
        }
        for (var a = 0; a < foundCount; a++) {
          var pick = 0;
          for (var k = 1; k < foundCount; k++) {
            if (foundT[k] < foundT[pick]) { pick = k; }
          }
          let c = found[pick];
          let enters = foundT[pick];
          foundT[pick] = 1e30;
          let site = cloudAt(c);
          let frame = siteFrame(site.centre);
          if (enters > u.r.x * mix(0.85, 1.15, dither)) {
            gathered = cloudImpostor(toLocalUp(ro - frame.centre, frame.up), toLocalUp(rd, frame.up), tMax, site,
                                     gathered, toLocalUp(sun, frame.up), hazeCol);
          } else {
            gathered = marchSite(toLocalUp(ro - frame.centre, frame.up), toLocalUp(rd, frame.up), tMax, site,
                                 steps, dither, gathered, toLocalUp(sun, frame.up), hazeCol);
          }
        }
        lastBase = base;
        if (gathered.a < 0.01) { break; }
      }
      return gathered;
    }
  ),
  CODE(
    struct CloudOut {
      @location(0) color : vec4f,
      @location(1) near : f32,
    }
    @fragment
    fn fs_cloud(@builtin(position) frag : vec4f) -> CloudOut {
      if (u.p.x < 0.5) { return CloudOut(vec4f(0.0, 0.0, 0.0, 1.0), 0.0); }
      let fullPx = frag.xy / u.e.yz * u.a.xy;
      let ro = u.b.xyz;
      let rd = viewRay(fullPx);
      let dither = ign(frag.xy);
      let steps = i32(u.a.w);
      let hit = waterHit(ro, rd);
      cloudFront = 1e9;
      var cloud = marchCloud(ro, rd, hit.t, steps, dither, skyView(ro, rd).color);
      let near = select(0.0, 1.0 / max(cloudFront, 0.1), cloudFront < 1e8);
      if (u.g.y > 0.5 && cloud.a > 0.01 && hit.t < HAZE_LIMIT) {
        let lrd = hit.rd;
        let footprint = hit.t / (u.b.w * u.e.z) / max(-lrd.y, 0.05);
        let wave = waveSlopes(hit.surf.xz, footprint);
        let wn = waterNormal(wave, -lrd);
        let r = reflect(lrd, wn);
        if (r.y > 0.01) {
          let refl = marchCloud(hit.world, fromLocalUp(r, hit.up), 150.0, max(steps / 2, 8), dither, skyGlow(normalize(vec3f(r.x, 0.0, r.z))));
          if (refl.a < 0.999) {
            let rough = waterRough(wave.z);
            let weight = mirrorWeight(max(dot(wn, -lrd), 1e-3), rough, WATER_F0) * (1.0 - rough);
            let f = hit.t * HAZE;
            let change = refl.rgb - (1.0 - refl.a) * reflectionBackdrop(r, hit.surf.xz);
            cloud = vec4f(cloud.rgb + cloud.a * weight * exp(-f * f) * change, cloud.a);
          }
        }
      }
      return CloudOut(cloud, near);
    }
  ),
  CODE(
    fn wingPanel(o : vec3f, d : vec3f, centre : vec3f, way : vec3f, radii : vec3f) -> EllipsoidHit {
      let fore = normalize(vec3f(1.0, 0.0, 0.0) - way * way.x);
      let face = cross(fore, way);
      let rel = o - centre;
      let hit = hitEllipsoid(vec3f(dot(rel, fore), dot(rel, way), dot(rel, face)), vec3f(dot(d, fore), dot(d, way), dot(d, face)), radii);
      return EllipsoidHit(fore * hit.normal.x + way * hit.normal.y + face * hit.normal.z, hit.s, hit.near);
    }
    struct BirdLook {
      color : vec4f,
      distance : f32,
    }
    const BIRD_SIGHT : f32 = 90.0;
    struct Feather {
      h : f32,
      loc : vec4f,
      fid : vec3f,
    }
    const FEATHER_LENGTH : f32 = 2.6;
    fn featherAt(p : vec3f, R : vec3f, rows : f32, px : f32) -> Feather {
      let q = normalize(p / R);
      let u = atan2(q.z, q.y);
      let v = acos(clamp(q.x, -1.0, 1.0));
      let dv = PI / rows;
      let s = v / dv;
      let j0 = floor(s);
      let around = max(R.y, R.z) / R.x;
      var f : Feather;
      f.h = 0.0;
      f.loc = vec4f(0.0);
      f.fid = vec3f(0.0, j0, 4.0);
      var occ = 0.0;
      for (var kk = 0; kk < 3; kk++) {
        let j = j0 - (2.0 - f32(kk));
        if (j < 0.0) { continue; }
        let n = max(4.0, floor(1.5 * rows * around * sin((j + 0.5) * dv) + 0.5));
        let ph = branchHash(vec2f(j, 7.0));
        let uc = (u / (2.0 * PI) + 0.5) * n + ph;
        let cell = floor(uc);
        let i = cell - n * floor(cell / n);
        let r1 = branchHash(vec2f(i, j) + 0.5);
        let r2 = branchHash(vec2f(j, i) + 13.7);
        let a = fract(uc) - 0.5 + (r1 - 0.5) * 0.1;
        let len = FEATHER_LENGTH * (0.86 + 0.28 * r2);
        let b = (s - j) / len;
        let w = 0.68 * pow(max(0.0, 1.0 - b * b), 0.62);
        if (b < 1.0 && abs(a) < w) {
          let lam = 0.10 * dv * R.x;
          let barbs = 1.0 - smoothstep(0.3, 0.8, px / lam);
          var height = mix(0.12, 1.0, b);
          height -= 0.22 * 4.0 * a * a;
          height += 0.07 * (1.0 - smoothstep(0.0, 0.06, abs(a)));
          height += 0.016 * barbs * sin(b * len * 48.0 - abs(a) * 62.0);
          f.h = height;
          f.loc = vec4f(a, b, occ, w);
          f.fid = vec3f(i, j, n);
          break;
        }
        let e = select(length(vec2f(a, (b - 1.0) * len)), abs(a) - w, b < 1.0);
        occ = max(occ, 1.0 - smoothstep(0.0, 0.2, e));
      }
      return f;
    }
    fn featheredSurface(p : vec3f, R : vec3f, rows : f32, px : f32, relief : f32) -> f32 {
      let k0 = length(p / R);
      let k1 = length(p / (R * R));
      return k0 * (k0 - 1.0) / k1 - relief * featherAt(p, R, rows, px).h;
    }
  ),
  CODE(
    struct BirdFrame {
      ahead : vec3f,
      left : vec3f,
      top : vec3f,
    }
    fn birdFrame(bird : Boid) -> BirdFrame {
      let speed = length(vec3f(bird.vel, bird.climb));
      let ahead = vec3f(bird.vel.x, bird.climb, bird.vel.y) / max(speed, 1e-4);
      let level = normalize(cross(ahead, vec3f(0.0, 1.0, 0.0)));
      let over = cross(level, ahead);
      let bank = clamp(atan(bird.turn * speed / 9.81), -1.0, 1.0);
      let left = level * cos(bank) - over * sin(bank);
      return BirdFrame(ahead, left, cross(left, ahead));
    }
    struct Plume {
      cover : f32,
      h : f32,
      a : f32,
      b : f32,
      occ : f32,
      w : f32,
      id : vec2f,
      flight : f32,
    }
    fn vane(a : f32, b : f32, shape : f32, barbPitch : f32, px : f32, wide : f32, long : f32, f : ptr<function, Plume>) -> bool {
      let w = 0.68 * pow(max(0.0, 1.0 - pow(max(b, 0.0), shape)), 0.62);
      if (b < 1.0 && abs(a) < w) {
        let barbs = 1.0 - smoothstep(0.3, 0.8, px / barbPitch);
        var height = mix(0.12, 1.0, b);
        height -= 0.22 * 4.0 * a * a;
        height += 0.07 * (1.0 - smoothstep(0.0, 0.06, abs(a)));
        height += 0.016 * barbs * sin(b * long / barbPitch * 0.82 - abs(a) * 62.0);
        var solid = smoothstep(0.0, px, (1.0 - b) * long);
        if (w < 0.55) { solid *= smoothstep(0.0, px, (w - abs(a)) * wide); }
        (*f).cover = solid;
        (*f).h = height;
        (*f).a = a;
        (*f).b = b;
        (*f).w = w;
        return true;
      }
      let e = select(length(vec2f(a, (b - 1.0) * long / wide)), abs(a) - w, b < 1.0);
      (*f).occ = max((*f).occ, 1.0 - smoothstep(0.0, 0.2, e));
      return false;
    }
    const WING_SPAN : f32 = 1.7;
    const WING_ROW : f32 = 0.10;
    fn wingLead(t : f32) -> f32 {
      return 0.40 + 0.10 * sin(2.6 * t) - 0.45 * t * t * t;
    }
  ),
  CODE(
    fn featherWing(q : vec2f, px : f32) -> Plume {
      var f = Plume(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec2f(0.0), 0.0);
      if (q.y < 0.0 || q.y > WING_SPAN + 0.2) { return f; }
      let t = clamp(q.y / WING_SPAN, 0.0, 1.0);
      let row = WING_ROW * (1.0 - 0.8 * pow(t, 2.4));
      let s = (wingLead(t) - q.x) / row;
      if (s < 0.0 || s > 14.5) { return f; }
      let j0 = min(floor(s), 4.0);
      for (var kk = 0; kk < 3; kk++) {
        let j = j0 - (2.0 - f32(kk));
        if (j < 0.0) { continue; }
        let flight = step(3.5, j);
        let wide = mix(0.075, 0.115, flight);
        let ph = branchHash(vec2f(j, 31.0));
        let uc = q.y / wide + ph;
        let i = floor(uc);
        if ((i + 0.5 - ph) * wide > WING_SPAN) { continue; }
        let r1 = branchHash(vec2f(i, j + 40.0) + 0.5);
        let r2 = branchHash(vec2f(j + 40.0, i) + 13.7);
        let a = fract(uc) - 0.5 + (r1 - 0.5) * 0.1;
        let len = mix(2.6 * (0.86 + 0.28 * r2), mix(6.0, 9.5, smoothstep(0.1, 0.85, t)) * (0.94 + 0.12 * r2), flight);
        if (vane(a, (s - j) / len, mix(2.0, 6.0, flight), 0.131 * row, px, wide, len * row, &f)) {
          f.id = vec2f(i, j + 40.0);
          f.flight = flight;
          f.cover *= smoothstep(0.0, px, s * row);
          break;
        }
      }
      return f;
    }
    fn featherTail(q : vec2f, spread : f32, px : f32) -> Plume {
      var f = Plume(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec2f(0.0), 1.0);
      let r = length(q);
      if (q.x <= 0.0 || r > 1.05) { return f; }
      let count = 12.0;
      let uc = (atan2(q.y, q.x) / (2.0 * spread) + 0.5) * count;
      if (uc < -0.3 || uc > count + 0.3) { return f; }
      let i = clamp(floor(uc), 0.0, count - 1.0);
      let a = uc - i - 0.5;
      let inward = select(-1.0, 1.0, i + 0.5 < 0.5 * count);
      let wide = 2.0 * spread * r / count;
      for (var kk = 0; kk < 2; kk++) {
        let ii = i + select(0.0, inward, kk == 0);
        if (ii < 0.0 || ii > count - 1.0) { continue; }
        let aa = a - select(0.0, inward, kk == 0);
        let r2 = branchHash(vec2f(ii, 77.0));
        let long = 0.92 + 0.08 * r2 - 0.05 * abs(ii + 0.5 - 0.5 * count) / count;
        if (vane(aa, r / long, 6.0, 0.012, px, max(wide, 1e-3), long, &f)) {
          f.id = vec2f(ii, 90.0);
          break;
        }
      }
      return f;
    }
    struct BirdHit {
      cover : f32,
      s : f32,
      normal : vec3f,
      albedo : vec3f,
      gloss : f32,
      thin : f32,
    }
  ),
  CODE(
    fn birdShape(o : vec3f, dir : vec3f, bird : Boid, climbing : f32, soft : f32, px : f32, sunWay : vec3f, detail : bool) -> BirdHit {
      let size = bird.size;
      let flapping = smoothstep(0.06, 0.3, bird.effort);
      let stroke = bird.phase;
      let late = sin(stroke - 1.1);
      let shoulder = 0.12 + (0.4 + 0.35 * bird.effort) * flapping * sin(stroke);
      let wrist = mix(-0.08, -0.25 + 0.4 * late, flapping);
      let fold = flapping * 0.5 * (1.0 - late);
      let tuck = clamp(-climbing * 1.6, 0.0, 0.7);
      let sweep = 0.15 + 0.45 * max(fold, tuck) + 0.5 * tuck;
      let bodyR = size * vec3f(0.27, 0.045, 0.045);
      let body = hitEllipsoid(o, dir, bodyR);
      var cover = 1.0 - smoothstep(1.0 - soft, 1.0 + soft, body.near);
      var s = body.s;
      var kind = 0;
      var armNormal = vec3f(0.0, 0.0, 1.0);
      let beakAt = size * vec3f(0.262, 0.0, -0.004);
      let beakR = size * vec3f(0.07, 0.012, 0.014);
      let beak = hitEllipsoid(o - beakAt, dir, beakR);
      let beakSoft = clamp(px / beakR.y, 0.05, 0.45);
      let beakCover = 1.0 - smoothstep(1.0 - beakSoft, 1.0 + beakSoft, beak.near);
      if (beakCover > 0.0 && (cover <= 0.0 || beak.s < s)) {
        cover = beakCover;
        s = beak.s;
        kind = 3;
      }
      var plume = Plume(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec2f(0.0), 0.0);
      var sheetX = vec3f(1.0, 0.0, 0.0);
      var sheetY = vec3f(0.0, 1.0, 0.0);
      var sheetZ = vec3f(0.0, 0.0, 1.0);
      var sheetAt = vec2f(0.0);
      var sheetPx = 1.0;
      var outward = 0.0;
      var tailSpread = 0.3;
      let chordUnit = 0.17 * size;
      let spanUnit = 0.482 * size;
  ),
  CODE(
      for (var w = 0; w < 2; w++) {
        let wingSide = select(-1.0, 1.0, w == 0);
        let root = size * vec3f(0.03, 0.035 * wingSide, 0.015);
        let innerWay = vec3f(0.0, wingSide * cos(shoulder), sin(shoulder));
        let wristAt = root + innerWay * (size * 0.38);
        let outerWay = normalize(vec3f(-sweep, wingSide * cos(shoulder + wrist), sin(shoulder + wrist)));
        for (var part = 0; part < 2; part++) {
          let way = select(innerWay, outerWay, part == 1);
          let base = select(root, wristAt, part == 1);
          let fore = normalize(vec3f(1.0, 0.0, 0.0) - way * way.x);
          let face = cross(fore, way) * wingSide;
          let armR = size * select(vec3f(0.042, 0.2, 0.012), vec3f(0.03, 0.235, 0.008), part == 1);
          let armAt = base + way * (size * select(0.19, 0.22, part == 1)) + fore * (size * select(0.04, 0.028, part == 1));
          let arm = wingPanel(o, dir, armAt, way, armR);
          let armSoft = clamp(px / armR.z, 0.05, 0.45);
          let armCover = 1.0 - smoothstep(1.0 - armSoft, 1.0 + armSoft, arm.near);
          if (armCover > 0.0 && (cover <= 0.0 || arm.s < s)) {
            cover = armCover;
            s = arm.s;
            kind = 4;
            armNormal = arm.normal;
            outward = clamp((dot(o + dir * arm.s - base, way) + select(0.0, 0.38 * size, part == 1)) / (0.82 * size), 0.0, 1.0);
          }
          let facing = dot(dir, face);
          if (abs(facing) < 1e-5) { continue; }
          let t = dot(base - o, face) / facing;
          let rel = o + dir * t - base;
          let along = dot(rel, way);
          if (along < select(-0.1 * size, 0.0, part == 1) || along > select(0.38 * size, 0.6 * size, part == 1)) { continue; }
          let q = vec2f(dot(rel, fore) / chordUnit, (along + select(0.0, 0.38 * size, part == 1)) / spanUnit);
          let f = featherWing(q, px / chordUnit);
          if (f.cover > 0.0 && (cover <= 0.0 || t < s)) {
            cover = f.cover;
            s = t;
            kind = 1;
            plume = f;
            sheetX = fore;
            sheetY = way;
            sheetZ = face;
            sheetAt = q;
            sheetPx = px / chordUnit;
            outward = clamp(q.y / WING_SPAN, 0.0, 1.0);
          }
        }
      }
      let fan = clamp(abs(bird.turn) * 0.6 + max(climbing, 0.0) * 1.5, 0.0, 1.0);
      let tailUnit = 0.27 * size;
      if (abs(dir.z) > 1e-5) {
        let t = -o.z / dir.z;
        let at = o + dir * t;
        let q = vec2f(-0.17 * size - at.x, at.y) / tailUnit;
        let spread = mix(0.2, 0.55, fan);
        let f = featherTail(q, spread, px / tailUnit);
        if (f.cover > 0.0 && (cover <= 0.0 || t < s)) {
          cover = f.cover;
          s = t;
          kind = 2;
          plume = f;
          sheetX = vec3f(1.0, 0.0, 0.0);
          sheetY = vec3f(0.0, 1.0, 0.0);
          sheetZ = vec3f(0.0, 0.0, 1.0);
          sheetAt = q;
          sheetPx = px / tailUnit;
          tailSpread = spread;
        }
      }
      if (cover <= 0.0 || !detail) {
        return BirdHit(cover, s, body.normal, vec3f(0.9), 0.0, 0.0);
      }
  ),
  CODE(
      var normal = body.normal;
      var albedo = vec3f(0.92, 0.92, 0.90);
      var gloss = 0.0;
      var thin = 0.0;
      if (kind == 0) {
        let rows = 22.0;
        let scale = bodyR.x;
        let pixel = px / scale;
        let show = 1.0 - smoothstep(0.12, 0.4, pixel * rows / 2.0);
        if (show > 0.0) {
          let R = bodyR / scale;
          let p = R * normalize((o + dir * s) / bodyR);
          let f = featherAt(p, R, rows, pixel);
          let relief = 0.03 * clamp(2.5 * min(R.y, R.z), 0.1, 1.0);
          let e = max(0.0008, 0.75 * pixel);
          let k = vec2f(1.0, -1.0);
          let n = normalize(k.xyy * featheredSurface(p + k.xyy * e, R, rows, pixel, relief) + k.yyx * featheredSurface(p + k.yyx * e, R, rows, pixel, relief) +
                            k.yxy * featheredSurface(p + k.yxy * e, R, rows, pixel, relief) + k.xxx * featheredSurface(p + k.xxx * e, R, rows, pixel, relief));
          normal = normalize(mix(body.normal, n, show));
          let own = branchHash(vec2f(f.fid.x * 1.7 + 3.0, f.fid.y * 2.3 + 1.0));
          let fringe = 1.0 - smoothstep(0.0, 0.09, min(f.loc.w - abs(f.loc.x), (1.0 - f.loc.y) * 1.2));
          let tone = mix(0.86, 1.05, own) * (1.0 - 0.45 * f.loc.z) * mix(1.0, 0.8, 1.0 - smoothstep(0.0, 0.04, abs(f.loc.x))) * mix(1.0, 1.12, fringe);
          albedo *= mix(1.0, tone, show);
          if (u.s.w > 0.5) {
            let q = normalize(p / R);
            let ru = atan2(q.z, q.y);
            let rv = acos(clamp(q.x, -1.0, 1.0));
            let alongWay = normalize(R * vec3f(-sin(rv), cos(rv) * cos(ru), cos(rv) * sin(ru)));
            var barb = alongWay * 0.78 + clamp(f.loc.x * 30.0, -1.0, 1.0) * vec3f(0.0, -sin(ru), cos(ru)) * 0.62;
            barb = normalize(barb - n * dot(barb, n));
            let between = dot(barb, normalize(sunWay - dir));
            gloss = show * 0.3 * pow(max(0.0, 1.0 - between * between), 36.0) * smoothstep(0.0, 0.3, dot(n, sunWay)) * (1.0 - 0.7 * f.loc.z);
          }
        }
        let at = o + dir * s;
        let eyeAt = size * vec3f(0.212, select(-0.0262, 0.0262, at.y > 0.0), 0.009);
        let away = length(at - eyeAt);
        let edge = max(px, 0.0006 * size);
        let ring = 1.0 - smoothstep(0.0085 * size - edge, 0.0085 * size + edge, away);
        let eye = 1.0 - smoothstep(0.0065 * size - edge, 0.0065 * size + edge, away);
        if (ring > 0.0) {
          normal = normalize(mix(normal, body.normal, ring));
          albedo = mix(albedo, vec3f(0.85, 0.8, 0.55), ring);
          albedo = mix(albedo, vec3f(0.015), eye);
          gloss = mix(gloss, 0.9 * pow(max(dot(body.normal, normalize(sunWay - dir)), 0.0), 90.0), eye);
        }
  ),
  CODE(
      } else if (kind == 4) {
        normal = armNormal;
        albedo = mix(vec3f(0.62, 0.64, 0.66), vec3f(0.07, 0.07, 0.08), smoothstep(0.6, 0.8, outward) * 0.75);
        if (armNormal.z < -0.2) { albedo = mix(albedo, vec3f(0.9), 0.55); }
      } else if (kind == 3) {
        normal = beak.normal;
        let toTip = clamp((o.x + dir.x * s - beakAt.x) / beakR.x, 0.0, 1.0);
        albedo = mix(vec3f(1.0, 0.4, 0.03), vec3f(0.9, 0.24, 0.02), toTip);
        gloss = 0.25 * pow(max(dot(normal, normalize(sunWay - dir)), 0.0), 40.0);
      } else {
        thin = 1.0;
        let below = dot(sheetZ, dir) > 0.0;
        let flat = select(sheetZ, -sheetZ, below);
        let e = max(0.0012, 0.75 * sheetPx);
        var hx = 0.0;
        var hy = 0.0;
        if (kind == 1) {
          hx = featherWing(sheetAt + vec2f(e, 0.0), sheetPx).h - plume.h;
          hy = featherWing(sheetAt + vec2f(0.0, e), sheetPx).h - plume.h;
        } else {
          hx = plume.h - featherTail(sheetAt + vec2f(e, 0.0), tailSpread, sheetPx).h;
          hy = featherTail(sheetAt + vec2f(0.0, e), tailSpread, sheetPx).h - plume.h;
        }
        normal = normalize(flat - 0.03 * clamp(vec2f(hx, hy) / e, vec2f(-8.0), vec2f(8.0)).x * sheetX
                                - 0.03 * clamp(hy / e, -8.0, 8.0) * sheetY);
        if (kind == 1) {
          let dark = smoothstep(0.6, 0.8, outward) * mix(0.5, 1.0, plume.flight);
          albedo = mix(vec3f(0.62, 0.64, 0.66), vec3f(0.07, 0.07, 0.08), dark);
          albedo = mix(albedo, vec3f(0.93), plume.flight * (1.0 - dark) * smoothstep(0.8, 0.96, plume.b));
          if (below) { albedo = mix(albedo, vec3f(0.9), 0.55); }
        }
        let own = branchHash(plume.id * vec2f(1.7, 2.3) + vec2f(3.0, 1.0));
        albedo *= mix(0.9, 1.05, own) * (1.0 - 0.45 * plume.occ) * mix(1.0, 0.75, 1.0 - smoothstep(0.0, 0.04, abs(plume.a)));
        if (u.s.w > 0.5) {
          var barb = -sheetX * 0.78 + clamp(plume.a * 30.0, -1.0, 1.0) * sheetY * 0.62;
          barb = normalize(barb - normal * dot(barb, normal));
          let between = dot(barb, normalize(sunWay - dir));
          gloss = 0.3 * pow(max(0.0, 1.0 - between * between), 36.0) * smoothstep(0.0, 0.3, dot(normal, sunWay)) * (1.0 - 0.7 * plume.occ)
                * (1.0 - smoothstep(0.02, 0.08, sheetPx));
        }
      }
      return BirdHit(cover, s, normal, albedo, gloss, thin);
    }
  ),
  CODE(
    fn birdsAlong(ro : vec3f, rd : vec3f, tSea : f32) -> BirdLook {
      var look = BirdLook(vec4f(0.0), 1e9);
      if (u.o.y < 0.5) { return look; }
      let reachUp = BIRD_DEPTH + 0.8;
      let slope = select(rd.y, select(-1e-4, 1e-4, rd.y >= 0.0), abs(rd.y) < 1e-4);
      let ta = (BIRD_HEIGHT - reachUp - ro.y) / slope;
      let tb = (BIRD_HEIGHT + reachUp - ro.y) / slope;
      let lo = max(min(ta, tb), 0.0);
      let hi = min(min(max(ta, tb), tSea), BIRD_SIGHT);
      if (hi <= lo) { return look; }
      let sun = u.c.xyz;
      let samples = clamp(i32(ceil((hi - lo) * length(rd.xz) / (0.85 * BIRD_CELL_SIZE))), 1, 16);
      let reach = (hi - lo) / f32(samples);
      var best = vec4f(0.0);
      var nearest = 1e9;
      for (var n = 0; n < samples; n++) {
        let tm = mix(lo, hi, (f32(n) + 0.5) / f32(samples));
        let here = ro + rd * tm;
        let m = worldToMap(here).xz;
        let p = m - FISH_WORLD * floor(m / FISH_WORLD);
        let corner = vec2i(floor(p / BIRD_CELL_SIZE - 0.5));
        for (var j = 0; j < 2; j++) {
          for (var i = 0; i < 2; i++) {
            let c = (corner + vec2i(i, j) + BIRD_GRID) % BIRD_GRID;
            let cell = u32(c.y * BIRD_GRID + c.x);
            let count = min(birdCells[cell * BIRD_STRIDE], BIRD_PER_CELL);
            for (var k = 0u; k < count; k++) {
              let bird = birds[birdCells[cell * BIRD_STRIDE + 1u + k]];
              var flat = p - bird.pos;
              flat -= FISH_WORLD * round(flat / FISH_WORLD);
              let d = vec3f(flat.x, here.y - BIRD_HEIGHT - bird.lift, flat.y);
              let alongRay = dot(d, rd);
              if (dot(d, d) - alongRay * alongRay > 1.3) { continue; }
              if (abs(alongRay) > 0.5 * reach + 0.01) { continue; }
              let frame = birdFrame(bird);
              let o = vec3f(dot(d, frame.ahead), dot(d, frame.left), dot(d, frame.top));
              let dir = vec3f(dot(rd, frame.ahead), dot(rd, frame.left), dot(rd, frame.top));
              let footprint = max(tm + dot(-o, dir), 0.1) / (u.b.w * u.a.y);
              let far = clamp(0.5 / footprint - 0.5, 0.0, 1.0) * (1.0 - smoothstep(0.75 * BIRD_SIGHT, BIRD_SIGHT, tm));
              if (far <= 0.0) { continue; }
              let sunWay = vec3f(dot(sun, frame.ahead), dot(sun, frame.left), dot(sun, frame.top));
              let hit = birdShape(o, dir, bird, frame.ahead.y, clamp(footprint / (0.045 * bird.size), 0.05, 0.45), footprint, sunWay, true);
              let distance = tm + hit.s;
              if (hit.cover <= 0.0 || (best.a > 0.5 && distance > nearest)) { continue; }
              nearest = distance;
              let nrm = frame.ahead * hit.normal.x + frame.left * hit.normal.y + frame.top * hit.normal.z;
              let ambient = mix(waterBounce() * 0.35 + skyAmbient() * 0.45, skyAmbient(), nrm.y * 0.5 + 0.5);
              let lit = hit.albedo * (sunColor() * 0.8 * (max(dot(nrm, sun), 0.0) + hit.thin * 0.4 * max(dot(-nrm, sun), 0.0)) + ambient) + sunColor() * hit.gloss;
              let f = distance * HAZE;
              best = vec4f(mix(skyGlow(normalize(vec3f(rd.x, 0.0, rd.z))), lit, exp(-f * f)), hit.cover * far);
            }
          }
        }
      }
      look.color = best;
      look.distance = nearest;
      return look;
    }
  ),
  CODE(
    fn birdShadow(mapXZ : vec2f, y : f32) -> f32 {
      let sun = u.c.xyz;
      if (u.o.y < 0.5 || u.s.z < 0.5 || sun.y < 0.08) { return 1.0; }
      let reachUp = BIRD_DEPTH + 0.8;
      let lo = max((BIRD_HEIGHT - reachUp - y) / sun.y, 0.0);
      let hi = min(BIRD_HEIGHT + reachUp - y, 9.0) / sun.y;
      if (hi <= lo) { return 1.0; }
      let samples = clamp(i32(ceil((hi - lo) * length(sun.xz) / (0.85 * BIRD_CELL_SIZE))), 1, 4);
      let reach = (hi - lo) / f32(samples);
      var light = 1.0;
      for (var n = 0; n < samples; n++) {
        let tm = mix(lo, hi, (f32(n) + 0.5) / f32(samples));
        let hereXZ = mapXZ + sun.xz * tm;
        let p = hereXZ - FISH_WORLD * floor(hereXZ / FISH_WORLD);
        let corner = vec2i(floor(p / BIRD_CELL_SIZE - 0.5));
        for (var j = 0; j < 2; j++) {
          for (var i = 0; i < 2; i++) {
            let c = (corner + vec2i(i, j) + BIRD_GRID) % BIRD_GRID;
            let cell = u32(c.y * BIRD_GRID + c.x);
            let count = min(birdCells[cell * BIRD_STRIDE], BIRD_PER_CELL);
            for (var k = 0u; k < count; k++) {
              let bird = birds[birdCells[cell * BIRD_STRIDE + 1u + k]];
              var flat = p - bird.pos;
              flat -= FISH_WORLD * round(flat / FISH_WORLD);
              let d = vec3f(flat.x, y + sun.y * tm - BIRD_HEIGHT - bird.lift, flat.y);
              let alongRay = dot(d, sun);
              if (dot(d, d) - alongRay * alongRay > 1.3) { continue; }
              if (abs(alongRay) > 0.5 * reach + 0.01) { continue; }
              let frame = birdFrame(bird);
              let o = vec3f(dot(d, frame.ahead), dot(d, frame.left), dot(d, frame.top));
              let dir = vec3f(dot(sun, frame.ahead), dot(sun, frame.left), dot(sun, frame.top));
              let blur = 0.0047 * max(tm + dot(-o, dir), 0.0) + 0.01;
              let hit = birdShape(o, dir, bird, frame.ahead.y, clamp(blur / (0.045 * bird.size), 0.05, 0.9), blur, vec3f(0.0), false);
              let lowness = 1.0 - smoothstep(6.5, 9.0, y + sun.y * tm - d.y - u.g.x);
              light = min(light, 1.0 - 0.85 * hit.cover * clamp(0.07 / blur, 0.2, 1.0) * lowness);
            }
          }
        }
      }
      return light;
    }
    @fragment
    fn fs_final(@builtin(position) frag : vec4f) -> @location(0) vec4f {
      let ro = u.b.xyz;
      let rd = viewRay(frag.xy);
      viewFocus = u.v.w * (1.0 - smoothstep(0.08, 0.32, abs(frag.y / u.a.y - 0.5)));
      let scene = backdrop(frag.xy, ro, rd);
      let bird = birdsAlong(ro, rd, waterHit(ro, rd).t);
      let uv = frag.xy / u.a.xy;
      let cloud = textureSampleLevel(cloudTex, linearSamp, uv, 0.0);
      let cloudNear = textureSampleLevel(cloudNearTex, linearSamp, uv, 0.0).r;
      let behind = cloud.rgb + mix(scene, bird.color.rgb, bird.color.a) * cloud.a;
      let inFront = mix(cloud.rgb + scene * cloud.a, bird.color.rgb, bird.color.a);
      var col = mix(behind, inFront, smoothstep(-0.002, 0.002, 1.0 / max(bird.distance, 0.1) - cloudNear));
      col = tonemap(col) + (ign(frag.xy) - 0.5) / 255.0;
      return vec4f(col, 1.0);
    }
  ),
  NULL
};
// clang-format on
