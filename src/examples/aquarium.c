#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* cimgui for GUI */
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* sokol_fetch for async file loading */
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

/* sokol_time for timing */
#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* cJSON for JSON parsing */
#include <cJSON.h>

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
 * WebGPU Example - Aquarium
 *
 * Aquarium is a complete port of the classic WebGL Aquarium to modern WebGPU,
 * showcasing advanced rendering techniques and efficient GPU programming.
 *
 * Ref:
 * https://github.com/webgfx/aquarium-web/tree/main/webgpu
 * https://github.com/webatintel/aquarium
 * https://webglsamples.org/aquarium/aquarium.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define DEPTH_STENCIL_FORMAT (WGPUTextureFormat_Depth24Plus)

#define OPTION_DEFINITION_COUNT (8u)
#define FISH_COUNT_PRESET_COUNT (10u)
#define VIEW_PRESET_COUNT (6u)
#define SCENE_DEFINITION_COUNT (28u)
#define FISH_SPECIES_COUNT (5u)

/* Asset loading constants */
#define MAX_MODELS_PER_SCENE 4
#define MAX_PROP_PLACEMENTS 200
#define MAX_VERTICES 65536
#define MAX_INDICES 65536
#define ASSET_FILE_BUFFER_SIZE (4 * 1024 * 1024) /* 4MB for asset files */
/* Path relative to executable in build/Desktop-Debug/Debugx64/ */
#define AQUARIUM_ASSETS_PATH "assets/models/Aquarium/"

/* -------------------------------------------------------------------------- *
 * Config
 * -------------------------------------------------------------------------- */

typedef struct {
  float speed;
  float target_height;
  float target_radius;
  float eye_height;
  float eye_radius;
  float eye_speed;
  float field_of_view;
  float ambient_red;
  float ambient_green;
  float ambient_blue;
  float fog_power;
  float fog_mult;
  float fog_offset;
  float fog_red;
  float fog_green;
  float fog_blue;
} globals_t;

static globals_t default_globals = {
  .speed         = 1.0f,
  .target_height = 0.0f,
  .target_radius = 88.0f,
  .eye_height    = 38.0f,
  .eye_radius    = 69.0f,
  .eye_speed     = 0.06f,
  .field_of_view = 85.0f,
  .ambient_red   = 0.22f,
  .ambient_green = 0.25f,
  .ambient_blue  = 0.39f,
  .fog_power     = 14.5f,
  .fog_mult      = 1.66f,
  .fog_offset    = 0.53f,
  .fog_red       = 0.54f,
  .fog_green     = 0.86f,
  .fog_blue      = 1.0f,
};

typedef struct {
  float fish_height_range;
  float fish_height;
  float fish_speed;
  float fish_offset;
  float fish_xclock;
  float fish_yclock;
  float fish_zclock;
  float fish_tail_speed;
} fish_t;

static fish_t default_fish = {
  .fish_height_range = 1.0f,
  .fish_height       = 25.0f,
  .fish_speed        = 0.124f,
  .fish_offset       = 0.52f,
  .fish_xclock       = 1.0f,
  .fish_yclock       = 0.556f,
  .fish_zclock       = 1.0f,
  .fish_tail_speed   = 1.0f,
};

typedef struct {
  float refraction_fudge;
  float eta;
  float tank_color_fudge;
} inner_const_t;

static inner_const_t default_inner_const = {
  .refraction_fudge = 3.0f,
  .eta              = 1.0f,
  .tank_color_fudge = 0.8f,
};

static struct {
  const char* id;
  const char* label;
  bool default_value;
} option_definitions[OPTION_DEFINITION_COUNT] = {
  // clang-format off
  { .id = "normalMaps", .label = "Normal Maps", .default_value = true  },
  { .id = "reflection", .label = "Reflection",  .default_value = true  },
  { .id = "tank",       .label = "Tank",        .default_value = true  },
  { .id = "museum",     .label = "Museum",      .default_value = true  },
  { .id = "fog",        .label = "Fog",         .default_value = true  },
  { .id = "bubbles",    .label = "Bubbles",     .default_value = true  },
  { .id = "lightRays",  .label = "Light Rays",  .default_value = true  },
  { .id = "lasers",     .label = "Lasers",      .default_value = false },
  // clang-format on
};

static uint32_t fish_count_presets[FISH_COUNT_PRESET_COUNT]
  = {1, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000};

typedef struct {
  const char* name;
  globals_t globals;
  inner_const_t inner_const;
} view_preset_t;

static view_preset_t view_presets[VIEW_PRESET_COUNT] = {
  {
    .name = "Inside (A)",
    .globals = {
      .target_height = 63.3f,
      .target_radius = 91.6f,
      .eye_height    = 7.5f,
      .eye_radius    = 13.2f,
      .eye_speed     = 0.0258f,
      .field_of_view = 82.699f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.502f,
      .ambient_blue  = 0.706f,
      .fog_power     = 16.5f,
      .fog_mult      = 1.5f,
      .fog_offset    = 0.738f,
      .fog_red       = 0.338f,
      .fog_green     = 0.81f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.796f,
    },
  },
  {
    .name = "Outside (A)",
    .globals = {
      .target_height = 17.1f,
      .target_radius = 69.2f,
      .eye_height    = 59.1f,
      .eye_radius    = 124.4f,
      .eye_speed     = 0.0258f,
      .field_of_view = 56.923f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.46f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
  {
    .name = "Inside (Original)",
    .globals = {
      .target_height = 0.0f,
      .target_radius = 88.0f,
      .eye_height    = 38.0f,
      .eye_radius    = 69.0f,
      .eye_speed     = 0.0258f,
      .field_of_view = 64.0f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 16.5f,
      .fog_mult      = 1.5f,
      .fog_offset    = 0.738f,
      .fog_red       = 0.338f,
      .fog_green     = 0.81f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.796f,
    },
  },
  {
    .name = "Outside (Original)",
    .globals = {
      .target_height = 72.0f,
      .target_radius = 73.0f,
      .eye_height    = 3.9f,
      .eye_radius    = 120.0f,
      .eye_speed     = 0.0258,
      .field_of_view = 74.0f,
      .ambient_red   = 0.218f,
      .ambient_green = 0.246f,
      .ambient_blue  = 0.394f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.46f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
  {
    .name = "Center (LG)",
    .globals = {
      .target_height = 24.0f,
      .target_radius = 73.0f,
      .eye_height    = 24.0f,
      .eye_radius    = 0.0f,
      .eye_speed     = 0.06f,
      .field_of_view = 60.0f,
      .ambient_red   = 0.22f,
      .ambient_green = 0.25f,
      .ambient_blue  = 0.39f,
      .fog_power     = 14.5f,
      .fog_mult      = 1.3f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.54f,
      .fog_green     = 0.86f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 0.8f,
    },
  },
  {
    .name = "Outside (LG)",
    .globals = {
      .target_height = 20.0f,
      .target_radius = 127.0f,
      .eye_height    = 39.9f,
      .eye_radius    = 124.0f,
      .eye_speed     = 0.06f,
      .field_of_view = 24.0f,
      .ambient_red   = 0.22f,
      .ambient_green = 0.25f,
      .ambient_blue  = 0.39f,
      .fog_power     = 27.1f,
      .fog_mult      = 1.2f,
      .fog_offset    = 0.53f,
      .fog_red       = 0.382f,
      .fog_green     = 0.602f,
      .fog_blue      = 1.0f,
    },
    .inner_const = {
      .refraction_fudge = 3.0f,
      .eta              = 1.0f,
      .tank_color_fudge = 1.0f,
    },
  },
};

/* -------------------------------------------------------------------------- *
 * Scene registry
 * -------------------------------------------------------------------------- */

static struct {
  const char* name;
  const char* program;
  bool blend;
  bool fog;
  bool lasers;
  const char* group;
} scene_definitions[SCENE_DEFINITION_COUNT] = {
  // clang-format off
  { .name = "SmallFishA",      .program = "fishReflection"                                     },
  { .name = "MediumFishA",     .program = "fishNormal"                                         },
  { .name = "MediumFishB",     .program = "fishReflection"                                     },
  { .name = "BigFishA",        .program = "fishNormal",    .lasers = true                      },
  { .name = "BigFishB",        .program = "fishNormal",    .lasers = true                      },
  { .name = "Arch",            .program = "diffuse"                                            },
  { .name = "Coral",           .program = "diffuse"                                            },
  { .name = "CoralStoneA",     .program = "diffuse"                                            },
  { .name = "CoralStoneB",     .program = "diffuse"                                            },
  { .name = "EnvironmentBox",  .program = "diffuse",       .fog    = false, .group = "outside" },
  { .name = "FloorBase_Baked", .program = "diffuse"                                            },
  { .name = "FloorCenter",     .program = "diffuse"                                            },
  { .name = "GlobeBase",       .program = "diffuse",       .fog    = false                     },
  { .name = "GlobeInner",      .program = "inner"                                              },
  { .name = "GlobeOuter",      .program = "outer",         .blend  = true                      },
  { .name = "RockA",           .program = "diffuse"                                            },
  { .name = "RockB",           .program = "diffuse"                                            },
  { .name = "RockC",           .program = "diffuse"                                            },
  { .name = "RuinColumn",      .program = "diffuse"                                            },
  { .name = "Skybox",          .program = "diffuse",       .fog    = false, .group = "outside" },
  { .name = "Stone",           .program = "diffuse"                                            },
  { .name = "Stones",          .program = "diffuse"                                            },
  { .name = "SunknShip",       .program = "diffuse"                                            },
  { .name = "SunknSub",        .program = "diffuse"                                            },
  { .name = "SupportBeams",    .program = "diffuse",       .fog    = false, .group = "outside" },
  { .name = "SeaweedA",        .program = "seaweed",       .blend  = true,  .group = "seaweed" },
  { .name = "SeaweedB",        .program = "seaweed",       .blend  = true,  .group = "seaweed" },
  { .name = "TreasureChest",   .program = "diffuse"                                            },
  // clang-format on
};

static struct {
  const char* name;
  float speed;
  float speed_range;
  float radius;
  float radius_range;
  float tail_speed;
  float height_offset;
  float height_range;
  bool lasers;
  float laser_rot;
  float laser_off[3];
  float laser_scale[3];
  struct {
    float fish_length;
    float fish_wave_length;
    float fish_bend_amount;
  } const_uniforms;
} fish_species[FISH_SPECIES_COUNT] = {
  {
    .name          = "SmallFishA",
    .speed         = 1.0f,
    .speed_range   = 1.5f,
    .radius        = 30.0f,
    .radius_range  = 25.0f,
    .tail_speed    = 10.0f,
    .height_offset = 0.0f,
    .height_range  = 16.0f,
    .const_uniforms = {
      .fish_length      = 10.0f,
      .fish_wave_length = 1.0f,
      .fish_bend_amount = 2.0f,
    },
  },
  {
    .name          = "MediumFishA",
    .speed         = 1.0f,
    .speed_range   = 2.0f,
    .radius        = 10.0f,
    .radius_range  = 20.0f,
    .tail_speed    = 1.0f,
    .height_offset = 0.0f,
    .height_range  = 16.0f,
    .const_uniforms = {
      .fish_length      = 10.0f,
      .fish_wave_length = -2.0f,
      .fish_bend_amount = 2.0f,
    },
  },
  {
    .name          = "MediumFishB",
    .speed         = 0.5f,
    .speed_range   = 4.0f,
    .radius        = 10.0f,
    .radius_range  = 20.0f,
    .tail_speed    = 3.0f,
    .height_offset = -8.0f,
    .height_range  = 5.0f,
    .const_uniforms = {
      .fish_length      = 10.0f,
      .fish_wave_length = -2.0f,
      .fish_bend_amount = 2.0f,
    },
  },
  {
    .name          = "BigFishA",
    .speed         = 0.5f,
    .speed_range   = 0.5f,
    .radius        = 50.0f,
    .radius_range  = 3.0f,
    .tail_speed    = 1.5f,
    .height_offset = 0.0f,
    .height_range  = 16.0f,
    .lasers        = true,
    .laser_rot     = 0.04f,
    .laser_off     = {0.0f, 0.1f, 9.0f},
    .laser_scale   = {0.3f, 0.3f, 1000.0f},
    .const_uniforms = {
      .fish_length      = 10.0f,
      .fish_wave_length = -1.0f,
      .fish_bend_amount = 0.5f,
    },
  },
  {
    .name          = "BigFishB",
    .speed         = 0.5f,
    .speed_range   = 0.5f,
    .radius        = 45.0f,
    .radius_range  = 3.0f,
    .tail_speed    = 1.0f,
    .height_offset = 0.0f,
    .height_range  = 16.0f,
    .lasers        = true,
    .laser_rot     = 0.04f,
    .laser_off     = {0.0f, -0.3f, 9.0f},
    .laser_scale   = {0.3f,  0.3f, 1000.0f},
    .const_uniforms = {
      .fish_length      = 10.0f,
      .fish_wave_length = -0.7f,
      .fish_bend_amount = 0.3f,
    },
  },
};

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* bubble_shader_wgsl;
static const char* diffuse_shader_wgsl;
static const char* fish_shader_p1_wgsl;
static const char* fish_shader_p2_wgsl;
static const char* inner_shader_p1_wgsl;
static const char* inner_shader_p2_wgsl;
static const char* laser_shader_wgsl;
static const char* light_ray_shader_wgsl;
static const char* outer_shader_wgsl;
static const char* seaweed_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Shader loader
 * -------------------------------------------------------------------------- */

static char* get_concatenated_shader(const char* s1, const char* s2)
{
  size_t len1      = strlen(s1);
  size_t len2      = strlen(s2);
  size_t total_len = len1 + len2 + 1; /* +1 for null terminator */

  char* full_shader = malloc(total_len);
  if (full_shader == NULL) {
    return NULL; /* Handle allocation failure */
  }

  snprintf(full_shader, total_len, "%s%s", s1, s2);

  return full_shader;
}

static WGPUShaderModule load_shader_module(WGPUDevice device, const char* path,
                                           const char* label)
{
  /* Get WGSL shader cpde */
  const char* wgsl_source_code  = NULL;
  char* wgsl_source_code_concat = NULL;
  if (strcmp(path, "shaders/bubble.wgsl") == 0) {
    wgsl_source_code = bubble_shader_wgsl;
  }
  else if (strcmp(path, "shaders/diffuse.wgsl") == 0) {
    wgsl_source_code = diffuse_shader_wgsl;
  }
  else if (strcmp(path, "shaders/fish.wgsl") == 0) {
    wgsl_source_code_concat
      = get_concatenated_shader(fish_shader_p1_wgsl, fish_shader_p2_wgsl);
  }
  else if (strcmp(path, "shaders/inner.wgsl") == 0) {
    wgsl_source_code_concat
      = get_concatenated_shader(inner_shader_p1_wgsl, inner_shader_p2_wgsl);
  }
  else if (strcmp(path, "shaders/laser.wgsl") == 0) {
    wgsl_source_code = laser_shader_wgsl;
  }
  else if (strcmp(path, "shaders/light_ray.wgsl") == 0) {
    wgsl_source_code = light_ray_shader_wgsl;
  }
  else if (strcmp(path, "shaders/outer.wgsl") == 0) {
    wgsl_source_code = outer_shader_wgsl;
  }
  else if (strcmp(path, "shaders/seaweed.wgsl") == 0) {
    wgsl_source_code = seaweed_shader_wgsl;
  }

  if (wgsl_source_code == NULL && wgsl_source_code_concat == NULL) {
    fprintf(stderr, "Failed to load shader from %s", path);
    return NULL;
  }

  /* Create shader module */
  WGPUShaderSourceWGSL shader_code_desc
    = {.chain = {.sType = WGPUSType_ShaderSourceWGSL},
       .code  = {
          .data   = (wgsl_source_code != NULL) ? wgsl_source_code :
                                                 wgsl_source_code_concat,
          .length = WGPU_STRLEN,
       }};
  WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
    device, &(WGPUShaderModuleDescriptor){
              .nextInChain = &shader_code_desc.chain,
              .label       = STRVIEW(label),
            });

  if (wgsl_source_code_concat != NULL) {
    free(wgsl_source_code_concat);
  }

  return shader_module;
}

/* -------------------------------------------------------------------------- *
 * Bindings
 * -------------------------------------------------------------------------- */

WGPUBindGroupLayout create_bind_group_layout(WGPUDevice device,
                                             const char* label,
                                             WGPUBindGroupLayoutEntry* entries,
                                             uint32_t entry_count)
{
  return wgpuDeviceCreateBindGroupLayout(device,
                                         &(WGPUBindGroupLayoutDescriptor){
                                           .label      = STRVIEW(label),
                                           .entryCount = entry_count,
                                           .entries    = entries,
                                         });
}

WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBindGroupLayout layout,
                                WGPUBindGroupEntry* entries,
                                uint32_t entry_count, const char* label)
{
  return wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                             .label      = STRVIEW(label),
                                             .layout     = layout,
                                             .entryCount = entry_count,
                                             .entries    = entries,
                                           });
}

WGPUBuffer create_uniform_buffer(WGPUDevice device, uint64_t size,
                                 const char* label)
{
  return wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW(label),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = size,
            });
}

/* -------------------------------------------------------------------------- *
 * Math functions
 * -------------------------------------------------------------------------- */

float math_random(void)
{
  return (float)rand() / (float)RAND_MAX;
}

typedef struct {
  float m[16];
} mat4_t;

static void mat4_identity(mat4_t* out)
{
  memset(out->m, 0, sizeof(out->m));
  out->m[0] = out->m[5] = out->m[10] = out->m[15] = 1.0f;
}

static void mat4_perspective_yfov(mat4_t* out, float fovy_rad, float aspect,
                                  float near, float far)
{
  float f         = 1.0f / tanf(fovy_rad / 2.0f);
  float range_inv = 1.0f / (near - far);
  memset(out->m, 0, sizeof(out->m));
  out->m[0]  = f / aspect;
  out->m[5]  = f;
  out->m[10] = (near + far) * range_inv;
  out->m[11] = -1.0f;
  out->m[14] = near * far * range_inv * 2.0f;
}

static void vec3_subtract(const float* a, const float* b, float* out)
{
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static void vec3_add(const float* a, const float* b, float* out)
{
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

static void vec3_scale(const float* v, float s, float* out)
{
  out[0] = v[0] * s;
  out[1] = v[1] * s;
  out[2] = v[2] * s;
}

static void vec3_cross(const float* a, const float* b, float* out)
{
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

static float vec3_dot(const float* a, const float* b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float vec3_length(const float* v)
{
  return sqrtf(vec3_dot(v, v));
}

static void vec3_normalize(const float* v, float* out)
{
  float len = vec3_length(v);
  if (len > 0.0f) {
    out[0] = v[0] / len;
    out[1] = v[1] / len;
    out[2] = v[2] / len;
  }
  else {
    out[0] = out[1] = out[2] = 0.0f;
  }
}

static float lerp(float a, float b, float t)
{
  return a + (b - a) * t;
}

static float clamp(float value, float min, float max)
{
  return fminf(max, fmaxf(min, value));
}

static void mat4_lookat(mat4_t* out, const float* eye, const float* target,
                        const float* up)
{
  float z[3], x[3], y[3];
  vec3_subtract(eye, target, z);
  vec3_normalize(z, z);
  vec3_cross(up, z, x);
  vec3_normalize(x, x);
  vec3_cross(z, x, y);

  out->m[0]  = x[0];
  out->m[1]  = y[0];
  out->m[2]  = z[0];
  out->m[3]  = 0.0f;
  out->m[4]  = x[1];
  out->m[5]  = y[1];
  out->m[6]  = z[1];
  out->m[7]  = 0.0f;
  out->m[8]  = x[2];
  out->m[9]  = y[2];
  out->m[10] = z[2];
  out->m[11] = 0.0f;
  out->m[12] = -vec3_dot(x, eye);
  out->m[13] = -vec3_dot(y, eye);
  out->m[14] = -vec3_dot(z, eye);
  out->m[15] = 1.0f;
}

static void mat4_multiply(const mat4_t* a, const mat4_t* b, mat4_t* out)
{
  /* Column-major matrix multiplication: out = a * b
   * out[col][row] = sum_k( a[k][row] * b[col][k] ) */
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      float sum = 0.0f;
      for (int k = 0; k < 4; ++k) {
        sum += a->m[k * 4 + row] * b->m[col * 4 + k];
      }
      out->m[col * 4 + row] = sum;
    }
  }
}

static void mat4_translate(const mat4_t* matrix, const float* translation,
                           mat4_t* out)
{
  memcpy(out->m, matrix->m, sizeof(out->m));
  out->m[12] = matrix->m[0] * translation[0] + matrix->m[4] * translation[1]
               + matrix->m[8] * translation[2] + matrix->m[12];
  out->m[13] = matrix->m[1] * translation[0] + matrix->m[5] * translation[1]
               + matrix->m[9] * translation[2] + matrix->m[13];
  out->m[14] = matrix->m[2] * translation[0] + matrix->m[6] * translation[1]
               + matrix->m[10] * translation[2] + matrix->m[14];
  out->m[15] = matrix->m[3] * translation[0] + matrix->m[7] * translation[1]
               + matrix->m[11] * translation[2] + matrix->m[15];
}

static void mat4_scale(const mat4_t* matrix, const float* scale, mat4_t* out)
{
  memcpy(out->m, matrix->m, sizeof(out->m));
  out->m[0] *= scale[0];
  out->m[1] *= scale[0];
  out->m[2] *= scale[0];
  out->m[3] *= scale[0];
  out->m[4] *= scale[1];
  out->m[5] *= scale[1];
  out->m[6] *= scale[1];
  out->m[7] *= scale[1];
  out->m[8] *= scale[2];
  out->m[9] *= scale[2];
  out->m[10] *= scale[2];
  out->m[11] *= scale[2];
}

static void mat4_transpose(const mat4_t* matrix, mat4_t* out)
{
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out->m[i * 4 + j] = matrix->m[j * 4 + i];
    }
  }
}

static void mat4_inverse(const mat4_t* m, mat4_t* out)
{
  const float* a = m->m;
  float inv[16], det;
  int i;
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
           + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];

  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
           - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];

  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
           + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];

  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
            - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];

  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
           - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];

  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
           + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];

  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
           - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];

  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
            + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];

  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
           + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];

  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
           - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];

  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
            + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];

  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
            - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];

  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
           - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];

  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
           + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];

  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
            - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];

  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
            + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

  det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (det == 0) {
    mat4_identity(out);
    return;
  }
  det = 1.0f / det;
  for (i = 0; i < 16; i++)
    out->m[i] = inv[i] * det;
}

/* -------------------------------------------------------------------------- *
 * Bubbles Animation.
 * -------------------------------------------------------------------------- */

typedef struct {
  float timer;
  float position[3];
} bubble_emitter_instance_t;

typedef struct {
  int num_sets;
  float trigger_interval[2];
  float radius_range[2];
  bubble_emitter_instance_t* emitters;
  int index;
  void (*trigger_callback)(float pos[3]);
} bubble_emitter_t;

static float bubble_emitter_random_interval(const bubble_emitter_t* this);

static void bubble_emitter_init(bubble_emitter_t* this, int num_sets,
                                float trigger_interval[2],
                                float radius_range[2])
{
  memset(this, 0, sizeof(bubble_emitter_t));
  this->num_sets = num_sets;
  memcpy(this->trigger_interval, trigger_interval, sizeof(float) * 2);
  memcpy(this->radius_range, radius_range, sizeof(float) * 2);
  if (num_sets > 0) {
    this->emitters = malloc(num_sets * sizeof(bubble_emitter_instance_t));
    for (int i = 0; i < num_sets; ++i) {
      bubble_emitter_instance_t* emitter = &this->emitters[i];
      emitter->timer                     = bubble_emitter_random_interval(this);
      emitter->position[0] = emitter->position[1] = emitter->position[2] = 0.0f;
    }
  }
}

static void bubble_emitter_destroy(bubble_emitter_t* this)
{
  if (this->emitters) {
    free(this->emitters);
    this->emitters = NULL;
  }
}

static void bubble_random_on_trigger(bubble_emitter_t* this,
                                     void (*callback)(float pos[3]))
{
  this->trigger_callback = callback;
}

static float bubble_emitter_random_interval(const bubble_emitter_t* this)
{
  const float min = this->trigger_interval[0], max = this->trigger_interval[1];
  return min + math_random() * (max - min);
}

static void bubble_emitter_update(bubble_emitter_t* this, float delta_seconds,
                                  globals_t* globals)
{
  for (int i = 0; i < this->num_sets; ++i) {
    bubble_emitter_instance_t* emitter = &this->emitters[i];
    emitter->timer -= delta_seconds * globals->speed;
    if (emitter->timer <= 0) {
      emitter->timer         = bubble_emitter_random_interval(this);
      const float min_radius = this->radius_range[0],
                  max_radius = this->radius_range[1];
      const float radius
        = min_radius + math_random() * (max_radius - min_radius);
      const float angle    = math_random() * PI2;
      emitter->position[0] = sinf(angle) * radius;
      emitter->position[1] = 0.0f;
      emitter->position[2] = cosf(angle) * radius;
      if (this->trigger_callback) {
        this->trigger_callback(emitter->position);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Light-rays Animation.
 * -------------------------------------------------------------------------- */

typedef struct {
  float duration;
  float timer;
  float rotation;
  float x;
  float y;
  float intensity;
} light_ray_t;

typedef struct {
  int count;
  float duration_min;
  float duration_range;
  float speed;
  float spread;
  float pos_range;
  float rot_range;
  float rot_lerp;
  float height;
  light_ray_t rays[20];
} light_ray_controller_t;

static void light_ray_controller_init_light_ray(light_ray_controller_t* this,
                                                light_ray_t* ray);

static void light_ray_controller_init(light_ray_controller_t* this, int count,
                                      float duration_min, float duration_range,
                                      float speed, float spread,
                                      float pos_range, float rot_range,
                                      float rot_lerp, float height)
{
  memset(this, 0, sizeof(light_ray_controller_t));
  this->count          = count > 20 ? 20 : count;
  this->duration_min   = duration_min;
  this->duration_range = duration_range;
  this->speed          = speed;
  this->spread         = spread;
  this->pos_range      = pos_range;
  this->rot_range      = rot_range;
  this->rot_lerp       = rot_lerp;
  this->height         = height;
  for (int i = 0; i < this->count; ++i) {
    light_ray_controller_init_light_ray(this, &this->rays[i]);
  }
}

static void light_ray_controller_init_light_ray(light_ray_controller_t* this,
                                                light_ray_t* ray)
{
  memset(ray, 0, sizeof(light_ray_t));
  ray->duration  = this->duration_min + math_random() * this->duration_range;
  ray->timer     = 0;
  ray->rotation  = math_random() * this->rot_range;
  ray->x         = (math_random() - 0.5) * this->pos_range;
  ray->intensity = 1.0f;
}

static void light_ray_controller_ray_reset(light_ray_controller_t* this,
                                           light_ray_t* ray)
{
  ray->duration = this->duration_min + math_random() * this->duration_range;
  ray->timer    = ray->duration;
  ray->rotation = math_random() * this->rot_range;
  ray->x        = (math_random() - 0.5f) * this->pos_range;
}

static void light_ray_controller_update(light_ray_controller_t* this,
                                        float delta_seconds, globals_t* globals)
{
  const float rot_lerp = this->rot_lerp, height = this->height;
  for (int i = 0; i < this->count; ++i) {
    light_ray_t* ray = &this->rays[i];
    ray->timer -= delta_seconds * globals->speed;
    if (ray->timer <= 0) {
      light_ray_controller_ray_reset(this, ray);
    }
    const float t  = fmaxf(0, fminf(1, ray->timer / ray->duration));
    ray->intensity = sinf(t * PI);
    ray->rotation
      = ray->rotation + (math_random() - 0.5f) * rot_lerp * delta_seconds;
    ray->y = fmaxf(70, fminf(120, height + globals->eye_height));
  }
}

/* -------------------------------------------------------------------------- *
 * Fish School Animation
 * Manages fish positions, rotations and tail animations for all species.
 * -------------------------------------------------------------------------- */

#define TAIL_DIRECTION_DELTA 0.04f
#define TARGET_HEIGHT_DELTA 0.01f
#define MAX_FISH_PER_SPECIES 1000

typedef struct {
  int index;
  float position[3];
  float target[3];
  float scale;
  float tail_time;
  float speed_factor;
  float radius_jitter_x;
  float radius_jitter_y;
  float radius_jitter_z;
  float scale_jitter;
  float tail_phase;
} fish_instance_t;

typedef struct {
  int species_index;
  fish_instance_t* fish;
  int fish_count;
  int fish_capacity;
} species_state_t;

typedef struct {
  species_state_t species_state[FISH_SPECIES_COUNT];
} fish_school_t;

static void fish_school_create_fish_instance(fish_instance_t* fish, int index,
                                             int species_index)
{
  memset(fish, 0, sizeof(fish_instance_t));
  fish->index       = index;
  fish->position[0] = 0.0f;
  fish->position[1] = fish_species[species_index].height_offset;
  fish->position[2] = 0.0f;
  fish->target[0]   = 0.0f;
  fish->target[1]   = fish_species[species_index].height_offset;
  fish->target[2]   = 1.0f;
  fish->scale       = 1.0f;
  fish->tail_time   = 0.0f;
  fish->speed_factor
    = fish_species[species_index].speed
      + math_random() * fish_species[species_index].speed_range;
  fish->radius_jitter_x = math_random();
  fish->radius_jitter_y = math_random();
  fish->radius_jitter_z = math_random();
  fish->scale_jitter    = math_random();
  fish->tail_phase      = math_random() * PI2;
}

static void fish_school_init(fish_school_t* this)
{
  memset(this, 0, sizeof(fish_school_t));
  for (uint32_t i = 0; i < FISH_SPECIES_COUNT; ++i) {
    this->species_state[i].species_index = i;
    this->species_state[i].fish          = NULL;
    this->species_state[i].fish_count    = 0;
    this->species_state[i].fish_capacity = 0;
  }
}

static void fish_school_destroy(fish_school_t* this)
{
  for (uint32_t i = 0; i < FISH_SPECIES_COUNT; ++i) {
    if (this->species_state[i].fish != NULL) {
      free(this->species_state[i].fish);
      this->species_state[i].fish          = NULL;
      this->species_state[i].fish_count    = 0;
      this->species_state[i].fish_capacity = 0;
    }
  }
}

static void fish_school_resize_fish_array(fish_school_t* this,
                                          species_state_t* state, int desired)
{
  const int current = state->fish_count;
  if (current > desired) {
    state->fish_count = desired;
    return;
  }

  /* Grow capacity if needed */
  if (desired > state->fish_capacity) {
    int new_capacity = desired + 16; /* Add some padding */
    if (new_capacity > MAX_FISH_PER_SPECIES) {
      new_capacity = MAX_FISH_PER_SPECIES;
    }
    fish_instance_t* new_fish
      = realloc(state->fish, new_capacity * sizeof(fish_instance_t));
    if (new_fish == NULL) {
      fprintf(stderr, "Failed to allocate fish array\n");
      return;
    }
    state->fish          = new_fish;
    state->fish_capacity = new_capacity;
  }

  /* Create new fish instances */
  for (int i = current; i < desired; ++i) {
    fish_school_create_fish_instance(&state->fish[i], i, state->species_index);
  }
  state->fish_count = desired;
  UNUSED_VAR(this);
}

static void fish_school_update_counts(fish_school_t* this, int total_fish)
{
  int remaining = total_fish > 0 ? total_fish : 0;

  /* Assign big fish */
  for (uint32_t i = 0; i < FISH_SPECIES_COUNT; ++i) {
    const char* name = fish_species[i].name;
    if (strncmp(name, "Big", 3) == 0) {
      int cap = remaining;
      if (total_fish < 100) {
        cap = 1;
      }
      else if (total_fish < 1000) {
        cap = 2;
      }
      else {
        cap = 4;
      }
      int desired = remaining < cap ? remaining : cap;
      if (remaining <= 0) {
        desired = 0;
      }
      fish_school_resize_fish_array(this, &this->species_state[i], desired);
      remaining -= desired;
    }
  }

  /* Assign medium fish */
  for (uint32_t i = 0; i < FISH_SPECIES_COUNT; ++i) {
    const char* name = fish_species[i].name;
    if (strncmp(name, "Medium", 6) == 0) {
      int cap = remaining;
      if (total_fish < 1000) {
        cap = total_fish / 10 > 0 ? total_fish / 10 : 0;
      }
      else if (total_fish < 10000) {
        cap = 80;
      }
      else {
        cap = 160;
      }
      int desired = remaining < cap ? remaining : cap;
      if (remaining <= 0) {
        desired = 0;
      }
      fish_school_resize_fish_array(this, &this->species_state[i], desired);
      remaining -= desired;
    }
  }

  /* Assign small fish */
  for (uint32_t i = 0; i < FISH_SPECIES_COUNT; ++i) {
    const char* name = fish_species[i].name;
    if (strncmp(name, "Small", 5) == 0) {
      int desired = remaining > 0 ? remaining : 0;
      fish_school_resize_fish_array(this, &this->species_state[i], desired);
      remaining -= desired;
    }
  }
}

static void fish_school_update(fish_school_t* this, float global_clock,
                               fish_t* fish_config)
{
  const float base_clock = global_clock * fish_config->fish_speed;

  for (uint32_t species_index = 0; species_index < FISH_SPECIES_COUNT;
       ++species_index) {
    species_state_t* state = &this->species_state[species_index];
    if (state->fish == NULL || state->fish_count == 0) {
      continue;
    }

    const float height_base
      = fish_config->fish_height + fish_species[species_index].height_offset;
    const float height_range = fish_config->fish_height_range
                               * fish_species[species_index].height_range;

    for (int i = 0; i < state->fish_count; ++i) {
      fish_instance_t* fish = &state->fish[i];
      const float speed     = fish->speed_factor;
      const float clock = (base_clock + i * fish_config->fish_offset) * speed;

      const float x_radius
        = fish_species[species_index].radius
          + fish->radius_jitter_x * fish_species[species_index].radius_range;
      const float y_radius = 2.0f + fish->radius_jitter_y * height_range;
      const float z_radius
        = fish_species[species_index].radius
          + fish->radius_jitter_z * fish_species[species_index].radius_range;

      const float x_clock = clock * fish_config->fish_xclock;
      const float y_clock = clock * fish_config->fish_yclock;
      const float z_clock = clock * fish_config->fish_zclock;

      fish->position[0] = sinf(x_clock) * x_radius;
      fish->position[1] = sinf(y_clock) * y_radius + height_base;
      fish->position[2] = cosf(z_clock) * z_radius;

      fish->target[0] = sinf(x_clock - TAIL_DIRECTION_DELTA) * x_radius;
      fish->target[1]
        = sinf(y_clock - TARGET_HEIGHT_DELTA) * y_radius + height_base;
      fish->target[2] = cosf(z_clock - TAIL_DIRECTION_DELTA) * z_radius;

      fish->scale = 1.0f + fish->scale_jitter;

      const float tail_base = (global_clock + i) * fish_config->fish_tail_speed
                                * fish_species[species_index].tail_speed * speed
                              + fish->tail_phase;
      float wrapped   = fmodf(tail_base, PI2);
      fish->tail_time = wrapped < 0 ? wrapped + PI2 : wrapped;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Bubble Particle pipeline
 * Renders billboarded particles with additive blending.
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout bind_group_layout_0;
  WGPUBindGroupLayout bind_group_layout_1;
} bubble_pipeline_result_t;

static WGPURenderPipeline cached_bubble_pipeline             = NULL;
static WGPUPipelineLayout cached_bubble_pipeline_layout      = NULL;
static WGPUBindGroupLayout cached_bubble_bind_group_layout_0 = NULL;
static WGPUBindGroupLayout cached_bubble_bind_group_layout_1 = NULL;

static bubble_pipeline_result_t create_bubble_pipeline(WGPUDevice device,
                                                       WGPUTextureFormat format)
{
  if (cached_bubble_pipeline) {
    return (bubble_pipeline_result_t){
      .pipeline            = cached_bubble_pipeline,
      .pipeline_layout     = cached_bubble_pipeline_layout,
      .bind_group_layout_0 = cached_bubble_bind_group_layout_0,
      .bind_group_layout_1 = cached_bubble_bind_group_layout_1,
    };
  }

  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/bubble.wgsl", "bubble-shader");

  /* Bind group 0: Frame uniforms (viewProjection, viewInverse, time) */
  {
    /* Size: 2 mat4x4 (32 floats) + time + padding (4 floats) = 36 floats
     * But std140 alignment requires 40 floats (160 bytes) */
    WGPUBindGroupLayoutEntry bgl_entries[1]
      = {{.binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {.type             = WGPUBufferBindingType_Uniform,
                         .hasDynamicOffset = false,
                         .minBindingSize   = sizeof(float) * 40}}};
    cached_bubble_bind_group_layout_0 = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Bubble Frame - Bind Group Layout"),
                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                .entries    = bgl_entries,
              });
    ASSERT(cached_bubble_bind_group_layout_0 != NULL);
  }

  /* Bind group 1: Particle texture and sampler */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Texture view */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout) {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    cached_bubble_bind_group_layout_1 = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Bubble Material Bind Group Layout"),
                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                .entries    = bgl_entries,
              });
    ASSERT(cached_bubble_bind_group_layout_1 != NULL);
  }

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_groups_layouts[2] = {
      cached_bubble_bind_group_layout_0, /* Group 0 */
      cached_bubble_bind_group_layout_1  /* Group 1 */
    };
    cached_bubble_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Bubble - Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(cached_bubble_pipeline_layout != NULL);
  }

  /* Render pipline */
  {
    WGPUVertexAttribute vertex_attributes[6] = {
      [0] = (WGPUVertexAttribute) {
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x2,
      },
      [1] = (WGPUVertexAttribute) {
        /* positionStartTime */
        .shaderLocation = 1,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x4,
      },
      [2] = (WGPUVertexAttribute) {
        /* velocityStartSize */
        .shaderLocation = 2,
        .offset         = 16,
        .format         = WGPUVertexFormat_Float32x4,
      },
      [3] = (WGPUVertexAttribute) {
        /* accelerationEndSize */
        .shaderLocation = 3,
        .offset         = 32,
        .format         = WGPUVertexFormat_Float32x4,
      },
      [4] = (WGPUVertexAttribute) {
        /* colorMult */
        .shaderLocation = 4,
        .offset         = 48,
        .format         = WGPUVertexFormat_Float32x4,
      },
      [5] = (WGPUVertexAttribute) {
        /* lifetimeFrameSpinStart */
        .shaderLocation = 5,
        .offset         = 64,
        .format         = WGPUVertexFormat_Float32x4,
      },
    };

    WGPUVertexBufferLayout vertex_buffer_layouts[2] = {
      [0] = (WGPUVertexBufferLayout) {
        /* Buffer 0: Corner vertices (shared quad) */
        .arrayStride    = 2 * 4, /* vec2<f32> */
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &vertex_attributes[0],
      },
      [1] = (WGPUVertexBufferLayout) {
        /* Buffer 1: Particle data (instanced) */
        .arrayStride    = 20 * 4, /* 5 vec4s = 80 bytes */
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 5,
        .attributes     = &vertex_attributes[1],
      },
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Bubble Particle Pipeline"),
      .layout = cached_bubble_pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts),
        .buffers     = vertex_buffer_layouts,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = format,
          .blend = &(WGPUBlendState) {
            /* Additive blending for particles */
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None, /* Billboards visible from both sides */
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = false, /* Particles don't write depth */
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    cached_bubble_pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(cached_bubble_pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (bubble_pipeline_result_t){
    .pipeline            = cached_bubble_pipeline,
    .pipeline_layout     = cached_bubble_pipeline_layout,
    .bind_group_layout_0 = cached_bubble_bind_group_layout_0,
    .bind_group_layout_1 = cached_bubble_bind_group_layout_1,
  };
}

/* -------------------------------------------------------------------------- *
 * Diffuse pipeline
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout material_layout;
  WGPUTextureFormat color_format;
  WGPUVertexBufferLayout* vertex_buffers;
  uint32_t vertex_buffer_count;
} diffuse_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
} diffuse_pipeline_result_t;

static diffuse_pipeline_result_t
create_diffuse_pipeline(WGPUDevice device, diffuse_pipeline_desc_t* desc)
{
  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/diffuse.wgsl", "diffuse-vertex");

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,    /* Group 0 */
      desc->model_layout,    /* Group 1 */
      desc->material_layout, /* Group 2 */
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Diffuse Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(pipeline_layout != NULL);
  }

  /* Render pipline */
  WGPURenderPipeline pipeline;
  {
    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Diffuse Pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = desc->vertex_buffer_count,
        .buffers     = desc->vertex_buffers,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->color_format,
          .blend = &(WGPUBlendState) {
            .color = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (diffuse_pipeline_result_t){
    .pipeline        = pipeline,
    .pipeline_layout = pipeline_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Fish pipeline
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout instance_layout;
  WGPUBindGroupLayout material_layout;
  WGPUTextureFormat color_format;
  WGPUVertexBufferLayout* vertex_buffers;
  uint32_t vertex_buffer_count;
} fish_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
} fish_pipeline_result_t;

static fish_pipeline_result_t create_fish_pipeline(WGPUDevice device,
                                                   fish_pipeline_desc_t* desc)
{
  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/fish.wgsl", "fish-shader");

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,    /* Group 0 */
      desc->instance_layout, /* Group 1 */
      desc->material_layout, /* Group 2 */
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Fish Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(pipeline_layout != NULL);
  }

  /* Render pipline */
  WGPURenderPipeline pipeline;
  {
    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Fish Pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = desc->vertex_buffer_count,
        .buffers     = desc->vertex_buffers,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->color_format,
          .blend = &(WGPUBlendState) {
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (fish_pipeline_result_t){
    .pipeline        = pipeline,
    .pipeline_layout = pipeline_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Inner Tank pipeline
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout material_layout;
  WGPUTextureFormat color_format;
  WGPUVertexBufferLayout* vertex_buffers;
  uint32_t vertex_buffer_count;
} inner_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
} inner_pipeline_result_t;

static inner_pipeline_result_t
create_inner_pipeline(WGPUDevice device, inner_pipeline_desc_t* desc)
{
  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/inner.wgsl", "inner-tank");

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,    /* Group 0 */
      desc->model_layout,    /* Group 1 */
      desc->material_layout, /* Group 2 */
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Inner Tank Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(pipeline_layout != NULL);
  }

  /* Render pipline */
  WGPURenderPipeline pipeline;
  {
    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Inner Tank Pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = desc->vertex_buffer_count,
        .buffers     = desc->vertex_buffers,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format    = desc->color_format,
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (inner_pipeline_result_t){
    .pipeline        = pipeline,
    .pipeline_layout = pipeline_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Laser Beam Pipeline
 * Renders laser beams with additive blending.
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUTextureFormat format;
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
} laser_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout material_bind_group_layout;
} laser_pipeline_result_t;

static WGPURenderPipeline cached_laser_pipeline                    = NULL;
static WGPUPipelineLayout cached_laser_pipeline_layout             = NULL;
static WGPUBindGroupLayout cached_laser_material_bind_group_layout = NULL;

static laser_pipeline_result_t
create_laser_pipeline(WGPUDevice device, laser_pipeline_desc_t* desc)
{
  if (cached_laser_pipeline) {
    return (laser_pipeline_result_t){
      .pipeline                   = cached_laser_pipeline,
      .pipeline_layout            = cached_laser_pipeline_layout,
      .material_bind_group_layout = cached_laser_material_bind_group_layout,
    };
  }

  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/laser.wgsl", "laser-shader");

  /* Material layout: texture, sampler, color multiplier */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Texture view */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout) {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: Uniform buffer */
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {
           .type             = WGPUBufferBindingType_Uniform,
           .minBindingSize   = sizeof(float) * 16 // 4x4 matrix
        }
      }
    };
    cached_laser_material_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Laser Material Bind Group Layout"),
                .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                .entries    = bgl_entries,
              });
    ASSERT(cached_laser_material_bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,                     /* Group 0 */
      desc->model_layout,                     /* Group 1 */
      cached_laser_material_bind_group_layout /* Group 2 */
    };
    cached_laser_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Laser Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(cached_laser_pipeline_layout != NULL);
  }

  /* Render pipline */
  {
    WGPUVertexAttribute vertex_attributes[2] = {
      [0] = (WGPUVertexAttribute) {
        /* position */
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x2,
      },
      [1] = (WGPUVertexAttribute) {
        /* texcoord */
        .shaderLocation = 1,
        .offset         = 8,
        .format         = WGPUVertexFormat_Float32x2,
      },
    };

    /* Vertex buffer layout for simple quad */
    WGPUVertexBufferLayout vertex_buffer_layout = {
      .arrayStride    = 16, /* 4 floats: position(2) + texcoord(2) */
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vertex_attributes),
      .attributes     = &vertex_attributes[0],
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Laser Pipeline"),
      .layout = cached_laser_pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->format,
          .blend = &(WGPUBlendState) {
            /* Additive blending for laser glow */
            .color = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None, /* Visible from both sides */
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = false, /* Particles don't write depth */
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    cached_laser_pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(cached_laser_pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (laser_pipeline_result_t){
    .pipeline                   = cached_laser_pipeline,
    .pipeline_layout            = cached_laser_pipeline_layout,
    .material_bind_group_layout = cached_laser_material_bind_group_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Light Ray (God Ray) Pipeline
 * Renders volumetric light shafts with alpha blending.
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUTextureFormat format;
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
} light_ray_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout material_bind_group_layout;
} light_ray_pipeline_result_t;

static WGPURenderPipeline cached_light_ray_pipeline                    = NULL;
static WGPUPipelineLayout cached_light_ray_pipeline_layout             = NULL;
static WGPUBindGroupLayout cached_light_ray_material_bind_group_layout = NULL;

static light_ray_pipeline_result_t
create_light_ray_pipeline(WGPUDevice device, light_ray_pipeline_desc_t* desc)
{
  if (cached_light_ray_pipeline) {
    return (light_ray_pipeline_result_t){
      .pipeline                   = cached_light_ray_pipeline,
      .pipeline_layout            = cached_light_ray_pipeline_layout,
      .material_bind_group_layout = cached_light_ray_material_bind_group_layout,
    };
  }

  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/light_ray.wgsl", "light-ray-shader");

  /* Material layout: texture and sampler only */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Texture view */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout) {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    cached_light_ray_material_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(
        device, &(WGPUBindGroupLayoutDescriptor){
                  .label      = STRVIEW("Light Ray Material Bind Group Layout"),
                  .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                  .entries    = bgl_entries,
                });
    ASSERT(cached_light_ray_material_bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,                         /* Group 0 */
      desc->model_layout,                         /* Group 1 */
      cached_light_ray_material_bind_group_layout /* Group 2 */
    };
    cached_light_ray_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Light Ray Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(cached_light_ray_pipeline_layout != NULL);
  }

  /* Render pipline */
  {
    WGPUVertexAttribute vertex_attributes[2] = {
      [0] = (WGPUVertexAttribute) {
        /* position */
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x2,
      },
      [1] = (WGPUVertexAttribute) {
        /* texcoord */
        .shaderLocation = 1,
        .offset         = 8,
        .format         = WGPUVertexFormat_Float32x2,
      },
    };

    /* Vertex buffer layout for simple quad */
    WGPUVertexBufferLayout vertex_buffer_layout = {
      .arrayStride    = 16, /* 4 floats: position(2) + texcoord(2) */
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vertex_attributes),
      .attributes     = &vertex_attributes[0],
    };

    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Light Ray Pipeline"),
      .layout = cached_light_ray_pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->format,
          .blend = &(WGPUBlendState) {
            /* Alpha blending for soft god rays */
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None, /* Visible from both sides */
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = false, /* Don't write to depth buffer */
        .depthCompare      = WGPUCompareFunction_Always, /* Always render, ignore depth */
      },
      .multisample = {
         .count = 1,
         .mask  = 0xffffffff
      },
    };

    cached_light_ray_pipeline
      = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(cached_light_ray_pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (light_ray_pipeline_result_t){
    .pipeline                   = cached_light_ray_pipeline,
    .pipeline_layout            = cached_light_ray_pipeline_layout,
    .material_bind_group_layout = cached_light_ray_material_bind_group_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Outer pipeline
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout material_layout;
  WGPUTextureFormat color_format;
  WGPUVertexBufferLayout* vertex_buffers;
  uint32_t vertex_buffer_count;
} outer_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
} outer_pipeline_result_t;

static outer_pipeline_result_t
create_outer_pipeline(WGPUDevice device, outer_pipeline_desc_t* desc)
{
  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/outer.wgsl", "outer-tank");

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,    /* Group 0 */
      desc->model_layout,    /* Group 1 */
      desc->material_layout, /* Group 2 */
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Outer Tank Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(pipeline_layout != NULL);
  }

  /* Render pipline */
  WGPURenderPipeline pipeline;
  {
    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Outer Tank Pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = desc->vertex_buffer_count,
        .buffers     = desc->vertex_buffers,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->color_format,
          .blend = &(WGPUBlendState) {
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (outer_pipeline_result_t){
    .pipeline        = pipeline,
    .pipeline_layout = pipeline_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Seaweed pipeline
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout material_layout;
  WGPUTextureFormat color_format;
  WGPUVertexBufferLayout* vertex_buffers;
  uint32_t vertex_buffer_count;
} seaweed_pipeline_desc_t;

typedef struct {
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;
} seaweed_pipeline_result_t;

static seaweed_pipeline_result_t
create_seaweed_pipeline(WGPUDevice device, seaweed_pipeline_desc_t* desc)
{
  /* Shader module */
  const WGPUShaderModule shader_module
    = load_shader_module(device, "shaders/seaweed.wgsl", "seaweed");

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout = NULL;
  {
    WGPUBindGroupLayout bind_groups_layouts[3] = {
      desc->frame_layout,    /* Group 0 */
      desc->model_layout,    /* Group 1 */
      desc->material_layout, /* Group 2 */
    };
    pipeline_layout = wgpuDeviceCreatePipelineLayout(
      device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Seaweed Pipeline Layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_groups_layouts),
        .bindGroupLayouts     = bind_groups_layouts,
      });
    ASSERT(pipeline_layout != NULL);
  }

  /* Render pipline */
  WGPURenderPipeline pipeline;
  {
    WGPURenderPipelineDescriptor rp_desc = {
      .label  = STRVIEW("Seaweed Pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = desc->vertex_buffer_count,
        .buffers     = desc->vertex_buffers,
      },
      .fragment = &(WGPUFragmentState) {
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState) {
          .format = desc->color_format,
          .blend = &(WGPUBlendState) {
            .color = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add,
            }
          },
          .writeMask = WGPUColorWriteMask_All
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW
      },
      .depthStencil = &(WGPUDepthStencilState) {
        .format            = DEPTH_STENCIL_FORMAT,
        .depthWriteEnabled = false, /* Seaweed uses alpha blending */
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xffffffff
      },
    };

    pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
    ASSERT(pipeline != NULL);

    wgpuShaderModuleRelease(shader_module);
  }

  return (seaweed_pipeline_result_t){
    .pipeline        = pipeline,
    .pipeline_layout = pipeline_layout,
  };
}

/* -------------------------------------------------------------------------- *
 * Texture Cache
 * -------------------------------------------------------------------------- */

#define MAX_TEXTURE_CACHE (64)

typedef struct {
  WGPUTexture texture;
  WGPUTextureView view;
  uint32_t width, height;
  int mip_levels;
  WGPUSampler sampler;
  char url[1024];
} texture_record_t;

typedef struct {
  WGPUDevice device;
  WGPUQueue queue;
  texture_record_t cache[MAX_TEXTURE_CACHE];
  int num_textures;
  WGPUSampler sampler;
  WGPUSampler linear_sampler;
  WGPUSampler cube_sampler;
  bool initialized;
} texture_cache_t;

static void texture_cache_init(texture_cache_t* this, WGPUDevice device,
                               WGPUQueue queue)
{
  memset(this, 0, sizeof(texture_cache_t));
  this->device  = device;
  this->queue   = queue;
  this->sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });
  this->linear_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .maxAnisotropy = 1,
            });
  this->cube_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .maxAnisotropy = 1,
            });
  this->initialized = true;
}

static texture_record_t* texture_cache_load_texture(texture_cache_t* this,
                                                    const char* url,
                                                    WGPUTextureFormat format)
{
  /* Check cache first */
  for (int32_t i = 0; i < this->num_textures; ++i) {
    if (strcmp(this->cache[i].url, url) == 0) {
      return &this->cache[i];
    }
  }

  /* Check cache capacity */
  if (this->num_textures >= MAX_TEXTURE_CACHE) {
    fprintf(stderr, "Texture cache full");
    return NULL;
  }

  /* Load texture data */
  int32_t img_width = 0, img_height = 0, img_channels = 0, depth = 4,
          mip_levels = 1;
  stbi_set_flip_vertically_on_load(true);
  uint8_t* img_data
    = stbi_load(url, &img_width, &img_height, &img_channels, depth);

  /* Check if image loaded successfully */
  if (img_data == NULL || img_width == 0 || img_height == 0) {
    fprintf(stderr, "Failed to load texture: %s (reason: %s)\n", url,
            stbi_failure_reason());
    return NULL;
  }

  /* Create the texture */
  WGPUTextureDescriptor texture_desc = {
    .size = {
       .width              = img_width,
       .height             = img_height,
       .depthOrArrayLayers = 1,
    },
    .format        = format,
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                     | WGPUTextureUsage_RenderAttachment,
    .mipLevelCount = mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
  };
  WGPUTexture texture = wgpuDeviceCreateTexture(this->device, &texture_desc);

  /* Upload pixel data to texture */
  wgpuQueueWriteTexture(this->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture = texture,
                          .aspect  = WGPUTextureAspect_All,
                        },
                        img_data, img_width * img_height * depth,
                        &(WGPUTexelCopyBufferLayout){
                          .bytesPerRow  = img_width * depth,
                          .rowsPerImage = img_height,
                        },
                        &(WGPUExtent3D){img_width, img_height, 1});

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_desc = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture_desc.mipLevelCount,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  WGPUTextureView view = wgpuTextureCreateView(texture, &texture_view_desc);

  /* Create texture record */
  texture_record_t* record = &this->cache[this->num_textures++];
  record->texture          = texture;
  record->view             = view;
  record->width            = img_width;
  record->height           = img_height;
  record->mip_levels       = mip_levels;
  record->sampler          = this->sampler;
  strncpy(record->url, url, sizeof(record->url) - 1);
  record->url[sizeof(record->url) - 1] = '\0';

  if (img_data) {
    stbi_image_free(img_data);
  }

  return record;
}

static texture_record_t*
texture_cache_load_cube_texture(texture_cache_t* this, const char* urls[6],
                                WGPUTextureFormat format)
{
#define NUM_FACES (6)

  /* Create key*/
  char key[1024] = "cube:";
  for (int i = 0; i < NUM_FACES; ++i) {
    strncat(key, urls[i], sizeof(key) - strlen(key) - 1);
    if (i < 5) {
      strncat(key, "|", sizeof(key) - strlen(key) - 1);
    }
  }

  /* Check cache first */
  for (int32_t i = 0; i < this->num_textures; ++i) {
    if (strcmp(this->cache[i].url, key) == 0) {
      return &this->cache[i];
    }
  }

  /* Check cache capacity */
  if (this->num_textures >= MAX_TEXTURE_CACHE) {
    fprintf(stderr, "Texture cache full");
    return NULL;
  }

  typedef struct {
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t depth;
    int32_t mip_levels;
    uint8_t* pixels;
  } bitmap_t;
  bitmap_t bitmaps[NUM_FACES] = {0};

  for (uint8_t face = 0; face < NUM_FACES; ++face) {
    bitmap_t* bitmap   = &bitmaps[face];
    bitmap->depth      = 4;
    bitmap->mip_levels = 1;
    stbi_set_flip_vertically_on_load(true);
    bitmap->pixels = stbi_load(urls[face], &bitmap->width, &bitmap->height,
                               &bitmap->channels, bitmap->depth);
  }

  const int32_t width      = bitmaps[0].width;
  const int32_t height     = bitmaps[0].height;
  const int32_t depth      = bitmaps[0].depth;
  const int32_t mip_levels = bitmaps[0].mip_levels;

  for (uint8_t i = 0; i < NUM_FACES; ++i) {
    ASSERT(bitmaps[i].width == width);
    ASSERT(bitmaps[i].height == height);
    ASSERT(bitmaps[i].depth == depth);
  }

  /* Create the texture */
  WGPUTextureDescriptor texture_desc = {
    .size = {
      .width              = width,
      .height             = height,
      .depthOrArrayLayers = 6,
    },
    .format        = format,
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                     | WGPUTextureUsage_RenderAttachment,
    .mipLevelCount = mip_levels,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
  };
  WGPUTexture texture = wgpuDeviceCreateTexture(this->device, &texture_desc);

  /* Upload pixel data to texture */
  {
    WGPUCommandEncoder cmd_encoder
      = wgpuDeviceCreateCommandEncoder(this->device, NULL);

    WGPUBuffer staging_buffers[6] = {0};
    for (uint32_t face = 0; face < NUM_FACES; ++face) {
      WGPUBufferDescriptor staging_buffer_desc = {
        .usage            = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite,
        .size             = width * height * depth,
        .mappedAtCreation = true,
      };
      staging_buffers[face]
        = wgpuDeviceCreateBuffer(this->device, &staging_buffer_desc);
      ASSERT(staging_buffers[face])
    }

    for (uint32_t face = 0; face < NUM_FACES; ++face) {
      /* Copy texture data into staging buffer */
      uint32_t face_num_bytes = width * height * depth;
      void* mapping
        = wgpuBufferGetMappedRange(staging_buffers[face], 0, face_num_bytes);
      ASSERT(mapping)
      memcpy(mapping, bitmaps[face].pixels, face_num_bytes);
      wgpuBufferUnmap(staging_buffers[face]);

      /* Upload staging buffer to texture */
      wgpuCommandEncoderCopyBufferToTexture(cmd_encoder,
          /* Source */
          &(WGPUTexelCopyBufferInfo) {
            .buffer = staging_buffers[face],
            .layout = (WGPUTexelCopyBufferLayout) {
              .offset       = 0,
              .bytesPerRow  = width * depth,
              .rowsPerImage = height,
            },
          },
          /* Destination */
          &(WGPUTexelCopyTextureInfo){
            .texture  = texture,
            .mipLevel = 0,
            .origin = (WGPUOrigin3D) {
                .x = 0,
                .y = 0,
                .z = face,
            },
            .aspect = WGPUTextureAspect_All,
          },
          /* Copy size */
          &(WGPUExtent3D){
            .width              = width,
            .height             = height,
            .depthOrArrayLayers = 1,
          });
    }

    WGPUCommandBuffer command_buffer
      = wgpuCommandEncoderFinish(cmd_encoder, NULL);
    WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

    /* Sumbit commmand buffer and cleanup */
    ASSERT(command_buffer != NULL)

    /* Submit to the queue */
    wgpuQueueSubmit(this->queue, 1, &command_buffer);

    /* Release command buffer */
    WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer)

    /* Clean up staging resources and pixel data */
    for (uint32_t face = 0; face < NUM_FACES; ++face) {
      WGPU_RELEASE_RESOURCE(Buffer, staging_buffers[face]);
      stbi_image_free(bitmaps[face].pixels);
    }
  }

  /* Create the texture view */
  WGPUTextureViewDescriptor texture_view_desc = {
    .format          = texture_desc.format,
    .dimension       = WGPUTextureViewDimension_Cube,
    .baseMipLevel    = 0,
    .mipLevelCount   = texture_desc.mipLevelCount,
    .baseArrayLayer  = 0,
    .arrayLayerCount = NUM_FACES,
    .aspect          = WGPUTextureAspect_All,
    .usage           = WGPUTextureUsage_TextureBinding,
  };
  WGPUTextureView view = wgpuTextureCreateView(texture, &texture_view_desc);

  /* Create texture record */
  texture_record_t* record = &this->cache[this->num_textures++];
  record->texture          = texture;
  record->view             = view;
  record->width            = width;
  record->height           = height;
  record->mip_levels       = mip_levels;
  record->sampler          = this->sampler;
  strncpy(record->url, key, sizeof(record->url) - 1);
  record->url[sizeof(record->url) - 1] = '\0';
  return record;
}

static void texture_cache_destroy(texture_cache_t* this)
{
  for (int i = 0; i < this->num_textures; ++i) {
    WGPU_RELEASE_RESOURCE(Texture, this->cache[i].texture)
    WGPU_RELEASE_RESOURCE(TextureView, this->cache[i].view)
    WGPU_RELEASE_RESOURCE(Sampler, this->cache[i].sampler)
  }
  this->num_textures = 0;

  WGPU_RELEASE_RESOURCE(Sampler, this->sampler)
  WGPU_RELEASE_RESOURCE(Sampler, this->linear_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, this->cube_sampler)
}

/* -------------------------------------------------------------------------- *
 * Aquarium model
 * -------------------------------------------------------------------------- */

typedef enum {
  ATTRIBUTE_SLOT_POSITION = 0,
  ATTRIBUTE_SLOT_NORMAL   = 1,
  ATTRIBUTE_SLOT_TEXCOORD = 2,
  ATTRIBUTE_SLOT_TANGENT  = 3,
  ATTRIBUTE_SLOT_BINORMAL = 4,
} ATTRIBUTE_SLOTS;

static WGPUVertexFormat vertex_format(const char* type, int num_components)
{
  if (strcmp(type, "Float32Array") == 0) {
    switch (num_components) {
      case 2:
        return WGPUVertexFormat_Float32x2;
      case 3:
        return WGPUVertexFormat_Float32x3;
      case 4:
        return WGPUVertexFormat_Float32x4;
      default:
        fprintf(stderr, "Unsupported float component count: %d\n",
                num_components);
    }
  }
  return WGPUVertexFormat_Force32;
}

/* -------------------------------------------------------------------------- *
 * Aquarium renderer
 * -------------------------------------------------------------------------- */

#define FRAME_UNIFORM_SIZE (256)
#define MODEL_UNIFORM_SIZE (256)
#define MATERIAL_UNIFORM_SIZE (32)
#define FISH_INSTANCE_STRIDE_FLOATS (8)
#define FISH_INSTANCE_STRIDE_BYTES (FISH_INSTANCE_STRIDE_FLOATS * 4)
#define FISH_MATERIAL_UNIFORM_SIZE (32)
#define TANK_MATERIAL_UNIFORM_SIZE (64)

typedef struct {
  WGPUDevice device;
  /* Uniform buffers */
  WGPUBuffer frame_uniform_buffer;
  WGPUBuffer model_uniform_buffer;
  /* Bind groups */
  WGPUBindGroup frame_bind_group;
  WGPUBindGroup model_bind_group;
  /* Bind group layouts */
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout diffuse_material_layout;
  WGPUBindGroupLayout fish_instance_layout;
  WGPUBindGroupLayout fish_material_layout;
  WGPUBindGroupLayout tank_material_layout;
} aquarium_renderer_t;

static void
aquarium_renderer_create_bind_group_layouts(aquarium_renderer_t* this)
{
  const WGPUShaderStage visibility
    = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
  const WGPUShaderStage Vertex_visibility   = WGPUShaderStage_Vertex;
  const WGPUShaderStage fragment_visibility = WGPUShaderStage_Fragment;

  /* Frame layout */
  {
    WGPUBindGroupLayoutEntry bgl_entry = {
      .binding    = 0,
      .visibility = visibility,
      .buffer = {
        .type = WGPUBufferBindingType_Uniform,
      },
    };
    this->frame_layout
      = create_bind_group_layout(this->device, "frame-layout", &bgl_entry, 1);
    ASSERT(this->frame_layout != NULL);
  }

  /* Model layout */
  {
    WGPUBindGroupLayoutEntry bgl_entry = {
      .binding    = 0,
      .visibility = visibility,
      .buffer = {
        .type = WGPUBufferBindingType_Uniform,
      },
    };
    this->model_layout
      = create_bind_group_layout(this->device, "model-layout", &bgl_entry, 1);
    ASSERT(this->model_layout != NULL);
  }

  /* Diffuse material layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = visibility,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = visibility,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };
    this->diffuse_material_layout = create_bind_group_layout(
      this->device, "diffuse-material-layout", bgl_entries,
      (uint32_t)ARRAY_SIZE(bgl_entries));
    ASSERT(this->diffuse_material_layout != NULL);
  }

  /* Fish instance layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = Vertex_visibility,
        .buffer = {
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = visibility,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };
    this->fish_instance_layout = create_bind_group_layout(
      this->device, "fish-instance-layout", bgl_entries,
      (uint32_t)ARRAY_SIZE(bgl_entries));
    ASSERT(this->fish_instance_layout != NULL);
  }

  /* Fish material layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[4] = {
      [0] = {
        .binding    = 0,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [3] = {
        .binding    = 3,
        .visibility = fragment_visibility,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    this->fish_material_layout = create_bind_group_layout(
      this->device, "fish-material-layout", bgl_entries,
      (uint32_t)ARRAY_SIZE(bgl_entries));
    ASSERT(this->fish_material_layout != NULL);
  }

  /* Tank material layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[6] = {
      [0] = {
        .binding    = 0,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [3] = {
        .binding    = 3,
        .visibility = fragment_visibility,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
        },
      },
      [4] = {
        .binding    = 4,
        .visibility = fragment_visibility,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [5] = {
        .binding    = 5,
        .visibility = fragment_visibility,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };
    this->tank_material_layout = create_bind_group_layout(
      this->device, "tank-material-layout", bgl_entries,
      (uint32_t)ARRAY_SIZE(bgl_entries));
    ASSERT(this->tank_material_layout != NULL);
  }
}

static void aquarium_renderer_create_uniform_buffers(aquarium_renderer_t* this)
{
  /* Frame uniform buffer */
  {
    this->frame_uniform_buffer = create_uniform_buffer(
      this->device, FRAME_UNIFORM_SIZE, "frame-uniform");
    ASSERT(this->frame_uniform_buffer != NULL);
  }

  /* Frame bind group */
  {
    WGPUBindGroupEntry bg_entry = {
      .binding = 0,
      .buffer  = this->frame_uniform_buffer,
      .size    = FRAME_UNIFORM_SIZE,
    };
    this->frame_bind_group = create_bind_group(
      this->device, this->frame_layout, &bg_entry, 1, "frame-bind-group");
    ASSERT(this->frame_bind_group != NULL);
  }

  /* Model uniform buffer */
  {
    this->model_uniform_buffer = create_uniform_buffer(
      this->device, MODEL_UNIFORM_SIZE, "model-uniform");
    ASSERT(this->model_uniform_buffer != NULL);
  }

  /* Model bind group */
  {
    WGPUBindGroupEntry bg_entry = {
      .binding = 0,
      .buffer  = this->model_uniform_buffer,
      .size    = MODEL_UNIFORM_SIZE,
    };
    this->model_bind_group = create_bind_group(
      this->device, this->model_layout, &bg_entry, 1, "model-bind-group");
    ASSERT(this->model_bind_group != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Aquarium Model - Manages vertex/index buffers for scene geometry
 * -------------------------------------------------------------------------- */

#define MAX_VERTEX_BUFFERS (5)

typedef struct {
  WGPUBuffer buffer;
  uint32_t stride;
  uint32_t num_components;
  WGPUVertexFormat format;
  uint32_t slot;
} vertex_buffer_info_t;

typedef struct {
  /* Vertex buffers */
  vertex_buffer_info_t vertex_buffers[MAX_VERTEX_BUFFERS];
  uint32_t vertex_buffer_count;
  WGPUVertexBufferLayout vertex_buffer_layouts[MAX_VERTEX_BUFFERS];
  /* Index buffer */
  WGPUBuffer index_buffer;
  uint32_t index_count;
  WGPUIndexFormat index_format;
  /* Texture names */
  char diffuse_texture[256];
  char normal_map_texture[256];
  char reflection_map_texture[256];
  /* Bounding box */
  float bounding_box_min[3];
  float bounding_box_max[3];
  bool has_bounding_box;
} aquarium_model_t;

static void aquarium_model_init(aquarium_model_t* this)
{
  memset(this, 0, sizeof(aquarium_model_t));
  this->index_format = WGPUIndexFormat_Uint16;
}

static void aquarium_model_destroy(aquarium_model_t* this)
{
  for (uint32_t i = 0; i < this->vertex_buffer_count; ++i) {
    WGPU_RELEASE_RESOURCE(Buffer, this->vertex_buffers[i].buffer);
  }
  this->vertex_buffer_count = 0;
  WGPU_RELEASE_RESOURCE(Buffer, this->index_buffer);
  this->index_count = 0;
}

static void aquarium_model_bind(aquarium_model_t* this,
                                WGPURenderPassEncoder pass)
{
  /* Bind vertex buffers at their slot index (position=0, normal=1,
   * texCoord=2...) matching the shaderLocation in the pipeline layout. */
  for (uint32_t i = 0; i < this->vertex_buffer_count; ++i) {
    wgpuRenderPassEncoderSetVertexBuffer(pass, this->vertex_buffers[i].slot,
                                         this->vertex_buffers[i].buffer, 0,
                                         WGPU_WHOLE_SIZE);
  }
  if (this->index_buffer != NULL) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, this->index_buffer,
                                        this->index_format, 0, WGPU_WHOLE_SIZE);
  }
}

/* -------------------------------------------------------------------------- *
 * Loaded Scene Asset - Contains parsed model data from JSON
 * -------------------------------------------------------------------------- */

typedef struct {
  char name[64];
  aquarium_model_t models[MAX_MODELS_PER_SCENE];
  uint32_t model_count;
  bool loading; /* Currently being fetched */
  bool loaded;  /* Fetch completed successfully */
} loaded_scene_t;

/* Loaded scenes storage */
static loaded_scene_t loaded_scenes[SCENE_DEFINITION_COUNT];

/* Prop placement from PropPlacement.js */
typedef struct {
  char name[64];
  float world_matrix[16];
} prop_placement_t;

static prop_placement_t prop_placements[MAX_PROP_PLACEMENTS];
static uint32_t prop_placement_count = 0;

/* Forward declarations for asset loading */
static void load_scene_assets(void);
static void load_prop_placements(void);

/* -------------------------------------------------------------------------- *
 * Asset Loading - Parse JSON asset files and create GPU buffers
 * -------------------------------------------------------------------------- */

/* Get vertex format from type string and component count */
static WGPUVertexFormat get_vertex_format(const char* type, int num_components)
{
  if (strcmp(type, "Float32Array") == 0) {
    switch (num_components) {
      case 2:
        return WGPUVertexFormat_Float32x2;
      case 3:
        return WGPUVertexFormat_Float32x3;
      case 4:
        return WGPUVertexFormat_Float32x4;
      default:
        return WGPUVertexFormat_Float32x3;
    }
  }
  return WGPUVertexFormat_Float32x3;
}

/* Get index format from type string */
static WGPUIndexFormat get_index_format(const char* type)
{
  if (strcmp(type, "Uint32Array") == 0) {
    return WGPUIndexFormat_Uint32;
  }
  return WGPUIndexFormat_Uint16;
}

/* Get attribute slot for vertex attribute name */
static uint32_t get_attribute_slot(const char* name)
{
  if (strcmp(name, "position") == 0)
    return 0;
  if (strcmp(name, "normal") == 0)
    return 1;
  if (strcmp(name, "texCoord") == 0)
    return 2;
  if (strcmp(name, "tangent") == 0)
    return 3;
  if (strcmp(name, "binormal") == 0)
    return 4;
  return 0;
}

/* Create a GPU buffer from float array data */
static WGPUBuffer create_vertex_buffer_from_data(WGPUDevice device,
                                                 const float* data,
                                                 size_t data_count,
                                                 const char* label)
{
  size_t byte_size = data_count * sizeof(float);
  /* Align to 4 bytes */
  size_t aligned_size = (byte_size + 3) & ~3;

  WGPUBufferDescriptor buffer_desc = {
    .label            = STRVIEW(label),
    .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    .size             = aligned_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
  if (buffer) {
    void* mapped = wgpuBufferGetMappedRange(buffer, 0, aligned_size);
    if (mapped) {
      memcpy(mapped, data, byte_size);
    }
    wgpuBufferUnmap(buffer);
  }
  return buffer;
}

/* Create an index buffer from uint16 or uint32 array data */
static WGPUBuffer create_index_buffer_from_data(WGPUDevice device,
                                                const void* data,
                                                size_t byte_size,
                                                const char* label)
{
  /* Align to 4 bytes */
  size_t aligned_size = (byte_size + 3) & ~3;

  WGPUBufferDescriptor buffer_desc = {
    .label            = STRVIEW(label),
    .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
    .size             = aligned_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
  if (buffer) {
    void* mapped = wgpuBufferGetMappedRange(buffer, 0, aligned_size);
    if (mapped) {
      memcpy(mapped, data, byte_size);
    }
    wgpuBufferUnmap(buffer);
  }
  return buffer;
}

/* Parse a model from JSON and create GPU buffers */
static bool parse_model_from_json(WGPUDevice device, cJSON* model_json,
                                  aquarium_model_t* model)
{
  aquarium_model_init(model);

  /* Parse textures */
  cJSON* textures = cJSON_GetObjectItem(model_json, "textures");
  if (textures) {
    cJSON* diffuse = cJSON_GetObjectItem(textures, "diffuse");
    if (diffuse && cJSON_IsString(diffuse)) {
      strncpy(model->diffuse_texture, diffuse->valuestring,
              sizeof(model->diffuse_texture) - 1);
    }
    cJSON* normal_map = cJSON_GetObjectItem(textures, "normalMap");
    if (normal_map && cJSON_IsString(normal_map)) {
      strncpy(model->normal_map_texture, normal_map->valuestring,
              sizeof(model->normal_map_texture) - 1);
    }
    cJSON* reflection_map = cJSON_GetObjectItem(textures, "reflectionMap");
    if (reflection_map && cJSON_IsString(reflection_map)) {
      strncpy(model->reflection_map_texture, reflection_map->valuestring,
              sizeof(model->reflection_map_texture) - 1);
    }
  }

  /* Parse fields (vertex attributes and indices) */
  cJSON* fields = cJSON_GetObjectItem(model_json, "fields");
  if (!fields) {
    fprintf(stderr, "Model has no fields\n");
    return false;
  }

  /* Temporary storage for float data */
  static float temp_float_data[MAX_VERTICES * 4];
  static uint16_t temp_index_data_16[MAX_INDICES];
  static uint32_t temp_index_data_32[MAX_INDICES];

  cJSON* field = NULL;
  cJSON_ArrayForEach(field, fields)
  {
    const char* field_name = field->string;
    if (!field_name)
      continue;

    cJSON* num_components_json = cJSON_GetObjectItem(field, "numComponents");
    cJSON* type_json           = cJSON_GetObjectItem(field, "type");
    cJSON* data_json           = cJSON_GetObjectItem(field, "data");

    if (!type_json || !data_json || !cJSON_IsArray(data_json)) {
      continue;
    }

    const char* type = type_json->valuestring;
    int num_components
      = num_components_json ? num_components_json->valueint : 3;
    int data_array_size = cJSON_GetArraySize(data_json);

    if (strcmp(field_name, "indices") == 0) {
      /* Parse index buffer */
      model->index_format = get_index_format(type);
      model->index_count  = (uint32_t)data_array_size;

      if (model->index_format == WGPUIndexFormat_Uint32) {
        int idx = 0;
        cJSON* item;
        cJSON_ArrayForEach(item, data_json)
        {
          if (idx < MAX_INDICES) {
            temp_index_data_32[idx++] = (uint32_t)item->valueint;
          }
        }
        model->index_buffer = create_index_buffer_from_data(
          device, temp_index_data_32, model->index_count * sizeof(uint32_t),
          "index-buffer");
      }
      else {
        int idx = 0;
        cJSON* item;
        cJSON_ArrayForEach(item, data_json)
        {
          if (idx < MAX_INDICES) {
            temp_index_data_16[idx++] = (uint16_t)item->valueint;
          }
        }
        model->index_buffer = create_index_buffer_from_data(
          device, temp_index_data_16, model->index_count * sizeof(uint16_t),
          "index-buffer");
      }
    }
    else {
      /* Parse vertex attribute */
      int idx = 0;
      cJSON* item;
      cJSON_ArrayForEach(item, data_json)
      {
        if (idx < MAX_VERTICES * 4) {
          temp_float_data[idx++] = (float)item->valuedouble;
        }
      }

      uint32_t slot = get_attribute_slot(field_name);

      /* Debug: print first model's vertex buffer slots */
      static bool first_model_debug = true;
      if (first_model_debug && model->vertex_buffer_count < 5) {
        printf("[VBUF] %s -> slot=%u, components=%d\n", field_name, slot,
               num_components);
        if (model->vertex_buffer_count == 4) {
          first_model_debug = false;
        }
      }

      /* Create vertex buffer */
      vertex_buffer_info_t* vb_info
        = &model->vertex_buffers[model->vertex_buffer_count];
      vb_info->buffer = create_vertex_buffer_from_data(device, temp_float_data,
                                                       idx, field_name);
      vb_info->stride = (uint32_t)(num_components * sizeof(float));
      vb_info->num_components = (uint32_t)num_components;
      vb_info->format         = get_vertex_format(type, num_components);
      vb_info->slot           = slot;

      /* Create vertex buffer layout */
      WGPUVertexBufferLayout* layout
        = &model->vertex_buffer_layouts[model->vertex_buffer_count];
      layout->arrayStride    = vb_info->stride;
      layout->stepMode       = WGPUVertexStepMode_Vertex;
      layout->attributeCount = 1;
      /* Note: attributes pointer needs to be set up separately */

      model->vertex_buffer_count++;
    }
  }

  /* Parse bounding box if available */
  cJSON* bounding_box = cJSON_GetObjectItem(model_json, "boundingBox");
  if (bounding_box) {
    cJSON* min_json = cJSON_GetObjectItem(bounding_box, "min");
    cJSON* max_json = cJSON_GetObjectItem(bounding_box, "max");
    if (min_json && max_json && cJSON_IsArray(min_json)
        && cJSON_IsArray(max_json)) {
      model->has_bounding_box = true;
      for (int i = 0; i < 3 && i < cJSON_GetArraySize(min_json); ++i) {
        model->bounding_box_min[i]
          = (float)cJSON_GetArrayItem(min_json, i)->valuedouble;
      }
      for (int i = 0; i < 3 && i < cJSON_GetArraySize(max_json); ++i) {
        model->bounding_box_max[i]
          = (float)cJSON_GetArrayItem(max_json, i)->valuedouble;
      }
    }
  }

  return true;
}

/* Scene indices for fetch callbacks - must be static for sfetch user_data */
static int scene_indices[SCENE_DEFINITION_COUNT];

/* Forward declarations for asset loading functions (implementations after state
 * struct) */
static void scene_fetch_callback(const sfetch_response_t* response);
static void placement_fetch_callback(const sfetch_response_t* response);

/* -------------------------------------------------------------------------- *
 * Render Items - Model + material + world matrix
 * -------------------------------------------------------------------------- */

#define MAX_DIFFUSE_ITEMS (64)
#define MAX_SEAWEED_ITEMS (32)
#define MAX_INNER_ITEMS (8)
#define MAX_OUTER_ITEMS (8)
#define MAX_FISH_RENDER_GROUPS (8)

typedef struct {
  aquarium_model_t* model;
  WGPUBindGroup material_bind_group;
  WGPUBuffer material_uniform_buffer;
  float world_matrix[16];
} diffuse_render_item_t;

typedef struct {
  aquarium_model_t* model;
  WGPUBindGroup material_bind_group;
  WGPUBuffer material_uniform_buffer;
  float world_matrix[16];
  float time_offset;
} seaweed_render_item_t;

typedef struct {
  aquarium_model_t* model;
  WGPUBindGroup material_bind_group;
  float world_matrix[16];
  WGPUBuffer uniform_buffer;
  float uniform_data[16]; /* Tank material uniforms */
} tank_render_item_t;

typedef struct {
  int species_index;
  const char* program;
  aquarium_model_t* model;
  WGPUBindGroup material_bind_group;
  WGPUBuffer instance_buffer;
  float* instance_data;
  uint32_t instance_capacity;
  uint32_t instance_count;
  WGPUBindGroup instance_bind_group;
  WGPUBuffer species_uniform_buffer;
  float species_uniform_data[8]; /* SpeciesUniforms */
  bool has_normal_map;
  bool has_reflection_map;
} fish_render_group_t;

/* -------------------------------------------------------------------------- *
 * Aquarium State
 * -------------------------------------------------------------------------- */

static struct {
  /* WebGPU context */
  wgpu_context_t* wgpu_context;
  WGPUDevice device;
  WGPUQueue queue;
  WGPUTextureFormat color_format;

  /* Depth texture */
  WGPUTexture depth_texture;
  WGPUTextureView depth_view;
  uint32_t depth_width, depth_height;

  /* Configuration */
  globals_t globals;
  fish_t fish_config;
  inner_const_t inner_const;
  struct {
    bool normal_maps;
    bool reflection;
    bool tank;
    bool museum;
    bool fog;
    bool bubbles;
    bool light_rays;
    bool lasers;
  } options;
  int fish_count;
  int view_index; /* Current camera view preset index */

  /* Timing */
  float clock;
  float eye_clock;
  uint64_t last_time;

  /* Texture cache */
  texture_cache_t texture_cache;
  texture_record_t* skybox_cubemap;

  /* Bind group layouts */
  WGPUBindGroupLayout frame_layout;
  WGPUBindGroupLayout model_layout;
  WGPUBindGroupLayout diffuse_material_layout;
  WGPUBindGroupLayout fish_instance_layout;
  WGPUBindGroupLayout fish_material_layout;
  WGPUBindGroupLayout tank_material_layout;

  /* Uniform buffers */
  WGPUBuffer frame_uniform_buffer;
  WGPUBuffer model_uniform_buffer;
  float frame_uniform_data[64]; /* FRAME_UNIFORM_SIZE / 4 */
  float model_uniform_data[64]; /* MODEL_UNIFORM_SIZE / 4 */
  float model_extra_default[4];
  float model_extra_scratch[4];

  /* Bind groups */
  WGPUBindGroup frame_bind_group;
  WGPUBindGroup model_bind_group;

  /* Pipelines */
  WGPURenderPipeline diffuse_pipeline;
  WGPUPipelineLayout diffuse_pipeline_layout;
  WGPURenderPipeline fish_pipeline;
  WGPUPipelineLayout fish_pipeline_layout;
  WGPURenderPipeline seaweed_pipeline;
  WGPUPipelineLayout seaweed_pipeline_layout;
  WGPURenderPipeline inner_pipeline;
  WGPUPipelineLayout inner_pipeline_layout;
  WGPURenderPipeline outer_pipeline;
  WGPUPipelineLayout outer_pipeline_layout;

  /* Render items */
  diffuse_render_item_t diffuse_items[MAX_DIFFUSE_ITEMS];
  uint32_t diffuse_item_count;
  seaweed_render_item_t seaweed_items[MAX_SEAWEED_ITEMS];
  uint32_t seaweed_item_count;
  tank_render_item_t inner_items[MAX_INNER_ITEMS];
  uint32_t inner_item_count;
  tank_render_item_t outer_items[MAX_OUTER_ITEMS];
  uint32_t outer_item_count;

  /* Fish rendering */
  fish_render_group_t fish_render_groups[MAX_FISH_RENDER_GROUPS];
  uint32_t fish_render_group_count;
  fish_school_t fish_school;

  /* Bubble system */
  bubble_pipeline_result_t bubble_pipeline_result;
  WGPUBuffer bubble_corner_buffer;
  WGPUBuffer bubble_particle_buffer;
  float* bubble_particle_data;
  WGPUBuffer bubble_frame_uniform_buffer;
  float bubble_frame_uniform_data[40];
  WGPUBindGroup bubble_frame_bind_group;
  WGPUBindGroup bubble_material_bind_group;
  texture_record_t* bubble_texture;
  bool bubble_material_bind_group_created;
  float bubble_timer;
  int bubble_index;
  int max_bubble_particles;
  int num_active_bubbles;
  bubble_emitter_t bubble_emitter;

  /* Laser system */
  laser_pipeline_result_t laser_pipeline_result;
  WGPUBuffer laser_vertex_buffer;
  WGPUBuffer laser_color_mult_buffer;
  WGPUBindGroup laser_material_bind_group;
  texture_record_t* laser_texture;

  /* Light ray system */
  light_ray_pipeline_result_t light_ray_pipeline_result;
  WGPUBuffer light_ray_quad_buffer;
  light_ray_controller_t light_ray_controller;
  WGPUBindGroup light_ray_material_bind_groups[20];
  texture_record_t* light_ray_texture;
  bool light_ray_bind_groups_created;

  /* Skybox texture for tank rendering */
  texture_record_t* skybox_texture;
  WGPUTextureView skybox_view;

  /* Asset loading */
  uint8_t file_buffer[ASSET_FILE_BUFFER_SIZE];
  struct {
    bool placement_loaded;
    bool scenes_loaded;
    int scenes_pending;
    int textures_pending;
  } loading_state;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* State flags */
  bool initialized;
  bool render_data_initialized;
} state = {
  /* Default configuration */
  .globals = {
    .speed         = 1.0f,
    .target_height = 0.0f,
    .target_radius = 88.0f,
    .eye_height    = 38.0f,
    .eye_radius    = 69.0f,
    .eye_speed     = 0.06f,
    .field_of_view = 85.0f,
    .ambient_red   = 0.22f,
    .ambient_green = 0.25f,
    .ambient_blue  = 0.39f,
    .fog_power     = 14.5f,
    .fog_mult      = 1.66f,
    .fog_offset    = 0.53f,
    .fog_red       = 0.54f,
    .fog_green     = 0.86f,
    .fog_blue      = 1.0f,
  },
  .fish_config = {
    .fish_height_range = 1.0f,
    .fish_height       = 25.0f,
    .fish_speed        = 0.124f,
    .fish_offset       = 0.52f,
    .fish_xclock       = 1.0f,
    .fish_yclock       = 0.556f,
    .fish_zclock       = 1.0f,
    .fish_tail_speed   = 1.0f,
  },
  .inner_const = {
    .refraction_fudge = 3.0f,
    .eta              = 1.0f,
    .tank_color_fudge = 0.8f,
  },
  .options = {
    .normal_maps = true,
    .reflection  = true,
    .tank        = true,
    .museum      = true,
    .fog         = true,
    .bubbles     = true,
    .light_rays  = true,
    .lasers      = false,
  },
  .fish_count           = 500,
  .view_index           = 0, /* Default view: "Inside (A)" */
  .max_bubble_particles = 1000,
  .color_attachment     = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.1, 0.2, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .initialized = false,
};

/* -------------------------------------------------------------------------- *
 * Asset Loading Callbacks - Implementations (require state struct)
 * -------------------------------------------------------------------------- */

/* Callback for scene asset file fetch */
static void scene_fetch_callback(const sfetch_response_t* response)
{
  /* Get scene index from user data first - we need it for all code paths */
  int scene_index       = *(int*)response->user_data;
  loaded_scene_t* scene = NULL;
  if (scene_index >= 0 && scene_index < (int)SCENE_DEFINITION_COUNT) {
    scene = &loaded_scenes[scene_index];
  }

  if (!response->fetched) {
    fprintf(stderr, "Failed to fetch scene: error %d\n", response->error_code);
    if (scene) {
      scene->loading = false;
      scene->loaded  = true; /* Mark as loaded (failed) to continue loading */
    }
    state.loading_state.scenes_pending--;
    return;
  }

  if (scene_index < 0 || scene_index >= (int)SCENE_DEFINITION_COUNT) {
    state.loading_state.scenes_pending--;
    return;
  }

  /* Parse JSON */
  char* json_str = (char*)response->data.ptr;
  /* Ensure null termination */
  if (response->data.size < ASSET_FILE_BUFFER_SIZE) {
    ((char*)response->data.ptr)[response->data.size] = '\0';
  }

  cJSON* root = cJSON_Parse(json_str);
  if (!root) {
    fprintf(stderr, "Failed to parse JSON for scene %s\n", scene->name);
    scene->loading = false;
    scene->loaded  = true; /* Mark as loaded (failed) to continue loading */
    state.loading_state.scenes_pending--;
    return;
  }

  /* Parse models array */
  cJSON* models_array = cJSON_GetObjectItem(root, "models");
  if (models_array && cJSON_IsArray(models_array)) {
    int model_count = cJSON_GetArraySize(models_array);
    scene->model_count
      = (uint32_t)(model_count > MAX_MODELS_PER_SCENE ? MAX_MODELS_PER_SCENE :
                                                        model_count);

    for (uint32_t i = 0; i < scene->model_count; ++i) {
      cJSON* model_json = cJSON_GetArrayItem(models_array, (int)i);
      if (model_json) {
        if (!parse_model_from_json(state.device, model_json,
                                   &scene->models[i])) {
          fprintf(stderr, "Failed to parse model %u in scene %s\n", i,
                  scene->name);
        }
      }
    }
  }

  cJSON_Delete(root);
  scene->loading = false;
  scene->loaded  = true;
  state.loading_state.scenes_pending--;
  state.loading_state.scenes_loaded++;
}

/* Callback for prop placement file fetch */
static void placement_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    fprintf(stderr, "Failed to fetch PropPlacement.js: error %d\n",
            response->error_code);
    return;
  }

  /* Parse JSON */
  char* json_str = (char*)response->data.ptr;
  if (response->data.size < ASSET_FILE_BUFFER_SIZE) {
    ((char*)response->data.ptr)[response->data.size] = '\0';
  }

  cJSON* root = cJSON_Parse(json_str);
  if (!root) {
    fprintf(stderr, "Failed to parse PropPlacement.js JSON\n");
    return;
  }

  /* Parse objects array */
  cJSON* objects_array = cJSON_GetObjectItem(root, "objects");
  if (objects_array && cJSON_IsArray(objects_array)) {
    prop_placement_count = 0;
    cJSON* obj;
    cJSON_ArrayForEach(obj, objects_array)
    {
      if (prop_placement_count >= MAX_PROP_PLACEMENTS)
        break;

      prop_placement_t* placement = &prop_placements[prop_placement_count];

      cJSON* name_json = cJSON_GetObjectItem(obj, "name");
      if (name_json && cJSON_IsString(name_json)) {
        strncpy(placement->name, name_json->valuestring,
                sizeof(placement->name) - 1);
      }

      cJSON* matrix_json = cJSON_GetObjectItem(obj, "worldMatrix");
      if (matrix_json && cJSON_IsArray(matrix_json)) {
        for (int i = 0; i < 16 && i < cJSON_GetArraySize(matrix_json); ++i) {
          placement->world_matrix[i]
            = (float)cJSON_GetArrayItem(matrix_json, i)->valuedouble;
        }
      }

      prop_placement_count++;
    }
  }

  cJSON_Delete(root);
  state.loading_state.placement_loaded = true;
}

/* Start loading prop placements */
static void load_prop_placements(void)
{
  sfetch_send(&(sfetch_request_t){
    .path     = AQUARIUM_ASSETS_PATH "PropPlacement.js",
    .callback = placement_fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

/* Start loading all scene assets */
static void load_scene_assets(void)
{
  state.loading_state.scenes_pending = SCENE_DEFINITION_COUNT;

  for (uint32_t i = 0; i < SCENE_DEFINITION_COUNT; ++i) {
    scene_indices[i] = (int)i;
    strncpy(loaded_scenes[i].name, scene_definitions[i].name,
            sizeof(loaded_scenes[i].name) - 1);
    loaded_scenes[i].model_count = 0;
    loaded_scenes[i].loading     = false;
    loaded_scenes[i].loaded      = false;
  }

  /* Note: We load scenes one at a time to reuse the file buffer */
  /* Start with the first scene */
  loaded_scenes[0].loading = true;
  char path[512];
  snprintf(path, sizeof(path), "%s%s.js", AQUARIUM_ASSETS_PATH,
           scene_definitions[0].name);
  sfetch_send(&(sfetch_request_t){
    .path      = path,
    .callback  = scene_fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {.ptr = &scene_indices[0], .size = sizeof(int)},
  });
}

/* Continue loading next scene (called from frame loop) */
static void continue_loading_scenes(void)
{
  if (state.loading_state.scenes_pending <= 0)
    return;

  /* Find next unloaded scene that isn't already being loaded */
  for (uint32_t i = 0; i < SCENE_DEFINITION_COUNT; ++i) {
    if (!loaded_scenes[i].loaded && !loaded_scenes[i].loading) {
      loaded_scenes[i].loading = true;
      char path[512];
      snprintf(path, sizeof(path), "%s%s.js", AQUARIUM_ASSETS_PATH,
               scene_definitions[i].name);
      sfetch_send(&(sfetch_request_t){
        .path      = path,
        .callback  = scene_fetch_callback,
        .buffer    = SFETCH_RANGE(state.file_buffer),
        .user_data = {.ptr = &scene_indices[i], .size = sizeof(int)},
      });
      break;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Depth texture management
 * -------------------------------------------------------------------------- */

static void setup_depth_texture_if_needed(void)
{
  uint32_t width  = state.wgpu_context->width;
  uint32_t height = state.wgpu_context->height;

  if (state.depth_texture != NULL && state.depth_width == width
      && state.depth_height == height) {
    return;
  }

  /* Release old depth resources */
  if (state.depth_texture != NULL) {
    WGPU_RELEASE_RESOURCE(TextureView, state.depth_view);
    WGPU_RELEASE_RESOURCE(Texture, state.depth_texture);
  }

  /* Create new depth texture */
  WGPUTextureDescriptor depth_texture_desc = {
    .label = STRVIEW("Aquarium Depth Texture"),
    .size = {
      .width              = width,
      .height             = height,
      .depthOrArrayLayers = 1,
    },
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = DEPTH_STENCIL_FORMAT,
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture
    = wgpuDeviceCreateTexture(state.device, &depth_texture_desc);
  ASSERT(state.depth_texture != NULL);

  /* Create depth texture view */
  WGPUTextureViewDescriptor depth_view_desc = {
    .label           = STRVIEW("Aquarium Depth Texture View"),
    .format          = DEPTH_STENCIL_FORMAT,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_view
    = wgpuTextureCreateView(state.depth_texture, &depth_view_desc);
  ASSERT(state.depth_view != NULL);

  state.depth_width  = width;
  state.depth_height = height;
}

/* -------------------------------------------------------------------------- *
 * Update model uniforms
 * -------------------------------------------------------------------------- */

static void update_model_uniforms(const float* world_matrix, const float* extra)
{
  mat4_t world, world_inverse, world_inverse_transpose;
  memcpy(world.m, world_matrix, sizeof(float) * 16);
  mat4_inverse(&world, &world_inverse);
  mat4_transpose(&world_inverse, &world_inverse_transpose);

  memcpy(&state.model_uniform_data[0], world.m, sizeof(float) * 16);
  memcpy(&state.model_uniform_data[16], world_inverse.m, sizeof(float) * 16);
  memcpy(&state.model_uniform_data[32], world_inverse_transpose.m,
         sizeof(float) * 16);
  memcpy(&state.model_uniform_data[48],
         extra != NULL ? extra : state.model_extra_default, sizeof(float) * 4);

  wgpuQueueWriteBuffer(state.queue, state.model_uniform_buffer, 0,
                       state.model_uniform_data, MODEL_UNIFORM_SIZE);
}

/* -------------------------------------------------------------------------- *
 * Compute frame uniforms
 * -------------------------------------------------------------------------- */

static void compute_frame_uniforms(void)
{
  const globals_t* g = &state.globals;

  /* Compute eye position and target */
  float eye_position[3] = {sinf(state.eye_clock) * g->eye_radius, g->eye_height,
                           cosf(state.eye_clock) * g->eye_radius};
  float target[3]
    = {sinf(state.eye_clock + PI) * g->target_radius, g->target_height,
       cosf(state.eye_clock + PI) * g->target_radius};
  float up[3] = {0.0f, 1.0f, 0.0f};

  /* Compute view and projection matrices */
  mat4_t view_matrix, view_inverse, projection, view_projection;
  mat4_lookat(&view_matrix, eye_position, target, up);
  mat4_inverse(&view_matrix, &view_inverse);

  float aspect = (float)state.wgpu_context->width
                 / (float)fmaxf(1.0f, (float)state.wgpu_context->height);
  mat4_perspective_yfov(&projection, g->field_of_view * PI / 180.0f, aspect,
                        1.0f, 25000.0f);
  mat4_multiply(&projection, &view_matrix, &view_projection);

  /* Debug: print matrices once */
  static bool first_uniform = true;
  if (first_uniform) {
    first_uniform = false;
    printf("[FRAME] eye=(%.1f,%.1f,%.1f) target=(%.1f,%.1f,%.1f)\n",
           eye_position[0], eye_position[1], eye_position[2], target[0],
           target[1], target[2]);
    printf("[FRAME] viewProj row0: %.4f %.4f %.4f %.4f\n", view_projection.m[0],
           view_projection.m[4], view_projection.m[8], view_projection.m[12]);
    printf("[FRAME] viewProj row1: %.4f %.4f %.4f %.4f\n", view_projection.m[1],
           view_projection.m[5], view_projection.m[9], view_projection.m[13]);
    printf("[FRAME] viewProj row2: %.4f %.4f %.4f %.4f\n", view_projection.m[2],
           view_projection.m[6], view_projection.m[10], view_projection.m[14]);
    printf("[FRAME] viewProj row3: %.4f %.4f %.4f %.4f\n", view_projection.m[3],
           view_projection.m[7], view_projection.m[11], view_projection.m[15]);
  }

  /* Fill frame uniform data */
  memcpy(&state.frame_uniform_data[0], view_projection.m, sizeof(float) * 16);
  memcpy(&state.frame_uniform_data[16], view_inverse.m, sizeof(float) * 16);

  /* Light world position (above camera) */
  state.frame_uniform_data[32] = eye_position[0];
  state.frame_uniform_data[33] = eye_position[1] + 20.0f;
  state.frame_uniform_data[34] = eye_position[2];
  state.frame_uniform_data[35] = 1.0f;

  /* Light color */
  state.frame_uniform_data[36] = 1.0f;
  state.frame_uniform_data[37] = 1.0f;
  state.frame_uniform_data[38] = 1.0f;
  state.frame_uniform_data[39] = 1.0f;

  /* Ambient color */
  state.frame_uniform_data[40] = g->ambient_red;
  state.frame_uniform_data[41] = g->ambient_green;
  state.frame_uniform_data[42] = g->ambient_blue;
  state.frame_uniform_data[43] = 1.0f;

  /* Fog color */
  state.frame_uniform_data[44] = g->fog_red;
  state.frame_uniform_data[45] = g->fog_green;
  state.frame_uniform_data[46] = g->fog_blue;
  state.frame_uniform_data[47] = 1.0f;

  /* Fog params (power, mult, offset, enabled) */
  state.frame_uniform_data[48] = g->fog_power;
  state.frame_uniform_data[49] = g->fog_mult;
  state.frame_uniform_data[50] = g->fog_offset;
  state.frame_uniform_data[51] = state.options.fog ? 1.0f : 0.0f;

  /* Upload to GPU */
  wgpuQueueWriteBuffer(state.queue, state.frame_uniform_buffer, 0,
                       state.frame_uniform_data, FRAME_UNIFORM_SIZE);
}

/* -------------------------------------------------------------------------- *
 * Find loaded scene by name
 * -------------------------------------------------------------------------- */

static loaded_scene_t* find_scene_by_name(const char* name)
{
  for (uint32_t i = 0; i < SCENE_DEFINITION_COUNT; ++i) {
    if (strcmp(loaded_scenes[i].name, name) == 0 && loaded_scenes[i].loaded) {
      return &loaded_scenes[i];
    }
  }
  return NULL;
}

/* -------------------------------------------------------------------------- *
 * Create vertex buffer layouts from model
 * -------------------------------------------------------------------------- */

static WGPUVertexAttribute temp_vertex_attributes[MAX_VERTEX_BUFFERS];

/* Helper to compare vertex buffers by slot for qsort */
static int compare_vertex_buffers_by_slot(const void* a, const void* b)
{
  const vertex_buffer_info_t* va = (const vertex_buffer_info_t*)a;
  const vertex_buffer_info_t* vb = (const vertex_buffer_info_t*)b;
  return (int)va->slot - (int)vb->slot;
}

static void setup_vertex_buffer_layouts(aquarium_model_t* model,
                                        WGPUVertexBufferLayout* layouts)
{
  /* First, create a sorted copy of vertex buffers by slot */
  static vertex_buffer_info_t sorted_buffers[MAX_VERTEX_BUFFERS];
  memcpy(sorted_buffers, model->vertex_buffers,
         model->vertex_buffer_count * sizeof(vertex_buffer_info_t));
  qsort(sorted_buffers, model->vertex_buffer_count,
        sizeof(vertex_buffer_info_t), compare_vertex_buffers_by_slot);

  /* Create layouts in slot-sorted order */
  for (uint32_t i = 0; i < model->vertex_buffer_count; ++i) {
    vertex_buffer_info_t* vb = &sorted_buffers[i];

    temp_vertex_attributes[i] = (WGPUVertexAttribute){
      .shaderLocation = vb->slot,
      .offset         = 0,
      .format         = vb->format,
    };

    layouts[i] = (WGPUVertexBufferLayout){
      .arrayStride    = vb->stride,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &temp_vertex_attributes[i],
    };
  }
}

/* Create diffuse material bind group for a render item */
static WGPUBindGroup
create_diffuse_material_bind_group(aquarium_model_t* model,
                                   WGPUBuffer material_uniform_buffer)
{
  if (model == NULL || model->diffuse_texture[0] == '\0') {
    return NULL;
  }

  /* Build full texture path */
  char texture_path[512];
  snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
           model->diffuse_texture);

  /* Load texture */
  texture_record_t* tex_record = texture_cache_load_texture(
    &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
  if (tex_record == NULL) {
    fprintf(stderr, "Failed to load texture: %s\n", texture_path);
    return NULL;
  }

  /* Create bind group */
  WGPUBindGroupEntry entries[3] = {
    [0] = {
      .binding     = 0,
      .textureView = tex_record->view,
    },
    [1] = {
      .binding = 1,
      .sampler = tex_record->sampler,
    },
    [2] = {
      .binding = 2,
      .buffer  = material_uniform_buffer,
      .size    = MATERIAL_UNIFORM_SIZE,
    },
  };

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    state.device, &(WGPUBindGroupDescriptor){
                    .label      = STRVIEW("diffuse-material-bind-group"),
                    .layout     = state.diffuse_material_layout,
                    .entryCount = (uint32_t)ARRAY_SIZE(entries),
                    .entries    = entries,
                  });

  return bind_group;
}

/* Create material uniform buffer with default values */
static WGPUBuffer create_material_uniform_buffer(void)
{
  /* MaterialUniforms layout (32 bytes):
   * specular: vec4<f32>   (16 bytes)
   * shininess: f32        (4 bytes)
   * specularFactor: f32   (4 bytes)
   * pad0: vec2<f32>       (8 bytes)
   */
  float material_data[8] = {
    1.0f,  1.0f, 1.0f, 1.0f, /* specular */
    50.0f,                   /* shininess */
    0.5f,                    /* specularFactor */
    0.0f,  0.0f              /* padding */
  };

  WGPUBuffer buffer = wgpuDeviceCreateBuffer(
    state.device, &(WGPUBufferDescriptor){
                    .label = STRVIEW("material-uniform-buffer"),
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = MATERIAL_UNIFORM_SIZE,
                  });

  wgpuQueueWriteBuffer(state.queue, buffer, 0, material_data,
                       MATERIAL_UNIFORM_SIZE);

  return buffer;
}

/* Create fish material bind group for a render group */
static WGPUBindGroup create_fish_material_bind_group(aquarium_model_t* model,
                                                     bool has_normal_map,
                                                     bool has_reflection_map)
{
  if (model == NULL || model->diffuse_texture[0] == '\0') {
    return NULL;
  }

  /* Load diffuse texture */
  char texture_path[512];
  snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
           model->diffuse_texture);
  texture_record_t* diffuse_tex = texture_cache_load_texture(
    &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
  if (diffuse_tex == NULL) {
    fprintf(stderr, "Failed to load fish diffuse texture: %s\n", texture_path);
    return NULL;
  }

  /* Load normal texture (or use diffuse as placeholder) */
  texture_record_t* normal_tex = diffuse_tex;
  if (has_normal_map && model->normal_map_texture[0] != '\0') {
    snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
             model->normal_map_texture);
    texture_record_t* loaded = texture_cache_load_texture(
      &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
    if (loaded != NULL) {
      normal_tex = loaded;
    }
  }

  /* Load reflection texture (or use diffuse as placeholder) */
  texture_record_t* reflection_tex = diffuse_tex;
  if (has_reflection_map && model->reflection_map_texture[0] != '\0') {
    snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
             model->reflection_map_texture);
    texture_record_t* loaded = texture_cache_load_texture(
      &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
    if (loaded != NULL) {
      reflection_tex = loaded;
    }
  }

  /* Create bind group */
  WGPUBindGroupEntry entries[4] = {
    [0] = {
      .binding     = 0,
      .textureView = diffuse_tex->view,
    },
    [1] = {
      .binding     = 1,
      .textureView = normal_tex->view,
    },
    [2] = {
      .binding     = 2,
      .textureView = reflection_tex->view,
    },
    [3] = {
      .binding = 3,
      .sampler = diffuse_tex->sampler,
    },
  };

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    state.device, &(WGPUBindGroupDescriptor){
                    .label      = STRVIEW("fish-material-bind-group"),
                    .layout     = state.fish_material_layout,
                    .entryCount = (uint32_t)ARRAY_SIZE(entries),
                    .entries    = entries,
                  });

  return bind_group;
}

/* Create tank material bind group for inner/outer tank */
static WGPUBindGroup
create_tank_material_bind_group(aquarium_model_t* model,
                                WGPUBuffer uniform_buffer,
                                texture_record_t* skybox_cubemap)
{
  if (model == NULL || model->diffuse_texture[0] == '\0'
      || skybox_cubemap == NULL) {
    return NULL;
  }

  /* Load diffuse texture */
  char texture_path[512];
  snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
           model->diffuse_texture);
  texture_record_t* diffuse_tex = texture_cache_load_texture(
    &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
  if (diffuse_tex == NULL) {
    fprintf(stderr, "Failed to load tank diffuse texture: %s\n", texture_path);
    return NULL;
  }

  /* Load normal texture (or use diffuse as placeholder) */
  texture_record_t* normal_tex = diffuse_tex;
  if (model->normal_map_texture[0] != '\0') {
    snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
             model->normal_map_texture);
    texture_record_t* loaded = texture_cache_load_texture(
      &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
    if (loaded != NULL) {
      normal_tex = loaded;
    }
  }

  /* Load reflection texture (or use diffuse as placeholder) */
  texture_record_t* reflection_tex = diffuse_tex;
  if (model->reflection_map_texture[0] != '\0') {
    snprintf(texture_path, sizeof(texture_path), "%s%s", AQUARIUM_ASSETS_PATH,
             model->reflection_map_texture);
    texture_record_t* loaded = texture_cache_load_texture(
      &state.texture_cache, texture_path, WGPUTextureFormat_RGBA8Unorm);
    if (loaded != NULL) {
      reflection_tex = loaded;
    }
  }

  /* Create bind group */
  WGPUBindGroupEntry entries[6] = {
    [0] = {
      .binding     = 0,
      .textureView = diffuse_tex->view,
    },
    [1] = {
      .binding     = 1,
      .textureView = normal_tex->view,
    },
    [2] = {
      .binding     = 2,
      .textureView = reflection_tex->view,
    },
    [3] = {
      .binding     = 3,
      .textureView = skybox_cubemap->view,
    },
    [4] = {
      .binding = 4,
      .sampler = diffuse_tex->sampler,
    },
    [5] = {
      .binding = 5,
      .buffer  = uniform_buffer,
      .size    = TANK_MATERIAL_UNIFORM_SIZE,
    },
  };

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    state.device, &(WGPUBindGroupDescriptor){
                    .label      = STRVIEW("tank-material-bind-group"),
                    .layout     = state.tank_material_layout,
                    .entryCount = (uint32_t)ARRAY_SIZE(entries),
                    .entries    = entries,
                  });

  return bind_group;
}

/* -------------------------------------------------------------------------- *
 * Initialize bubble rendering system
 * -------------------------------------------------------------------------- */

static void emit_bubbles(float position[3]);

static void init_bubble_system(void)
{
  /* Create bubble pipeline */
  state.bubble_pipeline_result
    = create_bubble_pipeline(state.device, state.color_format);

  /* Load bubble texture */
  const char* bubble_texture_path = AQUARIUM_ASSETS_PATH "bubble.png";
  state.bubble_texture            = texture_cache_load_texture(
    &state.texture_cache, bubble_texture_path, WGPUTextureFormat_RGBA8Unorm);

  /* Create corner buffer (shared quad vertices for billboards) */
  float corners[12] = {
    -0.5f, -0.5f, 0.5f, -0.5f, 0.5f,  0.5f,
    -0.5f, -0.5f, 0.5f, 0.5f,  -0.5f, 0.5f,
  };
  state.bubble_corner_buffer = wgpuDeviceCreateBuffer(
    state.device, &(WGPUBufferDescriptor){
                    .label = STRVIEW("bubble-corners"),
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(corners),
                  });
  wgpuQueueWriteBuffer(state.queue, state.bubble_corner_buffer, 0, corners,
                       sizeof(corners));

  /* Create particle data buffer */
  const int particle_stride  = 20; /* 5 vec4s per particle */
  state.max_bubble_particles = 1000;
  size_t particle_buffer_size
    = state.max_bubble_particles * particle_stride * sizeof(float);
  state.bubble_particle_data
    = calloc(state.max_bubble_particles * particle_stride, sizeof(float));
  state.bubble_particle_buffer = wgpuDeviceCreateBuffer(
    state.device, &(WGPUBufferDescriptor){
                    .label = STRVIEW("bubble-particles"),
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = particle_buffer_size,
                  });

  /* Initialize all particles as inactive */
  for (int i = 0; i < state.max_bubble_particles; i++) {
    int offset                              = i * particle_stride;
    state.bubble_particle_data[offset + 3]  = -1000.0f; /* startTime */
    state.bubble_particle_data[offset + 16] = 1.0f;     /* lifetime */
  }

  /* Create frame uniform buffer for bubbles (36 floats) */
  state.bubble_frame_uniform_buffer = wgpuDeviceCreateBuffer(
    state.device, &(WGPUBufferDescriptor){
                    .label = STRVIEW("bubble-frame-uniform"),
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(float) * 40, /* padded for alignment */
                  });

  /* Create bubble frame bind group */
  WGPUBindGroupEntry frame_entry = {
    .binding = 0,
    .buffer  = state.bubble_frame_uniform_buffer,
    .size    = sizeof(float) * 40,
  };
  state.bubble_frame_bind_group = wgpuDeviceCreateBindGroup(
    state.device, &(WGPUBindGroupDescriptor){
                    .label  = STRVIEW("bubble-frame-bind-group"),
                    .layout = state.bubble_pipeline_result.bind_group_layout_0,
                    .entryCount = 1,
                    .entries    = &frame_entry,
                  });

  /* Defer bubble material bind group creation until texture is loaded */
  state.bubble_material_bind_group_created = false;

  state.bubble_index       = 0;
  state.num_active_bubbles = 0;
  state.bubble_timer       = 2.0f;

  /* Set bubble emitter callback */
  bubble_random_on_trigger(&state.bubble_emitter, emit_bubbles);
}

static void create_bubble_material_bind_group(void)
{
  if (state.bubble_material_bind_group_created) {
    return;
  }

  /* Check if texture is loaded and valid */
  if (state.bubble_texture == NULL || state.bubble_texture->view == NULL
      || state.bubble_texture->sampler == NULL
      || state.bubble_texture->width == 0 || state.bubble_texture->height == 0
      || state.bubble_pipeline_result.bind_group_layout_1 == NULL) {
    return;
  }

  /* Create bubble material bind group */
  WGPUBindGroupEntry material_entries[2] = {
    [0] = {.binding = 0, .textureView = state.bubble_texture->view},
    [1] = {.binding = 1, .sampler = state.bubble_texture->sampler},
  };
  state.bubble_material_bind_group = wgpuDeviceCreateBindGroup(
    state.device, &(WGPUBindGroupDescriptor){
                    .label  = STRVIEW("bubble-material-bind-group"),
                    .layout = state.bubble_pipeline_result.bind_group_layout_1,
                    .entryCount = (uint32_t)ARRAY_SIZE(material_entries),
                    .entries    = material_entries,
                  });

  if (state.bubble_material_bind_group != NULL) {
    state.bubble_material_bind_group_created = true;
  }
}

static void emit_bubbles(float position[3])
{
  if (state.bubble_particle_data == NULL) {
    return;
  }

  const int particle_stride = 20;
  const int num_to_emit     = 100;

  for (int i = 0; i < num_to_emit; i++) {
    int particle_index = (state.bubble_index + i) % state.max_bubble_particles;
    int offset         = particle_index * particle_stride;

    /* Position (relative to emitter) */
    state.bubble_particle_data[offset + 0]
      = position[0] + (math_random() - 0.5f) * 0.2f;
    state.bubble_particle_data[offset + 1]
      = position[1] - 2.0f + math_random() * 4.0f;
    state.bubble_particle_data[offset + 2]
      = position[2] + (math_random() - 0.5f) * 0.2f;
    state.bubble_particle_data[offset + 3] = state.clock; /* startTime */

    /* Velocity */
    state.bubble_particle_data[offset + 4] = (math_random() - 0.5f) * 0.1f;
    state.bubble_particle_data[offset + 5] = 0.0f;
    state.bubble_particle_data[offset + 6] = (math_random() - 0.5f) * 0.1f;
    state.bubble_particle_data[offset + 7]
      = 0.01f + math_random() * 0.01f; /* startSize */

    /* Acceleration (buoyancy) */
    state.bubble_particle_data[offset + 8]  = 0.0f;
    state.bubble_particle_data[offset + 9]  = 0.05f + math_random() * 0.02f;
    state.bubble_particle_data[offset + 10] = 0.0f;
    state.bubble_particle_data[offset + 11]
      = 0.4f + math_random() * 0.2f; /* endSize */

    /* Color multiplier (bluish-white) */
    state.bubble_particle_data[offset + 12] = 0.7f;
    state.bubble_particle_data[offset + 13] = 0.8f;
    state.bubble_particle_data[offset + 14] = 1.0f;
    state.bubble_particle_data[offset + 15] = 1.0f;

    /* Lifetime, frameStart, spinStart, spinSpeed */
    state.bubble_particle_data[offset + 16] = 40.0f; /* lifetime */
    state.bubble_particle_data[offset + 17] = 0.0f;  /* frameStart */
    state.bubble_particle_data[offset + 18]
      = math_random() * PI2; /* spinStart */
    state.bubble_particle_data[offset + 19]
      = (math_random() - 0.5f) * 0.2f; /* spinSpeed */
  }

  state.bubble_index
    = (state.bubble_index + num_to_emit) % state.max_bubble_particles;
  state.num_active_bubbles = state.max_bubble_particles; /* All slots active */
}

static void update_bubble_uniforms(void)
{
  if (state.bubble_frame_uniform_buffer == NULL) {
    return;
  }

  /* Copy viewProjection matrix (16 floats) */
  memcpy(&state.bubble_frame_uniform_data[0], state.frame_uniform_data,
         sizeof(float) * 16);

  /* Copy viewInverse matrix (16 floats) */
  memcpy(&state.bubble_frame_uniform_data[16], &state.frame_uniform_data[16],
         sizeof(float) * 16);

  /* Set time */
  state.bubble_frame_uniform_data[32] = state.clock;

  wgpuQueueWriteBuffer(state.queue, state.bubble_frame_uniform_buffer, 0,
                       state.bubble_frame_uniform_data, sizeof(float) * 40);
}

static void upload_bubble_particles(void)
{
  if (state.bubble_particle_buffer == NULL
      || state.bubble_particle_data == NULL) {
    return;
  }

  const int particle_stride = 20;
  wgpuQueueWriteBuffer(
    state.queue, state.bubble_particle_buffer, 0, state.bubble_particle_data,
    state.max_bubble_particles * particle_stride * sizeof(float));
}

static void render_bubbles(WGPURenderPassEncoder pass)
{
  if (!state.options.bubbles || state.bubble_pipeline_result.pipeline == NULL
      || state.num_active_bubbles == 0) {
    return;
  }

  /* Try to create material bind group if texture is ready */
  create_bubble_material_bind_group();

  /* Only render if bind group is ready */
  if (!state.bubble_material_bind_group_created
      || state.bubble_material_bind_group == NULL) {
    return;
  }

  /* Update bubble uniforms */
  update_bubble_uniforms();

  /* Upload particle data */
  upload_bubble_particles();

  wgpuRenderPassEncoderSetPipeline(pass, state.bubble_pipeline_result.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bubble_frame_bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetBindGroup(pass, 1, state.bubble_material_bind_group,
                                    0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.bubble_corner_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 1, state.bubble_particle_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(pass, 6, state.num_active_bubbles, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Initialize light ray rendering system
 * -------------------------------------------------------------------------- */

static void init_light_ray_system(void)
{
  /* Create light ray pipeline */
  light_ray_pipeline_desc_t desc = {
    .frame_layout = state.frame_layout,
    .model_layout = state.model_layout,
    .format       = state.color_format,
  };
  state.light_ray_pipeline_result
    = create_light_ray_pipeline(state.device, &desc);

  /* Load light ray texture */
  const char* light_ray_texture_path = AQUARIUM_ASSETS_PATH "LightRay.png";
  state.light_ray_texture            = texture_cache_load_texture(
    &state.texture_cache, light_ray_texture_path, WGPUTextureFormat_RGBA8Unorm);
  state.light_ray_bind_groups_created = false;

  /* Create quad vertices for light rays */
  float vertices[24] = {
    /* position (xy), texcoord (uv) */
    -10.0f, 0.0f,   0.0f, 1.0f, /* bottom-left */
    10.0f,  0.0f,   1.0f, 1.0f, /* bottom-right */
    10.0f,  100.0f, 1.0f, 0.0f, /* top-right */
    -10.0f, 0.0f,   0.0f, 1.0f, /* bottom-left */
    10.0f,  100.0f, 1.0f, 0.0f, /* top-right */
    -10.0f, 100.0f, 0.0f, 0.0f, /* top-left */
  };
  state.light_ray_quad_buffer = wgpuDeviceCreateBuffer(
    state.device, &(WGPUBufferDescriptor){
                    .label = STRVIEW("light-ray-vertices"),
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertices),
                  });
  wgpuQueueWriteBuffer(state.queue, state.light_ray_quad_buffer, 0, vertices,
                       sizeof(vertices));
}

static void create_light_ray_bind_groups(void)
{
  if (state.light_ray_bind_groups_created) {
    return;
  }

  /* Check if texture is loaded and valid */
  if (state.light_ray_texture == NULL || state.light_ray_texture->view == NULL
      || state.light_ray_texture->sampler == NULL
      || state.light_ray_texture->width == 0
      || state.light_ray_texture->height == 0
      || state.light_ray_pipeline_result.material_bind_group_layout == NULL) {
    return;
  }

  /* Create material bind groups for each light ray */
  for (int i = 0; i < state.light_ray_controller.count; i++) {
    WGPUBindGroupEntry entries[2] = {
      [0] = {.binding = 0, .textureView = state.light_ray_texture->view},
      [1] = {.binding = 1, .sampler = state.light_ray_texture->sampler},
    };
    state.light_ray_material_bind_groups[i] = wgpuDeviceCreateBindGroup(
      state.device,
      &(WGPUBindGroupDescriptor){
        .label  = STRVIEW("light-ray-material-bind-group"),
        .layout = state.light_ray_pipeline_result.material_bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(entries),
        .entries    = entries,
      });
    if (state.light_ray_material_bind_groups[i] == NULL) {
      fprintf(stderr, "Failed to create light ray material bind group %d\n", i);
      return;
    }
  }

  state.light_ray_bind_groups_created = true;
}

static void render_light_rays(WGPURenderPassEncoder pass)
{
  if (!state.options.light_rays
      || state.light_ray_pipeline_result.pipeline == NULL
      || state.light_ray_controller.count == 0) {
    return;
  }

  /* Try to create bind groups if texture is ready */
  create_light_ray_bind_groups();

  /* Only render if bind groups are ready */
  if (!state.light_ray_bind_groups_created) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(pass,
                                   state.light_ray_pipeline_result.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.light_ray_quad_buffer, 0,
                                       WGPU_WHOLE_SIZE);

  for (int i = 0; i < state.light_ray_controller.count; i++) {
    light_ray_t* ray = &state.light_ray_controller.rays[i];

    if (ray->intensity <= 0.0f
        || state.light_ray_material_bind_groups[i] == NULL) {
      continue;
    }

    /* Calculate lerp based on timer */
    float lerp = ray->timer / ray->duration;
    if (lerp < 0.0f) {
      lerp = 0.0f;
    }
    if (lerp > 1.0f) {
      lerp = 1.0f;
    }

    /* Build world matrix with rotation and scaling */
    float rot_z = ray->rotation + lerp * 0.2f;
    float cos_r = cosf(rot_z);
    float sin_r = sinf(rot_z);

    mat4_t world_matrix;
    mat4_identity(&world_matrix);

    /* Rotation around Z axis and scale
     * Matrix layout (column-major):
     * [0] [4] [8]  [12]   (column 0, 1, 2, 3)
     * [1] [5] [9]  [13]
     * [2] [6] [10] [14]
     * [3] [7] [11] [15]
     */
    world_matrix.m[0]  = cos_r * 10.0f;
    world_matrix.m[1]  = sin_r * 10.0f;
    world_matrix.m[4]  = -sin_r * 10.0f;
    world_matrix.m[5]  = cos_r * -100.0f; /* negative for downward rays */
    world_matrix.m[10] = 10.0f;

    /* Translation */
    world_matrix.m[12] = ray->x;
    world_matrix.m[13] = ray->y;
    world_matrix.m[14] = 0.0f;

    /* Update model uniforms with alpha in extra data */
    state.model_extra_scratch[0] = ray->intensity;
    update_model_uniforms(world_matrix.m, state.model_extra_scratch);

    wgpuRenderPassEncoderSetBindGroup(pass, 1, state.model_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetBindGroup(
      pass, 2, state.light_ray_material_bind_groups[i], 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialize render data after assets are loaded
 * -------------------------------------------------------------------------- */

static void init_render_data(void)
{
  if (state.render_data_initialized) {
    return;
  }

  /* Check if all scenes are loaded */
  bool all_loaded = true;
  for (uint32_t i = 0; i < SCENE_DEFINITION_COUNT; ++i) {
    if (!loaded_scenes[i].loaded) {
      all_loaded = false;
    }
  }
  if (!all_loaded || !state.loading_state.placement_loaded) {
    return;
  }

  /* Find a diffuse model to create the diffuse pipeline */
  loaded_scene_t* arch_scene = find_scene_by_name("Arch");
  if (arch_scene && arch_scene->model_count > 0) {
    aquarium_model_t* model = &arch_scene->models[0];
    WGPUVertexBufferLayout layouts[MAX_VERTEX_BUFFERS];
    setup_vertex_buffer_layouts(model, layouts);

    diffuse_pipeline_desc_t desc = {
      .frame_layout        = state.frame_layout,
      .model_layout        = state.model_layout,
      .material_layout     = state.diffuse_material_layout,
      .color_format        = state.color_format,
      .vertex_buffers      = layouts,
      .vertex_buffer_count = model->vertex_buffer_count,
    };
    diffuse_pipeline_result_t result
      = create_diffuse_pipeline(state.device, &desc);
    state.diffuse_pipeline        = result.pipeline;
    state.diffuse_pipeline_layout = result.pipeline_layout;
  }

  /* Find a seaweed model to create the seaweed pipeline */
  loaded_scene_t* seaweed_scene = find_scene_by_name("SeaweedA");
  if (seaweed_scene && seaweed_scene->model_count > 0) {
    aquarium_model_t* model = &seaweed_scene->models[0];
    WGPUVertexBufferLayout layouts[MAX_VERTEX_BUFFERS];
    setup_vertex_buffer_layouts(model, layouts);

    seaweed_pipeline_desc_t desc = {
      .frame_layout        = state.frame_layout,
      .model_layout        = state.model_layout,
      .material_layout     = state.diffuse_material_layout,
      .color_format        = state.color_format,
      .vertex_buffers      = layouts,
      .vertex_buffer_count = model->vertex_buffer_count,
    };
    seaweed_pipeline_result_t result
      = create_seaweed_pipeline(state.device, &desc);
    state.seaweed_pipeline        = result.pipeline;
    state.seaweed_pipeline_layout = result.pipeline_layout;
  }

  /* Create render items from placements */
  state.diffuse_item_count    = 0;
  state.seaweed_item_count    = 0;
  state.inner_item_count      = 0;
  state.outer_item_count      = 0;
  uint32_t seaweed_time_index = 0;

  for (uint32_t p = 0; p < prop_placement_count; ++p) {
    prop_placement_t* placement = &prop_placements[p];
    loaded_scene_t* scene       = find_scene_by_name(placement->name);
    if (!scene || scene->model_count == 0) {
      continue;
    }

    /* Find scene definition */
    int def_index = -1;
    for (int32_t i = 0; i < (int32_t)SCENE_DEFINITION_COUNT; ++i) {
      if (strcmp(scene_definitions[i].name, placement->name) == 0) {
        def_index = i;
        break;
      }
    }
    if (def_index < 0) {
      continue;
    }

    const char* program = scene_definitions[def_index].program;

    if (strcmp(program, "diffuse") == 0) {
      if (state.diffuse_item_count < MAX_DIFFUSE_ITEMS) {
        diffuse_render_item_t* item
          = &state.diffuse_items[state.diffuse_item_count++];
        item->model = &scene->models[0];
        memcpy(item->world_matrix, placement->world_matrix, sizeof(float) * 16);
        item->material_bind_group = NULL; /* Will create when texture is loaded
                                           */
      }
    }
    else if (strcmp(program, "seaweed") == 0) {
      if (state.seaweed_item_count < MAX_SEAWEED_ITEMS) {
        seaweed_render_item_t* item
          = &state.seaweed_items[state.seaweed_item_count++];
        item->model = &scene->models[0];
        memcpy(item->world_matrix, placement->world_matrix, sizeof(float) * 16);
        item->time_offset         = (float)seaweed_time_index++;
        item->material_bind_group = NULL;
      }
    }
    else if (strcmp(program, "inner") == 0) {
      if (state.inner_item_count < MAX_INNER_ITEMS) {
        tank_render_item_t* item = &state.inner_items[state.inner_item_count++];
        item->model              = &scene->models[0];
        memcpy(item->world_matrix, placement->world_matrix, sizeof(float) * 16);
        item->material_bind_group = NULL;
      }
    }
    else if (strcmp(program, "outer") == 0) {
      if (state.outer_item_count < MAX_OUTER_ITEMS) {
        tank_render_item_t* item = &state.outer_items[state.outer_item_count++];
        item->model              = &scene->models[0];
        memcpy(item->world_matrix, placement->world_matrix, sizeof(float) * 16);
        item->material_bind_group = NULL;
      }
    }
  }

  /* Create material bind groups for diffuse items */
  for (uint32_t i = 0; i < state.diffuse_item_count; ++i) {
    diffuse_render_item_t* item = &state.diffuse_items[i];
    if (item->model != NULL && item->material_bind_group == NULL) {
      item->material_uniform_buffer = create_material_uniform_buffer();
      item->material_bind_group     = create_diffuse_material_bind_group(
        item->model, item->material_uniform_buffer);
    }
  }

  /* Create material bind groups for seaweed items */
  for (uint32_t i = 0; i < state.seaweed_item_count; ++i) {
    seaweed_render_item_t* item = &state.seaweed_items[i];
    if (item->model != NULL && item->material_bind_group == NULL) {
      item->material_uniform_buffer = create_material_uniform_buffer();
      item->material_bind_group     = create_diffuse_material_bind_group(
        item->model, item->material_uniform_buffer);
    }
  }

  /* Load skybox cubemap texture (needed for tank materials) */
  {
    const char* skybox_urls[6] = {
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_positive_x.jpg",
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_negative_x.jpg",
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_positive_y.jpg",
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_negative_y.jpg",
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_positive_z.jpg",
      AQUARIUM_ASSETS_PATH "GlobeOuter_EM_negative_z.jpg",
    };
    texture_record_t* skybox_cubemap = texture_cache_load_cube_texture(
      &state.texture_cache, skybox_urls, WGPUTextureFormat_RGBA8Unorm);
    if (skybox_cubemap != NULL) {
      state.skybox_cubemap = skybox_cubemap;
    }
    else {
      fprintf(stderr, "Warning: Failed to load skybox cubemap\n");
    }
  }

  /* Create material bind groups for inner tank items */
  for (uint32_t i = 0; i < state.inner_item_count; ++i) {
    tank_render_item_t* item = &state.inner_items[i];
    if (item->model != NULL && item->material_bind_group == NULL
        && state.skybox_cubemap != NULL) {
      /* Create tank uniform buffer with default values */
      item->uniform_buffer = wgpuDeviceCreateBuffer(
        state.device,
        &(WGPUBufferDescriptor){
          .label = STRVIEW("inner-tank-uniform"),
          .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
          .size  = TANK_MATERIAL_UNIFORM_SIZE,
        });
      /* TankMaterialUniforms: specular(16) + params0(16) + params1(16) +
       * extra(16) = 64 bytes */
      float tank_uniforms[16] = {
        1.0f,  1.0f, 1.0f,
        1.0f, /* specular */
        50.0f, 0.5f, 3.0f,
        1.5f, /* shininess, specularFactor, refractionFudge, eta */
        0.8f,  1.0f, 1.0f,
        0.0f, /* tankColorFudge, useNormalMap, useReflectionMap, padding */
        0.0f,  0.0f, 0.0f,
        0.0f, /* extra padding */
      };
      wgpuQueueWriteBuffer(state.queue, item->uniform_buffer, 0, tank_uniforms,
                           TANK_MATERIAL_UNIFORM_SIZE);

      item->material_bind_group = create_tank_material_bind_group(
        item->model, item->uniform_buffer, state.skybox_cubemap);
    }
  }

  /* Create material bind groups for outer tank items */
  for (uint32_t i = 0; i < state.outer_item_count; ++i) {
    tank_render_item_t* item = &state.outer_items[i];
    if (item->model != NULL && item->material_bind_group == NULL
        && state.skybox_cubemap != NULL) {
      /* Create tank uniform buffer with default values */
      item->uniform_buffer = wgpuDeviceCreateBuffer(
        state.device,
        &(WGPUBufferDescriptor){
          .label = STRVIEW("outer-tank-uniform"),
          .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
          .size  = TANK_MATERIAL_UNIFORM_SIZE,
        });
      float tank_uniforms[16] = {
        1.0f,  1.0f, 1.0f,
        1.0f, /* specular */
        50.0f, 0.5f, 3.0f,
        1.5f, /* shininess, specularFactor, refractionFudge, eta */
        0.8f,  1.0f, 1.0f,
        0.0f, /* tankColorFudge, useNormalMap, useReflectionMap, padding */
        0.0f,  0.0f, 0.0f,
        0.0f, /* extra padding */
      };
      wgpuQueueWriteBuffer(state.queue, item->uniform_buffer, 0, tank_uniforms,
                           TANK_MATERIAL_UNIFORM_SIZE);

      item->material_bind_group = create_tank_material_bind_group(
        item->model, item->uniform_buffer, state.skybox_cubemap);
    }
  }

  /* Create fish pipeline */
  {
    /* Build vertex buffer layouts for fish model */
    loaded_scene_t* fish_scene = find_scene_by_name("SmallFishA");
    if (fish_scene && fish_scene->model_count > 0) {
      aquarium_model_t* fish_model = &fish_scene->models[0];
      WGPUVertexBufferLayout layouts[MAX_VERTEX_BUFFERS];
      setup_vertex_buffer_layouts(fish_model, layouts);

      fish_pipeline_desc_t desc = {
        .frame_layout        = state.frame_layout,
        .instance_layout     = state.fish_instance_layout,
        .material_layout     = state.fish_material_layout,
        .color_format        = state.color_format,
        .vertex_buffers      = layouts,
        .vertex_buffer_count = fish_model->vertex_buffer_count,
      };
      fish_pipeline_result_t result = create_fish_pipeline(state.device, &desc);
      state.fish_pipeline           = result.pipeline;
      state.fish_pipeline_layout    = result.pipeline_layout;
    }
  }

  /* Create inner pipeline */
  if (state.inner_item_count > 0) {
    loaded_scene_t* inner_scene = find_scene_by_name("GlobeInner");
    if (inner_scene && inner_scene->model_count > 0) {
      aquarium_model_t* inner_model = &inner_scene->models[0];
      WGPUVertexBufferLayout layouts[MAX_VERTEX_BUFFERS];
      setup_vertex_buffer_layouts(inner_model, layouts);

      inner_pipeline_desc_t desc = {
        .frame_layout        = state.frame_layout,
        .model_layout        = state.model_layout,
        .material_layout     = state.tank_material_layout,
        .color_format        = state.color_format,
        .vertex_buffers      = layouts,
        .vertex_buffer_count = inner_model->vertex_buffer_count,
      };
      inner_pipeline_result_t result
        = create_inner_pipeline(state.device, &desc);
      state.inner_pipeline        = result.pipeline;
      state.inner_pipeline_layout = result.pipeline_layout;
    }
  }

  /* Create outer pipeline */
  if (state.outer_item_count > 0) {
    loaded_scene_t* outer_scene = find_scene_by_name("GlobeOuter");
    if (outer_scene && outer_scene->model_count > 0) {
      aquarium_model_t* outer_model = &outer_scene->models[0];
      WGPUVertexBufferLayout layouts[MAX_VERTEX_BUFFERS];
      setup_vertex_buffer_layouts(outer_model, layouts);

      outer_pipeline_desc_t desc = {
        .frame_layout        = state.frame_layout,
        .model_layout        = state.model_layout,
        .material_layout     = state.tank_material_layout,
        .color_format        = state.color_format,
        .vertex_buffers      = layouts,
        .vertex_buffer_count = outer_model->vertex_buffer_count,
      };
      outer_pipeline_result_t result
        = create_outer_pipeline(state.device, &desc);
      state.outer_pipeline        = result.pipeline;
      state.outer_pipeline_layout = result.pipeline_layout;
    }
  }

  /* Create fish render groups - one per species */
  state.fish_render_group_count = 0;
  for (int species = 0;
       species < FISH_SPECIES_COUNT
       && state.fish_render_group_count < MAX_FISH_RENDER_GROUPS;
       ++species) {
    const char* species_name = fish_species[species].name;
    loaded_scene_t* scene    = find_scene_by_name(species_name);
    if (!scene || scene->model_count == 0) {
      continue;
    }

    fish_render_group_t* group
      = &state.fish_render_groups[state.fish_render_group_count++];
    group->species_index       = species;
    group->program             = scene_definitions[species].program;
    group->model               = &scene->models[0];
    group->instance_capacity   = 500; /* Max fish per species */
    group->instance_count      = 0;
    group->material_bind_group = NULL;
    group->instance_bind_group = NULL;

    /* Determine which textures are available */
    group->has_normal_map = (group->model->normal_map_texture[0] != '\0');
    group->has_reflection_map
      = (group->model->reflection_map_texture[0] != '\0');

    /* Initialize species uniforms */
    group->species_uniform_data[0]
      = fish_species[species].const_uniforms.fish_length;
    group->species_uniform_data[1]
      = fish_species[species].const_uniforms.fish_wave_length;
    group->species_uniform_data[2]
      = fish_species[species].const_uniforms.fish_bend_amount;
    group->species_uniform_data[3] = group->has_normal_map ? 1.0f : 0.0f;
    group->species_uniform_data[4] = group->has_reflection_map ? 1.0f : 0.0f;
    group->species_uniform_data[5] = 50.0f; /* shininess */
    group->species_uniform_data[6] = 0.5f;  /* specularFactor */
    group->species_uniform_data[7] = 0.0f;  /* padding */

    /* Create species uniform buffer */
    group->species_uniform_buffer = wgpuDeviceCreateBuffer(
      state.device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("fish-species-uniform"),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = sizeof(group->species_uniform_data),
      });
    wgpuQueueWriteBuffer(state.queue, group->species_uniform_buffer, 0,
                         group->species_uniform_data,
                         sizeof(group->species_uniform_data));

    /* Create instance buffer */
    size_t instance_buffer_size
      = group->instance_capacity * 32; /* 8 floats per instance */
    group->instance_buffer = wgpuDeviceCreateBuffer(
      state.device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("fish-instance-buffer"),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = instance_buffer_size,
      });
    group->instance_data = calloc(group->instance_capacity, 8 * sizeof(float));

    /* Create instance bind group */
    WGPUBindGroupEntry instance_entries[2] = {
      [0] = {
        .binding = 0,
        .buffer  = group->instance_buffer,
        .size    = instance_buffer_size,
      },
      [1] = {
        .binding = 1,
        .buffer  = group->species_uniform_buffer,
        .size    = sizeof(group->species_uniform_data),
      },
    };
    group->instance_bind_group = wgpuDeviceCreateBindGroup(
      state.device, &(WGPUBindGroupDescriptor){
                      .label      = STRVIEW("fish-instance-bind-group"),
                      .layout     = state.fish_instance_layout,
                      .entryCount = (uint32_t)ARRAY_SIZE(instance_entries),
                      .entries    = instance_entries,
                    });

    /* Create material bind group */
    group->material_bind_group = create_fish_material_bind_group(
      group->model, group->has_normal_map, group->has_reflection_map);
  }

  /* Initialize bubble system */
  init_bubble_system();

  /* Initialize light ray system */
  init_light_ray_system();

  state.render_data_initialized = true;

  /* Debug: Print render data statistics */
  printf("=== Render Data Initialized ===\n");
  printf("Diffuse items: %u\n", state.diffuse_item_count);
  printf("Seaweed items: %u\n", state.seaweed_item_count);
  printf("Inner items: %u\n", state.inner_item_count);
  printf("Outer items: %u\n", state.outer_item_count);
  printf("Fish render groups: %u\n", state.fish_render_group_count);
  printf("Diffuse pipeline: %s\n", state.diffuse_pipeline ? "OK" : "NULL");
  printf("Fish pipeline: %s\n", state.fish_pipeline ? "OK" : "NULL");
  printf("Seaweed pipeline: %s\n", state.seaweed_pipeline ? "OK" : "NULL");
  printf("Inner pipeline: %s\n", state.inner_pipeline ? "OK" : "NULL");
  printf("Outer pipeline: %s\n", state.outer_pipeline ? "OK" : "NULL");

  /* Count items with valid bind groups */
  int valid_diffuse = 0;
  int valid_models  = 0;
  for (uint32_t i = 0; i < state.diffuse_item_count; ++i) {
    if (state.diffuse_items[i].material_bind_group != NULL) {
      valid_diffuse++;
    }
    if (state.diffuse_items[i].model
        && state.diffuse_items[i].model->index_buffer) {
      valid_models++;
    }
  }
  printf("Diffuse items with bind groups: %d\n", valid_diffuse);
  printf("Diffuse items with index buffers: %d\n", valid_models);
  if (state.diffuse_item_count > 0 && state.diffuse_items[0].model) {
    aquarium_model_t* m = state.diffuse_items[0].model;
    printf("First model: vb_count=%u, idx_count=%u\n", m->vertex_buffer_count,
           m->index_count);
  }
}

/* -------------------------------------------------------------------------- *
 * Aquarium example - Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context == NULL) {
    return EXIT_FAILURE;
  }

  /* Initialize sokol_time and sokol_fetch */
  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 32,
    .num_channels = 1,
    .num_lanes    = 1,
  });

  state.wgpu_context = wgpu_context;
  state.device       = wgpu_context->device;
  state.queue        = wgpu_context->queue;
  state.color_format = wgpu_context->render_format;
  state.last_time    = stm_now();

  /* Initialize texture cache */
  texture_cache_init(&state.texture_cache, state.device, state.queue);

  /* Initialize fish school */
  fish_school_init(&state.fish_school);
  fish_school_update_counts(&state.fish_school, state.fish_count);

  /* Setup depth texture */
  setup_depth_texture_if_needed();

  /* Create bind group layouts (from
   * aquarium_renderer_create_bind_group_layouts) */
  aquarium_renderer_t temp_renderer = {.device = state.device};
  aquarium_renderer_create_bind_group_layouts(&temp_renderer);
  state.frame_layout            = temp_renderer.frame_layout;
  state.model_layout            = temp_renderer.model_layout;
  state.diffuse_material_layout = temp_renderer.diffuse_material_layout;
  state.fish_instance_layout    = temp_renderer.fish_instance_layout;
  state.fish_material_layout    = temp_renderer.fish_material_layout;
  state.tank_material_layout    = temp_renderer.tank_material_layout;

  /* Create uniform buffers */
  state.frame_uniform_buffer
    = create_uniform_buffer(state.device, FRAME_UNIFORM_SIZE, "frame-uniform");
  state.model_uniform_buffer
    = create_uniform_buffer(state.device, MODEL_UNIFORM_SIZE, "model-uniform");

  /* Create frame bind group */
  {
    WGPUBindGroupEntry bg_entry = {
      .binding = 0,
      .buffer  = state.frame_uniform_buffer,
      .size    = FRAME_UNIFORM_SIZE,
    };
    state.frame_bind_group = create_bind_group(
      state.device, state.frame_layout, &bg_entry, 1, "frame-bind-group");
    ASSERT(state.frame_bind_group != NULL);
  }

  /* Create model bind group */
  {
    WGPUBindGroupEntry bg_entry = {
      .binding = 0,
      .buffer  = state.model_uniform_buffer,
      .size    = MODEL_UNIFORM_SIZE,
    };
    state.model_bind_group = create_bind_group(
      state.device, state.model_layout, &bg_entry, 1, "model-bind-group");
    ASSERT(state.model_bind_group != NULL);
  }

  /* Initialize light ray controller */
  light_ray_controller_init(&state.light_ray_controller, 5, 1.0f, 1.0f, 4.0f,
                            1.0f, 40.0f, 1.0f, 0.2f, 80.0f);

  /* Initialize bubble emitter */
  float trigger_interval[2] = {2.0f, 10.0f};
  float radius_range[2]     = {0.0f, 50.0f};
  bubble_emitter_init(&state.bubble_emitter, 10, trigger_interval,
                      radius_range);

  /* Start loading assets */
  load_prop_placements();
  load_scene_assets();

  /* Initialize imgui overlay */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Aquarium example - Frame
 * -------------------------------------------------------------------------- */

/* Forward declaration for render_gui */
static void render_gui(wgpu_context_t* wgpu_context);

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loading */
  sfetch_dowork();

  /* Continue loading scenes if still pending */
  if (state.loading_state.scenes_pending > 0) {
    continue_loading_scenes();
  }

  /* Initialize render data once all assets are loaded */
  if (!state.render_data_initialized && state.loading_state.placement_loaded
      && state.loading_state.scenes_pending <= 0) {
    init_render_data();
  }

  /* Handle window resize */
  setup_depth_texture_if_needed();

  /* Update timing using sokol_time */
  uint64_t now        = stm_now();
  float delta_seconds = (float)stm_sec(stm_diff(now, state.last_time));
  state.last_time     = now;
  delta_seconds       = fminf(delta_seconds, 0.1f); /* Cap to 100ms */

  /* Update clock */
  state.clock += delta_seconds * state.globals.speed;
  state.eye_clock += delta_seconds * state.globals.eye_speed;

  /* Start new ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_seconds);

  /* Render GUI */
  render_gui(wgpu_context);

  /* Debug: print loading state once per second */
  static float debug_timer = 0.0f;
  debug_timer += delta_seconds;
  if (debug_timer > 1.0f && !state.render_data_initialized) {
    debug_timer = 0.0f;
    printf("[DEBUG] Loading: placement=%d, scenes_pending=%d, render_init=%d\n",
           state.loading_state.placement_loaded,
           state.loading_state.scenes_pending, state.render_data_initialized);
  }

  /* Early return if not ready to render - still show GUI */
  if (!state.render_data_initialized) {
    imgui_overlay_render(wgpu_context);
    return EXIT_SUCCESS;
  }

  /* Update fish school */
  fish_school_update(&state.fish_school, state.clock, &state.fish_config);

  /* Update fish instance data and upload to GPU */
  for (uint32_t g = 0; g < state.fish_render_group_count; ++g) {
    fish_render_group_t* group = &state.fish_render_groups[g];
    if (group->model == NULL || group->instance_buffer == NULL) {
      continue;
    }

    /* Get fish data from simulation */
    int species_index = group->species_index;
    species_state_t* sim_state
      = &state.fish_school.species_state[species_index];
    if (sim_state->fish == NULL) {
      group->instance_count = 0;
      continue;
    }

    int fish_count = sim_state->fish_count;
    if (fish_count > (int)group->instance_capacity) {
      fish_count = group->instance_capacity;
    }
    group->instance_count = fish_count;

    /* Copy fish data to instance buffer format:
     * FishInstance { worldPosition: vec3, scale: f32, nextPosition: vec3, time:
     * f32 }
     */
    for (int i = 0; i < fish_count; ++i) {
      fish_instance_t* fish = &sim_state->fish[i];
      float* dest           = &group->instance_data[i * 8];
      dest[0]               = fish->position[0];
      dest[1]               = fish->position[1];
      dest[2]               = fish->position[2];
      dest[3]               = fish->scale;
      dest[4]               = fish->target[0];
      dest[5]               = fish->target[1];
      dest[6]               = fish->target[2];
      dest[7]               = fish->tail_time;
    }

    /* Upload to GPU */
    if (fish_count > 0) {
      wgpuQueueWriteBuffer(state.queue, group->instance_buffer, 0,
                           group->instance_data,
                           fish_count * 8 * sizeof(float));
    }
  }

  /* Update light rays */
  light_ray_controller_update(&state.light_ray_controller, delta_seconds,
                              &state.globals);

  /* Update bubbles */
  bubble_emitter_update(&state.bubble_emitter, delta_seconds, &state.globals);

  /* Compute frame uniforms */
  compute_frame_uniforms();

  /* Get current texture view */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_view;

  /* Create command encoder */
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(state.device, NULL);

  /* Begin render pass */
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(encoder, &state.render_pass_descriptor);

  /* Render diffuse items */
  for (uint32_t i = 0; i < state.diffuse_item_count; ++i) {
    diffuse_render_item_t* item = &state.diffuse_items[i];
    if (item->model && state.diffuse_pipeline && item->material_bind_group) {
      wgpuRenderPassEncoderSetPipeline(pass, state.diffuse_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0,
                                        NULL);

      /* Update model uniforms with this item's world matrix */
      update_model_uniforms(item->world_matrix, state.model_extra_default);
      wgpuRenderPassEncoderSetBindGroup(pass, 1, state.model_bind_group, 0,
                                        NULL);

      wgpuRenderPassEncoderSetBindGroup(pass, 2, item->material_bind_group, 0,
                                        NULL);

      aquarium_model_bind(item->model, pass);

      if (item->model->index_buffer) {
        wgpuRenderPassEncoderDrawIndexed(pass, item->model->index_count, 1, 0,
                                         0, 0);
      }
    }
  }

  /* Render fish (instanced) */
  for (uint32_t g = 0; g < state.fish_render_group_count; ++g) {
    fish_render_group_t* group = &state.fish_render_groups[g];
    if (group->model && group->instance_count > 0 && state.fish_pipeline
        && group->material_bind_group && group->instance_bind_group) {
      wgpuRenderPassEncoderSetPipeline(pass, state.fish_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0,
                                        NULL);
      wgpuRenderPassEncoderSetBindGroup(pass, 1, group->instance_bind_group, 0,
                                        NULL);
      wgpuRenderPassEncoderSetBindGroup(pass, 2, group->material_bind_group, 0,
                                        NULL);

      aquarium_model_bind(group->model, pass);

      if (group->model->index_buffer) {
        wgpuRenderPassEncoderDrawIndexed(pass, group->model->index_count,
                                         group->instance_count, 0, 0, 0);
      }
    }
  }

  /* Render seaweed items */
  for (uint32_t i = 0; i < state.seaweed_item_count; ++i) {
    seaweed_render_item_t* item = &state.seaweed_items[i];
    if (item->model && state.seaweed_pipeline && item->material_bind_group) {
      wgpuRenderPassEncoderSetPipeline(pass, state.seaweed_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0,
                                        NULL);

      /* Seaweed uses time offset for animation */
      state.model_extra_scratch[0] = state.clock + item->time_offset;
      update_model_uniforms(item->world_matrix, state.model_extra_scratch);
      wgpuRenderPassEncoderSetBindGroup(pass, 1, state.model_bind_group, 0,
                                        NULL);

      wgpuRenderPassEncoderSetBindGroup(pass, 2, item->material_bind_group, 0,
                                        NULL);

      aquarium_model_bind(item->model, pass);

      if (item->model->index_buffer) {
        wgpuRenderPassEncoderDrawIndexed(pass, item->model->index_count, 1, 0,
                                         0, 0);
      }
    }
  }

  /* Render inner tank items (with refraction) */
  if (state.options.tank && state.inner_pipeline) {
    for (uint32_t i = 0; i < state.inner_item_count; ++i) {
      tank_render_item_t* item = &state.inner_items[i];
      if (item->model && item->material_bind_group) {
        wgpuRenderPassEncoderSetPipeline(pass, state.inner_pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0,
                                          NULL);
        update_model_uniforms(item->world_matrix, state.model_extra_default);
        wgpuRenderPassEncoderSetBindGroup(pass, 1, state.model_bind_group, 0,
                                          NULL);
        wgpuRenderPassEncoderSetBindGroup(pass, 2, item->material_bind_group, 0,
                                          NULL);

        aquarium_model_bind(item->model, pass);

        if (item->model->index_buffer) {
          wgpuRenderPassEncoderDrawIndexed(pass, item->model->index_count, 1, 0,
                                           0, 0);
        }
      }
    }
  }

  /* Render outer tank items (skybox reflection with blending) */
  if (state.options.tank && state.outer_pipeline) {
    for (uint32_t i = 0; i < state.outer_item_count; ++i) {
      tank_render_item_t* item = &state.outer_items[i];
      if (item->model && item->material_bind_group) {
        wgpuRenderPassEncoderSetPipeline(pass, state.outer_pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, state.frame_bind_group, 0,
                                          NULL);
        update_model_uniforms(item->world_matrix, state.model_extra_default);
        wgpuRenderPassEncoderSetBindGroup(pass, 1, state.model_bind_group, 0,
                                          NULL);
        wgpuRenderPassEncoderSetBindGroup(pass, 2, item->material_bind_group, 0,
                                          NULL);

        aquarium_model_bind(item->model, pass);

        if (item->model->index_buffer) {
          wgpuRenderPassEncoderDrawIndexed(pass, item->model->index_count, 1, 0,
                                           0, 0);
        }
      }
    }
  }

  /* Render bubbles */
  /* DISABLED FOR DEBUGGING: render_bubbles(pass); */

  /* Render light rays */
  /* DISABLED FOR DEBUGGING: render_light_rays(pass); */

  /* End render pass */
  wgpuRenderPassEncoderEnd(pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);

  /* Submit command buffer */
  WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder);
  wgpuQueueSubmit(state.queue, 1, &command_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, command_buffer);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Aquarium example - GUI
 * -------------------------------------------------------------------------- */

/* Apply a camera view preset */
static void apply_view_preset(int index)
{
  if (index < 0 || index >= VIEW_PRESET_COUNT) {
    return;
  }

  const view_preset_t* preset = &view_presets[index];

  /* Update globals from the preset */
  state.globals.target_height = preset->globals.target_height;
  state.globals.target_radius = preset->globals.target_radius;
  state.globals.eye_height    = preset->globals.eye_height;
  state.globals.eye_radius    = preset->globals.eye_radius;
  state.globals.eye_speed     = preset->globals.eye_speed;
  state.globals.field_of_view = preset->globals.field_of_view;
  state.globals.ambient_red   = preset->globals.ambient_red;
  state.globals.ambient_green = preset->globals.ambient_green;
  state.globals.ambient_blue  = preset->globals.ambient_blue;
  state.globals.fog_power     = preset->globals.fog_power;
  state.globals.fog_mult      = preset->globals.fog_mult;
  state.globals.fog_offset    = preset->globals.fog_offset;
  state.globals.fog_red       = preset->globals.fog_red;
  state.globals.fog_green     = preset->globals.fog_green;
  state.globals.fog_blue      = preset->globals.fog_blue;

  /* Update inner constants if needed */
  state.inner_const.refraction_fudge = preset->inner_const.refraction_fudge;
  state.inner_const.eta              = preset->inner_const.eta;
  state.inner_const.tank_color_fudge = preset->inner_const.tank_color_fudge;
}

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Set window position */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Aquarium Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Camera settings */
  if (igCollapsingHeaderBoolPtr("Camera", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    /* Change View button */
    if (igButton("Change View", (ImVec2){0, 0})) {
      state.view_index = (state.view_index + 1) % VIEW_PRESET_COUNT;
      apply_view_preset(state.view_index);
    }
    igSameLine(0.0f, -1.0f);
    igText("%s", view_presets[state.view_index].name);

    igSeparator();

    imgui_overlay_slider_float("Eye Height", &state.globals.eye_height, 1.0f,
                               100.0f, "%.1f");
    imgui_overlay_slider_float("Eye Radius", &state.globals.eye_radius, 10.0f,
                               200.0f, "%.1f");
    imgui_overlay_slider_float("Eye Speed", &state.globals.eye_speed, 0.0f,
                               1.0f, "%.2f");
    imgui_overlay_slider_float("FOV", &state.globals.field_of_view, 30.0f,
                               120.0f, "%.0f");
  }

  /* Animation settings */
  if (igCollapsingHeaderBoolPtr("Animation", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_float("Speed", &state.globals.speed, 0.0f, 5.0f,
                               "%.1f");
  }

  /* Fog settings */
  if (igCollapsingHeaderBoolPtr("Fog", NULL, 0)) {
    imgui_overlay_slider_float("Fog Power", &state.globals.fog_power, 0.0f,
                               50.0f, "%.1f");
    imgui_overlay_slider_float("Fog Mult", &state.globals.fog_mult, 0.0f, 5.0f,
                               "%.2f");
    imgui_overlay_slider_float("Fog Offset", &state.globals.fog_offset, 0.0f,
                               2.0f, "%.2f");
  }

  /* Toggle options */
  if (igCollapsingHeaderBoolPtr("Options", NULL, 0)) {
    igCheckbox("Tank", &state.options.tank);
    igCheckbox("Bubbles", &state.options.bubbles);
    igCheckbox("Light Rays", &state.options.light_rays);
    igCheckbox("Fog", &state.options.fog);
    igCheckbox("Normal Maps", &state.options.normal_maps);
    igCheckbox("Reflection", &state.options.reflection);
  }

  /* Statistics */
  igSeparator();
  igText("Fish Count: %d", state.fish_count);
  igText("Active Bubbles: %d", state.num_active_bubbles);
  igText("FPS: %.1f", igGetIO()->Framerate);

  igEnd();
}

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    setup_depth_texture_if_needed();
  }
}

/* -------------------------------------------------------------------------- *
 * Aquarium example - Cleanup
 * -------------------------------------------------------------------------- */

static void cleanup(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shut down imgui overlay */
  imgui_overlay_shutdown();

  /* Shut down sokol_fetch */
  sfetch_shutdown();

  /* Release loaded scene models */
  for (uint32_t i = 0; i < SCENE_DEFINITION_COUNT; ++i) {
    loaded_scene_t* scene = &loaded_scenes[i];
    for (uint32_t m = 0; m < scene->model_count; ++m) {
      aquarium_model_destroy(&scene->models[m]);
    }
    scene->model_count = 0;
    scene->loaded      = false;
  }

  /* Release fish school */
  fish_school_destroy(&state.fish_school);

  /* Release bubble emitter */
  bubble_emitter_destroy(&state.bubble_emitter);

  /* Release texture cache */
  texture_cache_destroy(&state.texture_cache);

  /* Release uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.frame_uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_uniform_buffer);

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.frame_bind_group);
  WGPU_RELEASE_RESOURCE(BindGroup, state.model_bind_group);

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.frame_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.model_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.diffuse_material_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.fish_instance_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.fish_material_layout);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.tank_material_layout);

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.diffuse_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.diffuse_pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.fish_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.fish_pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.seaweed_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.seaweed_pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.inner_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.inner_pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.outer_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.outer_pipeline_layout);

  /* Release depth resources */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_view);
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture);

  state.initialized = false;
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "WebGPU Aquarium",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = cleanup,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* bubble_shader_wgsl = CODE(
  // Bubble Particle Shader for WebGPU Aquarium
  // Billboarded particles with lifetime animation

  struct Uniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    time: f32,
    padding: vec3<f32>,
  };

  struct ParticleData {
    positionStartTime: vec4<f32>,      // xyz = position, w = start time
    velocityStartSize: vec4<f32>,      // xyz = velocity, w = start size
    accelerationEndSize: vec4<f32>,    // xyz = acceleration, w = end size
    colorMult: vec4<f32>,              // rgba multiplier
    lifetimeFrameSpinStart: vec4<f32>, // x = lifetime, y = frameStart, z = spinStart, w = spinSpeed
  };

  struct VertexInput {
    @location(0) corner: vec2<f32>,    // Corner position (-0.5 to 0.5)
    @location(1) positionStartTime: vec4<f32>,
    @location(2) velocityStartSize: vec4<f32>,
    @location(3) accelerationEndSize: vec4<f32>,
    @location(4) colorMult: vec4<f32>,
    @location(5) lifetimeFrameSpinStart: vec4<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) percentLife: f32,
    @location(2) colorMult: vec4<f32>,
  };

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Unpack particle data
    let position = input.positionStartTime.xyz;
    let startTime = input.positionStartTime.w;
    let velocity = input.velocityStartSize.xyz;
    let startSize = input.velocityStartSize.w;
    let acceleration = input.accelerationEndSize.xyz;
    let endSize = input.accelerationEndSize.w;
    let lifetime = input.lifetimeFrameSpinStart.x;
    let spinStart = input.lifetimeFrameSpinStart.z;
    let spinSpeed = input.lifetimeFrameSpinStart.w;

    // Calculate particle age and percent life
    let age = uniforms.time - startTime;
    let percentLife = age / lifetime;

    // Hide particles that are not alive
    var size = mix(startSize, endSize, percentLife);
    if (percentLife < 0.0 || percentLife > 1.0) {
      size = 0.0;
    }

    // Calculate particle position with physics
    let currentPosition = position + velocity * age + acceleration * age * age;

    // Calculate rotation
    let angle = spinStart + spinSpeed * age;
    let s = sin(angle);
    let c = cos(angle);
    let rotatedCorner = vec2<f32>(
      input.corner.x * c + input.corner.y * s,
      -input.corner.x * s + input.corner.y * c
    );

    // Billboard - face the camera
    let basisX = uniforms.viewInverse[0].xyz;
    let basisY = uniforms.viewInverse[1].xyz;
    let offsetPosition = (basisX * rotatedCorner.x + basisY * rotatedCorner.y) * size;

    // Final world position
    let worldPosition = currentPosition + offsetPosition;

    // Output
    output.position = uniforms.viewProjection * vec4<f32>(worldPosition, 1.0);
    output.texCoord = input.corner + vec2<f32>(0.5, 0.5);  // Convert from -0.5..0.5 to 0..1
    output.percentLife = percentLife;
    output.colorMult = input.colorMult;

    return output;
  }

  @group(1) @binding(0) var particleTexture: texture_2d<f32>;
  @group(1) @binding(1) var particleSampler: sampler;

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the particle texture
    let texColor = textureSample(particleTexture, particleSampler, input.texCoord);

    // Fade out at end of life
    let alpha = 1.0 - smoothstep(0.7, 1.0, input.percentLife);

    // Apply color multiplier and lifetime alpha
    var color = texColor * input.colorMult;
    color.a *= alpha;

    return color;
  }
);

static const char* diffuse_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct MaterialUniforms {
    specular: vec4<f32>,
    shininess: f32,
    specularFactor: f32,
    pad0: vec2<f32>,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var linearSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) worldPosition: vec3<f32>,
    @location(5) clipPosition: vec4<f32>,
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    output.worldPosition = worldPosition.xyz;
    output.clipPosition = output.position;
    return output;
  }

  fn lit(l: f32, h: f32, shininess: f32) -> vec3<f32> {
    let ambient = 1.0;
    let diffuse = max(l, 0.0);
    let specular = select(0.0, pow(max(h, 0.0), shininess), l > 0.0);
    return vec3<f32>(ambient, diffuse, specular);
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let normal = normalize(input.normal);
    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);

    let lighting = lit(dot(normal, surfaceToLight), dot(normal, halfVector), materialUniforms.shininess);
    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = vec3<f32>(0.0);
    color += lightColor * diffuseColor.rgb * lighting.y;
    color += diffuseColor.rgb * ambientColor;
    color += frameUniforms.lightColor.rgb * materialUniforms.specular.rgb * lighting.z * materialUniforms.specularFactor;

    var outColor = vec4<f32>(color, diffuseColor.a);

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      let foggedColor = mix(outColor.rgb, frameUniforms.fogColor.rgb, fogFactor);
      outColor = vec4<f32>(foggedColor, outColor.a);
    }

    return outColor;
  }
);

static const char* fish_shader_p1_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct FishInstance {
    worldPosition: vec3<f32>,
    scale: f32,
    nextPosition: vec3<f32>,
    time: f32,
  }

  struct SpeciesUniforms {
    fishLength: f32,
    fishWaveLength: f32,
    fishBendAmount: f32,
    useNormalMap: f32,
    useReflectionMap: f32,
    shininess: f32,
    specularFactor: f32,
    padding: f32,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<storage, read> fishInstances: array<FishInstance>;
  @group(1) @binding(1) var<uniform> speciesUniforms: SpeciesUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var linearSampler: sampler;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) binormal: vec3<f32>,
    @location(6) clipPosition: vec4<f32>,
  }

  fn safeForward(forward: vec3<f32>) -> vec3<f32> {
    let lenSq = dot(forward, forward);
    if (lenSq < 1e-6) {
      return vec3<f32>(0.0, 0.0, 1.0);
    }
    return forward / sqrt(lenSq);
  }

  fn computeBasis(forward: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    var right = cross(up, forward);
    var rightLenSq = dot(right, right);
    if (rightLenSq < 1e-6) {
      up = vec3<f32>(0.0, 0.0, 1.0);
      right = cross(up, forward);
      rightLenSq = dot(right, right);
      if (rightLenSq < 1e-6) {
        right = vec3<f32>(1.0, 0.0, 0.0);
      }
    }
    right = normalize(right);
    let realUp = normalize(cross(forward, right));
    return mat3x3<f32>(right, realUp, forward);
  }
);

static const char* fish_shader_p2_wgsl = CODE(
  @vertex
  fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
    let instance = fishInstances[instanceIndex];

    var forward = safeForward(instance.worldPosition - instance.nextPosition);
    let basis = computeBasis(forward);
    let right = basis[0];
    let trueUp = basis[1];

    let worldMatrix = mat4x4<f32>(
      vec4<f32>(right * instance.scale, 0.0),
      vec4<f32>(trueUp * instance.scale, 0.0),
      vec4<f32>(forward * instance.scale, 0.0),
      vec4<f32>(instance.worldPosition, 1.0)
    );

    var mult = input.position.z / max(speciesUniforms.fishLength, 0.0001);
    if (input.position.z <= 0.0) {
      mult = (-input.position.z / max(speciesUniforms.fishLength, 0.0001)) * 2.0;
    }

    let s = sin(instance.time + mult * speciesUniforms.fishWaveLength);
    let offset = (mult * mult) * s * speciesUniforms.fishBendAmount;
    let bentPosition = vec4<f32>(input.position + vec3<f32>(offset, 0.0, 0.0), 1.0);

    let worldPosition = worldMatrix * bentPosition;
    let normalMatrix = basis;

    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    output.normal = normalize(normalMatrix * input.normal);
    output.tangent = normalize(normalMatrix * input.tangent);
    output.binormal = normalize(normalMatrix * input.binormal);
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseSample = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);

    var normal = normalize(input.normal);
    var specStrength = 0.0;
    if (speciesUniforms.useNormalMap > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      var tangentNormal = normalSample.xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
      tangentNormal = normalize(tangentNormal + vec3<f32>(0.0, 0.0, 2.0));
      normal = normalize(tangentToWorld * tangentNormal);
      specStrength = normalSample.a;
    }

    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);

    let diffuseFactor = max(dot(normal, surfaceToLight), 0.0);
    let specularTerm = select(0.0, pow(max(dot(normal, halfVector), 0.0), speciesUniforms.shininess), diffuseFactor > 0.0);

    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = diffuseSample.rgb * ambientColor;
    color += diffuseSample.rgb * lightColor * diffuseFactor;
    color += lightColor * specularTerm * speciesUniforms.specularFactor * specStrength;

    if (speciesUniforms.useReflectionMap > 0.5) {
      let reflectionSample = textureSample(reflectionTexture, linearSampler, input.texCoord);
      let mixFactor = clamp(1.0 - reflectionSample.r, 0.0, 1.0);
      color = mix(reflectionSample.rgb, color, mixFactor);
    }

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      color = mix(color, frameUniforms.fogColor.rgb, fogFactor);
    }

    return vec4<f32>(color, diffuseSample.a);
  }
);

static const char* inner_shader_p1_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct TankMaterialUniforms {
    specular: vec4<f32>,
    params0: vec4<f32>, // x: shininess, y: specularFactor, z: refractionFudge, w: eta
    params1: vec4<f32>, // x: tankColorFudge, y: useNormalMap, z: useReflectionMap, w: outerFudge (unused)
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var skyboxTexture: texture_cube<f32>;
  @group(2) @binding(4) var linearSampler: sampler;
  @group(2) @binding(5) var<uniform> tankUniforms: TankMaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) binormal: vec3<f32>,
    @location(4) surfaceToLight: vec3<f32>,
    @location(5) surfaceToView: vec3<f32>,
    @location(6) clipPosition: vec4<f32>,
  }
);

static const char* inner_shader_p2_wgsl = CODE(
  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.tangent = (modelUniforms.worldInverseTranspose * vec4<f32>(input.tangent, 0.0)).xyz;
    output.binormal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.binormal, 0.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - worldPosition.xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);
    let tankColorFudge = tankUniforms.params1.x;
    diffuseColor = vec4<f32>(diffuseColor.rgb + vec3<f32>(tankColorFudge, tankColorFudge, tankColorFudge), 1.0);

    var normal = normalize(input.normal);
    let useNormalMap = tankUniforms.params1.y;
    if (useNormalMap > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);
      var tangentNormal = normalSample.xyz - vec3<f32>(0.5, 0.5, 0.5);
      tangentNormal = normalize(tangentNormal + vec3<f32>(0.0, 0.0, tankUniforms.params0.z));
      normal = normalize(tangentToWorld * tangentNormal);
    }

    let surfaceToView = normalize(input.surfaceToView);
    let eta = max(tankUniforms.params0.w, 0.0001);
    var refractionDir = refract(surfaceToView, normal, eta);
    if (dot(refractionDir, refractionDir) < 1e-6) {
      refractionDir = -surfaceToView;
    }
    refractionDir = normalize(refractionDir);

    let skySample = textureSample(skyboxTexture, linearSampler, refractionDir);

    var refractionMask = 1.0;
    let useReflectionMap = tankUniforms.params1.z;
    if (useReflectionMap > 0.5) {
      refractionMask = textureSample(reflectionTexture, linearSampler, input.texCoord).r;
    }
    refractionMask = clamp(refractionMask, 0.0, 1.0);

    let skyContribution = skySample.rgb * diffuseColor.rgb;
    let mixedColor = mix(skyContribution, diffuseColor.rgb, refractionMask);
    var outColor = vec4<f32>(mixedColor, diffuseColor.a);

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      let foggedColor = mix(outColor.rgb, frameUniforms.fogColor.rgb, fogFactor);
      outColor = vec4<f32>(foggedColor, outColor.a);
    }

    return outColor;
  }
);

static const char* laser_shader_wgsl = CODE(
  // Laser Beam Shader for WebGPU Aquarium
  // Simple textured beam with color modulation

  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
  };

  struct ModelUniforms {
    world: mat4x4<f32>,
  };

  struct MaterialUniforms {
    colorMult: vec4<f32>,
  };

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texCoord: vec2<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
  };

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var beamTexture: texture_2d<f32>;
  @group(2) @binding(1) var beamSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;

    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let texColor = textureSample(beamTexture, beamSampler, input.texCoord);
    return texColor * materialUniforms.colorMult;
  }
);

static const char* light_ray_shader_wgsl = CODE(
  // Light Ray (God Ray) Shader for WebGPU Aquarium
  // Animated volumetric light shafts from above

  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
  };

  struct ModelUniforms {
    world: mat4x4<f32>,
  };

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texCoord: vec2<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
  };

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var lightRayTexture: texture_2d<f32>;
  @group(2) @binding(1) var lightRaySampler: sampler;

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;

    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(lightRayTexture, lightRaySampler, input.texCoord);
  }
);

static const char* outer_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct TankMaterialUniforms {
    specular: vec4<f32>,
    params0: vec4<f32>, // x: shininess, y: specularFactor, z: refractionFudge (unused), w: eta (unused)
    params1: vec4<f32>, // x: tankColorFudge (unused), y: useNormalMap, z: useReflectionMap, w: outerFudge
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var normalTexture: texture_2d<f32>;
  @group(2) @binding(2) var reflectionTexture: texture_2d<f32>;
  @group(2) @binding(3) var skyboxTexture: texture_cube<f32>;
  @group(2) @binding(4) var linearSampler: sampler;
  @group(2) @binding(5) var<uniform> tankUniforms: TankMaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) binormal: vec3<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) binormal: vec3<f32>,
    @location(4) surfaceToView: vec3<f32>,
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPosition = modelUniforms.world * vec4<f32>(input.position, 1.0);
    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.texCoord = input.texCoord;
    output.normal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.normal, 0.0)).xyz;
    output.tangent = (modelUniforms.worldInverseTranspose * vec4<f32>(input.tangent, 0.0)).xyz;
    output.binormal = (modelUniforms.worldInverseTranspose * vec4<f32>(input.binormal, 0.0)).xyz;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - worldPosition.xyz;
    return output;
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseColor = textureSample(diffuseTexture, linearSampler, input.texCoord);

    var normal = normalize(input.normal);
    if (tankUniforms.params1.y > 0.5) {
      let tangent = normalize(input.tangent);
      let binormal = normalize(input.binormal);
      let tangentToWorld = mat3x3<f32>(tangent, binormal, normal);
      let normalSample = textureSample(normalTexture, linearSampler, input.texCoord);
      var tangentNormal = normalSample.xyz - vec3<f32>(0.5, 0.5, 0.5);
      normal = normalize(tangentToWorld * tangentNormal);
    }

    let surfaceToView = normalize(input.surfaceToView);
    let reflectionDir = normalize(-reflect(surfaceToView, normal));
    var skyColor = textureSample(skyboxTexture, linearSampler, reflectionDir);

    let fudgeAmount = tankUniforms.params1.w;
    let fudge = skyColor.rgb * fudgeAmount;
    let bright = min(1.0, fudge.r * fudge.g * fudge.b);

    var reflectionAmount = 0.0;
    if (tankUniforms.params1.z > 0.5) {
      reflectionAmount = textureSample(reflectionTexture, linearSampler, input.texCoord).r;
    }
    reflectionAmount = clamp(reflectionAmount, 0.0, 1.0);

    let reflectColor = mix(vec4<f32>(skyColor.rgb, bright), diffuseColor, 1.0 - reflectionAmount);
    let viewDot = clamp(abs(dot(surfaceToView, normal)), 0.0, 1.0);
    var reflectMix = clamp((viewDot + 0.3) * reflectionAmount, 0.0, 1.0);
    if (tankUniforms.params1.z <= 0.5) {
      reflectMix = 1.0;
    }

    let finalColor = mix(skyColor.rgb, reflectColor.rgb, reflectMix);
    let alpha = clamp(1.0 - viewDot, 0.0, 1.0);
    return vec4<f32>(finalColor, alpha);
  }
);

static const char* seaweed_shader_wgsl = CODE(
  struct FrameUniforms {
    viewProjection: mat4x4<f32>,
    viewInverse: mat4x4<f32>,
    lightWorldPos: vec4<f32>,
    lightColor: vec4<f32>,
    ambient: vec4<f32>,
    fogColor: vec4<f32>,
    fogParams: vec4<f32>,
  }

  struct ModelUniforms {
    world: mat4x4<f32>,
    worldInverse: mat4x4<f32>,
    worldInverseTranspose: mat4x4<f32>,
    extra: vec4<f32>,
  }

  struct MaterialUniforms {
    specular: vec4<f32>,
    shininess: f32,
    specularFactor: f32,
    pad0: vec2<f32>,
  }

  @group(0) @binding(0) var<uniform> frameUniforms: FrameUniforms;
  @group(1) @binding(0) var<uniform> modelUniforms: ModelUniforms;
  @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
  @group(2) @binding(1) var linearSampler: sampler;
  @group(2) @binding(2) var<uniform> materialUniforms: MaterialUniforms;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texCoord: vec2<f32>,
  }

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) surfaceToLight: vec3<f32>,
    @location(3) surfaceToView: vec3<f32>,
    @location(4) clipPosition: vec4<f32>,
  }

  fn safeNormalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 1e-5);
  }

  @vertex
  fn vs_main(input: VertexInput) -> VertexOutput {
    let worldPos = modelUniforms.world;
    let time = modelUniforms.extra.x;
    let toCamera = safeNormalize(frameUniforms.viewInverse[3].xyz - worldPos[3].xyz, vec3<f32>(0.0, 0.0, 1.0));
    let yAxis = vec3<f32>(0.0, 1.0, 0.0);
    let xAxis = safeNormalize(cross(yAxis, toCamera), vec3<f32>(1.0, 0.0, 0.0));
    let zAxis = safeNormalize(cross(xAxis, yAxis), vec3<f32>(0.0, 0.0, 1.0));

    let newWorld = mat4x4<f32>(
      vec4<f32>(xAxis, 0.0),
      vec4<f32>(yAxis, 0.0),
      vec4<f32>(zAxis, 0.0),
      vec4<f32>(worldPos[3].xyz, 1.0)
    );

    var bentPosition = vec4<f32>(input.position, 1.0);
    let sway = sin(time * 0.5) * pow(input.position.y * 0.07, 2.0);
    bentPosition.x += sway;
    bentPosition.y += -4.0;

    let worldPosition = newWorld * bentPosition;

    var output: VertexOutput;
    output.position = frameUniforms.viewProjection * worldPosition;
    output.clipPosition = output.position;
    output.texCoord = input.texCoord;
    let normalMatrix = mat3x3<f32>(newWorld[0].xyz, newWorld[1].xyz, newWorld[2].xyz);
    output.normal = normalize(normalMatrix * input.normal);
    let baseWorldPosition = (modelUniforms.world * vec4<f32>(input.position, 1.0)).xyz;
    output.surfaceToLight = frameUniforms.lightWorldPos.xyz - baseWorldPosition;
    output.surfaceToView = frameUniforms.viewInverse[3].xyz - baseWorldPosition;
    return output;
  }

  fn lit(l: f32, h: f32, shininess: f32) -> vec3<f32> {
    let diffuse = max(l, 0.0);
    let specular = select(0.0, pow(max(h, 0.0), shininess), l > 0.0);
    return vec3<f32>(1.0, diffuse, specular);
  }

  @fragment
  fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let diffuseSample = textureSample(diffuseTexture, linearSampler, input.texCoord);
    if (diffuseSample.a < 0.3) {
      discard;
    }

    let normal = normalize(input.normal);
    let surfaceToLight = normalize(input.surfaceToLight);
    let surfaceToView = normalize(input.surfaceToView);
    let halfVector = normalize(surfaceToLight + surfaceToView);
    let lighting = lit(dot(normal, surfaceToLight), dot(normal, halfVector), materialUniforms.shininess);

    let lightColor = frameUniforms.lightColor.rgb;
    let ambientColor = frameUniforms.ambient.rgb;

    var color = diffuseSample.rgb * ambientColor;
    color += diffuseSample.rgb * lightColor * lighting.y;
    color += lightColor * materialUniforms.specular.rgb * lighting.z * materialUniforms.specularFactor;

    if (frameUniforms.fogParams.w > 0.5) {
      let fogCoord = input.clipPosition.z / input.clipPosition.w;
      let fogFactor = clamp(pow(fogCoord, frameUniforms.fogParams.x) * frameUniforms.fogParams.y - frameUniforms.fogParams.z, 0.0, 1.0);
      color = mix(color, frameUniforms.fogColor.rgb, fogFactor);
    }

    return vec4<f32>(color, diffuseSample.a);
  }
);

// clang-format on
