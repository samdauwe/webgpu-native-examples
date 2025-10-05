#include "webgpu/wgpu_common.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

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

static struct {
  const char* name;
  globals_t globals;
  inner_const_t inner_const;
} view_presets[VIEW_PRESET_COUNT] = {
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
 * Math functions
 * -------------------------------------------------------------------------- */

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
  // Column-major
  out->m[0]  = x[0];
  out->m[4]  = x[1];
  out->m[8]  = x[2];
  out->m[12] = -vec3_dot(x, eye);
  out->m[1]  = y[0];
  out->m[5]  = y[1];
  out->m[9]  = y[2];
  out->m[13] = -vec3_dot(y, eye);
  out->m[2]  = z[0];
  out->m[6]  = z[1];
  out->m[10] = z[2];
  out->m[14] = -vec3_dot(z, eye);
  out->m[3]  = 0.0f;
  out->m[7]  = 0.0f;
  out->m[11] = 0.0f;
  out->m[15] = 1.0f;
}

static void mat4_multiply(const mat4_t* a, const mat4_t* b, mat4_t* out)
{
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out->m[i + j * 4] = a->m[0 + j * 4] * b->m[i + 0 * 4]
                          + a->m[1 + j * 4] * b->m[i + 1 * 4]
                          + a->m[2 + j * 4] * b->m[i + 2 * 4]
                          + a->m[3 + j * 4] * b->m[i + 3 * 4];
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
    WGPUBindGroupLayoutEntry bgl_entries[1]
      = {{.binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {
                .type             = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize   = sizeof(float) * 16 // 4x4 matrix
          }}};
    cached_bubble_bind_group_layout_0 = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Bubble Frame Bind Group Layout"),
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
        .label                = STRVIEW("Bubble Pipeline Layout"),
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
    .bind_group_layout_1 = cached_bubble_bind_group_layout_0,
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
 * Aquarium example
 * -------------------------------------------------------------------------- */

int main(void)
{
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
    .fish_length = f32,
    .fish_wave_length = f32,
    .fish_bend_amount = f32,
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

    var mult = input.position.z / max(speciesUniforms..fish_length =, 0.0001);
    if (input.position.z <= 0.0) {
      mult = (-input.position.z / max(speciesUniforms..fish_length =, 0.0001)) * 2.0;
    }

    let s = sin(instance.time + mult * speciesUniforms..fish_wave_length =);
    let offset = (mult * mult) * s * speciesUniforms..fish_bend_amount =;
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
