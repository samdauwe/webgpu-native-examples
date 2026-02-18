#include "webgpu/wgpu_common.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../webgpu/imgui_overlay.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Metaballs
 *
 * WebGPU demo featuring marching cubes and bloom post-processing via compute
 * shaders, physically based shading, deferred rendering, gamma correction and
 * shadow mapping.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Metaballs count - matches TypeScript reference (256) */
#define MAX_METABALLS 256
#define MAX_POINT_LIGHTS_COUNT 128
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus
#define SHADOW_MAP_SIZE 512
#define METABALLS_COMPUTE_WORKGROUP_SIZE_X 4
#define METABALLS_COMPUTE_WORKGROUP_SIZE_Y 4
#define METABALLS_COMPUTE_WORKGROUP_SIZE_Z 4

static const uint32_t METABALLS_COMPUTE_WORKGROUP_SIZE[3] = {
  METABALLS_COMPUTE_WORKGROUP_SIZE_X,
  METABALLS_COMPUTE_WORKGROUP_SIZE_Y,
  METABALLS_COMPUTE_WORKGROUP_SIZE_Z,
};

/* Volume resolution - matches TypeScript reference (100x100x80) */
static const uint32_t VOLUME_WIDTH  = 100;
static const uint32_t VOLUME_HEIGHT = 100;
static const uint32_t VOLUME_DEPTH  = 80;

/* -------------------------------------------------------------------------- *
 * Quality Settings
 * -------------------------------------------------------------------------- */

typedef enum {
  QualitySettings_Low    = 0,
  QualitySettings_Medium = 1,
  QualitySettings_High   = 2,
} quality_settings_enum;

typedef struct {
  bool bloom_toggle;
  uint32_t shadow_res;
  uint32_t point_lights_count;
  float output_scale;
  bool update_metaballs;
} quality_option_t;

static const quality_option_t QUALITIES[3] = {
  [QualitySettings_Low] = {
    .bloom_toggle       = false,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 1.0f,
    .update_metaballs   = false,
  },
  [QualitySettings_Medium] = {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 1.0f,
    .update_metaballs   = true,
   },
  [QualitySettings_High] = {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 128,
    .output_scale       = 1.0f,
    .update_metaballs   = true,
  },
};

static quality_settings_enum _quality = QualitySettings_Low;

static quality_option_t settings_get_quality_level(void)
{
  return QUALITIES[_quality];
}

static quality_settings_enum settings_get_quality(void)
{
  return _quality;
}

static void settings_set_quality(quality_settings_enum v)
{
  _quality = v;
}

/* -------------------------------------------------------------------------- *
 * Helper Functions
 * -------------------------------------------------------------------------- */

static float deg_to_rad(float deg)
{
  return (deg * PI) / 180.0f;
}

/* -------------------------------------------------------------------------- *
 * Shader variables - declared at top, code at bottom of file
 * Currently only compute + basic rendering shaders are used.
 * Additional shaders for post-processing will be enabled when ported.
 * -------------------------------------------------------------------------- */

static const char* metaball_field_compute_shader;
static const char* metaballs_fragment_shader;
static const char* metaballs_vertex_shader;
static const char* update_point_lights_compute_shader;
static const char* marching_cubes_create_compute_shader(void);

/* -------------------------------------------------------------------------- *
 * Marching Cubes Tables
 * -------------------------------------------------------------------------- */

// clang-format off
static const int32_t MARCHING_CUBES_EDGE_TABLE[256] = {
  0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

/* Marching cubes tri table with count prepended to each 16-entry row.
 * Format: [count, idx0, idx1, ..., idx14] where count is the number of
 * valid triangle indices. This matches the TypeScript source format.
 * The shader reads tables.tris[cubeIndex * 16] as the index count,
 * then reads indices starting at offset cubeIndex * 16 + 1.
 */
static const int32_t MARCHING_CUBES_TRI_TABLE[4096] = {
   0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1,
   3,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1,
   6,  3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1,
   9,  3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1,
   6,  9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1,
   6,  1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1,
   9,  9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1,
  12,  2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1,
   6,  8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9, 11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1,
   9,  9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1,
  12,  4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1,
   9,  3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1,
  12,  1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1,
  12,  4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1,
   9,  4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1,
   3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1,
   6,  1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1,
   9,  5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1,
  12,  2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1,
   6,  9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1,
   9,  0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1,
  12,  2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1,
   9, 10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1,
  12,  4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1,
  12,  5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1,
   9,  5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1,
   6,  9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1,
   9,  0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1,
   6,  1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1,
  12, 10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1,
  12,  8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1,
   9,  2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1,
   9,  7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1,
  12,  9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1,
  12,  2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1,
   9, 11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1,
  12,  9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1,
  15,  5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0,
  15, 11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0,
   6, 11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1,
   6,  1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1,
   9,  9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1,
  12,  5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1,
   6,  2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9, 11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1,
   9,  0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1,
  12,  5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1,
   9,  6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1,
  12,  0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1,
  12,  3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1,
   9,  6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1,
   6,  5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1,
   9,  1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1,
  12, 10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1,
   9,  6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1,
  12,  1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1,
  12,  8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1,
  15,  7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,
   9,  3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1,
  12,  5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1,
  12,  0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1,
  15,  9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6,
  12,  8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1,
  15,  5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11,
  15,  0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7,
  12,  6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1,
   6, 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1,
   9, 10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1,
  12,  8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1,
   9,  1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1,
  12,  3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1,
   6,  0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1,
   9, 10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1,
  12,  0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1,
  12,  3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1,
  15,  6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1,
  12,  9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1,
  15,  8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1,
   9,  3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1,
   6,  6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1,
  12,  0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1,
  12, 10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1,
   9, 10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1,
  12,  1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1,
  15,  2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9,
   9,  7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1,
   6,  7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1,
  15,  2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7,
  15,  1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11,
  12, 11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1,
  15,  8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6,
   6,  0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1,
   3,  7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1,
   6, 10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1,
   9,  2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1,
  12,  6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1,
   6,  7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1,
   9,  2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1,
  12,  1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1,
   9, 10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1,
  12, 10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1,
  12,  0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1,
   9,  7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1,
   6,  6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1,
   9,  8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1,
  12,  9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1,
   9,  6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1,
  12,  1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1,
  12,  4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1,
  15, 10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3,
   9,  8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1,
   6,  0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1,
   9,  1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1,
  12,  8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1,
   9, 10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1,
  15,  4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3,
   6, 10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1,
   9,  5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1,
  12, 11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1,
   9,  9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1,
  12,  6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1,
  12,  7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1,
  15,  3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6,
   9,  7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1,
  12,  9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1,
  12,  3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1,
  15,  6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,
  12,  9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1,
  15,  1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,
  15,  4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10,
  12,  7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1,
   9,  6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1,
  12,  3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1,
  12,  0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1,
   9,  6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1,
  12,  1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1,
  15,  0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10,
  15, 11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5,
  12,  6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1,
  12,  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1,
   9,  9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1,
  15,  1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8,
   6,  1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  15,  1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6,
  12, 10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1,
   6,  0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3, 10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6, 11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9, 11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1,
   9,  5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1,
  12, 10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1,
   9, 11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1,
  12,  0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1,
  12,  9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1,
  15,  7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2,
   9,  2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1,
  12,  8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1,
  12,  9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1,
  15,  9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2,
   6,  1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1,
   9,  9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1,
   6,  9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1,
  12,  5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1,
  12,  0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1,
  15, 10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4,
  12,  2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1,
  15,  0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11,
  15,  0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5,
   6,  9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1,
   9,  5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1,
  15,  3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9,
  12,  5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1,
   9,  8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1,
   6,  0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1,
   3,  9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1,
  12,  0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1,
  12,  1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1,
  15,  3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4,
  12,  4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1,
  15,  9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3,
   9, 11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1,
  12, 11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1,
  12,  2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1,
  15,  9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7,
  15,  3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10,
   6,  1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1,
  12,  4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1,
   6,  4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1,
   9,  0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1,
   6,  3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1,
  12,  3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1,
   6,  0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1,
   6,  9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  12,  2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1,
   3,  1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   6,  1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};
// clang-format on

/* -------------------------------------------------------------------------- *
 * Protocol Structs - Data structures used across components
 * -------------------------------------------------------------------------- */

typedef struct {
  float width;
  float height;
  float depth;
  uint32_t res_x;
  uint32_t res_y;
  uint32_t res_z;
  float iso_level;
} volume_settings_t;

typedef struct {
  float position[3];
  float radius;
  float strength;
  float subtract;
} metaball_pos_t;

typedef struct {
  float screen_width;
  float screen_height;
  float screen_effect_threshold;
  float enable_screen_effect;
} screen_effect_settings_t;

/* Spot light uniforms - must match WGSL SpotLight struct alignment:
 *   position: vec3<f32>   -> offset  0 (align 16, size 12)
 *   (pad 4 bytes)         -> offset 12
 *   direction: vec3<f32>  -> offset 16 (align 16, size 12)
 *   (pad 4 bytes)         -> offset 28
 *   color: vec3<f32>      -> offset 32 (align 16, size 12)
 *   cutOff: f32           -> offset 44 (align 4, size 4)
 *   outerCutOff: f32      -> offset 48 (align 4, size 4)
 *   intensity: f32        -> offset 52 (align 4, size 4)
 *   (struct pads to 64 bytes for alignment 16)
 */
typedef struct {
  float position[3];
  float _pad0; /* vec3 padding */
  float direction[3];
  float _pad1; /* vec3 padding */
  float color[3];
  float cutoff;
  float outer_cutoff;
  float intensity;
  float _pad2[2]; /* struct padding to 64 bytes */
} spot_light_uniforms_t;

/* Input point light data for compute shader */
typedef struct {
  vec4 position;
  vec4 velocity;
  vec3 color;
  float range;
  float intensity;
  float _padding[3]; /* Alignment padding */
} input_point_light_t;

/* Spot light initialization settings */
typedef struct {
  vec3 position;
  vec3 direction;
  vec3 color;
  float cut_off;
  float outer_cut_off;
  float intensity;
} ispot_light_t;

/* Screen effect configuration */
typedef struct {
  const char* fragment_shader_wgsl;
  struct {
    WGPUBindGroupLayout* items;
    uint32_t item_count;
  } bind_group_layouts;
  struct {
    WGPUBindGroup* items;
    uint32_t item_count;
  } bind_groups;
  WGPUTextureFormat presentation_format;
  const char* label;
} iscreen_effect_t;

/* Projection uniforms for shaders - matches WGSL alignment (144 bytes) */
typedef struct {
  mat4 matrix;
  mat4 inverse_matrix;
  vec2 output_size;
  float z_near;
  float z_far;
} projection_uniforms_t;

/* View uniforms for shaders */
typedef struct {
  mat4 matrix;
  mat4 inverse_matrix;
  vec3 position;
  float time;
  float delta_time;
  float _padding[3]; /* Alignment */
} view_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Camera Structures
 * -------------------------------------------------------------------------- */

typedef struct {
  float left;
  float right;
  float bottom;
  float top;
  float near;
  float far;
  mat4 view_matrix;
  mat4 projection_matrix;
  vec3 position;
} orthographic_camera_t;

typedef struct {
  float fov;
  float aspect;
  float near;
  float far;
  mat4 view_matrix;
  mat4 view_inv_matrix;
  mat4 projection_matrix;
  mat4 projection_inv_matrix;
  vec3 position;
  vec3 look_at_position;
} perspective_camera_t;

typedef struct {
  float damping;
  perspective_camera_t* camera;
  vec3 target;
  float spherical_radius;
  float spherical_theta;
  float spherical_phi;
  float damped_theta;
  float damped_phi;
  float damped_radius;
  bool is_mouse_down;
  float prev_pointer_x;
  float prev_pointer_y;
} camera_controller_t;

/* -------------------------------------------------------------------------- *
 * WebGPU Renderer State
 * -------------------------------------------------------------------------- */

typedef struct {
  /* Device reference */
  wgpu_context_t* wgpu_context;

  /* Uniform buffers */
  wgpu_buffer_t projection_ubo;
  wgpu_buffer_t view_ubo;
  wgpu_buffer_t screen_projection_ubo;
  wgpu_buffer_t screen_view_ubo;

  /* Depth texture */
  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;

  /* Bind groups */
  WGPUBindGroup frame_bind_group;
  WGPUBindGroupLayout frame_bind_group_layout;

  /* Screen-space bind groups */
  WGPUBindGroup screen_frame_bind_group;
  WGPUBindGroupLayout screen_frame_bind_group_layout;

  /* Screen size */
  uint32_t screen_width;
  uint32_t screen_height;

  /* Default sampler */
  WGPUSampler default_sampler;

  /* Presentation format */
  WGPUTextureFormat presentation_format;

  /* Framebuffer for final composite pass */
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;
} webgpu_renderer_t;

/* -------------------------------------------------------------------------- *
 * Metaballs Compute
 * -------------------------------------------------------------------------- */

/* GPU-side metaball representation
 * Must match WGSL struct alignment:
 *   position: vec3<f32>  -> offset 0,  size 12
 *   radius: f32          -> offset 12, size 4
 *   strength: f32        -> offset 16, size 4
 *   subtract: f32        -> offset 20, size 4
 *   (struct padded to 32 bytes for array stride alignment 16)
 */
typedef struct {
  float position[3];
  float radius;
  float strength;
  float subtract;
  float _pad[2]; /* padding to reach struct size 32 (array stride) */
} gpu_metaball_t;

/* Note: Volume data is written directly to the GPU buffer matching WGSL
 * alignment for IsosurfaceVolume struct. See init_metaballs_compute(). */

/* Metaball simulation position/velocity */
typedef struct {
  float x, y, z;
  float vx, vy, vz;
  float speed;
} metaball_sim_t;

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;
  volume_settings_t* volume_settings;

  /* Marching cubes tables as GPU buffers */
  WGPUBuffer tables_buffer; /* Combined edge + tri tables */
  uint64_t tables_buffer_size;

  /* Volume data */
  WGPUBuffer volume_buffer;
  uint64_t volume_buffer_size;

  /* Metaball buffers */
  wgpu_buffer_t metaballs_buffer;
  metaball_sim_t ball_positions[MAX_METABALLS];
  gpu_metaball_t ball_data[MAX_METABALLS];
  uint32_t metaball_count;
  float strength;
  float subtract;
  float strength_target;
  float subtract_target;
  bool has_calced_once;

  /* Vertex output buffers */
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t normal_buffer;
  wgpu_buffer_t index_buffer;
  wgpu_buffer_t indirect_render_buffer;
  uint32_t index_count;

  /* Compute pipelines */
  WGPUComputePipeline metaball_field_pipeline;
  WGPUComputePipeline marching_cubes_pipeline;

  /* Bind groups */
  WGPUBindGroup metaball_field_bind_group;
  WGPUBindGroupLayout metaball_field_bind_group_layout;
  WGPUBindGroup marching_cubes_bind_group;
  WGPUBindGroupLayout marching_cubes_bind_group_layout;
} metaballs_compute_t;

/* -------------------------------------------------------------------------- *
 * Point Lights
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;

  /* Light data */
  wgpu_buffer_t lights_buffer;
  wgpu_buffer_t lights_config_uniform_buffer;
  int32_t lights_count;

  /* Compute resources */
  WGPUBindGroupLayout lights_buffer_compute_bind_group_layout;
  WGPUBindGroup lights_buffer_compute_bind_group;
  WGPUPipelineLayout update_compute_pipeline_layout;
  WGPUComputePipeline update_compute_pipeline;
} point_lights_t;

/* -------------------------------------------------------------------------- *
 * Spot Light
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;
  perspective_camera_t camera;

  /* Light properties */
  vec3 _position;
  vec3 _direction;
  vec3 _color;
  float _cut_off;
  float _outer_cut_off;
  float _intensity;

  /* UBOs */
  struct {
    wgpu_buffer_t light_info;
    wgpu_buffer_t projection;
    wgpu_buffer_t view;
  } ubos;
  struct {
    spot_light_uniforms_t light_info;
    projection_uniforms_t projection;
    view_uniforms_t view;
  } ubos_data;

  /* Depth texture for shadow mapping */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;

  /* Render pass */
  struct {
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout ubos;
    WGPUBindGroupLayout depth_texture;
  } bind_group_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup ubos;
    WGPUBindGroup depth_texture;
  } bind_groups;
} spot_light_t;

/* -------------------------------------------------------------------------- *
 * Box Outline
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/box-outline.ts
 * -------------------------------------------------------------------------- */

#define BOX_OUTLINE_RADIUS 2.5f
#define BOX_OUTLINE_SIDE_COUNT 13u

typedef struct {
  webgpu_renderer_t* renderer;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
    wgpu_buffer_t instance_buffer;
  } buffers;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
} box_outline_t;

/* -------------------------------------------------------------------------- *
 * Ground
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/ground.ts
 * -------------------------------------------------------------------------- */

#define GROUND_WORLD_Y -7.5f
#define GROUND_WIDTH 100u
#define GROUND_HEIGHT 100u
#define GROUND_COUNT 100u
#define GROUND_SPACING 0

typedef struct {
  webgpu_renderer_t* renderer;
  spot_light_t* spot_light;

  struct {
    WGPUPipelineLayout render_pipeline;
    WGPUPipelineLayout render_shadow_pipeline;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline render_pipeline;
    WGPURenderPipeline render_shadow_pipeline;
  } render_pipelines;

  WGPUBindGroupLayout model_bind_group_layout;
  WGPUBindGroup model_bind_group;

  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t normal_buffer;
    wgpu_buffer_t instance_offsets_buffer;
    wgpu_buffer_t instance_material_buffer;
    wgpu_buffer_t uniform_buffer;
  } buffers;

  uint32_t instance_count;
  mat4 model_matrix;
} ground_t;

/* -------------------------------------------------------------------------- *
 * Metaballs Rendering
 * -------------------------------------------------------------------------- */

/* Material properties for metaballs */
typedef struct {
  vec3 color_rgb;
  float roughness;
  float metallic;
} metaballs_material_t;

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;
  metaballs_compute_t* compute;

  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPURenderPipeline shadow_pipeline;

  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
  wgpu_buffer_t ubo; /* Material UBO */

  metaballs_material_t material;
  metaballs_material_t target_material;

  mat4 model_matrix;
  WGPUBuffer model_matrix_buffer;
} metaballs_t;

/* -------------------------------------------------------------------------- *
 * Particles
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/meshes/particles.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
} particles_t;

/* -------------------------------------------------------------------------- *
 * Effect - Base class for post-processing effects
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/effect.ts
 * -------------------------------------------------------------------------- */

#define EFFECT_MAX_BIND_GROUP_COUNT 5u

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  struct {
    WGPUBindGroup items[EFFECT_MAX_BIND_GROUP_COUNT];
    uint32_t item_count;
  } bind_groups;
  WGPUTextureFormat presentation_format;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
  } buffers;
} effect_t;

/* -------------------------------------------------------------------------- *
 * Post-Processing Effects
 * -------------------------------------------------------------------------- */

/* Copy Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/copy-pass.ts
 */
typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } copy_texture;
} copy_pass_t;

/* Bloom Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/bloom-pass.ts
 */
#define BLOOM_PASS_TILE_DIM 128u
#define BLOOM_PASS_BATCH {4, 4}
#define BLOOM_PASS_FILTER_SIZE 10u
#define BLOOM_PASS_ITERATIONS 2u

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t* point_lights;
  spot_light_t* spot_light;
  WGPURenderPassDescriptor framebuffer_descriptor;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } bloom_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } input_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } blur_textures[2];

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  WGPUPipelineLayout blur_pipeline_layout;
  WGPUComputePipeline blur_pipeline;

  WGPUBindGroupLayout blur_constants_bind_group_layout;
  WGPUBindGroup blur_compute_constants_bindGroup;

  WGPUBindGroupLayout blur_compute_bind_group_layout;
  WGPUBindGroup blur_compute_bind_groups[3];

  wgpu_buffer_t blur_params_buffer;
  wgpu_buffer_t buffer_0;
  wgpu_buffer_t buffer_1;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUSampler sampler;
  uint32_t block_dim;
} bloom_pass_t;

/* Deferred Pass - G-buffer lighting
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/deferred-pass.ts
 */
typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t point_lights;
  spot_light_t spot_light;

  struct {
    WGPURenderPassColorAttachment color_attachments[2];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_normal;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_diffuse;

  vec3 spot_light_target;
  vec3 spot_light_color_target;
} deferred_pass_t;

/* Result Pass - Final composite
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/result-pass.ts
 */
typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } empty_texture;
} result_pass_t;

/* -------------------------------------------------------------------------- *
 * Main Application State
 * -------------------------------------------------------------------------- */

static struct {
  wgpu_context_t* wgpu_context;
  bool prepared;

  /* Settings */
  volume_settings_t volume_settings;
  screen_effect_settings_t screen_effect_settings;

  /* Cameras */
  perspective_camera_t main_camera;
  orthographic_camera_t screen_camera;
  camera_controller_t camera_controller;

  /* Renderer */
  webgpu_renderer_t renderer;

  /* Compute */
  metaballs_compute_t metaballs_compute;

  /* Scene objects */
  box_outline_t box_outline;
  ground_t ground;
  metaballs_t metaballs;
  particles_t particles;

  /* Post-processing */
  copy_pass_t copy_pass;
  bloom_pass_t bloom_pass;
  deferred_pass_t deferred_pass;
  result_pass_t result_pass;

  /* GUI settings */
  struct {
    int32_t point_lights_count;
    float bloom_threshold;
    bool enable_bloom;
    float iso_level;
  } gui_settings;

  /* Time tracking */
  float last_frame_time;
  float delta_time;
  float rearrange_countdown;
} state = {
  .prepared            = false,
  .rearrange_countdown = 5.0f,
  .volume_settings = {
    .width     = 6.0f,
    .height    = 6.0f,
    .depth     = 6.0f,
    .res_x     = VOLUME_WIDTH,
    .res_y     = VOLUME_HEIGHT,
    .res_z     = VOLUME_DEPTH,
    .iso_level = 20.0f,
  },
  .screen_effect_settings = {
    .screen_width            = 1280.0f,
    .screen_height           = 720.0f,
    .screen_effect_threshold = 0.95f,
    .enable_screen_effect    = 1.0f,
  },
  .gui_settings = {
    .point_lights_count = 50,
    .bloom_threshold    = 0.95f,
    .enable_bloom       = true,
    .iso_level          = 20.0f,
  },
};

/* -------------------------------------------------------------------------- *
 * Forward Declarations
 * -------------------------------------------------------------------------- */

static void init_cameras(void);
static void init_renderer(wgpu_context_t* wgpu_context);
static void init_metaballs_compute(wgpu_context_t* wgpu_context);
static void init_metaballs(wgpu_context_t* wgpu_context);

static void update_uniform_buffers(wgpu_context_t* wgpu_context);

static void render_gui(wgpu_context_t* wgpu_context, float delta_time);

static void cleanup_metaballs_compute(void);
static void cleanup_metaballs(void);

/* -------------------------------------------------------------------------- *
 * Camera Implementation
 * -------------------------------------------------------------------------- */

static void orthographic_camera_init(orthographic_camera_t* camera, float left,
                                     float right, float bottom, float top,
                                     float near, float far)
{
  camera->left   = left;
  camera->right  = right;
  camera->bottom = bottom;
  camera->top    = top;
  camera->near   = near;
  camera->far    = far;

  glm_vec3_zero(camera->position);
  glm_mat4_identity(camera->view_matrix);
  glm_ortho(left, right, bottom, top, near, far, camera->projection_matrix);
}

/* TODO: Will be used when post-processing is ported */
#if 0
static void
orthographic_camera_update_view_matrix(orthographic_camera_t* camera)
{
  vec3 center = {0.0f, 0.0f, -1.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_vec3_add(camera->position, center, center);
  glm_lookat(camera->position, center, up, camera->view_matrix);
}
#endif

static void perspective_camera_init(perspective_camera_t* camera, float fov,
                                    float aspect, float near, float far)
{
  camera->fov    = fov;
  camera->aspect = aspect;
  camera->near   = near;
  camera->far    = far;

  glm_vec3_zero(camera->position);
  glm_vec3_zero(camera->look_at_position);
  glm_mat4_identity(camera->view_matrix);
  glm_mat4_identity(camera->view_inv_matrix);
  glm_perspective(glm_rad(fov), aspect, near, far, camera->projection_matrix);
  glm_mat4_inv(camera->projection_matrix, camera->projection_inv_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* camera)
{
  glm_perspective(glm_rad(camera->fov), camera->aspect, camera->near,
                  camera->far, camera->projection_matrix);
  glm_mat4_inv(camera->projection_matrix, camera->projection_inv_matrix);
}

static void perspective_camera_update_view_matrix(perspective_camera_t* camera)
{
  vec3 up = {0.0f, 1.0f, 0.0f};
  glm_lookat(camera->position, camera->look_at_position, up,
             camera->view_matrix);
  glm_mat4_inv(camera->view_matrix, camera->view_inv_matrix);
}

static void perspective_camera_set_position(perspective_camera_t* camera,
                                            vec3 position)
{
  glm_vec3_copy(position, camera->position);
}

static void perspective_camera_look_at(perspective_camera_t* camera,
                                       vec3 target)
{
  glm_vec3_copy(target, camera->look_at_position);
  perspective_camera_update_view_matrix(camera);
}

static void camera_controller_init(camera_controller_t* controller,
                                   perspective_camera_t* camera, float damping)
{
  controller->camera  = camera;
  controller->damping = damping;

  glm_vec3_zero(controller->target);
  controller->spherical_radius = 10.0f;
  controller->spherical_theta  = GLM_PI / 4.0f;
  controller->spherical_phi    = GLM_PI / 4.0f;

  controller->damped_theta  = controller->spherical_theta;
  controller->damped_phi    = controller->spherical_phi;
  controller->damped_radius = controller->spherical_radius;

  controller->is_mouse_down  = false;
  controller->prev_pointer_x = 0.0f;
  controller->prev_pointer_y = 0.0f;
}

static void camera_controller_update(camera_controller_t* controller, float dt)
{
  /* Damping interpolation */
  float t = 1.0f - powf(controller->damping, dt);

  controller->damped_theta
    += (controller->spherical_theta - controller->damped_theta) * t;
  controller->damped_phi
    += (controller->spherical_phi - controller->damped_phi) * t;
  controller->damped_radius
    += (controller->spherical_radius - controller->damped_radius) * t;

  /* Convert spherical to Cartesian */
  float sin_phi   = sinf(controller->damped_phi);
  float cos_phi   = cosf(controller->damped_phi);
  float sin_theta = sinf(controller->damped_theta);
  float cos_theta = cosf(controller->damped_theta);

  controller->camera->position[0]
    = controller->target[0] + controller->damped_radius * sin_phi * sin_theta;
  controller->camera->position[1]
    = controller->target[1] + controller->damped_radius * cos_phi;
  controller->camera->position[2]
    = controller->target[2] + controller->damped_radius * sin_phi * cos_theta;

  perspective_camera_look_at(controller->camera, controller->target);
}

static void camera_controller_on_mouse_down(camera_controller_t* controller,
                                            float x, float y)
{
  controller->is_mouse_down  = true;
  controller->prev_pointer_x = x;
  controller->prev_pointer_y = y;
}

static void camera_controller_on_mouse_up(camera_controller_t* controller)
{
  controller->is_mouse_down = false;
}

static void camera_controller_on_mouse_move(camera_controller_t* controller,
                                            float x, float y)
{
  if (!controller->is_mouse_down) {
    return;
  }

  float dx = x - controller->prev_pointer_x;
  float dy = y - controller->prev_pointer_y;

  controller->spherical_theta -= dx * 0.005f;
  controller->spherical_phi -= dy * 0.005f;

  /* Clamp phi to avoid flipping */
  controller->spherical_phi
    = fmaxf(0.1f, fminf(GLM_PI - 0.1f, controller->spherical_phi));

  controller->prev_pointer_x = x;
  controller->prev_pointer_y = y;
}

static void camera_controller_on_wheel(camera_controller_t* controller,
                                       float delta)
{
  controller->spherical_radius *= 1.0f + delta * 0.001f;
  controller->spherical_radius
    = fmaxf(2.0f, fminf(30.0f, controller->spherical_radius));
}

/* -------------------------------------------------------------------------- *
 * Initialization Functions
 * -------------------------------------------------------------------------- */

static void init_cameras(void)
{
  uint32_t width  = (uint32_t)state.wgpu_context->width;
  uint32_t height = (uint32_t)state.wgpu_context->height;
  float aspect    = (float)width / (float)height;

  /* Main perspective camera */
  perspective_camera_init(&state.main_camera, 45.0f, aspect, 0.1f, 100.0f);
  state.main_camera.position[0] = 0.0f;
  state.main_camera.position[1] = 3.0f;
  state.main_camera.position[2] = 10.0f;

  /* Screen orthographic camera */
  orthographic_camera_init(&state.screen_camera, -1.0f, 1.0f, -1.0f, 1.0f,
                           -1.0f, 1.0f);

  /* Camera controller */
  camera_controller_init(&state.camera_controller, &state.main_camera, 0.02f);
}

static void init_renderer(wgpu_context_t* wgpu_context)
{
  state.renderer.wgpu_context  = wgpu_context;
  state.renderer.screen_width  = (uint32_t)wgpu_context->width;
  state.renderer.screen_height = (uint32_t)wgpu_context->height;

  /* Create projection uniform buffer */
  state.renderer.projection_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Projection UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(projection_uniforms_t),
                  });

  /* Create view uniform buffer */
  state.renderer.view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "View UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(view_uniforms_t),
                  });

  /* Create screen projection uniform buffer */
  state.renderer.screen_projection_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Screen Projection UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4),
                  });

  /* Create screen view uniform buffer */
  state.renderer.screen_view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Screen View UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4),
                  });

  /* Create depth texture */
  WGPUTextureDescriptor depth_texture_desc = {
    .label = STRVIEW("Depth - Texture"),
    .usage = WGPUTextureUsage_RenderAttachment
             | WGPUTextureUsage_TextureBinding,
    .dimension      = WGPUTextureDimension_2D,
    .size           = {
      .width              = state.renderer.screen_width,
      .height             = state.renderer.screen_height,
      .depthOrArrayLayers = 1,
    },
    .format         = DEPTH_FORMAT,
    .mipLevelCount  = 1,
    .sampleCount    = 1,
  };
  state.renderer.depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_texture_desc);
  ASSERT(state.renderer.depth_texture);

  WGPUTextureViewDescriptor depth_view_desc = {
    .label           = STRVIEW("Depth texture view"),
    .format          = DEPTH_FORMAT,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.renderer.depth_texture_view
    = wgpuTextureCreateView(state.renderer.depth_texture, &depth_view_desc);
  ASSERT(state.renderer.depth_texture_view);

  /* Create default sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Default sampler"),
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  state.renderer.default_sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(state.renderer.default_sampler);

  /* Create frame bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                    | WGPUShaderStage_Compute,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(projection_uniforms_t),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                    | WGPUShaderStage_Compute,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(view_uniforms_t),
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Frame bind group layout"),
    .entryCount = 3,
    .entries    = bgl_entries,
  };
  state.renderer.frame_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.renderer.frame_bind_group_layout);

  /* Create frame bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.renderer.projection_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(projection_uniforms_t),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.renderer.view_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(view_uniforms_t),
    },
    [2] = (WGPUBindGroupEntry){
      .binding = 2,
      .sampler = state.renderer.default_sampler,
    },
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Frame bind group"),
    .layout     = state.renderer.frame_bind_group_layout,
    .entryCount = 3,
    .entries    = bg_entries,
  };
  state.renderer.frame_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.renderer.frame_bind_group);

  /* Create screen frame bind group layout */
  {
    WGPUBindGroupLayoutEntry screen_bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                      | WGPUShaderStage_Compute,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
      },
      [1] = (WGPUBindGroupLayoutEntry){
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                      | WGPUShaderStage_Compute,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(mat4),
        },
      },
    };
    WGPUBindGroupLayoutDescriptor screen_bgl_desc = {
      .label      = STRVIEW("Screen frame bind group layout"),
      .entryCount = 2,
      .entries    = screen_bgl_entries,
    };
    state.renderer.screen_frame_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &screen_bgl_desc);
    ASSERT(state.renderer.screen_frame_bind_group_layout);
  }

  /* Create screen frame bind group */
  WGPUBindGroupEntry screen_bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.renderer.screen_projection_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(mat4),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.renderer.screen_view_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(mat4),
    },
  };

  WGPUBindGroupDescriptor screen_bg_desc = {
    .label      = STRVIEW("Screen frame bind group"),
    .layout     = state.renderer.screen_frame_bind_group_layout,
    .entryCount = 2,
    .entries    = screen_bg_entries,
  };
  state.renderer.screen_frame_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &screen_bg_desc);
  ASSERT(state.renderer.screen_frame_bind_group);

  /* Store presentation format */
  state.renderer.presentation_format = wgpu_context->render_format;

  /* Initialize framebuffer descriptor for final composite pass */
  state.renderer.framebuffer.color_attachments[0]
    = (WGPURenderPassColorAttachment){
      .view       = NULL, /* Will be set per frame to swapchain view */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    };
  state.renderer.framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = state.renderer.depth_texture_view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };
  state.renderer.framebuffer.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Final composite pass"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = state.renderer.framebuffer.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update projection uniforms */
  projection_uniforms_t proj_uniforms = {0};
  glm_mat4_copy(state.main_camera.projection_matrix, proj_uniforms.matrix);
  glm_mat4_copy(state.main_camera.projection_inv_matrix,
                proj_uniforms.inverse_matrix);
  proj_uniforms.output_size[0] = (float)state.renderer.screen_width;
  proj_uniforms.output_size[1] = (float)state.renderer.screen_height;
  proj_uniforms.z_near         = state.main_camera.near;
  proj_uniforms.z_far          = state.main_camera.far;
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.renderer.projection_ubo.buffer, 0, &proj_uniforms,
                       sizeof(proj_uniforms));

  /* Update view uniforms */
  view_uniforms_t view_uniforms = {0};
  glm_mat4_copy(state.main_camera.view_matrix, view_uniforms.matrix);
  glm_mat4_copy(state.main_camera.view_inv_matrix,
                view_uniforms.inverse_matrix);
  glm_vec3_copy(state.main_camera.position, view_uniforms.position);
  view_uniforms.time       = state.last_frame_time;
  view_uniforms.delta_time = state.delta_time;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.renderer.view_ubo.buffer, 0,
                       &view_uniforms, sizeof(view_uniforms));

  /* Update screen projection and view matrices */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.renderer.screen_projection_ubo.buffer, 0,
                       &state.screen_camera.projection_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.renderer.screen_view_ubo.buffer, 0,
                       &state.screen_camera.view_matrix, sizeof(mat4));
}

/* -------------------------------------------------------------------------- *
 * Input Handling
 * -------------------------------------------------------------------------- */

static void recreate_depth_texture(wgpu_context_t* wgpu_context)
{
  uint32_t width  = (uint32_t)wgpu_context->width;
  uint32_t height = (uint32_t)wgpu_context->height;

  /* Recreate depth texture */
  if (state.renderer.depth_texture_view) {
    wgpuTextureViewRelease(state.renderer.depth_texture_view);
    state.renderer.depth_texture_view = NULL;
  }
  if (state.renderer.depth_texture) {
    wgpuTextureRelease(state.renderer.depth_texture);
    state.renderer.depth_texture = NULL;
  }

  WGPUTextureDescriptor depth_texture_desc = {
    .label = STRVIEW("Depth texture"),
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    .dimension               = WGPUTextureDimension_2D,
    .size.width              = width,
    .size.height             = height,
    .size.depthOrArrayLayers = 1,
    .format                  = DEPTH_FORMAT,
    .mipLevelCount           = 1,
    .sampleCount             = 1,
  };
  state.renderer.depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &depth_texture_desc);

  WGPUTextureViewDescriptor depth_view_desc = {
    .label           = STRVIEW("Depth texture view"),
    .format          = DEPTH_FORMAT,
    .dimension       = WGPUTextureViewDimension_2D,
    .mipLevelCount   = 1,
    .arrayLayerCount = 1,
    .aspect          = WGPUTextureAspect_All,
  };
  state.renderer.depth_texture_view
    = wgpuTextureCreateView(state.renderer.depth_texture, &depth_view_desc);

  state.renderer.screen_width  = width;
  state.renderer.screen_height = height;
}

/* Forward declarations for resize helpers */
static void copy_pass_recreate_textures(copy_pass_t* this);
static void bloom_pass_recreate_textures(bloom_pass_t* this,
                                         copy_pass_t* copy_pass);
static void deferred_pass_recreate_textures(deferred_pass_t* this);
static void result_pass_recreate_bind_group(result_pass_t* this,
                                            copy_pass_t* copy_pass,
                                            bloom_pass_t* bloom_pass);

/* -------------------------------------------------------------------------- *
 * Metaballs Compute Implementation
 * -------------------------------------------------------------------------- */

static void init_metaballs_compute(wgpu_context_t* wgpu_context)
{
  metaballs_compute_t* mc = &state.metaballs_compute;
  mc->wgpu_context        = wgpu_context;
  mc->renderer            = &state.renderer;
  mc->volume_settings     = &state.volume_settings;
  mc->metaball_count      = MAX_METABALLS;
  mc->strength            = 1.0f;
  mc->subtract            = 1.0f;
  mc->strength_target     = 1.0f;
  mc->subtract_target     = 1.0f;
  mc->has_calced_once     = false;

  /* Initialize volume dimensions */
  volume_settings_t* vol = &state.volume_settings;

  /* Create tables buffer (edge table + tri table combined) */
  {
    mc->tables_buffer_size = (ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE)
                              + ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE))
                             * sizeof(int32_t);
    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Metaballs table - Storage buffer"),
      .usage            = WGPUBufferUsage_Storage,
      .size             = mc->tables_buffer_size,
      .mappedAtCreation = true,
    };
    mc->tables_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(mc->tables_buffer);

    int32_t* tables_array = (int32_t*)wgpuBufferGetMappedRange(
      mc->tables_buffer, 0, mc->tables_buffer_size);
    size_t j = 0;
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE); ++i) {
      tables_array[j++] = (int32_t)MARCHING_CUBES_EDGE_TABLE[i];
    }
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE); ++i) {
      tables_array[j++] = (int32_t)MARCHING_CUBES_TRI_TABLE[i];
    }
    wgpuBufferUnmap(mc->tables_buffer);
  }

  /* Create metaballs buffer */
  {
    mc->metaballs_buffer = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Metaballs - Storage buffer",
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size  = sizeof(uint32_t) * 4 + sizeof(gpu_metaball_t) * MAX_METABALLS,
      });
  }

  /* Create volume buffer
   *
   * WGSL struct alignment for IsosurfaceVolume:
   *   min: vec3<f32>   -> offset  0, size 12, pad 4 (align 16)
   *   max: vec3<f32>   -> offset 16, size 12, pad 4 (align 16)
   *   step: vec3<f32>  -> offset 32, size 12, pad 4 (align 16)
   *   size: vec3<u32>  -> offset 48, size 12
   *   threshold: f32   -> offset 60, size  4
   *   values: array    -> offset 64
   *
   * Header = 16 floats (64 bytes), then volume_elements floats for values
   */
  {
    const uint32_t volume_elements      = vol->res_x * vol->res_y * vol->res_z;
    const uint32_t volume_header_floats = 16; /* 64 bytes for struct header */
    const uint64_t volume_buffer_size
      = sizeof(float) * (volume_header_floats + volume_elements);

    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Volume - Storage buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = volume_buffer_size,
      .mappedAtCreation = true,
    };
    mc->volume_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(mc->volume_buffer);

    float* v = (float*)wgpuBufferGetMappedRange(mc->volume_buffer, 0,
                                                volume_buffer_size);
    memset(v, 0, (size_t)volume_buffer_size);

    /* min: vec3<f32> at float[0..2], pad at float[3] */
    v[0] = -vol->width / 2.0f;
    v[1] = -vol->height / 2.0f;
    v[2] = -vol->depth / 2.0f;

    /* max: vec3<f32> at float[4..6], pad at float[7] */
    v[4] = vol->width / 2.0f;
    v[5] = vol->height / 2.0f;
    v[6] = vol->depth / 2.0f;

    /* step: vec3<f32> at float[8..10], pad at float[11] */
    v[8]  = vol->width / (float)(vol->res_x - 1);
    v[9]  = vol->height / (float)(vol->res_y - 1);
    v[10] = vol->depth / (float)(vol->res_z - 1);

    /* size: vec3<u32> at float[12..14] (byte offset 48-59) */
    uint32_t* size_u32 = (uint32_t*)&v[12];
    size_u32[0]        = vol->res_x;
    size_u32[1]        = vol->res_y;
    size_u32[2]        = vol->res_z;

    /* threshold: f32 at float[15] (byte offset 60) */
    v[15] = vol->iso_level;

    wgpuBufferUnmap(mc->volume_buffer);

    /* Store volume buffer size for bind group */
    mc->volume_buffer_size = volume_buffer_size;
  }

  /* Calculate buffer sizes */
  const uint32_t marching_cube_cells
    = (vol->res_x - 1) * (vol->res_y - 1) * (vol->res_z - 1);
  const size_t vertex_buffer_size
    = sizeof(float) * 3 * 12 * marching_cube_cells;
  const size_t index_buffer_size = sizeof(uint32_t) * 15 * marching_cube_cells;

  /* Create vertex buffer */
  mc->vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Metaballs - Vertex buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  /* Create normal buffer */
  mc->normal_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Metaballs - Normal buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  /* Create index buffer */
  mc->index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Metaballs - Index buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
                    .size  = index_buffer_size,
                  });
  mc->index_count = (uint32_t)(index_buffer_size / sizeof(uint32_t));

  /* Create indirect render buffer - must match DrawIndirectArgs struct in
   * shader (36 bytes) */
  /* Layout: vc, vertexCount, firstVertex, firstInstance, indexCount,
     indexedInstanceCount, indexedFirstIndex, indexedBaseVertex,
     indexedFirstInstance */
  uint32_t indirect_render_array[9] = {
    0, /* vc (unused) */
    0, /* vertexCount (atomic, set by compute shader) */
    0, /* firstVertex */
    0, /* firstInstance */
    0, /* indexCount (atomic, set by compute shader) */
    1, /* indexedInstanceCount */
    0, /* indexedFirstIndex */
    0, /* indexedBaseVertex */
    0, /* indexedFirstInstance */
  };
  mc->indirect_render_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Metaballs indirect draw - Storage buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect
                             | WGPUBufferUsage_CopyDst,
                    .size         = sizeof(indirect_render_array),
                    .initial.data = indirect_render_array,
                  });

  /* Initialize ball positions with random velocities */
  for (uint32_t i = 0; i < MAX_METABALLS; ++i) {
    mc->ball_positions[i].x
      = (random_float() * 2 - 1) * (vol->width / 2.0f - 0.5f);
    mc->ball_positions[i].y
      = (random_float() * 2 - 1) * (vol->height / 2.0f - 0.5f);
    mc->ball_positions[i].z
      = (random_float() * 2 - 1) * (vol->depth / 2.0f - 0.5f);
    mc->ball_positions[i].vx    = random_float() * 1000;
    mc->ball_positions[i].vy    = (random_float() * 2 - 1) * 10;
    mc->ball_positions[i].vz    = random_float() * 1000;
    mc->ball_positions[i].speed = random_float() * 2 + 0.3f;
  }

  /* Compute metaballs pipeline */
  {
    WGPUShaderModule comp_shader = wgpu_create_shader_module(
      wgpu_context->device, metaball_field_compute_shader);
    ASSERT(comp_shader);

    mc->metaball_field_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Metaball field - Compute pipeline"),
        .compute = {
          .module     = comp_shader,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(mc->metaball_field_pipeline);

    wgpuShaderModuleRelease(comp_shader);
  }

  /* Compute metaballs bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = mc->metaballs_buffer.buffer,
        .size    = mc->metaballs_buffer.size,
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = mc->volume_buffer,
        .size    = mc->volume_buffer_size,
      },
    };

    mc->metaball_field_bind_group_layout
      = wgpuComputePipelineGetBindGroupLayout(mc->metaball_field_pipeline, 0);

    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Metaball field - Bind group"),
      .layout     = mc->metaball_field_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    mc->metaball_field_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(mc->metaball_field_bind_group);
  }

  /* Compute marching cubes pipeline */
  {
    WGPUShaderModule comp_shader = wgpu_create_shader_module(
      wgpu_context->device, marching_cubes_create_compute_shader());
    ASSERT(comp_shader);

    mc->marching_cubes_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Marching cubes - Compute pipeline"),
        .compute = {
          .module     = comp_shader,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(mc->marching_cubes_pipeline);

    wgpuShaderModuleRelease(comp_shader);
  }

  /* Compute marching cubes bind group */
  {
    WGPUBindGroupEntry bg_entries[6] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = mc->tables_buffer,
        .size    = mc->tables_buffer_size,
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = mc->volume_buffer,
        .size    = mc->volume_buffer_size,
      },
      [2] = (WGPUBindGroupEntry){
        .binding = 2,
        .buffer  = mc->vertex_buffer.buffer,
        .size    = mc->vertex_buffer.size,
      },
      [3] = (WGPUBindGroupEntry){
        .binding = 3,
        .buffer  = mc->normal_buffer.buffer,
        .size    = mc->normal_buffer.size,
      },
      [4] = (WGPUBindGroupEntry){
        .binding = 4,
        .buffer  = mc->index_buffer.buffer,
        .size    = mc->index_buffer.size,
      },
      [5] = (WGPUBindGroupEntry){
        .binding = 5,
        .buffer  = mc->indirect_render_buffer.buffer,
        .size    = mc->indirect_render_buffer.size,
      },
    };

    mc->marching_cubes_bind_group_layout
      = wgpuComputePipelineGetBindGroupLayout(mc->marching_cubes_pipeline, 0);

    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Marching cubes - Bind group"),
      .layout     = mc->marching_cubes_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    mc->marching_cubes_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(mc->marching_cubes_bind_group);
  }
}

static void update_metaballs_sim(float time_delta)
{
  metaballs_compute_t* mc = &state.metaballs_compute;
  volume_settings_t* vol  = mc->volume_settings;

  /* Smooth interpolation of subtract and strength towards targets */
  mc->subtract += (mc->subtract_target - mc->subtract) * time_delta * 4.0f;
  mc->strength += (mc->strength_target - mc->strength) * time_delta * 4.0f;

  /* Update metaball positions */
  for (uint32_t i = 0; i < MAX_METABALLS; i++) {
    metaball_sim_t* pos = &mc->ball_positions[i];

    /* Spring-like force towards center */
    pos->vx += -pos->x * pos->speed * 20.0f;
    pos->vy += -pos->y * pos->speed * 20.0f;
    pos->vz += -pos->z * pos->speed * 20.0f;

    /* Update position */
    pos->x += pos->vx * pos->speed * time_delta * 0.0001f;
    pos->y += pos->vy * pos->speed * time_delta * 0.0001f;
    pos->z += pos->vz * pos->speed * time_delta * 0.0001f;

    /* Bounce off walls */
    const float padding = 0.9f;
    const float width   = vol->width / 2.0f - padding;
    const float height  = vol->height / 2.0f - padding;
    const float depth   = vol->depth / 2.0f - padding;

    if (pos->x > width) {
      pos->x = width;
      pos->vx *= -1.0f;
    }
    else if (pos->x < -width) {
      pos->x = -width;
      pos->vx *= -1.0f;
    }

    if (pos->y > height) {
      pos->y = height;
      pos->vy *= -1.0f;
    }
    else if (pos->y < -height) {
      pos->y = -height;
      pos->vy *= -1.0f;
    }

    if (pos->z > depth) {
      pos->z = depth;
      pos->vz *= -1.0f;
    }
    else if (pos->z < -depth) {
      pos->z = -depth;
      pos->vz *= -1.0f;
    }
  }

  /* Fill GPU buffer data */
  uint32_t metaball_header[4] = {MAX_METABALLS, 0, 0, 0};
  wgpuQueueWriteBuffer(mc->wgpu_context->queue, mc->metaballs_buffer.buffer, 0,
                       metaball_header, sizeof(metaball_header));

  for (uint32_t i = 0; i < MAX_METABALLS; i++) {
    metaball_sim_t* position = &mc->ball_positions[i];
    gpu_metaball_t* metaball = &mc->ball_data[i];

    metaball->position[0] = position->x;
    metaball->position[1] = position->y;
    metaball->position[2] = position->z;
    metaball->radius      = sqrtf(mc->strength / mc->subtract);
    metaball->strength    = mc->strength;
    metaball->subtract    = mc->subtract;
    metaball->_pad[0]     = 0.0f;
    metaball->_pad[1]     = 0.0f;
  }

  wgpuQueueWriteBuffer(mc->wgpu_context->queue, mc->metaballs_buffer.buffer,
                       sizeof(uint32_t) * 4, mc->ball_data,
                       sizeof(gpu_metaball_t) * MAX_METABALLS);

  /* Reset indirect draw buffer - must match DrawIndirectArgs struct (36 bytes)
   */
  uint32_t indirect_reset[9] = {
    0, /* vc (unused) */
    0, /* vertexCount (will be set by compute shader) */
    0, /* firstVertex */
    0, /* firstInstance */
    0, /* indexCount (unused in our indexed draw) */
    1, /* indexedInstanceCount */
    0, /* indexedFirstIndex */
    0, /* indexedBaseVertex */
    0, /* indexedFirstInstance */
  };
  wgpuQueueWriteBuffer(mc->wgpu_context->queue,
                       mc->indirect_render_buffer.buffer, 0, indirect_reset,
                       sizeof(indirect_reset));
}

/* Randomize metaballs compute targets for smooth transitions */
static void metaballs_compute_rearrange(metaballs_compute_t* mc)
{
  mc->subtract_target = 3.0f + random_float() * 3.0f;
  mc->strength_target = 3.0f + random_float() * 3.0f;
}

/* Randomize metaballs material targets for smooth color transitions */
static void metaballs_rearrange(metaballs_t* mb)
{
  mb->target_material.color_rgb[0] = random_float();
  mb->target_material.color_rgb[1] = random_float();
  mb->target_material.color_rgb[2] = random_float();

  mb->target_material.metallic  = 0.08f + random_float() * 0.92f;
  mb->target_material.roughness = 0.08f + random_float() * 0.92f;

  metaballs_compute_rearrange(&state.metaballs_compute);
}

static void dispatch_metaballs_compute(WGPUComputePassEncoder compute_pass)
{
  metaballs_compute_t* mc = &state.metaballs_compute;
  volume_settings_t* vol  = mc->volume_settings;

  const uint32_t dispatch_x = vol->res_x / METABALLS_COMPUTE_WORKGROUP_SIZE[0];
  const uint32_t dispatch_y = vol->res_y / METABALLS_COMPUTE_WORKGROUP_SIZE[1];
  const uint32_t dispatch_z = vol->res_z / METABALLS_COMPUTE_WORKGROUP_SIZE[2];

  /* Metaball field compute */
  wgpuComputePassEncoderSetPipeline(compute_pass, mc->metaball_field_pipeline);
  wgpuComputePassEncoderSetBindGroup(compute_pass, 0,
                                     mc->metaball_field_bind_group, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass, dispatch_x, dispatch_y,
                                           dispatch_z);

  /* Marching cubes compute */
  wgpuComputePassEncoderSetPipeline(compute_pass, mc->marching_cubes_pipeline);
  wgpuComputePassEncoderSetBindGroup(compute_pass, 0,
                                     mc->marching_cubes_bind_group, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(compute_pass, dispatch_x, dispatch_y,
                                           dispatch_z);

  mc->has_calced_once = true;
}

static void cleanup_metaballs_compute(void)
{
  metaballs_compute_t* mc = &state.metaballs_compute;

  WGPU_RELEASE_RESOURCE(Buffer, mc->tables_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->metaballs_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->volume_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->normal_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->index_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, mc->indirect_render_buffer.buffer);
  WGPU_RELEASE_RESOURCE(ComputePipeline, mc->metaball_field_pipeline);
  WGPU_RELEASE_RESOURCE(ComputePipeline, mc->marching_cubes_pipeline);
  WGPU_RELEASE_RESOURCE(BindGroup, mc->metaball_field_bind_group);
  WGPU_RELEASE_RESOURCE(BindGroup, mc->marching_cubes_bind_group);
}

/* -------------------------------------------------------------------------- *
 * Metaballs Render Implementation
 * -------------------------------------------------------------------------- */

static void init_metaballs(wgpu_context_t* wgpu_context)
{
  metaballs_t* mb  = &state.metaballs;
  mb->wgpu_context = wgpu_context;
  mb->renderer     = &state.renderer;

  /* Metaballs material UBO */
  {
    /* padding required for struct alignment: size % 8 == 0 */
    const float metaballs_ubo_data[5] = {1.0f, 1.0f, 1.0f, 0.3f, 0.1f};
    mb->ubo
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Metaballs - UBO",
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size         = 8 * sizeof(float),
                                           .initial.data = metaballs_ubo_data,
                                         });
  }

  /* Bind group layout for metaballs material (group 1) */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = mb->ubo.size,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Metaballs - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    mb->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(mb->bind_group_layout);
  }

  /* Bind group for metaballs material (group 1) */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = mb->ubo.buffer,
        .size    = mb->ubo.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Metaballs - Bind group"),
      .layout     = mb->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    mb->bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(mb->bind_group);
  }

  /* Pipeline layout - frame bind group (group 0) + material (group 1) */
  WGPUBindGroupLayout bind_group_layouts[2] = {
    state.renderer.frame_bind_group_layout, /* Group 0 */
    mb->bind_group_layout,                  /* Group 1 */
  };

  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = STRVIEW("Metaballs render - Pipeline layout"),
    .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
    .bindGroupLayouts     = bind_group_layouts,
  };
  mb->pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                       &pipeline_layout_desc);
  ASSERT(mb->pipeline_layout);

  /* Create shader module */
  WGPUShaderModule vertex_shader_module
    = wgpu_create_shader_module(wgpu_context->device, metaballs_vertex_shader);
  ASSERT(vertex_shader_module);

  WGPUShaderModule fragment_shader_module = wgpu_create_shader_module(
    wgpu_context->device, metaballs_fragment_shader);
  ASSERT(fragment_shader_module);

  /* Vertex buffers layout */
  WGPUVertexAttribute vertex_attributes[2] = {
    [0] = (WGPUVertexAttribute){
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    },
    [1] = (WGPUVertexAttribute){
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    },
  };

  WGPUVertexBufferLayout vertex_buffers[2] = {
    [0] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
  };

  /* Color target state - 2 GBuffer outputs */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_states[2] = {
    [0] = (WGPUColorTargetState){
      /* normal + material id */
      .format    = WGPUTextureFormat_RGBA16Float,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    },
    [1] = (WGPUColorTargetState){
      /* albedo */
      .format    = WGPUTextureFormat_BGRA8Unorm,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    },
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = DEPTH_FORMAT,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Create render pipeline */
  WGPURenderPipelineDescriptor pipeline_desc = {
    .label  = STRVIEW("Metaballs - Render pipeline"),
    .layout = mb->pipeline_layout,
    .vertex = {
      .module      = vertex_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffers),
      .buffers     = vertex_buffers,
    },
    .fragment = &(WGPUFragmentState){
      .module      = fragment_shader_module,
      .entryPoint  = STRVIEW("main"),
      .targetCount = (uint32_t)ARRAY_SIZE(color_target_states),
      .targets     = color_target_states,
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .depthStencil = &depth_stencil_state,
    .multisample  = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  mb->render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
  ASSERT(mb->render_pipeline);

  /* Cleanup shader modules */
  wgpuShaderModuleRelease(vertex_shader_module);
  wgpuShaderModuleRelease(fragment_shader_module);

  /* Shadow pipeline will be initialized later when spot_light is available */
  mb->shadow_pipeline = NULL;
}

/* Shadow vertex shader for metaballs */
// clang-format off
static const char* metaballs_shadow_vertex_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  @group(0) @binding(1) var<uniform> projection: ProjectionUniformsStruct;
  @group(0) @binding(2) var<uniform> view: ViewUniformsStruct;

  struct Inputs {
    @location(0) position: vec3<f32>,
  }

  struct Output {
    @builtin(position) position: vec4<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;
    output.position = projection.matrix *
                      view.matrix *
                      vec4(input.position, 1.0);

    return output;
  }
);
// clang-format on

/* Initialize metaballs shadow pipeline after spot_light is available */
static void init_metaballs_shadow(wgpu_context_t* wgpu_context,
                                  spot_light_t* spot_light)
{
  metaballs_t* mb = &state.metaballs;

  /* Shadow pipeline layout - uses spot light's ubos bind group */
  WGPUBindGroupLayout shadow_bind_group_layouts[1] = {
    spot_light->bind_group_layouts.ubos,
  };

  WGPUPipelineLayoutDescriptor shadow_layout_desc = {
    .label                = STRVIEW("Metaballs shadow - Pipeline layout"),
    .bindGroupLayoutCount = 1,
    .bindGroupLayouts     = shadow_bind_group_layouts,
  };
  WGPUPipelineLayout shadow_pipeline_layout
    = wgpuDeviceCreatePipelineLayout(wgpu_context->device, &shadow_layout_desc);
  ASSERT(shadow_pipeline_layout);

  /* Shadow vertex shader - use embedded shader from later in file */
  /* Note: The shader string will be defined after Point Lights section */

  /* Vertex state for shadow rendering */
  WGPUVertexAttribute shadow_vertex_attr = {
    .format         = WGPUVertexFormat_Float32x3,
    .offset         = 0,
    .shaderLocation = 0,
  };
  WGPUVertexBufferLayout shadow_vertex_buffer_layout = {
    .arrayStride    = sizeof(float) * 3,
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = &shadow_vertex_attr,
  };

  /* Depth stencil state for shadow pass */
  WGPUDepthStencilState shadow_depth_stencil
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  shadow_depth_stencil.depthCompare = WGPUCompareFunction_Less;

  /* Shadow pipeline descriptor */
  WGPURenderPipelineDescriptor shadow_pipeline_desc = {
    .label  = STRVIEW("Metaballs shadow - Render pipeline"),
    .layout = shadow_pipeline_layout,
    .vertex = {
      .bufferCount = 1,
      .buffers     = &shadow_vertex_buffer_layout,
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .depthStencil = &shadow_depth_stencil,
    .multisample  = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  /* Create shadow vertex shader module from embedded WGSL */
  WGPUShaderModuleDescriptor shadow_shader_desc = {
    .label = STRVIEW("Metaballs shadow - Vertex shader"),
    .nextInChain
      = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
          .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = {metaballs_shadow_vertex_shader_wgsl, strlen(metaballs_shadow_vertex_shader_wgsl)},
      },
  };
  WGPUShaderModule shadow_shader
    = wgpuDeviceCreateShaderModule(wgpu_context->device, &shadow_shader_desc);
  ASSERT(shadow_shader);

  shadow_pipeline_desc.vertex.module     = shadow_shader;
  shadow_pipeline_desc.vertex.entryPoint = STRVIEW("main");

  mb->shadow_pipeline = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
                                                       &shadow_pipeline_desc);
  ASSERT(mb->shadow_pipeline);

  wgpuShaderModuleRelease(shadow_shader);
  wgpuPipelineLayoutRelease(shadow_pipeline_layout);
}

static void render_metaballs(WGPURenderPassEncoder render_pass)
{
  metaballs_t* mb         = &state.metaballs;
  metaballs_compute_t* mc = &state.metaballs_compute;

  if (!mc->has_calced_once) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, mb->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    state.renderer.frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, mb->bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, mc->vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, mc->normal_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(render_pass, mc->index_buffer.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  /* Draw all indices - the compute shader pads unused slots with degenerate
   * triangles (all vertices the same), so drawing the full buffer is safe */
  wgpuRenderPassEncoderDrawIndexed(render_pass, mc->index_count, 1, 0, 0, 0);
}

static void cleanup_metaballs(void)
{
  metaballs_t* mb = &state.metaballs;

  WGPU_RELEASE_RESOURCE(RenderPipeline, mb->render_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, mb->shadow_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, mb->pipeline_layout);
  WGPU_RELEASE_RESOURCE(Buffer, mb->ubo.buffer);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, mb->bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, mb->bind_group);
}

static void render_metaballs_shadow(WGPURenderPassEncoder render_pass,
                                    spot_light_t* spot_light)
{
  metaballs_t* mb         = &state.metaballs;
  metaballs_compute_t* mc = &state.metaballs_compute;

  if (!mc->has_calced_once || !mb->shadow_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, mb->shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    spot_light->bind_groups.ubos, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, mc->vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(render_pass, mc->index_buffer.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  /* Draw all indices - matching the main pass */
  wgpuRenderPassEncoderDrawIndexed(render_pass, mc->index_count, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Point Lights Implementation
 * -------------------------------------------------------------------------- */

/* Point lights update compute shader - embedded */
static const char* update_point_lights_compute_shader = CODE(
  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct InputPointLight {
    position: vec4<f32>,
    velocity: vec4<f32>,
    color: vec3<f32>,
    range: f32,
    intensity: f32,
  }

  struct LightsBuffer {
    lights: array<InputPointLight>,
  }

  struct LightsConfig {
    numLights: u32,
  }

  @group(0) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;
  @group(0) @binding(1) var<uniform> config: LightsConfig;

  @group(1) @binding(1) var<uniform> view: ViewUniformsStruct;

  @compute @workgroup_size(64, 1, 1)
  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var index = GlobalInvocationID.x;
    if (index >= config.numLights) {
      return;
    }

    lightsBuffer.lights[index].position.x += lightsBuffer.lights[index].velocity.x * view.deltaTime;
    lightsBuffer.lights[index].position.z += lightsBuffer.lights[index].velocity.z * view.deltaTime;

    let size = 42.0;
    var halfSize = size / 2.0;

    if (lightsBuffer.lights[index].position.x < -halfSize) {
      lightsBuffer.lights[index].position.x = -halfSize;
      lightsBuffer.lights[index].velocity.x *= -1.0;
    } else if (lightsBuffer.lights[index].position.x > halfSize) {
      lightsBuffer.lights[index].position.x = halfSize;
      lightsBuffer.lights[index].velocity.x *= -1.0;
    }

    if (lightsBuffer.lights[index].position.z < -halfSize) {
      lightsBuffer.lights[index].position.z = -halfSize;
      lightsBuffer.lights[index].velocity.z *= -1.0;
    } else if (lightsBuffer.lights[index].position.z > halfSize) {
      lightsBuffer.lights[index].position.z = halfSize;
      lightsBuffer.lights[index].velocity.z *= -1.0;
    }
  }
);

static bool point_lights_is_ready(point_lights_t* this)
{
  return this->update_compute_pipeline != NULL;
}

static void point_lights_set_lights_count(point_lights_t* this, uint32_t v)
{
  this->lights_count = v;
  wgpuQueueWriteBuffer(this->wgpu_context->queue,
                       this->lights_config_uniform_buffer.buffer, 0, &v,
                       sizeof(uint32_t));
}

static void point_lights_init(point_lights_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->lights_buffer_compute_bind_group_layout, /* Group 0 */
      this->renderer->frame_bind_group_layout,       /* Group 1 */
    };
    this->update_compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Point light update - Pipeline layout"),
        .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
        .bindGroupLayouts     = bind_group_layouts,
      });
    ASSERT(this->update_compute_pipeline_layout != NULL);
  }

  /* Compute pipeline */
  {
    WGPUShaderModule comp_shader = wgpu_create_shader_module(
      wgpu_context->device, update_point_lights_compute_shader);
    ASSERT(comp_shader);

    this->update_compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Point light update - Compute pipeline"),
        .layout  = this->update_compute_pipeline_layout,
        .compute = {
          .module     = comp_shader,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(this->update_compute_pipeline);

    wgpuShaderModuleRelease(comp_shader);
  }
}

static void point_lights_init_defaults(point_lights_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void point_lights_create(point_lights_t* this,
                                webgpu_renderer_t* renderer)
{
  point_lights_init_defaults(this);
  this->renderer     = renderer;
  this->wgpu_context = renderer->wgpu_context;

  wgpu_context_t* wgpu_context = this->wgpu_context;

  /* Lights uniform buffer */
  {
    input_point_light_t lights_data[MAX_POINT_LIGHTS_COUNT] = {0};
    float x, y, z, vel_x, vel_y, vel_z, r, g, b, radius, intensity;
    for (uint32_t i = 0; i < MAX_POINT_LIGHTS_COUNT; ++i) {
      input_point_light_t* light_data = &lights_data[i];

      x = (random_float() * 2 - 1) * 20;
      y = -2.0f;
      z = (random_float() * 2 - 1) * 20;

      vel_x = random_float() * 4 - 2;
      vel_y = random_float() * 4 - 2;
      vel_z = random_float() * 4 - 2;

      r = random_float();
      g = random_float();
      b = random_float();

      radius    = 5 + random_float() * 3;
      intensity = 10 + random_float() * 10;

      /* Position */
      light_data->position[0] = x;
      light_data->position[1] = y;
      light_data->position[2] = z;
      light_data->position[3] = 0.0f;
      /* Velocity */
      light_data->velocity[0] = vel_x;
      light_data->velocity[1] = vel_y;
      light_data->velocity[2] = vel_z;
      light_data->velocity[3] = 0.0f;
      /* Color */
      light_data->color[0] = r;
      light_data->color[1] = g;
      light_data->color[2] = b;
      /* Radius */
      light_data->range = radius;
      /* Intensity */
      light_data->intensity = intensity;
    }
    this->lights_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Lights - Uniform buffer",
                                           .usage = WGPUBufferUsage_Storage,
                                           .size  = sizeof(lights_data),
                                           .initial.data = lights_data,
                                         });
  }

  /* Lights config uniform buffer */
  {
    this->lights_count                 = state.gui_settings.point_lights_count;
    this->lights_config_uniform_buffer = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label        = "Lights config - Uniform buffer",
        .usage        = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size         = sizeof(uint32_t),
        .initial.data = &this->lights_count,
      });
  }

  /* Lights buffer compute bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
           .type           = WGPUBufferBindingType_Storage,
           .minBindingSize = this->lights_buffer.size,
         },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
             .type           = WGPUBufferBindingType_Uniform,
             .minBindingSize = this->lights_config_uniform_buffer.size,
         },
        .sampler = {0},
      },
    };
    this->lights_buffer_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(
        wgpu_context->device,
        &(WGPUBindGroupLayoutDescriptor){
          .label      = STRVIEW("Lights update compute - Bind group layout"),
          .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
          .entries    = bgl_entries,
        });
    ASSERT(this->lights_buffer_compute_bind_group_layout != NULL);
  }

  /* Lights buffer compute bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->lights_buffer.buffer,
        .size    = this->lights_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->lights_config_uniform_buffer.buffer,
        .size    = this->lights_config_uniform_buffer.size,
      },
    };
    this->lights_buffer_compute_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("lights buffer compute - Bind group"),
        .layout     = this->lights_buffer_compute_bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(this->lights_buffer_compute_bind_group != NULL);
  }

  point_lights_init(this);
}

static void point_lights_destroy(point_lights_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        this->lights_buffer_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->lights_buffer_compute_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->update_compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->update_compute_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_config_uniform_buffer.buffer)
}

static point_lights_t*
point_lights_update_sim(point_lights_t* this,
                        WGPUComputePassEncoder compute_pass)
{
  if (!point_lights_is_ready(this)) {
    return this;
  }
  wgpuComputePassEncoderSetPipeline(compute_pass,
                                    this->update_compute_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->lights_buffer_compute_bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(compute_pass, 1,
                                     this->renderer->frame_bind_group, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass, (uint32_t)ceil(state.gui_settings.point_lights_count / 64.0f),
    1, 1);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Spot Light Implementation
 * -------------------------------------------------------------------------- */

#define SPOT_LIGHT_SHADOW_RES (1024u)

static void spot_light_set_position(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_position);
  glm_vec3_copy(v, this->ubos_data.light_info.position);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));

  perspective_camera_set_position(&this->camera,
                                  (vec3){-v[0] * 15, v[1], -v[2] * 15});
  perspective_camera_update_view_matrix(&this->camera);

  view_uniforms_t* view_uniforms = &this->ubos_data.view;
  glm_mat4_copy(this->camera.view_matrix, view_uniforms->matrix);
  glm_mat4_copy(this->camera.view_inv_matrix, view_uniforms->inverse_matrix);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.view.buffer, 0,
                       view_uniforms, sizeof(view_uniforms_t));
}

static void spot_light_set_direction(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_direction);
  glm_vec3_copy(v, this->ubos_data.light_info.direction);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));

  glm_vec3_copy(v, this->camera.look_at_position);
  perspective_camera_update_view_matrix(&this->camera);

  view_uniforms_t* view_uniforms = &this->ubos_data.view;
  glm_mat4_copy(this->camera.view_matrix, view_uniforms->matrix);
  glm_mat4_copy(this->camera.view_inv_matrix, view_uniforms->inverse_matrix);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.view.buffer, 0,
                       view_uniforms, sizeof(view_uniforms_t));
}

static void spot_light_set_color(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_color);
  glm_vec3_copy(v, this->ubos_data.light_info.color);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));
}

static void spot_light_set_cut_off(spot_light_t* this, float v)
{
  this->_cut_off                    = v;
  this->ubos_data.light_info.cutoff = cosf(v);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));
}

static void spot_light_set_outer_cut_off(spot_light_t* this, float v)
{
  this->_outer_cut_off                    = v;
  this->ubos_data.light_info.outer_cutoff = cosf(v);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));

  this->camera.fov = glm_deg(v * 1.5f);
  perspective_camera_update_projection_matrix(&this->camera);

  projection_uniforms_t* projection_uniforms = &this->ubos_data.projection;
  glm_mat4_copy(this->camera.projection_matrix, projection_uniforms->matrix);
  glm_mat4_copy(this->camera.projection_inv_matrix,
                projection_uniforms->inverse_matrix);
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.projection.buffer,
                       0, projection_uniforms, sizeof(projection_uniforms_t));
}

static void spot_light_set_intensity(spot_light_t* this, float v)
{
  this->_intensity                     = v;
  this->ubos_data.light_info.intensity = v;
  wgpuQueueWriteBuffer(this->wgpu_context->queue, this->ubos.light_info.buffer,
                       0, &this->ubos_data.light_info,
                       sizeof(spot_light_uniforms_t));
}

static void spot_light_init_defaults(spot_light_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void spot_light_create(spot_light_t* this, webgpu_renderer_t* renderer,
                              ispot_light_t* ispot_light)
{
  spot_light_init_defaults(this);

  this->renderer               = renderer;
  this->wgpu_context           = renderer->wgpu_context;
  wgpu_context_t* wgpu_context = this->wgpu_context;

  perspective_camera_init(&this->camera, 56.0f, 1.0f, 0.1f, 120.0f);
  perspective_camera_update_view_matrix(&this->camera);
  perspective_camera_update_projection_matrix(&this->camera);

  /* Depth texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = SPOT_LIGHT_SHADOW_RES,
      .height             = SPOT_LIGHT_SHADOW_RES,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Spot light - Depth texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth32Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->depth_texture.texture != NULL);

    this->depth_texture.view = wgpuTextureCreateView(
      this->depth_texture.texture,
      &(WGPUTextureViewDescriptor){
        .label           = STRVIEW("Spot light - Depth texture view"),
        .dimension       = WGPUTextureViewDimension_2D,
        .format          = texture_desc.format,
        .baseMipLevel    = 0,
        .mipLevelCount   = 1,
        .baseArrayLayer  = 0,
        .arrayLayerCount = 1,
      });
    ASSERT(this->depth_texture.view != NULL);
  }

  /* Light info UBO */
  this->ubos.light_info = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Light info - UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(spot_light_uniforms_t),
                  });

  /* Projection UBO */
  this->ubos.projection = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Projection - UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(projection_uniforms_t),
                  });
  {
    projection_uniforms_t* proj = &this->ubos_data.projection;
    glm_mat4_copy(this->camera.projection_matrix, proj->matrix);
    glm_mat4_copy(this->camera.projection_inv_matrix, proj->inverse_matrix);
    proj->output_size[0] = (float)SPOT_LIGHT_SHADOW_RES;
    proj->output_size[1] = (float)SPOT_LIGHT_SHADOW_RES;
    proj->z_near         = this->camera.near;
    proj->z_far          = this->camera.far;
    wgpuQueueWriteBuffer(wgpu_context->queue, this->ubos.projection.buffer, 0,
                         proj, sizeof(projection_uniforms_t));
  }

  /* View UBO */
  this->ubos.view = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "View - UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(view_uniforms_t),
                  });
  {
    view_uniforms_t* view = &this->ubos_data.view;
    glm_mat4_copy(this->camera.view_matrix, view->matrix);
    glm_mat4_copy(this->camera.view_inv_matrix, view->inverse_matrix);
    glm_vec3_copy(this->camera.position, view->position);
    view->time       = 0.0f;
    view->delta_time = 0.0f;
    wgpuQueueWriteBuffer(wgpu_context->queue, this->ubos.view.buffer, 0, view,
                         sizeof(view_uniforms_t));
  }

  /* Set spot light properties */
  spot_light_set_position(this, ispot_light->position);
  spot_light_set_direction(this, ispot_light->direction);
  spot_light_set_color(this, ispot_light->color);
  spot_light_set_cut_off(this, ispot_light->cut_off);
  spot_light_set_outer_cut_off(this, ispot_light->outer_cut_off);
  spot_light_set_intensity(this, ispot_light->intensity);

  /* Render pass descriptor */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = this->depth_texture.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Spot light - Shadow pass"),
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
    .occlusionQuerySet      = NULL,
  };

  /* Spot light ubos bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->ubos.light_info.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->ubos.projection.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->ubos.view.size,
        },
        .sampler = {0},
      }
    };
    this->bind_group_layouts.ubos = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Spot light ubos - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layouts.ubos != NULL);
  }

  /* Spot light depth texture bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Depth,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.depth_texture = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Spot light depth texture - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layouts.depth_texture != NULL);
  }

  /* Spot light ubos bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->ubos.light_info.buffer,
        .size    = this->ubos.light_info.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->ubos.projection.buffer,
        .size    = this->ubos.projection.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->ubos.view.buffer,
        .size    = this->ubos.view.size,
      },
    };
    this->bind_groups.ubos = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Spot light ubos - Bind group"),
                              .layout = this->bind_group_layouts.ubos,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_groups.ubos != NULL);
  }

  /* Spot light depth texture bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->depth_texture.view,
      },
    };
    this->bind_groups.depth_texture = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Spot light depth texture - Bind group"),
        .layout     = this->bind_group_layouts.depth_texture,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(this->bind_groups.depth_texture != NULL);
  }
}

static void spot_light_destroy(spot_light_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.light_info.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.projection.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.view.buffer);
  WGPU_RELEASE_RESOURCE(Texture, this->depth_texture.texture);
  WGPU_RELEASE_RESOURCE(TextureView, this->depth_texture.view);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.ubos);
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        this->bind_group_layouts.depth_texture);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.ubos);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.depth_texture);
}

/* -------------------------------------------------------------------------- *
 * Box Outline Implementation
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* box_outline_vertex_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  @group(0) @binding(0) var<uniform> projection : ProjectionUniformsStruct;
  @group(0) @binding(1) var<uniform> view : ViewUniformsStruct;

  struct Inputs {
    @location(0) position: vec3<f32>,
    @location(1) instanceMat0: vec4<f32>,
    @location(2) instanceMat1: vec4<f32>,
    @location(3) instanceMat2: vec4<f32>,
    @location(4) instanceMat3: vec4<f32>,
  }

  struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) localPosition: vec3<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;

    var instanceMatrix = mat4x4(
      input.instanceMat0,
      input.instanceMat1,
      input.instanceMat2,
      input.instanceMat3,
    );

    var worldPosition = vec4<f32>(input.position, 1.0);
    output.position = projection.matrix *
                      view.matrix *
                      instanceMatrix *
                      worldPosition;

    output.localPosition = input.position;
    return output;
  }
);

static const char* box_outline_fragment_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct Output {
    @location(0) GBuffer_OUT0: vec4<f32>,
    @location(1) GBuffer_OUT1: vec4<f32>,
  }

  fn encodeNormals(n: vec3<f32>) -> vec2<f32> {
    var p = sqrt(n.z * 8.0 + 8.0);
    return vec2(n.xy / p + 0.5);
  }

  fn encodeGBufferOutput(
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ID: f32
  ) -> Output {
    var output: Output;
    output.GBuffer_OUT0 = vec4(encodeNormals(normal), metallic, ID);
    output.GBuffer_OUT1 = vec4(albedo, roughness);
    return output;
  }

  struct Input {
    @location(0) localPosition: vec3<f32>,
  }
  @group(0) @binding(0) var<uniform> projection : ProjectionUniformsStruct;
  @group(0) @binding(1) var<uniform> view : ViewUniformsStruct;

  @fragment
  fn main(input: Input) -> Output {
    var output: Output;
    var spacing = step(sin(input.localPosition.x * 10.0 + view.time * 2.0), 0.1);
    if (spacing < 0.5) {
      discard;
    }
    var normal = vec3(0.0);
    var albedo = vec3(1.0);
    var metallic = 0.0;
    var roughness = 0.0;
    var ID = 0.1;
    return encodeGBufferOutput(
      normal,
      albedo,
      metallic,
      roughness,
      ID
    );
  }
);
// clang-format on

static void box_outline_init(box_outline_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->renderer->frame_bind_group_layout,
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Box outline render - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Box outline render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_LineStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    /* Color target state */
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    /* Vertex attributes */
    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 4 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 8 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 12 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
    };

    /* Vertex buffer layouts */
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 16 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 4,
        .attributes     = &attributes[1],
      },
    };

    /* Vertex shader module */
    WGPUShaderModule vs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Box outline vertex shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {box_outline_vertex_shader_wgsl, strlen(box_outline_vertex_shader_wgsl)},
          },
      });

    /* Fragment shader module */
    WGPUShaderModule fs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Box outline fragment shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {box_outline_fragment_shader_wgsl, strlen(box_outline_fragment_shader_wgsl)},
          },
      });

    /* Render pipeline */
    this->render_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Box outline - Render pipeline"),
        .layout   = this->pipeline_layout,
        .primitive = primitive_state,
        .vertex = (WGPUVertexState){
          .module      = vs_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers     = vertex_buffers,
        },
        .fragment = &(WGPUFragmentState){
          .module      = fs_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = (uint32_t)ARRAY_SIZE(color_target_states),
          .targets     = color_target_states,
        },
        .depthStencil = &depth_stencil_state,
        .multisample = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      });
    ASSERT(this->render_pipeline != NULL);

    /* Cleanup shader modules */
    WGPU_RELEASE_RESOURCE(ShaderModule, vs_module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fs_module);
  }
}

static void box_outline_init_defaults(box_outline_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void box_outline_create(box_outline_t* this, webgpu_renderer_t* renderer)
{
  box_outline_init_defaults(this);
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  const float vertices[2 * 3]
    = {-BOX_OUTLINE_RADIUS, 0.0f, 0.0f, BOX_OUTLINE_RADIUS, 0.0f, 0.0f};

  const uint16_t indices[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  };

  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box outline - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box outline - Index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  float instance_matrices[BOX_OUTLINE_SIDE_COUNT * 16] = {0};
  mat4 instance_matrix                                 = GLM_MAT4_IDENTITY_INIT;

  /* Top ring */
  /* Instance 0: front top edge - translate only (TS rotates around (0,0,0)
   * which is a no-op in gl-matrix) */
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  memcpy(&instance_matrices[0 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 1: right top edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[1 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 2: left top edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[2 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 3: back top edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[3 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Bottom ring */
  /* Instance 4: front bottom edge - translate only (TS rotates around (0,0,0)
   * which is a no-op in gl-matrix) */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  memcpy(&instance_matrices[4 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 5: right bottom edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[5 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 6: left bottom edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[6 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 7: back bottom edge */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[7 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Vertical sides */
  /* Instance 8: front-right vertical */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[8 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 9: front-right vertical (TS duplicate position with extra Y
   * rotation) */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[9 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 10: front-left vertical */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[10 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 11: back-left vertical */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[11 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Instance 12: back-right vertical */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, GLM_PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, GLM_PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[12 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  this->buffers.instance_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Box outline instance matrices - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_matrices),
                    .initial.data = instance_matrices,
                  });

  /* Init render pipeline */
  box_outline_init(this);
}

static void box_outline_destroy(box_outline_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_buffer.buffer);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline);
}

static void box_outline_render(box_outline_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->frame_bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 2, BOX_OUTLINE_SIDE_COUNT, 0, 0,
                                   0);
}

/* -------------------------------------------------------------------------- *
 * Cube Geometry
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    float data[18 * 6];
    size_t data_size;
    size_t count;
  } positions;
  struct {
    float data[18 * 6];
    size_t data_size;
    size_t count;
  } normals;
  struct {
    float data[12 * 6];
    size_t data_size;
    size_t count;
  } uvs;
} cube_geometry_t;

static void cube_geometry_init_defaults(cube_geometry_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void cube_geometry_create_cube(cube_geometry_t* this, vec3 dimensions)
{
  cube_geometry_init_defaults(this);

  const vec3 position
    = {-dimensions[0] / 2.0f, -dimensions[1] / 2.0f, -dimensions[2] / 2.0f};
  const float x      = position[0];
  const float y      = position[1];
  const float z      = position[2];
  const float width  = dimensions[0];
  const float height = dimensions[1];
  const float depth  = dimensions[2];

  const vec3 fbl = {x, y, z + depth};
  const vec3 fbr = {x + width, y, z + depth};
  const vec3 ftl = {x, y + height, z + depth};
  const vec3 ftr = {x + width, y + height, z + depth};
  const vec3 bbl = {x, y, z};
  const vec3 bbr = {x + width, y, z};
  const vec3 btl = {x, y + height, z};
  const vec3 btr = {x + width, y + height, z};

  // clang-format off
  const float positions[18 * 6] = {
    /* front */
    fbl[0], fbl[1], fbl[2], fbr[0], fbr[1], fbr[2], ftl[0], ftl[1], ftl[2],
    ftl[0], ftl[1], ftl[2], fbr[0], fbr[1], fbr[2], ftr[0], ftr[1], ftr[2],
    /* right */
    fbr[0], fbr[1], fbr[2], bbr[0], bbr[1], bbr[2], ftr[0], ftr[1], ftr[2],
    ftr[0], ftr[1], ftr[2], bbr[0], bbr[1], bbr[2], btr[0], btr[1], btr[2],
    /* back */
    bbr[0], bbr[1], bbr[2], bbl[0], bbl[1], bbl[2], btr[0], btr[1], btr[2],
    btr[0], btr[1], btr[2], bbl[0], bbl[1], bbl[2], btl[0], btl[1], btl[2],
    /* left */
    bbl[0], bbl[1], bbl[2], fbl[0], fbl[1], fbl[2], btl[0], btl[1], btl[2],
    btl[0], btl[1], btl[2], fbl[0], fbl[1], fbl[2], ftl[0], ftl[1], ftl[2],
    /* top */
    ftl[0], ftl[1], ftl[2], ftr[0], ftr[1], ftr[2], btl[0], btl[1], btl[2],
    btl[0], btl[1], btl[2], ftr[0], ftr[1], ftr[2], btr[0], btr[1], btr[2],
    /* bottom */
    bbl[0], bbl[1], bbl[2], bbr[0], bbr[1], bbr[2], fbl[0], fbl[1], fbl[2],
    fbl[0], fbl[1], fbl[2], bbr[0], bbr[1], bbr[2], fbr[0], fbr[1], fbr[2],
  };
  // clang-format on
  memcpy(this->positions.data, positions, sizeof(positions));
  this->positions.data_size = sizeof(positions);
  this->positions.count     = (size_t)ARRAY_SIZE(positions);

  static const float uvs[12 * 6] = {
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* front */
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* right */
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* back */
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* left */
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* top */
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* bottom */
  };
  memcpy(this->uvs.data, uvs, sizeof(uvs));
  this->uvs.data_size = sizeof(uvs);
  this->uvs.count     = (size_t)ARRAY_SIZE(uvs);

  static const float normals[18 * 6] = {
    0,  0,  1,  0,  0,  1,  0,  0,  1,
    0,  0,  1,  0,  0,  1,  0,  0,  1, /* front */
    1,  0,  0,  1,  0,  0,  1,  0,  0,
    1,  0,  0,  1,  0,  0,  1,  0,  0, /* right */
    0,  0,  -1, 0,  0,  -1, 0,  0,  -1,
    0,  0,  -1, 0,  0,  -1, 0,  0,  -1, /* back */
    -1, 0,  0,  -1, 0,  0,  -1, 0,  0,
    -1, 0,  0,  -1, 0,  0,  -1, 0,  0, /* left */
    0,  1,  0,  0,  1,  0,  0,  1,  0,
    0,  1,  0,  0,  1,  0,  0,  1,  0, /* top */
    0,  -1, 0,  0,  -1, 0,  0,  -1, 0,
    0,  -1, 0,  0,  -1, 0,  0,  -1, 0, /* bottom */
  };
  memcpy(this->normals.data, normals, sizeof(normals));
  this->normals.data_size = sizeof(normals);
  this->normals.count     = (size_t)ARRAY_SIZE(normals);
}

/* -------------------------------------------------------------------------- *
 * Ground Implementation
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* ground_vertex_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct ModelUniforms {
    matrix: mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> projection: ProjectionUniformsStruct;
  @group(0) @binding(1) var<uniform> view: ViewUniformsStruct;
  @group(1) @binding(0) var<uniform> model: ModelUniforms;

  struct Inputs {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instanceOffset: vec3<f32>,
    @location(3) metallic: f32,
    @location(4) roughness: f32,
  }

  struct Output {
    @location(0) normal: vec3<f32>,
    @location(1) metallic: f32,
    @location(2) roughness: f32,
    @builtin(position) position: vec4<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;
    var dist = distance(input.instanceOffset.xy, vec2(0.0));
    var offsetX = input.instanceOffset.x;
    var offsetZ = input.instanceOffset.y;
    var scaleY = input.instanceOffset.z;
    var offsetPos = vec3(offsetX, abs(dist) * 0.06 + scaleY * 0.01, offsetZ);
    var scaleMatrix = mat4x4(
      1.0, 0.0, 0.0, 0.0,
      0.0, scaleY, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    );
    var worldPosition = model.matrix * scaleMatrix * vec4(input.position + offsetPos, 1.0);
    output.position = projection.matrix *
                      view.matrix *
                      worldPosition;

    output.normal = input.normal;
    output.metallic = input.metallic;
    output.roughness = input.roughness;
    return output;
  }
);

static const char* ground_fragment_shader_wgsl = CODE(
  struct Output {
    @location(0) GBuffer_OUT0: vec4<f32>,
    @location(1) GBuffer_OUT1: vec4<f32>,
  }

  fn encodeNormals(n: vec3<f32>) -> vec2<f32> {
    var p = sqrt(n.z * 8.0 + 8.0);
    return vec2(n.xy / p + 0.5);
  }

  fn encodeGBufferOutput(
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ID: f32
  ) -> Output {
    var output: Output;
    output.GBuffer_OUT0 = vec4(encodeNormals(normal), metallic, ID);
    output.GBuffer_OUT1 = vec4(albedo, roughness);
    return output;
  }

  struct Inputs {
    @location(0) normal: vec3<f32>,
    @location(1) metallic: f32,
    @location(2) roughness: f32,
  }

  @fragment
  fn main(input: Inputs) -> Output {
    var normal = normalize(input.normal);
    var albedo = vec3(1.0);
    var metallic = 1.0;
    var roughness = input.roughness;
    var ID = 0.0;

    return encodeGBufferOutput(
      normal,
      albedo,
      metallic,
      roughness,
      ID
    );
  }
);

static const char* ground_shadow_vertex_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct ModelUniforms {
    matrix: mat4x4<f32>,
  }

  @group(0) @binding(1) var<uniform> projection: ProjectionUniformsStruct;
  @group(0) @binding(2) var<uniform> view: ViewUniformsStruct;
  @group(1) @binding(0) var<uniform> model: ModelUniforms;

  struct Inputs {
    @location(0) position: vec3<f32>,
    @location(1) instanceOffset: vec3<f32>,
  }

  struct Output {
    @builtin(position) position: vec4<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;
    var dist = distance(input.instanceOffset.xy, vec2(0.0));
    var offsetX = input.instanceOffset.x;
    var offsetZ = input.instanceOffset.y;
    var scaleY = input.instanceOffset.z;
    var offsetPos = vec3(offsetX, abs(dist) * 0.06 + scaleY * 0.01, offsetZ);
    var scaleMatrix = mat4x4(
      1.0, 0.0, 0.0, 0.0,
      0.0, scaleY, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    );
    var worldPosition = model.matrix * scaleMatrix * vec4(input.position + offsetPos, 1.0);
    output.position = projection.matrix *
                      view.matrix *
                      worldPosition;

    return output;
  }
);
// clang-format on

static void ground_init(ground_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Ground render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->frame_bind_group_layout,
      this->model_bind_group_layout,
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Ground render - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_pipeline = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);
  }

  /* Ground render pipeline */
  {
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 1 * sizeof(float),
        .format         = WGPUVertexFormat_Float32,
      },
    };

    WGPUVertexBufferLayout vertex_buffers[4] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
      [2] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &attributes[2],
      },
      [3] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 2,
        .attributes     = &attributes[3],
      },
    };

    WGPUShaderModule vs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Ground vertex shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {ground_vertex_shader_wgsl, strlen(ground_vertex_shader_wgsl)},
          },
      });

    WGPUShaderModule fs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Ground fragment shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {ground_fragment_shader_wgsl, strlen(ground_fragment_shader_wgsl)},
          },
      });

    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Ground - Render pipeline"),
        .layout   = this->pipeline_layouts.render_pipeline,
        .primitive = primitive_state,
        .vertex = (WGPUVertexState){
          .module      = vs_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers     = vertex_buffers,
        },
        .fragment = &(WGPUFragmentState){
          .module      = fs_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = (uint32_t)ARRAY_SIZE(color_target_states),
          .targets     = color_target_states,
        },
        .depthStencil = &depth_stencil_state,
        .multisample = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      });
    ASSERT(this->render_pipelines.render_pipeline != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vs_module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fs_module);
  }

  /* Ground shadow render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->spot_light->bind_group_layouts.ubos,
      this->model_bind_group_layout,
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label = STRVIEW("Ground shadow rendering - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_shadow_pipeline
      = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                       &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);
  }

  /* Ground shadow render pipeline */
  {
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    WGPUVertexAttribute shadow_attributes[2] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
    };

    WGPUVertexBufferLayout shadow_vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &shadow_attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &shadow_attributes[1],
      },
    };

    WGPUShaderModule shadow_vs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Ground shadow vertex shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {ground_shadow_vertex_shader_wgsl, strlen(ground_shadow_vertex_shader_wgsl)},
          },
      });

    this->render_pipelines.render_shadow_pipeline
      = wgpuDeviceCreateRenderPipeline(
        wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label    = STRVIEW("Ground shadow - Render pipeline"),
          .layout   = this->pipeline_layouts.render_shadow_pipeline,
          .primitive = primitive_state,
          .vertex = (WGPUVertexState){
            .module      = shadow_vs_module,
            .entryPoint  = STRVIEW("main"),
            .bufferCount = (uint32_t)ARRAY_SIZE(shadow_vertex_buffers),
            .buffers     = shadow_vertex_buffers,
          },
          .fragment = NULL,
          .depthStencil = &depth_stencil_state,
          .multisample = (WGPUMultisampleState){
            .count = 1,
            .mask  = 0xFFFFFFFF,
          },
        });
    ASSERT(this->render_pipelines.render_shadow_pipeline != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, shadow_vs_module);
  }
}

static void ground_init_defaults(ground_t* this)
{
  memset(this, 0, sizeof(*this));
  glm_mat4_identity(this->model_matrix);
}

static void ground_create(ground_t* this, webgpu_renderer_t* renderer,
                          spot_light_t* spot_light)
{
  ground_init_defaults(this);

  this->renderer   = renderer;
  this->spot_light = spot_light;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Create cube */
  cube_geometry_t cube_geometry;
  vec3 cube_dimensions = GLM_VEC3_ONE_INIT;
  cube_geometry_create_cube(&cube_geometry, cube_dimensions);

  /* Ground vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Ground - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = cube_geometry.positions.data_size,
                    .initial.data = cube_geometry.positions.data,
                  });

  /* Ground normal buffer */
  this->buffers.normal_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Ground - Normal buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = cube_geometry.normals.data_size,
                    .initial.data = cube_geometry.normals.data,
                  });

  /* Ground instance buffers */
  float instance_offsets[GROUND_WIDTH * GROUND_HEIGHT * 3]           = {0};
  float instance_metallic_rougness[GROUND_WIDTH * GROUND_HEIGHT * 2] = {0};

  const float spacing_x
    = (float)GROUND_WIDTH / (float)GROUND_COUNT + GROUND_SPACING;
  const float spacing_y
    = (float)GROUND_HEIGHT / (float)GROUND_COUNT + GROUND_SPACING;

  float x_pos = 0.0f, y_pos = 0.0f;
  for (uint32_t x = 0, i = 0; x < GROUND_COUNT; x++) {
    for (uint32_t y = 0; y < GROUND_COUNT; y++) {
      x_pos = (float)x * spacing_x;
      y_pos = (float)y * spacing_y;

      instance_offsets[i * 3 + 0] = x_pos - (float)GROUND_WIDTH / 2.0f;
      instance_offsets[i * 3 + 1] = y_pos - (float)GROUND_HEIGHT / 2.0f;
      instance_offsets[i * 3 + 2] = random_float() * 3.0f + 1.0f;

      instance_metallic_rougness[i * 2 + 0] = 1.0f;
      instance_metallic_rougness[i * 2 + 1] = 0.5f;

      ++i;
    }
  }

  this->instance_count = ARRAY_SIZE(instance_offsets) / 3;

  this->buffers.instance_offsets_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Ground instance xyz - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_offsets),
                    .initial.data = instance_offsets,
                  });

  this->buffers.instance_material_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Ground instance material - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_metallic_rougness),
                    .initial.data = instance_metallic_rougness,
                  });

  /* Ground uniform buffer */
  glm_translate(this->model_matrix, (vec3){0, GROUND_WORLD_Y, 0});
  this->buffers.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Ground - Uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(this->model_matrix),
                    .initial.data = this->model_matrix,
                  });

  /* Ground bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->buffers.uniform_buffer.size,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Ground - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->model_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->model_bind_group_layout != NULL);
  }

  /* Ground bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = this->buffers.uniform_buffer.buffer,
        .size    = this->buffers.uniform_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Ground - Bind group"),
      .layout     = this->model_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->model_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->model_bind_group != NULL);
  }

  /* Init render pipeline */
  ground_init(this);
}

static void ground_destroy(ground_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layouts.render_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        this->pipeline_layouts.render_shadow_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipelines.render_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline,
                        this->render_pipelines.render_shadow_pipeline);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->model_bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, this->model_bind_group);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.normal_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_offsets_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_material_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.uniform_buffer.buffer);
}

static ground_t* ground_render_shadow(ground_t* this,
                                      WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_shadow_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(
    render_pass, this->render_pipelines.render_shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->spot_light->bind_groups.ubos, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_offsets_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

static ground_t* ground_render(ground_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass,
                                   this->render_pipelines.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->frame_bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.normal_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, this->buffers.instance_offsets_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, this->buffers.instance_material_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Particles Implementation
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* particles_vertex_shader_wgsl = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct InputPointLight {
    position: vec4<f32>,
    velocity: vec4<f32>,
    color: vec3<f32>,
    range: f32,
    intensity: f32,
  }

  struct LightsBuffer {
    lights: array<InputPointLight>,
  }

  @group(0) @binding(0) var<uniform> projection: ProjectionUniformsStruct;
  @group(0) @binding(1) var<uniform> view: ViewUniformsStruct;

  @group(1) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;

  struct Inputs {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
  }

  struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
  }

  var<private> normalisedPosition: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
  );

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;

    var inputPosition = normalisedPosition[input.vertexIndex];

    var sc = clamp(lightsBuffer.lights[input.instanceIndex].intensity * 0.01, 0.01, 0.1);
    var scaleMatrix = mat4x4(
      sc,  0.0, 0.0, 0.0,
      0.0, sc,  0.0, 0.0,
      0.0, 0.0, sc,  0.0,
      0.0, 0.0, 0.0, 1.0,
    );

    var instancePosition = lightsBuffer.lights[input.instanceIndex].position;
    var worldPosition = vec4(instancePosition.xyz, 0.0);

    var viewMatrix = view.matrix;

    output.position = projection.matrix *
                      (
                        viewMatrix *
                        (worldPosition +
                        vec4(0.0, 0.0, 0.0, 1.0)) +
                        scaleMatrix * vec4(inputPosition, 0.0, 0.0)
                      );

    var instanceColor = lightsBuffer.lights[input.instanceIndex].color;
    output.color = instanceColor;
    output.uv = inputPosition * vec2(0.5, -0.5) + vec2(0.5);
    return output;
  }
);

static const char* particles_fragment_shader_wgsl = CODE(
  struct Input {
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
  }

  struct Output {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
  }

  @fragment
  fn main(input: Input) -> Output {
    var dist = distance(input.uv, vec2(0.5));
    if (dist > 0.5) {
      discard;
    }
    var output: Output;
    output.normal = vec4(0.0, 0.0, 0.0, 0.1);
    output.albedo = vec4(input.color, 1.0);
    return output;
  }
);
// clang-format on

static void particles_init(particles_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->frame_bind_group_layout,
      this->bind_group_layout,
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Particles render - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    WGPUShaderModule vs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Particles vertex shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {particles_vertex_shader_wgsl, strlen(particles_vertex_shader_wgsl)},
          },
      });

    WGPUShaderModule fs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Particles fragment shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {particles_fragment_shader_wgsl, strlen(particles_fragment_shader_wgsl)},
          },
      });

    this->render_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = STRVIEW("Particles - Render pipeline"),
        .layout   = this->pipeline_layout,
        .primitive = primitive_state,
        .vertex = (WGPUVertexState){
          .module      = vs_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .fragment = &(WGPUFragmentState){
          .module      = fs_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = (uint32_t)ARRAY_SIZE(color_target_states),
          .targets     = color_target_states,
        },
        .depthStencil = &depth_stencil_state,
        .multisample = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      });
    ASSERT(this->render_pipeline != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vs_module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fs_module);
  }
}

static void particles_init_defaults(particles_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void particles_create(particles_t* this, webgpu_renderer_t* renderer,
                             wgpu_buffer_t* lights_buffer)
{
  particles_init_defaults(this);

  this->renderer               = renderer;
  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Particles bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = lights_buffer->size,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Particles - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Particles bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = lights_buffer->buffer,
        .size    = lights_buffer->size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Particles - Bind group"),
      .layout     = this->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
  }

  /* Init render pipeline */
  particles_init(this);
}

static void particles_destroy(particles_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);
}

static void particles_render(particles_t* this,
                             WGPURenderPassEncoder render_pass,
                             uint32_t point_lights_count)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->frame_bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(render_pass, 4, point_lights_count, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Effect Implementation
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* effect_vertex_shader_wgsl = CODE(
  struct Inputs {
    @location(0) position: vec2<f32>,
  }

  struct Output {
    @builtin(position) position: vec4<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;
    output.position = vec4(input.position, 0.0, 1.0);
    return output;
  }
);

static const char* copy_pass_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var texture: texture_2d<f32>;

  struct Inputs {
    @builtin(position) coords: vec4<f32>,
  }
  struct Output {
    @location(0) color: vec4<f32>,
  }

  @fragment
  fn main(input: Inputs) -> Output {
    var output: Output;
    let albedo = textureLoad(
      texture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );
    output.color = vec4(albedo.rgb, 1.0);
    return output;
  }
);
// clang-format on

static void effect_init(effect_t* this, const char* fragment_shader_wgsl,
                        WGPUBindGroupLayout* bind_group_layouts,
                        uint32_t bind_group_layout_count, const char* label)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Render pipeline layout */
  {
    char pipeline_layout_lbl[256] = {0};
    snprintf(pipeline_layout_lbl, sizeof(pipeline_layout_lbl),
             "%s - Pipeline layout", label);
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label = {pipeline_layout_lbl, strlen(pipeline_layout_lbl)},
      .bindGroupLayoutCount = bind_group_layout_count,
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = this->presentation_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUVertexAttribute attribute = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    WGPUVertexBufferLayout vertex_buffers[1] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute,
      },
    };

    WGPUShaderModule vs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Effect vertex shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {effect_vertex_shader_wgsl, strlen(effect_vertex_shader_wgsl)},
          },
      });

    WGPUShaderModule fs_module = wgpuDeviceCreateShaderModule(
      wgpu_context->device,
      &(WGPUShaderModuleDescriptor){
        .label = STRVIEW("Effect fragment shader"),
        .nextInChain
          = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
              .sType = WGPUSType_ShaderSourceWGSL,
            },
            .code = {fragment_shader_wgsl, strlen(fragment_shader_wgsl)},
          },
      });

    this->render_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label    = {label, strlen(label)},
        .layout   = this->pipeline_layout,
        .primitive = primitive_state,
        .vertex = (WGPUVertexState){
          .module      = vs_module,
          .entryPoint  = STRVIEW("main"),
          .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers     = vertex_buffers,
        },
        .fragment = &(WGPUFragmentState){
          .module      = fs_module,
          .entryPoint  = STRVIEW("main"),
          .targetCount = 1,
          .targets     = &color_target_state,
        },
        .multisample = (WGPUMultisampleState){
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
      });
    ASSERT(this->render_pipeline != NULL);

    WGPU_RELEASE_RESOURCE(ShaderModule, vs_module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fs_module);
  }
}

static void effect_set_bind_groups(effect_t* this,
                                   iscreen_effect_t* screen_effect)
{
  const uint32_t max_bind_groups = (uint32_t)EFFECT_MAX_BIND_GROUP_COUNT;
  for (uint32_t i = 0;
       i < screen_effect->bind_groups.item_count && i < max_bind_groups; i++) {
    this->bind_groups.items[i] = screen_effect->bind_groups.items[i];
  }
  this->bind_groups.item_count
    = MIN(screen_effect->bind_groups.item_count, max_bind_groups);
}

static void effect_create(effect_t* this, webgpu_renderer_t* renderer,
                          iscreen_effect_t* screen_effect)
{
  memset(this, 0, sizeof(*this));
  this->renderer            = renderer;
  this->presentation_format = screen_effect->presentation_format;

  effect_set_bind_groups(this, screen_effect);

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Vertex data & indices */
  const float vertex_data[2 * 4] = {
    -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
  };

  const uint16_t indices[3 * 2] = {
    3, 2, 1, 3, 1, 0,
  };

  /* Effect vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fullscreen effect - Vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertex_data),
                    .initial.data = vertex_data,
                  });

  /* Effect index buffer */
  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fullscreen effect - Index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  /* Init render pipeline */
  effect_init(this, screen_effect->fragment_shader_wgsl,
              screen_effect->bind_group_layouts.items,
              screen_effect->bind_group_layouts.item_count,
              screen_effect->label);
}

static void effect_destroy(effect_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout);
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer);
}

static void effect_pre_render(effect_t* this, WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  for (uint32_t i = 0; i < this->bind_groups.item_count; ++i) {
    wgpuRenderPassEncoderSetBindGroup(render_pass, i,
                                      this->bind_groups.items[i], 0, 0);
  }
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
}

/* -------------------------------------------------------------------------- *
 * Copy Pass Implementation
 * -------------------------------------------------------------------------- */

static bool copy_pass_is_ready(copy_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void copy_pass_init_defaults(copy_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void copy_pass_create(copy_pass_t* this, webgpu_renderer_t* renderer)
{
  copy_pass_init_defaults(this);

  this->renderer               = renderer;
  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Copy texture */
  WGPUExtent3D texture_extent = {
    .width              = renderer->screen_width,
    .height             = renderer->screen_height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Copy pass - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->copy_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->copy_texture.texture != NULL);

  /* Copy texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Copy pass - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->copy_texture.view
    = wgpuTextureCreateView(this->copy_texture.texture, &texture_view_dec);
  ASSERT(this->copy_texture.view != NULL);

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Copy pass - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = this->copy_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Copy pass - Bind group"),
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->frame_bind_group_layout,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->frame_bind_group,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_wgsl          = copy_pass_fragment_shader_wgsl,
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = WGPUTextureFormat_RGBA16Float,
      .label                         = "Copy pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Frame buffer Color attachments */
  this->framebuffer.color_attachments[0] = (WGPURenderPassColorAttachment){
    .view       = this->copy_texture.view,
    .depthSlice = ~0,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor){
      .r = 0.0f,
      .g = 0.0f,
      .b = 0.0f,
      .a = 1.0f,
    },
  };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Copy pass frame buffer"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };
}

static void copy_pass_destroy(copy_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);
  WGPU_RELEASE_RESOURCE(Texture, this->copy_texture.texture);
  WGPU_RELEASE_RESOURCE(TextureView, this->copy_texture.view);
}

static void copy_pass_render(copy_pass_t* this,
                             WGPURenderPassEncoder render_pass)
{
  if (!copy_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

static void copy_pass_recreate_textures(copy_pass_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Release old texture resources */
  WGPU_RELEASE_RESOURCE(TextureView, this->copy_texture.view);
  WGPU_RELEASE_RESOURCE(Texture, this->copy_texture.texture);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);

  /* Recreate copy texture */
  WGPUExtent3D texture_extent = {
    .width              = this->renderer->screen_width,
    .height             = this->renderer->screen_height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Copy pass - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->copy_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);

  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Copy pass - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->copy_texture.view
    = wgpuTextureCreateView(this->copy_texture.texture, &texture_view_dec);

  /* Recreate bind group */
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = this->copy_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Copy pass - Bind group"),
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });

  /* Update effect bind group reference */
  this->effect.bind_groups.items[0] = this->bind_group;

  /* Update framebuffer color attachment view */
  this->framebuffer.color_attachments[0].view = this->copy_texture.view;
}

/* -------------------------------------------------------------------------- *
 * Bloom Pass Implementation
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/bloom-pass.ts
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* bloom_pass_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var texture: texture_2d<f32>;

  struct Inputs {
    @builtin(position) coords: vec4<f32>,
  }
  struct Output {
    @location(0) color: vec4<f32>,
  }

  @fragment
  fn main(input: Inputs) -> Output {
    var output: Output;
    var albedo = textureLoad(
      texture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );
    var brightness = dot(albedo.rgb, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0) {
      output.color = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
      output.color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    return output;
  }
);

static const char* bloom_blur_compute_shader_wgsl = CODE(
  struct Params {
    filterDim : u32,
    blockDim : u32,
  }

  struct Flip {
    value : u32,
  }

  @group(0) @binding(0) var samp: sampler;
  @group(0) @binding(1) var<uniform> params: Params;
  @group(1) @binding(0) var inputTex: texture_2d<f32>;
  @group(1) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
  @group(1) @binding(2) var<uniform> flip : Flip;

  var<workgroup> tile : array<array<vec3<f32>, 128>, 4>;

  @compute @workgroup_size(32, 1, 1)
  fn main(
    @builtin(workgroup_id) WorkGroupID : vec3<u32>,
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>
  ) {
    var filterOffset : u32 = (params.filterDim - 1u) / 2u;
    var dims : vec2<i32> = vec2<i32>(textureDimensions(inputTex, 0));

    var baseIndex = vec2<i32>(
      WorkGroupID.xy * vec2<u32>(params.blockDim, 4u) +
      LocalInvocationID.xy * vec2<u32>(4u, 1u)
    ) - vec2<i32>(i32(filterOffset), 0);

    for (var r : u32 = 0u; r < 4u; r = r + 1u) {
      for (var c : u32 = 0u; c < 4u; c = c + 1u) {
        var loadIndex = baseIndex + vec2<i32>(i32(c), i32(r));
        if (flip.value != 0u) {
          loadIndex = loadIndex.yx;
        }

        tile[r][4u * LocalInvocationID.x + c] =
          textureSampleLevel(inputTex, samp,
            (vec2<f32>(loadIndex) + vec2<f32>(0.25, 0.25)) / vec2<f32>(dims), 0.0).rgb;
      }
    }

    workgroupBarrier();

    for (var r : u32 = 0u; r < 4u; r = r + 1u) {
      for (var c : u32 = 0u; c < 4u; c = c + 1u) {
        var writeIndex = baseIndex + vec2<i32>(i32(c), i32(r));
        if (flip.value != 0u) {
          writeIndex = writeIndex.yx;
        }

        var center : u32 = 4u * LocalInvocationID.x + c;
        if (center >= filterOffset &&
            center < 128u - filterOffset &&
            all(writeIndex < dims)) {
          var acc : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
          for (var f : u32 = 0u; f < params.filterDim; f = f + 1u) {
            var i : u32 = center + f - filterOffset;
            acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
          }
          textureStore(outputTex, writeIndex, vec4<f32>(acc, 1.0));
        }
      }
    }
  }
);
// clang-format on

static bool bloom_pass_is_ready(bloom_pass_t* this)
{
  return (this->effect.render_pipeline != NULL)
         && (this->blur_pipeline != NULL);
}

static void bloom_pass_init_compute_pipeline(bloom_pass_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Bloom pass blur pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->blur_constants_bind_group_layout, // Group 0
      this->blur_compute_bind_group_layout,   // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Bloom pass blur - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->blur_pipeline_layout != NULL);
  }

  /* Bloom pass blur pipeline */
  {
    WGPUShaderModuleDescriptor shader_desc = {
      .label = STRVIEW("Bloom blur - Compute shader"),
      .nextInChain
        = (const WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
          .chain = (WGPUChainedStruct){
            .sType = WGPUSType_ShaderSourceWGSL,
          },
          .code = {bloom_blur_compute_shader_wgsl, strlen(bloom_blur_compute_shader_wgsl)},
        },
    };
    WGPUShaderModule comp_shader
      = wgpuDeviceCreateShaderModule(wgpu_context->device, &shader_desc);
    ASSERT(comp_shader != NULL);

    this->blur_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Bloom pass blur - Compute pipeline"),
        .layout  = this->blur_pipeline_layout,
        .compute = (WGPUComputeState){
          .module     = comp_shader,
          .entryPoint = STRVIEW("main"),
        },
      });
    ASSERT(this->blur_pipeline != NULL);
    wgpuShaderModuleRelease(comp_shader);
  }

  /* Horizontal flip */
  const uint32_t horizontal_flip_data[1] = {
    0 //
  };
  this->buffer_0 = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Horizontal flip - Uniform buffer",
                    .usage        = WGPUBufferUsage_Uniform,
                    .size         = sizeof(horizontal_flip_data),
                    .initial.data = &horizontal_flip_data[0],
                  });

  /* Vertical flip */
  const uint32_t vertical_flip_data[1] = {
    1 //
  };
  this->buffer_1 = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Vertical flip - Uniform buffer",
                    .usage        = WGPUBufferUsage_Uniform,
                    .size         = sizeof(vertical_flip_data),
                    .initial.data = &vertical_flip_data[0],
                  });

  /* Blur compute bind group 0 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->bloom_texture.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_0.buffer,
        .size    = this->buffer_0.size,
      },
    };
    this->blur_compute_bind_groups[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 0"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[0] != NULL);
  }

  /* Blur compute bind group 1 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->blur_textures[0].view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[1].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_1.buffer,
        .size    = this->buffer_1.size,
      },
    };
    this->blur_compute_bind_groups[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 1"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[1] != NULL);
  }

  /* Blur compute bind group 2 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->blur_textures[1].view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_0.buffer,
        .size    = this->buffer_0.size,
      },
    };
    this->blur_compute_bind_groups[2] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 2"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[2] != NULL);
  }
}

static void bloom_pass_init_defaults(bloom_pass_t* this)
{
  memset(this, 0, sizeof(*this));
  this->block_dim = 0;
}

static void bloom_pass_create(bloom_pass_t* this, webgpu_renderer_t* renderer,
                              copy_pass_t* copy_pass)
{
  bloom_pass_init_defaults(this);
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Bloom texture */
  WGPUExtent3D texture_extent = {
    .width              = renderer->screen_width,
    .height             = renderer->screen_height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Bloom - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->bloom_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->bloom_texture.texture != NULL);

  /* Bloom texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Bloom - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->bloom_texture.view
    = wgpuTextureCreateView(this->bloom_texture.texture, &texture_view_dec);
  ASSERT(this->bloom_texture.view != NULL);

  /* Bloom pass bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Texture view
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Bloom pass - Bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Blur texture and blur texture views */
  for (uint8_t i = 0; i < 2; ++i) {
    /* Blur texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->screen_width,
      .height             = renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Blur - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding
               | WGPUTextureUsage_TextureBinding,
    };
    this->blur_textures[i].texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->blur_textures[i].texture != NULL);

    /* Blur texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Blur - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->blur_textures[i].view = wgpuTextureCreateView(
      this->blur_textures[i].texture, &texture_view_dec);
    ASSERT(this->blur_textures[i].view != NULL);
  }

  /* G-buffer bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = copy_pass->copy_texture.view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Gbuffer - Bind group"),
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->frame_bind_group_layout,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->frame_bind_group,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_wgsl          = bloom_pass_fragment_shader_wgsl,
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = WGPUTextureFormat_RGBA16Float,
      .label                         = "Bloom - Pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Input texture */
  this->input_texture.texture = copy_pass->copy_texture.texture;
  this->input_texture.view    = copy_pass->copy_texture.view;

  /* Frame buffer descriptor */
  this->framebuffer.color_attachments[0] =
    (WGPURenderPassColorAttachment) {
      .view       = this->bloom_texture.view,
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

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };

  /* Blur params buffer */
  this->block_dim = BLOOM_PASS_TILE_DIM - (BLOOM_PASS_FILTER_SIZE - 1);
  const uint32_t blur_params[2] = {BLOOM_PASS_FILTER_SIZE, this->block_dim};
  this->blur_params_buffer      = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                         .label = "Blur params - Uniform buffer",
                         .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                         .size  = sizeof(blur_params),
                         .initial.data = &blur_params[0],
                  });

  /* Bloom sampler */
  this->sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Bloom - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(this->sampler != NULL);

  /* Blur constants bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->blur_params_buffer.size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Blur constants - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->blur_constants_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->blur_constants_bind_group_layout != NULL);
  }

  /* Blur constants bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = this->sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->blur_params_buffer.buffer,
        .size    = this->blur_params_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = STRVIEW("Blur constants - Bind group"),
      .layout     = this->blur_constants_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->blur_compute_constants_bindGroup
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->blur_compute_constants_bindGroup != NULL);
  }

  /* Blur compute bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = (WGPUStorageTextureBindingLayout) {
           .access        = WGPUStorageTextureAccess_WriteOnly,
           .format        = WGPUTextureFormat_RGBA8Unorm,
           .viewDimension = WGPUTextureViewDimension_2D,
         },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(float),
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Blur compute - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->blur_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->blur_compute_bind_group_layout != NULL);
  }

  /* Init compute pipeline */
  bloom_pass_init_compute_pipeline(this);
}

static void bloom_pass_destroy(bloom_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(Texture, this->bloom_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->bloom_texture.view)
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(this->blur_textures); ++i) {
    WGPU_RELEASE_RESOURCE(Texture, this->blur_textures[i].texture)
    WGPU_RELEASE_RESOURCE(TextureView, this->blur_textures[i].view)
  }
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->blur_pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->blur_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->blur_constants_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_constants_bindGroup)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->blur_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[2])
  WGPU_RELEASE_RESOURCE(Buffer, this->blur_params_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer_0.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer_1.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, this->sampler)
}

static void bloom_pass_update_bloom(bloom_pass_t* this,
                                    WGPUComputePassEncoder compute_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  const webgpu_renderer_t* renderer = this->renderer;
  const uint32_t block_dim          = this->block_dim;
  const uint32_t batch[2]           = BLOOM_PASS_BATCH;
  const uint32_t src_width          = renderer->screen_width;
  const uint32_t src_height         = renderer->screen_height;

  wgpuComputePassEncoderSetPipeline(compute_pass, this->blur_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->blur_compute_constants_bindGroup, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->blur_compute_bind_groups[0], 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_width / (float)block_dim), /* workgroupCountX */
    (uint32_t)ceil(src_height / (float)batch[1]), /* workgroupCountY */
    1                                             /* workgroupCountZ */
  );
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->blur_compute_bind_groups[1], 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_height / (float)block_dim), /* workgroupCountX */
    (uint32_t)ceil(src_width / (float)batch[1]),   /* workgroupCountY */
    1                                              /* workgroupCountZ */
  );
  for (uint32_t i = 0; i < BLOOM_PASS_ITERATIONS - 1; ++i) {
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_groups[2], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_width / (float)block_dim), /* workgroupCountX */
      (uint32_t)ceil(src_height / (float)batch[1]), /* workgroupCountY */
      1                                             /* workgroupCountZ */
    );
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_height / (float)block_dim), /* workgroupCountX */
      (uint32_t)ceil(src_width / (float)batch[1]),   /* workgroupCountY */
      1                                              /* workgroupCountZ */
    );
  }
}

static void bloom_pass_render(bloom_pass_t* this,
                              WGPURenderPassEncoder render_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

static void bloom_pass_recreate_textures(bloom_pass_t* this,
                                         copy_pass_t* copy_pass)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Release old texture resources */
  WGPU_RELEASE_RESOURCE(TextureView, this->bloom_texture.view);
  WGPU_RELEASE_RESOURCE(Texture, this->bloom_texture.texture);
  for (uint8_t i = 0; i < 2; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, this->blur_textures[i].view);
    WGPU_RELEASE_RESOURCE(Texture, this->blur_textures[i].texture);
  }
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[0]);
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[1]);
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[2]);

  /* Recreate bloom texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = this->renderer->screen_width,
      .height             = this->renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Bloom - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->bloom_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    WGPUTextureViewDescriptor view_desc = {
      .label           = STRVIEW("Bloom - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->bloom_texture.view
      = wgpuTextureCreateView(this->bloom_texture.texture, &view_desc);
  }

  /* Recreate blur textures */
  for (uint8_t i = 0; i < 2; ++i) {
    WGPUExtent3D texture_extent = {
      .width              = this->renderer->screen_width,
      .height             = this->renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Blur - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding
               | WGPUTextureUsage_TextureBinding,
    };
    this->blur_textures[i].texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    WGPUTextureViewDescriptor view_desc = {
      .label           = STRVIEW("Blur - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->blur_textures[i].view
      = wgpuTextureCreateView(this->blur_textures[i].texture, &view_desc);
  }

  /* Recreate bind group (references copy pass texture) */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding     = 0,
        .textureView = copy_pass->copy_texture.view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Gbuffer - Bind group"),
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
  }

  /* Update effect bind group reference */
  this->effect.bind_groups.items[0] = this->bind_group;

  /* Update input texture reference */
  this->input_texture.texture = copy_pass->copy_texture.texture;
  this->input_texture.view    = copy_pass->copy_texture.view;

  /* Update framebuffer color attachment view */
  this->framebuffer.color_attachments[0].view = this->bloom_texture.view;

  /* Recreate blur compute bind groups */
  {
    WGPUBindGroupEntry bg0[3] = {
      [0] = {.binding = 0, .textureView = this->bloom_texture.view},
      [1] = {.binding = 1, .textureView = this->blur_textures[0].view},
      [2] = {.binding = 2,
             .buffer  = this->buffer_0.buffer,
             .size    = this->buffer_0.size},
    };
    this->blur_compute_bind_groups[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 0"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = 3,
                              .entries    = bg0,
                            });
  }
  {
    WGPUBindGroupEntry bg1[3] = {
      [0] = {.binding = 0, .textureView = this->blur_textures[0].view},
      [1] = {.binding = 1, .textureView = this->blur_textures[1].view},
      [2] = {.binding = 2,
             .buffer  = this->buffer_1.buffer,
             .size    = this->buffer_1.size},
    };
    this->blur_compute_bind_groups[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 1"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = 3,
                              .entries    = bg1,
                            });
  }
  {
    WGPUBindGroupEntry bg2[3] = {
      [0] = {.binding = 0, .textureView = this->blur_textures[1].view},
      [1] = {.binding = 1, .textureView = this->blur_textures[0].view},
      [2] = {.binding = 2,
             .buffer  = this->buffer_0.buffer,
             .size    = this->buffer_0.size},
    };
    this->blur_compute_bind_groups[2] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blur compute - Bind group 2"),
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = 3,
                              .entries    = bg2,
                            });
  }
}

/* -------------------------------------------------------------------------- *
 * Deferred Pass Implementation
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/deferred-pass.ts
 * -------------------------------------------------------------------------- */

// clang-format off
/* Part 1: Struct definitions, utility functions, and bindings */
static const char* deferred_pass_fragment_shader_part1 = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  struct InputPointLight {
    position: vec4<f32>,
    velocity: vec4<f32>,
    color: vec3<f32>,
    range: f32,
    intensity: f32,
  }

  struct LightsBuffer {
    lights: array<InputPointLight>,
  }

  struct LightsConfig {
    numLights: u32,
  }

  struct PointLight {
    pointToLight: vec3<f32>,
    color: vec3<f32>,
    range: f32,
    intensity: f32,
  }

  struct DirectionalLight {
    direction: vec3<f32>,
    color: vec3<f32>,
  }

  struct SpotLight {
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    cutOff: f32,
    outerCutOff: f32,
    intensity: f32,
  }

  struct Surface {
    albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    worldPos: vec4<f32>,
    ID: f32,
    N: vec3<f32>,
    F0: vec3<f32>,
    V: vec3<f32>,
  }

  fn decodeNormals(enc: vec2<f32>) -> vec3<f32> {
    var fenc = enc * 4.0 - 2.0;
    var f = dot(fenc, fenc);
    var g = sqrt(1.0 - f / 4.0);
    return vec3(fenc*g, 1.0 - f / 2.0);
  }

  fn reconstructWorldPosFromZ(
    coords: vec2<f32>,
    size: vec2<f32>,
    depthTexture: texture_depth_2d,
    projInverse: mat4x4<f32>,
    viewInverse: mat4x4<f32>
  ) -> vec4<f32> {
    var uv = coords.xy / projection.outputSize;
    var depth = textureLoad(depthTexture, vec2<i32>(floor(coords)), 0);
    var x = uv.x * 2.0 - 1.0;
    var y = (1.0 - uv.y) * 2.0 - 1.0;
    var projectedPos = vec4(x, y, depth, 1.0);
    var worldPosition = projInverse * projectedPos;
    worldPosition = vec4(worldPosition.xyz / worldPosition.w, 1.0);
    worldPosition = viewInverse * worldPosition;
    return worldPosition;
  }

  @group(0) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;
  @group(0) @binding(1) var<uniform> lightsConfig: LightsConfig;
  @group(0) @binding(2) var normalTexture: texture_2d<f32>;
  @group(0) @binding(3) var diffuseTexture: texture_2d<f32>;
  @group(0) @binding(4) var depthTexture: texture_depth_2d;

  @group(1) @binding(0) var<uniform> projection: ProjectionUniformsStruct;
  @group(1) @binding(1) var<uniform> view: ViewUniformsStruct;
  @group(1) @binding(2) var depthSampler: sampler;

  @group(2) @binding(0) var<uniform> spotLight: SpotLight;
  @group(2) @binding(1) var<uniform> spotLightProjection: ProjectionUniformsStruct;
  @group(2) @binding(2) var<uniform> spotLightView: ViewUniformsStruct;

  @group(3) @binding(0) var spotLightDepthTexture: texture_depth_2d;

  struct Inputs {
    @builtin(position) coords: vec4<f32>,
  }
  struct Output {
    @location(0) color: vec4<f32>,
  }

  const PI = 3.141592653589793;
  const LOG2 = 1.4426950408889634;

  fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    var a      = roughness*roughness;
    var a2     = a*a;
    var NdotH  = max(dot(N, H), 0.0);
    var NdotH2 = NdotH*NdotH;

    var num   = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
  }

  fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    var r = (roughness + 1.0);
    var k = (r*r) / 8.0;

    var num   = NdotV;
    var denom = NdotV * (1.0 - k) + k;

    return num / denom;
  }

  fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    var NdotV = max(dot(N, V), 0.0);
    var NdotL = max(dot(N, L), 0.0);
    var ggx2  = GeometrySchlickGGX(NdotV, roughness);
    var ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
  }

  fn FresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
  }

  fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (1.0 + x);
  }

  fn rangeAttenuation(range : f32, distance : f32) -> f32 {
    if (range <= 0.0) {
        return 1.0 / pow(distance, 2.0);
    }
    return clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0) / pow(distance, 2.0);
  }
  );
  /* Part 2: PBR light radiance functions */
  static const char* deferred_pass_fragment_shader_part2 = CODE(
  fn PointLightRadiance(light : PointLight, surface : Surface) -> vec3<f32> {
    var L = normalize(light.pointToLight);
    var H = normalize(surface.V + L);
    var distance = length(light.pointToLight);

    var NDF = DistributionGGX(surface.N, H, surface.roughness);
    var G = GeometrySmith(surface.N, surface.V, L, surface.roughness);
    var F = FresnelSchlick(max(dot(H, surface.V), 0.0), surface.F0);

    var kD = (vec3(1.0, 1.0, 1.0) - F) * (1.0 - surface.metallic);

    var NdotL = max(dot(surface.N, L), 0.0);

    var numerator = NDF * G * F;
    var denominator = max(4.0 * max(dot(surface.N, surface.V), 0.0) * NdotL, 0.001);
    var specular = numerator / vec3(denominator, denominator, denominator);

    var attenuation = rangeAttenuation(light.range, distance);
    var radiance = light.color * light.intensity * attenuation;
    return (kD * surface.albedo.rgb / vec3(PI, PI, PI) + specular) * radiance * NdotL;
  }

  fn SpotLightRadiance(light: SpotLight, surface: Surface) -> vec3<f32> {
    var L = normalize(light.position - surface.worldPos.xyz);
    var H = normalize(surface.V + L);

    var theta = dot(L, normalize(light.direction));
    var attenuation = smoothstep(light.outerCutOff, light.cutOff, theta);

    var NDF = DistributionGGX(surface.N, H, surface.roughness);
    var G = GeometrySmith(surface.N, surface.V, L, surface.roughness);
    var F = FresnelSchlick(max(dot(H, surface.V), 0.0), surface.F0);

    var kD = (vec3(1.0, 1.0, 1.0) - F) * (1.0 - surface.metallic);

    var NdotL = max(dot(surface.N, L), 0.0);

    var numerator = NDF * G * F;
    var denominator = max(4.0 * max(dot(surface.N, surface.V), 0.0) * NdotL, 0.001);
    var specular = numerator / denominator;

    var radiance = light.color * light.intensity * attenuation;

    return (kD * surface.albedo.rgb / vec3(PI, PI, PI) + specular) * radiance * NdotL;
  }

  fn DirectionalLightRadiance(light: DirectionalLight, surface : Surface) -> vec3<f32> {
    var L = normalize(light.direction);
    var H = normalize(surface.V + L);

    var NDF = DistributionGGX(surface.N, H, surface.roughness);
    var G = GeometrySmith(surface.N, surface.V, L, surface.roughness);
    var F = FresnelSchlick(max(dot(H, surface.V), 0.0), surface.F0);

    var kD = (vec3(1.0, 1.0, 1.0) - F) * (1.0 - surface.metallic);

    var NdotL = max(dot(surface.N, L), 0.0);

    var numerator = NDF * G * F;
    var denominator = max(4.0 * max(dot(surface.N, surface.V), 0.0) * NdotL, 0.001);
    var specular = numerator / vec3(denominator, denominator, denominator);

    var radiance = light.color;
    return (kD * surface.albedo.rgb / vec3(PI, PI, PI) + specular) * radiance * NdotL;
  }

  const GAMMA = 2.2;
  fn linearTosRGB(linear: vec3<f32>) -> vec3<f32> {
    var INV_GAMMA = 1.0 / GAMMA;
    return pow(linear, vec3<f32>(INV_GAMMA, INV_GAMMA, INV_GAMMA));
  }

  fn LinearizeDepth(depth: f32) -> f32 {
    var z = depth * 2.0 - 1.0;
    var near_plane = 0.001;
    var far_plane = 0.4;
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
  }
  );
  /* Part 3: Fragment shader entry point */
  static const char* deferred_pass_fragment_shader_part3 = CODE(
  @fragment
  fn main(input: Inputs) -> Output {
    var worldPosition = reconstructWorldPosFromZ(
      input.coords.xy,
      projection.outputSize,
      depthTexture,
      projection.inverseMatrix,
      view.inverseMatrix
    );

    var normalRoughnessMatID = textureLoad(
      normalTexture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );

    var albedo = textureLoad(
      diffuseTexture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );

    var surface: Surface;
    surface.ID = normalRoughnessMatID.w;

    var output: Output;

    var posFromLight = spotLightProjection.matrix * spotLightView.matrix * vec4(worldPosition.xyz, 1.0);
    posFromLight = vec4(posFromLight.xyz / posFromLight.w, 1.0);
    var shadowPos = vec3(
      posFromLight.xy * vec2(0.5,-0.5) + vec2(0.5, 0.5),
      posFromLight.z
    );

    var shadowMapSize = vec2<f32>(textureDimensions(spotLightDepthTexture, 0));
    var projectedDepth = textureLoad(spotLightDepthTexture, vec2<i32>(shadowPos.xy * shadowMapSize), 0);

    if (surface.ID == 0.0) {
      var inRange =
        shadowPos.x >= 0.0 &&
        shadowPos.x <= 1.0 &&
        shadowPos.y >= 0.0 &&
        shadowPos.y <= 1.0;
      var visibility = 1.0;
      if (inRange && projectedDepth <= posFromLight.z - 0.000009) {
        visibility = 0.0;
      }

      surface.albedo = albedo;
      surface.metallic = normalRoughnessMatID.z;
      surface.roughness = albedo.a;
      surface.worldPos = worldPosition;
      surface.N = decodeNormals(normalRoughnessMatID.xy);
      surface.F0 = mix(vec3(0.04), surface.albedo.rgb, vec3(surface.metallic));
      surface.V = normalize(view.position - worldPosition.xyz);

      var Lo = vec3(0.0);

      for (var i : u32 = 0u; i < lightsConfig.numLights; i = i + 1u) {
          var light = lightsBuffer.lights[i];
        var pointLight: PointLight;

        if (distance(light.position.xyz, worldPosition.xyz) > light.range) {
          continue;
        }

        pointLight.pointToLight = light.position.xyz - worldPosition.xyz;
        pointLight.color = light.color;
        pointLight.range = light.range;
        pointLight.intensity = light.intensity;
        Lo += PointLightRadiance(pointLight, surface);
      }

      var dirLight: DirectionalLight;
      dirLight.direction = vec3(2.0, 20.0, 0.0);
      dirLight.color = vec3(0.1);
      Lo += DirectionalLightRadiance(dirLight, surface) * visibility;

      Lo += SpotLightRadiance(spotLight, surface) * visibility;

      var ambient = vec3(0.09) * albedo.rgb;
      var color = ambient + Lo;
      output.color = vec4(color.rgb, 1.0);

      var fogDensity = 0.085;
      var fogDistance = length(worldPosition.xyz);
      var fogAmount = 1.0 - exp2(-fogDensity * fogDensity * fogDistance * fogDistance * LOG2);
      fogAmount = clamp(fogAmount, 0.0, 1.0);
      var fogColor = vec4(vec3(0.005), 1.0);
      output.color = mix(output.color, fogColor, fogAmount);


    } else if (0.1 - surface.ID < 0.01 && surface.ID < 0.1) {
      output.color = vec4(albedo.rgb, 1.0);
    } else {
      output.color = vec4(vec3(0.005), 1.0);
    }
    return output;
  }
);
// clang-format on

/* Combine deferred pass fragment shader chunks into one buffer */
#define DEFERRED_PASS_SHADER_SIZE (16 * 1024)
static const char* deferred_pass_create_fragment_shader(void)
{
  static char shader_source[DEFERRED_PASS_SHADER_SIZE];
  snprintf(shader_source, sizeof(shader_source), "%s%s%s",
           deferred_pass_fragment_shader_part1,
           deferred_pass_fragment_shader_part2,
           deferred_pass_fragment_shader_part3);
  return shader_source;
}

static bool deferred_pass_is_ready(deferred_pass_t* this)
{
  return point_lights_is_ready(&this->point_lights)
         && (this->effect.render_pipeline != NULL);
}

static void deferred_pass_init_defaults(deferred_pass_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_copy((vec3){0.0f, 80.0f, 0.0f}, this->spot_light_target);
  glm_vec3_one(this->spot_light_color_target);
}

static void deferred_pass_create(deferred_pass_t* this,
                                 webgpu_renderer_t* renderer)
{
  deferred_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Point light */
  point_lights_create(&this->point_lights, renderer);

  /* Spot light */
  ispot_light_t ispot_light = {
    .position      = {0.0f, 80.0f, 1.0f},
    .direction     = {0.0f, 1.0f, 0.0f},
    .color         = GLM_VEC3_ONE_INIT,
    .cut_off       = deg_to_rad(1.0f),
    .outer_cut_off = deg_to_rad(4.0f),
    .intensity     = 40.0f,
  };
  spot_light_create(&this->spot_light, renderer, &ispot_light);

  /* G-Buffer normal texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = renderer->screen_width,
      .height             = renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Gbuffer normal - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_normal.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_normal.texture != NULL);

    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Gbuffer normal - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_normal.view = wgpuTextureCreateView(
      this->g_buffer_texture_normal.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_normal.view != NULL);
  }

  /* G-Buffer diffuse texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = renderer->screen_width,
      .height             = renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Gbuffer diffuse - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_diffuse.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_diffuse.texture != NULL);

    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Gbuffer diffuse - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_diffuse.view = wgpuTextureCreateView(
      this->g_buffer_texture_diffuse.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_diffuse.view != NULL);
  }

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[5] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = this->point_lights.lights_buffer.size,
       },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = this->point_lights.lights_config_uniform_buffer.size,
       },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Depth,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Gbuffer - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[5] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = this->point_lights.lights_buffer.buffer,
      .size    = this->point_lights.lights_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .buffer  = this->point_lights.lights_config_uniform_buffer.buffer,
      .size    = this->point_lights.lights_config_uniform_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = this->g_buffer_texture_normal.view,
    },
    [3] = (WGPUBindGroupEntry) {
      .binding     = 3,
      .textureView = this->g_buffer_texture_diffuse.view,
    },
    [4] = (WGPUBindGroupEntry) {
      .binding     = 4,
      .textureView = renderer->depth_texture_view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Gbuffer - Bind group"),
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[4] = {
      this->bind_group_layout,
      this->renderer->frame_bind_group_layout,
      this->spot_light.bind_group_layouts.ubos,
      this->spot_light.bind_group_layouts.depth_texture,
    };
    WGPUBindGroup bind_groups[4] = {
      this->bind_group,
      this->renderer->frame_bind_group,
      this->spot_light.bind_groups.ubos,
      this->spot_light.bind_groups.depth_texture,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_wgsl          = deferred_pass_create_fragment_shader(),
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format = settings_get_quality_level().bloom_toggle ?
                               WGPUTextureFormat_RGBA16Float :
                               WGPUTextureFormat_BGRA8Unorm,
      .label               = "Deferred - Pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Frame buffer Color attachments */
  {
    this->framebuffer.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_normal.view,
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
  }
  {
    this->framebuffer.color_attachments[1] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_diffuse.view,
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
  }

  /* Frame buffer depth stencil attachment */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = renderer->depth_texture_view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 2,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
  };
}

static void deferred_pass_destroy(deferred_pass_t* this)
{
  effect_destroy(&this->effect);
  point_lights_destroy(&this->point_lights);
  spot_light_destroy(&this->spot_light);

  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_normal.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_normal.view)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_diffuse.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_diffuse.view)
}

static void deferred_pass_recreate_textures(deferred_pass_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;
  webgpu_renderer_t* renderer  = this->renderer;

  /* Release old texture resources */
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_normal.view);
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_normal.texture);
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_diffuse.view);
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_diffuse.texture);
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);

  /* Recreate G-Buffer normal texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = renderer->screen_width,
      .height             = renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Gbuffer normal - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_normal.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    WGPUTextureViewDescriptor view_desc = {
      .label           = STRVIEW("Gbuffer normal - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_normal.view = wgpuTextureCreateView(
      this->g_buffer_texture_normal.texture, &view_desc);
  }

  /* Recreate G-Buffer diffuse texture */
  {
    WGPUExtent3D texture_extent = {
      .width              = renderer->screen_width,
      .height             = renderer->screen_height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Gbuffer diffuse - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_diffuse.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    WGPUTextureViewDescriptor view_desc = {
      .label           = STRVIEW("Gbuffer diffuse - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_diffuse.view = wgpuTextureCreateView(
      this->g_buffer_texture_diffuse.texture, &view_desc);
  }

  /* Recreate bind group (with new texture views + depth texture view) */
  {
    WGPUBindGroupEntry bg_entries[5] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = this->point_lights.lights_buffer.buffer,
        .size    = this->point_lights.lights_buffer.size,
      },
      [1] = (WGPUBindGroupEntry){
        .binding = 1,
        .buffer  = this->point_lights.lights_config_uniform_buffer.buffer,
        .size    = this->point_lights.lights_config_uniform_buffer.size,
      },
      [2] = (WGPUBindGroupEntry){
        .binding     = 2,
        .textureView = this->g_buffer_texture_normal.view,
      },
      [3] = (WGPUBindGroupEntry){
        .binding     = 3,
        .textureView = this->g_buffer_texture_diffuse.view,
      },
      [4] = (WGPUBindGroupEntry){
        .binding     = 4,
        .textureView = renderer->depth_texture_view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Gbuffer - Bind group"),
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
  }

  /* Update effect bind group reference */
  this->effect.bind_groups.items[0] = this->bind_group;

  /* Update framebuffer color attachment views */
  this->framebuffer.color_attachments[0].view
    = this->g_buffer_texture_normal.view;
  this->framebuffer.color_attachments[1].view
    = this->g_buffer_texture_diffuse.view;

  /* Update framebuffer depth stencil attachment */
  this->framebuffer.depth_stencil_attachment.view
    = renderer->depth_texture_view;
}

static void deferred_pass_rearrange(deferred_pass_t* this)
{
  this->spot_light_target[0]       = (random_float() * 2 - 1) * 3;
  this->spot_light_target[2]       = (random_float() * 2 - 1) * 3;
  this->spot_light_color_target[0] = random_float();
  this->spot_light_color_target[1] = random_float();
  this->spot_light_color_target[2] = random_float();
}

static void deferred_pass_update_lights_sim(deferred_pass_t* this,
                                            WGPUComputePassEncoder compute_pass,
                                            float _time, float time_delta)
{
  UNUSED_VAR(_time);

  point_lights_update_sim(&this->point_lights, compute_pass);
  const float speed = time_delta * 2.0f;

  /* Update spot light position - must use setter to update UBO */
  vec3 new_position = {
    this->spot_light._position[0]
      + (this->spot_light_target[0] - this->spot_light._position[0]) * speed,
    this->spot_light._position[1]
      + (this->spot_light_target[1] - this->spot_light._position[1]) * speed,
    this->spot_light._position[2]
      + (this->spot_light_target[2] - this->spot_light._position[2]) * speed,
  };
  spot_light_set_position(&this->spot_light, new_position);

  /* Update spot light color - must use setter to update UBO */
  vec3 new_color = {
    (this->spot_light_color_target[0] - this->spot_light._color[0]) * speed * 4,
    (this->spot_light_color_target[1] - this->spot_light._color[1]) * speed * 4,
    (this->spot_light_color_target[2] - this->spot_light._color[2]) * speed * 4,
  };
  spot_light_set_color(&this->spot_light, new_color);
}

static void deferred_pass_render(deferred_pass_t* this,
                                 WGPURenderPassEncoder render_pass)
{
  if (!deferred_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Result Pass Implementation
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/result-pass.ts
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* result_pass_fragment_shader_wgsl = CODE(
  const GAMMA = 2.2;
  fn linearTosRGB(linear: vec3<f32>) -> vec3<f32> {
    var INV_GAMMA = 1.0 / GAMMA;
    return pow(linear, vec3<f32>(INV_GAMMA, INV_GAMMA, INV_GAMMA));
  }

  @group(0) @binding(0) var copyTexture: texture_2d<f32>;
  @group(0) @binding(1) var bloomTexture: texture_2d<f32>;

  struct Inputs {
    @builtin(position) coords: vec4<f32>,
  }
  struct Output {
    @location(0) color: vec4<f32>,
  }

  @fragment
  fn main(input: Inputs) -> Output {
    var output: Output;
    var hdrColor = textureLoad(
      copyTexture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );
    var bloomColor = textureLoad(
      bloomTexture,
      vec2<i32>(floor(input.coords.xy)),
      0
    );

    hdrColor += bloomColor;

    var result = vec3(1.0) - exp(-hdrColor.rgb * 1.0);

    output.color = vec4(result, 1.0);
    return output;
  }
);
// clang-format on

static bool result_pass_is_ready(result_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void result_pass_init_defaults(result_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void result_pass_create(result_pass_t* this, webgpu_renderer_t* renderer,
                               copy_pass_t* copy_pass, bloom_pass_t* bloom_pass)
{
  result_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Result pass - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Empty texture (1x1 placeholder) */
  WGPUExtent3D texture_extent = {
    .width              = 1,
    .height             = 1,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = STRVIEW("Empty - Texture"),
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_BGRA8Unorm,
    .usage         = WGPUTextureUsage_TextureBinding,
  };
  this->empty_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->empty_texture.texture != NULL);

  /* Empty texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = STRVIEW("Empty - Texture view"),
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->empty_texture.view
    = wgpuTextureCreateView(this->empty_texture.texture, &texture_view_dec);
  ASSERT(this->empty_texture.view != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = copy_pass->copy_texture.view,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = bloom_pass ? bloom_pass->blur_textures[1].view
                                  : this->empty_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Result pass - Bind group"),
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->frame_bind_group_layout,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->frame_bind_group,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_wgsl          = result_pass_fragment_shader_wgsl,
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = wgpu_context->render_format,
      .label                         = "Result - Pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }
}

static void result_pass_destroy(result_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->empty_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->empty_texture.view)
}

static void result_pass_render(result_pass_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!result_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->frame_bind_group, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

static void result_pass_recreate_bind_group(result_pass_t* this,
                                            copy_pass_t* copy_pass,
                                            bloom_pass_t* bloom_pass)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Release old bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group);

  /* Recreate bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = copy_pass->copy_texture.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding     = 1,
      .textureView = bloom_pass ? bloom_pass->blur_textures[1].view
                                : this->empty_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Result pass - Bind group"),
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });

  /* Update effect bind group reference */
  this->effect.bind_groups.items[0] = this->bind_group;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  /* Forward to imgui */
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Check if ImGui wants to capture the input */
  ImGuiIO* io            = igGetIO();
  bool imgui_wants_mouse = io->WantCaptureMouse;

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Update camera aspect ratio */
    uint32_t width  = (uint32_t)wgpu_context->width;
    uint32_t height = (uint32_t)wgpu_context->height;
    if (width > 0 && height > 0) {
      state.main_camera.aspect = (float)width / (float)height;
      perspective_camera_update_projection_matrix(&state.main_camera);

      /* Update screen effect settings */
      state.screen_effect_settings.screen_width  = (float)width;
      state.screen_effect_settings.screen_height = (float)height;

      /* Recreate depth texture (also updates renderer screen_width/height) */
      recreate_depth_texture(wgpu_context);

      /* Recreate deferred pass GBuffer textures + bind group */
      deferred_pass_recreate_textures(&state.deferred_pass);

      /* Recreate copy pass texture + bind group */
      copy_pass_recreate_textures(&state.copy_pass);

      /* Recreate bloom pass textures + bind groups (if enabled) */
      if (settings_get_quality_level().bloom_toggle) {
        bloom_pass_recreate_textures(&state.bloom_pass, &state.copy_pass);
      }

      /* Recreate result pass bind group (references copy + bloom textures) */
      result_pass_recreate_bind_group(
        &state.result_pass, &state.copy_pass,
        settings_get_quality_level().bloom_toggle ? &state.bloom_pass : NULL);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN
           && !imgui_wants_mouse) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      camera_controller_on_mouse_down(&state.camera_controller,
                                      (float)input_event->mouse_x,
                                      (float)input_event->mouse_y);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP
           && !imgui_wants_mouse) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      camera_controller_on_mouse_up(&state.camera_controller);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && !imgui_wants_mouse) {
    camera_controller_on_mouse_move(&state.camera_controller,
                                    (float)input_event->mouse_x,
                                    (float)input_event->mouse_y);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_SCROLL
           && !imgui_wants_mouse) {
    camera_controller_on_wheel(&state.camera_controller,
                               input_event->scroll_y * 50.0f);
  }
}

/* -------------------------------------------------------------------------- *
 * GUI Rendering
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context, float delta_time)
{
  imgui_overlay_new_frame(wgpu_context, delta_time);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 180.0f}, ImGuiCond_FirstUseEver);

  igBegin("Compute Metaballs Settings", NULL, 0);

  /* Point lights count slider */
  if (igSliderInt("Point Lights", &state.gui_settings.point_lights_count, 1,
                  MAX_POINT_LIGHTS_COUNT, "%d")) {
    point_lights_set_lights_count(
      &state.deferred_pass.point_lights,
      (uint32_t)state.gui_settings.point_lights_count);
  }

  /* Bloom threshold slider */
  igSliderFloat("Bloom Threshold", &state.gui_settings.bloom_threshold, 0.0f,
                1.0f, "%.2f", 1.0f);
  state.screen_effect_settings.screen_effect_threshold
    = state.gui_settings.bloom_threshold;

  /* Enable bloom checkbox */
  igCheckbox("Enable Bloom", &state.gui_settings.enable_bloom);
  state.screen_effect_settings.enable_screen_effect
    = state.gui_settings.enable_bloom ? 1.0f : 0.0f;

  /* Iso level slider */
  if (igSliderFloat("Iso Level", &state.gui_settings.iso_level, 5.0f, 50.0f,
                    "%.1f", 1.0f)) {
    state.volume_settings.iso_level = state.gui_settings.iso_level;
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Main Callbacks
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  state.wgpu_context = wgpu_context;

  /* Initialize sokol_time */
  stm_setup();

  /* Initialize random seed */
  srand((unsigned int)time(NULL));

  /* Initialize components */
  init_cameras();
  init_renderer(wgpu_context);
  init_metaballs_compute(wgpu_context);
  init_metaballs(wgpu_context);

  /* Initialize deferred pass (creates spot_light and point_lights internally)
   */
  deferred_pass_create(&state.deferred_pass, &state.renderer);

  /* Initialize copy pass */
  copy_pass_create(&state.copy_pass, &state.renderer);

  /* Initialize bloom pass (if enabled) */
  if (settings_get_quality_level().bloom_toggle) {
    bloom_pass_create(&state.bloom_pass, &state.renderer, &state.copy_pass);
  }

  /* Initialize result pass */
  result_pass_create(
    &state.result_pass, &state.renderer, &state.copy_pass,
    settings_get_quality_level().bloom_toggle ? &state.bloom_pass : NULL);

  /* Initialize scene objects - these need the spot light from deferred_pass */
  box_outline_create(&state.box_outline, &state.renderer);
  ground_create(&state.ground, &state.renderer,
                &state.deferred_pass.spot_light);
  particles_create(&state.particles, &state.renderer,
                   &state.deferred_pass.point_lights.lights_buffer);

  /* Initialize metaballs shadow pipeline */
  init_metaballs_shadow(wgpu_context, &state.deferred_pass.spot_light);

  /* Initialize imgui */
  imgui_overlay_init(wgpu_context);

  state.prepared = true;

  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.prepared) {
    return EXIT_SUCCESS;
  }

  /* Update time */
  static uint64_t start_time = 0;
  if (start_time == 0) {
    start_time = stm_now();
  }
  float current_time    = (float)stm_sec(stm_since(start_time));
  state.delta_time      = current_time - state.last_frame_time;
  state.last_frame_time = current_time;

  /* Periodically rearrange geometry */
  if (state.rearrange_countdown < 0.0f) {
    deferred_pass_rearrange(&state.deferred_pass);
    metaballs_rearrange(&state.metaballs);
    state.rearrange_countdown = 5.0f;
  }
  state.rearrange_countdown -= state.delta_time;

  /* Update camera */
  camera_controller_update(&state.camera_controller, state.delta_time);
  update_uniform_buffers(wgpu_context);

  /* Update renderer framebuffer view to current swapchain texture */
  state.renderer.framebuffer.color_attachments[0].view
    = wgpu_context->swapchain_view;

  /* Update metaballs simulation */
  update_metaballs_sim(state.delta_time);

  /* Render GUI */
  render_gui(wgpu_context, state.delta_time);

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Main command encoder"),
                          });

  /* Compute pass - update metaballs field, run marching cubes, and update
   * lights */
  {
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(
      cmd_encoder, &(WGPUComputePassDescriptor){
                     .label = STRVIEW("Metaballs compute pass"),
                   });

    /* Update metaballs if enabled */
    if (settings_get_quality_level().update_metaballs) {
      dispatch_metaballs_compute(compute_pass);
    }
    else if (!state.metaballs_compute.has_calced_once) {
      /* Compute at least once */
      dispatch_metaballs_compute(compute_pass);
    }

    /* Update point lights simulation */
    deferred_pass_update_lights_sim(&state.deferred_pass, compute_pass,
                                    state.last_frame_time, state.delta_time);

    /* Update bloom if enabled */
    if (settings_get_quality_level().bloom_toggle) {
      bloom_pass_update_bloom(&state.bloom_pass, compute_pass);
    }

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
  }

  /* Shadow pass - render scene from spot light POV */
  {
    WGPURenderPassEncoder spot_light_shadow_pass
      = wgpuCommandEncoderBeginRenderPass(
        cmd_encoder, &state.deferred_pass.spot_light.framebuffer.descriptor);
    render_metaballs_shadow(spot_light_shadow_pass,
                            &state.deferred_pass.spot_light);
    ground_render_shadow(&state.ground, spot_light_shadow_pass);
    wgpuRenderPassEncoderEnd(spot_light_shadow_pass);
    wgpuRenderPassEncoderRelease(spot_light_shadow_pass);
  }

  /* G-buffer pass - render scene to G-buffer textures */
  {
    WGPURenderPassEncoder g_buffer_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_encoder, &state.deferred_pass.framebuffer.descriptor);
    render_metaballs(g_buffer_pass);
    box_outline_render(&state.box_outline, g_buffer_pass);
    ground_render(&state.ground, g_buffer_pass);
    particles_render(&state.particles, g_buffer_pass,
                     state.deferred_pass.point_lights.lights_count);
    wgpuRenderPassEncoderEnd(g_buffer_pass);
    wgpuRenderPassEncoderRelease(g_buffer_pass);
  }

  /* Post-processing passes */
  if (settings_get_quality_level().bloom_toggle) {
    /* Copy pass - render deferred lighting to copy buffer */
    {
      WGPURenderPassEncoder copy_render_pass
        = wgpuCommandEncoderBeginRenderPass(
          cmd_encoder, &state.copy_pass.framebuffer.descriptor);
      deferred_pass_render(&state.deferred_pass, copy_render_pass);
      wgpuRenderPassEncoderEnd(copy_render_pass);
      wgpuRenderPassEncoderRelease(copy_render_pass);
    }

    /* Bloom pass */
    {
      WGPURenderPassEncoder bloom_render_pass
        = wgpuCommandEncoderBeginRenderPass(
          cmd_encoder, &state.bloom_pass.framebuffer.descriptor);
      bloom_pass_render(&state.bloom_pass, bloom_render_pass);
      wgpuRenderPassEncoderEnd(bloom_render_pass);
      wgpuRenderPassEncoderRelease(bloom_render_pass);
    }

    /* Final composite pass */
    {
      WGPURenderPassEncoder final_pass = wgpuCommandEncoderBeginRenderPass(
        cmd_encoder, &state.renderer.framebuffer.descriptor);
      result_pass_render(&state.result_pass, final_pass);
      wgpuRenderPassEncoderEnd(final_pass);
      wgpuRenderPassEncoderRelease(final_pass);
    }
  }
  else {
    /* Final composite pass - render deferred lighting directly */
    {
      WGPURenderPassEncoder final_pass = wgpuCommandEncoderBeginRenderPass(
        cmd_encoder, &state.renderer.framebuffer.descriptor);
      deferred_pass_render(&state.deferred_pass, final_pass);
      wgpuRenderPassEncoderEnd(final_pass);
      wgpuRenderPassEncoderRelease(final_pass);
    }
  }

  /* Submit commands */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_encoder);

  /* Render imgui - must be after main submit since it creates its own encoder
   * and submits to the same swapchain texture with loadOp=Load */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  if (!state.prepared) {
    return;
  }

  /* Cleanup imgui */
  imgui_overlay_shutdown();

  /* Cleanup scene objects */
  particles_destroy(&state.particles);
  ground_destroy(&state.ground);
  box_outline_destroy(&state.box_outline);

  /* Cleanup metaballs */
  cleanup_metaballs();
  cleanup_metaballs_compute();

  /* Cleanup post-processing passes */
  result_pass_destroy(&state.result_pass);
  if (settings_get_quality_level().bloom_toggle) {
    bloom_pass_destroy(&state.bloom_pass);
  }
  copy_pass_destroy(&state.copy_pass);
  deferred_pass_destroy(&state.deferred_pass);

  /* Cleanup renderer */
  WGPU_RELEASE_RESOURCE(Sampler, state.renderer.default_sampler);
  WGPU_RELEASE_RESOURCE(TextureView, state.renderer.depth_texture_view);
  WGPU_RELEASE_RESOURCE(Texture, state.renderer.depth_texture);
  WGPU_RELEASE_RESOURCE(BindGroup, state.renderer.frame_bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.renderer.frame_bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, state.renderer.screen_frame_bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        state.renderer.screen_frame_bind_group_layout);
  WGPU_RELEASE_RESOURCE(Buffer, state.renderer.projection_ubo.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.renderer.view_ubo.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.renderer.screen_projection_ubo.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.renderer.screen_view_ubo.buffer);

  state.prepared = false;
}

int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Metaballs",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Inline WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* metaball_field_compute_shader = CODE(
  struct Metaball {
    position: vec3<f32>,
    radius: f32,
    strength: f32,
    subtract: f32,
  }

  struct Metaballs {
    count: u32,
    balls: array<Metaball>,
  }

  struct IsosurfaceVolume {
    min: vec3<f32>,
    max: vec3<f32>,
    step: vec3<f32>,
    size: vec3<u32>,
    threshold: f32,
    values: array<f32>,
  }

  @group(0) @binding(0) var<storage> metaballs : Metaballs;
  @group(0) @binding(1) var<storage, read_write> volume : IsosurfaceVolume;

  fn positionAt(index : vec3<u32>) -> vec3<f32> {
    return volume.min + (volume.step * vec3<f32>(index.xyz));
  }

  fn surfaceFunc(position : vec3<f32>) -> f32 {
    var result = 0.0;
    for (var i = 0u; i < metaballs.count; i = i + 1u) {
      var ball = metaballs.balls[i];
      var dist = distance(position, ball.position);
      var val = ball.strength / (0.000001 + (dist * dist)) - ball.subtract;
      if (val > 0.0) {
        result = result + val;
      }
    }
    return result;
  }

  @compute @workgroup_size(4, 4, 4)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    var position = positionAt(global_id);
    var valueIndex = global_id.x +
                    (global_id.y * volume.size.x) +
                    (global_id.z * volume.size.x * volume.size.y);

    volume.values[valueIndex] = surfaceFunc(position);
  }
);

static const char* marching_cubes_compute_shader_part1 =
  /* Part 1: Struct definitions and helper functions */
  CODE(
  struct Tables {
    edges: array<u32, 256>,
    tris: array<i32, 4096>,
  };
  @group(0) @binding(0) var<storage> tables : Tables;

  struct IsosurfaceVolume {
    min: vec3<f32>,
    max: vec3<f32>,
    step: vec3<f32>,
    size: vec3<u32>,
    threshold: f32,
    values: array<f32>,
  }
  @group(0) @binding(1) var<storage, read_write> volume : IsosurfaceVolume;

  struct PositionBuffer {
    values : array<f32>,
  };
  @group(0) @binding(2) var<storage, read_write> positionsOut : PositionBuffer;

  struct NormalBuffer {
    values : array<f32>,
  };
  @group(0) @binding(3) var<storage, read_write> normalsOut : NormalBuffer;

  struct IndexBuffer {
    tris : array<u32>,
  };
  @group(0) @binding(4) var<storage, read_write> indicesOut : IndexBuffer;

  struct DrawIndirectArgs {
    vc : u32,
    vertexCount : atomic<u32>,
    firstVertex : u32,
    firstInstance : u32,

    indexCount : atomic<u32>,
    indexedInstanceCount : u32,
    indexedFirstIndex : u32,
    indexedBaseVertex : u32,
    indexedFirstInstance : u32,
  };
  @group(0) @binding(5) var<storage, read_write> drawOut : DrawIndirectArgs;

  fn valueAt(index : vec3<u32>) -> f32 {
    if (any(index >= volume.size)) { return 0.0; }

    var valueIndex = index.x +
                    (index.y * volume.size.x) +
                    (index.z * volume.size.x * volume.size.y);
    return volume.values[valueIndex];
  }

  fn positionAt(index : vec3<u32>) -> vec3<f32> {
    return volume.min + (volume.step * vec3<f32>(index.xyz));
  }

  fn normalAt(index : vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
      valueAt(index - vec3<u32>(1u, 0u, 0u)) - valueAt(index + vec3<u32>(1u, 0u, 0u)),
      valueAt(index - vec3<u32>(0u, 1u, 0u)) - valueAt(index + vec3<u32>(0u, 1u, 0u)),
      valueAt(index - vec3<u32>(0u, 0u, 1u)) - valueAt(index + vec3<u32>(0u, 0u, 1u))
    );
  }

  var<private> positions : array<vec3<f32>, 12>;
  var<private> normals : array<vec3<f32>, 12>;
  var<private> indices : array<u32, 12>;
  var<private> cubeVerts : u32 = 0u;
  );
  /* Part 2: Interpolation functions */
  static const char* marching_cubes_compute_shader_part2 = CODE(
  fn interpX(index : u32, i : vec3<u32>, va : f32, vb : f32) {
    var mu = (volume.threshold - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3<f32>(volume.step.x * mu, 0.0, 0.0);

    var na = normalAt(i);
    var nb = normalAt(i + vec3<u32>(1u, 0u, 0u));
    normals[cubeVerts] = mix(na, nb, vec3<f32>(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
  }

  fn interpY(index : u32, i : vec3<u32>, va : f32, vb : f32) {
    var mu = (volume.threshold - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3<f32>(0.0, volume.step.y * mu, 0.0);

    var na = normalAt(i);
    var nb = normalAt(i + vec3<u32>(0u, 1u, 0u));
    normals[cubeVerts] = mix(na, nb, vec3<f32>(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
  }

  fn interpZ(index : u32, i : vec3<u32>, va : f32, vb : f32) {
    var mu = (volume.threshold - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3<f32>(0.0, 0.0, volume.step.z * mu);

    var na = normalAt(i);
    var nb = normalAt(i + vec3<u32>(0u, 0u, 1u));
    normals[cubeVerts] = mix(na, nb, vec3<f32>(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
  }
  );
  /* Part 3: Main compute entry point */
  static const char* marching_cubes_compute_shader_part3 = CODE(
  @compute @workgroup_size(4, 4, 4)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    var i0 = global_id;
    var i1 = global_id + vec3<u32>(1u, 0u, 0u);
    var i2 = global_id + vec3<u32>(1u, 1u, 0u);
    var i3 = global_id + vec3<u32>(0u, 1u, 0u);
    var i4 = global_id + vec3<u32>(0u, 0u, 1u);
    var i5 = global_id + vec3<u32>(1u, 0u, 1u);
    var i6 = global_id + vec3<u32>(1u, 1u, 1u);
    var i7 = global_id + vec3<u32>(0u, 1u, 1u);

    var v0 = valueAt(i0);
    var v1 = valueAt(i1);
    var v2 = valueAt(i2);
    var v3 = valueAt(i3);
    var v4 = valueAt(i4);
    var v5 = valueAt(i5);
    var v6 = valueAt(i6);
    var v7 = valueAt(i7);

    var cubeIndex = 0u;
    if (v0 < volume.threshold) { cubeIndex = cubeIndex | 1u; }
    if (v1 < volume.threshold) { cubeIndex = cubeIndex | 2u; }
    if (v2 < volume.threshold) { cubeIndex = cubeIndex | 4u; }
    if (v3 < volume.threshold) { cubeIndex = cubeIndex | 8u; }
    if (v4 < volume.threshold) { cubeIndex = cubeIndex | 16u; }
    if (v5 < volume.threshold) { cubeIndex = cubeIndex | 32u; }
    if (v6 < volume.threshold) { cubeIndex = cubeIndex | 64u; }
    if (v7 < volume.threshold) { cubeIndex = cubeIndex | 128u; }

    var edges = tables.edges[cubeIndex];

    if ((edges & 1u) != 0u) { interpX(0u, i0, v0, v1); }
    if ((edges & 2u) != 0u) { interpY(1u, i1, v1, v2); }
    if ((edges & 4u) != 0u) { interpX(2u, i3, v3, v2); }
    if ((edges & 8u) != 0u) { interpY(3u, i0, v0, v3); }
    if ((edges & 16u) != 0u) { interpX(4u, i4, v4, v5); }
    if ((edges & 32u) != 0u) { interpY(5u, i5, v5, v6); }
    if ((edges & 64u) != 0u) { interpX(6u, i7, v7, v6); }
    if ((edges & 128u) != 0u) { interpY(7u, i4, v4, v7); }
    if ((edges & 256u) != 0u) { interpZ(8u, i0, v0, v4); }
    if ((edges & 512u) != 0u) { interpZ(9u, i1, v1, v5); }
    if ((edges & 1024u) != 0u) { interpZ(10u, i2, v2, v6); }
    if ((edges & 2048u) != 0u) { interpZ(11u, i3, v3, v7); }

    var triTableOffset = (cubeIndex << 4u) + 1u;
    var indexCount = u32(tables.tris[triTableOffset - 1u]);

    var firstVertex = atomicAdd(&drawOut.vertexCount, cubeVerts);

    var bufferOffset = (global_id.x +
                        global_id.y * volume.size.x +
                        global_id.z * volume.size.x * volume.size.y);
    var firstIndex = bufferOffset * 15u;

    for (var i = 0u; i < cubeVerts; i = i + 1u) {
      positionsOut.values[firstVertex*3u + i*3u] = positions[i].x;
      positionsOut.values[firstVertex*3u + i*3u + 1u] = positions[i].y;
      positionsOut.values[firstVertex*3u + i*3u + 2u] = positions[i].z;

      normalsOut.values[firstVertex*3u + i*3u] = normals[i].x;
      normalsOut.values[firstVertex*3u + i*3u + 1u] = normals[i].y;
      normalsOut.values[firstVertex*3u + i*3u + 2u] = normals[i].z;
    }

    for (var i = 0u; i < indexCount; i = i + 1u) {
      var index = tables.tris[triTableOffset + i];
      indicesOut.tris[firstIndex + i] = firstVertex + indices[index];
    }

    for (var i = indexCount; i < 15u; i = i + 1u) {
      indicesOut.tris[firstIndex + i] = firstVertex;
    }
  }
);

/* Combine marching cubes compute shader chunks into one buffer */
#define MARCHING_CUBES_SHADER_SIZE (8 * 1024)
static const char* marching_cubes_create_compute_shader(void)
{
  static char shader_source[MARCHING_CUBES_SHADER_SIZE];
  snprintf(shader_source, sizeof(shader_source), "%s%s%s",
           marching_cubes_compute_shader_part1,
           marching_cubes_compute_shader_part2,
           marching_cubes_compute_shader_part3);
  return shader_source;
}

static const char* metaballs_vertex_shader = CODE(
  struct ProjectionUniformsStruct {
    matrix : mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    outputSize : vec2<f32>,
    zNear : f32,
    zFar : f32,
  }

  struct ViewUniformsStruct {
    matrix: mat4x4<f32>,
    inverseMatrix: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
    deltaTime: f32,
  }

  @group(0) @binding(0) var<uniform> projection : ProjectionUniformsStruct;
  @group(0) @binding(1) var<uniform> view : ViewUniformsStruct;

  struct Inputs {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
  }

  struct Output {
    @location(0) normal: vec3<f32>,
    @builtin(position) position: vec4<f32>,
  }

  @vertex
  fn main(input: Inputs) -> Output {
    var output: Output;
    output.position = projection.matrix *
                    view.matrix *
                    vec4(input.position, 1.0);
    output.normal = input.normal;
    return output;
  }
);

static const char* metaballs_fragment_shader = CODE(
  struct Output {
    @location(0) GBuffer_OUT0: vec4<f32>,
    @location(1) GBuffer_OUT1: vec4<f32>,
  }

  fn encodeNormals(n: vec3<f32>) -> vec2<f32> {
    var p = sqrt(n.z * 8.0 + 8.0);
    return vec2(n.xy / p + 0.5);
  }

  fn encodeGBufferOutput(
    normal: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ID: f32
  ) -> Output {
    var output: Output;
    output.GBuffer_OUT0 = vec4(encodeNormals(normal), metallic, ID);
    output.GBuffer_OUT1 = vec4(albedo, roughness);
    return output;
  }

  struct Uniforms {
    color: vec3<f32>,
    roughness: f32,
    metallic: f32,
  }

  @group(1) @binding(0) var<uniform> ubo: Uniforms;

  struct Inputs {
    @location(0) normal: vec3<f32>,
  }

  @fragment
  fn main(input: Inputs) -> Output {
    var normal = normalize(input.normal);
    var albedo = ubo.color;
    var metallic = ubo.metallic;
    var roughness = ubo.roughness;
    var ID = 0.0;
    return encodeGBufferOutput(
      normal,
      albedo,
      metallic,
      roughness,
      ID
    );
  }
);

// clang-format on
