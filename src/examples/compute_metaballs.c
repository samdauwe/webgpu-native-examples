/**
 * @file compute_metaballs.c
 * @brief WebGPU demo featuring marching cubes and bloom post-processing via
 * compute shaders, physically based shading, deferred rendering, gamma
 * correction and shadow mapping.
 *
 * @ref
 * https://github.com/gnikoloff/webgpu-compute-metaballs/tree/main/src
 */

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
 * Constants
 * -------------------------------------------------------------------------- */

#define MAX_METABALLS 256
#define MAX_POINT_LIGHTS_COUNT 256
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus
#define SHADOW_MAP_SIZE 128
#define METABALLS_COMPUTE_WORKGROUP_SIZE_X 4
#define METABALLS_COMPUTE_WORKGROUP_SIZE_Y 4
#define METABALLS_COMPUTE_WORKGROUP_SIZE_Z 4

static const uint32_t METABALLS_COMPUTE_WORKGROUP_SIZE[3] = {
  METABALLS_COMPUTE_WORKGROUP_SIZE_X,
  METABALLS_COMPUTE_WORKGROUP_SIZE_Y,
  METABALLS_COMPUTE_WORKGROUP_SIZE_Z,
};

/* Configurable volume settings */
static const uint32_t VOLUME_WIDTH  = 64;
static const uint32_t VOLUME_HEIGHT = 64;
static const uint32_t VOLUME_DEPTH  = 64;

/* -------------------------------------------------------------------------- *
 * Shader variables - declared at top, code at bottom of file
 * -------------------------------------------------------------------------- */

static const char* bloom_blur_compute_shader;
static const char* bloom_pass_fragment_shader;
static const char* box_outline_fragment_shader;
static const char* box_outline_vertex_shader;
static const char* copy_pass_fragment_shader;
static const char* deferred_pass_fragment_shader;
static const char* effect_vertex_shader;
static const char* ground_fragment_shader;
static const char* ground_shadow_vertex_shader;
static const char* ground_vertex_shader;
static const char* marching_cubes_compute_shader;
static const char* metaball_field_compute_shader;
static const char* metaballs_fragment_shader;
static const char* metaballs_shadow_vertex_shader;
static const char* metaballs_vertex_shader;
static const char* particles_fragment_shader;
static const char* particles_vertex_shader;
static const char* result_pass_fragment_shader;
static const char* update_point_lights_compute_shader;

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

static const int32_t MARCHING_CUBES_TRI_TABLE[4096] = {
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1,
   3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1,
   3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1,
   3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1,
   9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1,
   9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1,
   2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1,
   8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1,
   9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1,
   4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1,
   3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1,
   1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1,
   4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1,
   4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1,
   5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1,
   2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1,
   9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1,
   0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1,
   2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1,
  10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1,
   4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1,
   5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1,
   5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1,
   9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1,
   0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1,
   1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1,
  10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1,
   8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1,
   2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1,
   7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1,
   2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1,
  11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1,
   5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1,
  11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1,
  11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1,
   1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1,
   9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1,
   5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1,
   2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1,
   0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1,
   5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1,
   6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1,
   0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1,
   3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1,
   6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1,
   5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1,
   1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1,
  10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1,
   6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1,
   1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1,
   8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1,
   7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1,
   3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1,
   5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1,
   0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1,
   9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1,
   8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1,
   5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1,
   0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1,
   6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1,
  10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1,
  10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1,
   8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1,
   1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1,
   0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1,
  10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1,
   3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1,
   6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1,
   9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1,
   8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1,
   3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1,
   6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1,
   0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1,
  10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1,
  10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1,
   1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1,
   2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1,
   7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1,
   7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1,
   2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1,
   1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1,
  11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1,
   8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1,
   0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1,
   7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1,
  10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1,
   2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1,
   6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1,
   7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1,
   2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1,
   1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1,
  10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1,
  10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1,
   0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1,
   7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1,
   6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1,
   8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1,
   9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1,
   6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1,
   4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1,
  10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1,
   8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1,
   0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1,
   1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1,
   8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1,
  10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1,
   4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1,
  10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1,
   5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1,
  11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1,
   9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1,
   6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1,
   7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1,
   3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1,
   7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1,
   9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1,
   3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1,
   6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1,
   9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1,
   1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1,
   4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1,
   7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1,
   6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1,
   3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1,
   0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1,
   6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1,
   0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1,
  11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1,
   6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1,
   5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1,
   9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1,
   1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1,
   1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1,
  10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1,
   0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1,
   5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1,
  10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1,
  11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1,
   9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1,
   7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1,
   2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1,
   8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1,
   9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1,
   9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1,
   1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1,
   9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1,
   9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1,
   5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1,
   0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1,
  10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1,
   2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1,
   0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1,
   0,  2, 5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1,
   9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1,
   5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1,
   3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1,
   5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1,
   8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1,
   0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1,
   9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1,
   0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1,
   1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1,
   3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1,
   4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1,
   9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1,
  11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1,
  11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1,
   2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1,
   9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1,
   3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1,
   1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1,
   4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1,
   4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1,
   0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1,
   3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1,
   3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1,
   0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1,
   9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1,
   1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
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

typedef struct {
  float position[3];
  float cutoff;
  float direction[3];
  float outer_cutoff;
  float color[3];
  float intensity;
  vec4 shadow_view_proj_matrix[4]; /* 4x4 matrix as 4 vec4s */
} spot_light_uniforms_t;

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
  mat4 projection_matrix;
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
} webgpu_renderer_t;

/* -------------------------------------------------------------------------- *
 * Metaballs Compute
 * -------------------------------------------------------------------------- */

/* GPU-side metaball representation */
typedef struct {
  float position[3];
  float radius;
  float strength;
  float subtract;
  float padding[2];
} gpu_metaball_t;

/* Volume settings for GPU */
typedef struct {
  float x_min;
  float y_min;
  float z_min;
  float x_step;
  float y_step;
  float z_step;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  float iso_level;
  float padding[2];
} gpu_volume_settings_t;

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
  float positions[MAX_POINT_LIGHTS_COUNT * 4];
  float colors[MAX_POINT_LIGHTS_COUNT * 4];
  uint32_t count;
} point_lights_data_t;

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;

  point_lights_data_t data;

  WGPUBuffer positions_buffer;
  WGPUBuffer colors_buffer;
  WGPUBuffer count_buffer;

  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;

  /* Render resources */
  WGPURenderPipeline render_pipeline;
  WGPUBuffer sphere_vertex_buffer;
  WGPUBuffer sphere_index_buffer;
  uint32_t sphere_index_count;
} point_lights_t;

/* -------------------------------------------------------------------------- *
 * Spot Light
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;

  spot_light_uniforms_t uniforms;
  WGPUBuffer uniforms_buffer;

  /* Shadow map */
  WGPUTexture shadow_map_texture;
  WGPUTextureView shadow_map_texture_view;
  WGPUSampler shadow_map_sampler;

  /* Bind groups */
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;

  /* Render resources (light representation) */
  WGPURenderPipeline render_pipeline;
  WGPUBuffer cone_vertex_buffer;
  WGPUBuffer cone_index_buffer;
  uint32_t cone_index_count;

  /* Shadow pass pipeline */
  WGPURenderPipeline shadow_pipeline;
} spot_light_t;

/* -------------------------------------------------------------------------- *
 * Box Outline
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;
  volume_settings_t* volume_settings;

  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t index_count;

  WGPURenderPipeline render_pipeline;
} box_outline_t;

/* -------------------------------------------------------------------------- *
 * Ground
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;

  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t index_count;

  mat4 model_matrix;
  WGPUBuffer model_matrix_buffer;

  WGPURenderPipeline render_pipeline;
  WGPURenderPipeline shadow_pipeline;
  WGPUBindGroup bind_group;
} ground_t;

/* -------------------------------------------------------------------------- *
 * Metaballs Rendering
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;
  metaballs_compute_t* compute;

  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPURenderPipeline shadow_pipeline;

  WGPUBindGroup bind_group;

  mat4 model_matrix;
  WGPUBuffer model_matrix_buffer;
} metaballs_t;

/* -------------------------------------------------------------------------- *
 * Particles
 * -------------------------------------------------------------------------- */

typedef struct {
  float position[3];
  float velocity[3];
  float life;
  float max_life;
} particle_t;

typedef struct {
  wgpu_context_t* wgpu_context;
  webgpu_renderer_t* renderer;

  particle_t particles[512];
  uint32_t particle_count;

  WGPUBuffer vertex_buffer;
  WGPUBuffer instance_buffer;

  WGPURenderPipeline render_pipeline;
} particles_t;

/* -------------------------------------------------------------------------- *
 * Post-Processing Effects
 * -------------------------------------------------------------------------- */

/* Copy pass - copies a texture */
typedef struct {
  wgpu_context_t* wgpu_context;

  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
} copy_pass_t;

/* Bloom pass */
typedef struct {
  wgpu_context_t* wgpu_context;
  uint32_t width;
  uint32_t height;

  WGPUTexture blur_textures[2];
  WGPUTextureView blur_texture_views[2];

  WGPUComputePipeline blur_h_pipeline;
  WGPUComputePipeline blur_v_pipeline;

  WGPUBindGroup blur_bind_groups[2];
  WGPUBindGroupLayout blur_bind_group_layout;

  /* Combine pass */
  WGPURenderPipeline combine_pipeline;
  WGPUBindGroup combine_bind_group;
} bloom_pass_t;

/* Deferred pass - G-buffer rendering */
typedef struct {
  wgpu_context_t* wgpu_context;
  uint32_t width;
  uint32_t height;

  /* G-buffer textures */
  WGPUTexture position_texture;
  WGPUTextureView position_texture_view;

  WGPUTexture normal_texture;
  WGPUTextureView normal_texture_view;

  WGPUTexture albedo_texture;
  WGPUTextureView albedo_texture_view;

  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;

  /* Output texture */
  WGPUTexture output_texture;
  WGPUTextureView output_texture_view;

  /* Pipeline */
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;
} deferred_pass_t;

/* Result pass - final composite */
typedef struct {
  wgpu_context_t* wgpu_context;

  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;

  WGPUBuffer settings_buffer;
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

  /* Lighting */
  point_lights_t point_lights;
  spot_light_t spot_light;

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
} state = {
  .prepared = false,
  .volume_settings = {
    .width     = 6.0f,
    .height    = 6.0f,
    .depth     = 6.0f,
    .res_x     = 64,
    .res_y     = 64,
    .res_z     = 64,
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
static void init_point_lights(wgpu_context_t* wgpu_context);
static void init_spot_light(wgpu_context_t* wgpu_context);
static void init_box_outline(wgpu_context_t* wgpu_context);
static void init_ground(wgpu_context_t* wgpu_context);
static void init_metaballs(wgpu_context_t* wgpu_context);
static void init_particles(wgpu_context_t* wgpu_context);
static void init_post_processing(wgpu_context_t* wgpu_context);

static void update_uniform_buffers(wgpu_context_t* wgpu_context);
static void update_camera(float dt);
static void update_metaballs(float time);
static void update_particles(float dt);

static void render_gui(wgpu_context_t* wgpu_context, float delta_time);

static void cleanup_renderer(void);
static void cleanup_metaballs_compute(void);
static void cleanup_point_lights(void);
static void cleanup_spot_light(void);
static void cleanup_box_outline(void);
static void cleanup_ground(void);
static void cleanup_metaballs(void);
static void cleanup_particles(void);
static void cleanup_post_processing(void);

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

static void
orthographic_camera_update_view_matrix(orthographic_camera_t* camera)
{
  vec3 center = {0.0f, 0.0f, -1.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_vec3_add(camera->position, center, center);
  glm_lookat(camera->position, center, up, camera->view_matrix);
}

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
  glm_perspective(glm_rad(fov), aspect, near, far, camera->projection_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* camera)
{
  glm_perspective(glm_rad(camera->fov), camera->aspect, camera->near,
                  camera->far, camera->projection_matrix);
}

static void perspective_camera_update_view_matrix(perspective_camera_t* camera)
{
  vec3 up = {0.0f, 1.0f, 0.0f};
  glm_lookat(camera->position, camera->look_at_position, up,
             camera->view_matrix);
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
                    .size  = sizeof(mat4),
                  });

  /* Create view uniform buffer */
  state.renderer.view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "View UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4),
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
    .usage = WGPUTextureUsage_RenderAttachment,
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

  /* Create frame bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(mat4),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(mat4),
      },
    },
  };

  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Frame bind group layout"),
    .entryCount = 2,
    .entries    = bgl_entries,
  };
  state.renderer.frame_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.renderer.frame_bind_group_layout);

  /* Create frame bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.renderer.projection_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(mat4),
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.renderer.view_ubo.buffer,
      .offset  = 0,
      .size    = sizeof(mat4),
    },
  };

  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Frame bind group"),
    .layout     = state.renderer.frame_bind_group_layout,
    .entryCount = 2,
    .entries    = bg_entries,
  };
  state.renderer.frame_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.renderer.frame_bind_group);

  /* Create screen frame bind group layout */
  state.renderer.screen_frame_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.renderer.screen_frame_bind_group_layout);

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
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update projection and view matrices */
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.renderer.projection_ubo.buffer, 0,
                       &state.main_camera.projection_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(wgpu_context->queue, state.renderer.view_ubo.buffer, 0,
                       &state.main_camera.view_matrix, sizeof(mat4));

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
    .label                   = STRVIEW("Depth texture"),
    .usage                   = WGPUTextureUsage_RenderAttachment,
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

  /* Create volume buffer */
  {
    const uint32_t volume_elements    = vol->res_x * vol->res_y * vol->res_z;
    const uint64_t volume_buffer_size = sizeof(float) * 12
                                        + sizeof(uint32_t) * 4
                                        + sizeof(float) * volume_elements;

    WGPUBufferDescriptor buffer_desc = {
      .label            = STRVIEW("Volume - Storage buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = volume_buffer_size,
      .mappedAtCreation = true,
    };
    mc->volume_buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(mc->volume_buffer);

    float* volume_mapped = (float*)wgpuBufferGetMappedRange(
      mc->volume_buffer, 0, volume_buffer_size);
    uint32_t* volume_size = (uint32_t*)(&volume_mapped[12]);

    /* Set volume parameters */
    volume_mapped[0] = -vol->width / 2.0f;  /* x_min */
    volume_mapped[1] = -vol->height / 2.0f; /* y_min */
    volume_mapped[2] = -vol->depth / 2.0f;  /* z_min */

    volume_mapped[8]  = vol->width / (float)(vol->res_x - 1);  /* x_step */
    volume_mapped[9]  = vol->height / (float)(vol->res_y - 1); /* y_step */
    volume_mapped[10] = vol->depth / (float)(vol->res_z - 1);  /* z_step */

    volume_size[0] = vol->res_x; /* width */
    volume_size[1] = vol->res_y; /* height */
    volume_size[2] = vol->res_z; /* depth */

    volume_mapped[15] = vol->iso_level; /* iso_level */

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
      wgpu_context->device, marching_cubes_compute_shader);
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

  /* Pipeline layout - only uses the frame bind group */
  WGPUBindGroupLayout bind_group_layouts[1] = {
    state.renderer.frame_bind_group_layout,
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

  /* Color target state */
  WGPUBlendState blend_state        = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
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
      .targetCount = 1,
      .targets     = &color_target,
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
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, mc->vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, mc->normal_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(render_pass, mc->index_buffer.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  /* Draw all indices - the compute shader pads unused slots with degenerate
   * triangles */
  wgpuRenderPassEncoderDrawIndexed(render_pass, mc->index_count, 1, 0, 0, 0);
}

static void cleanup_metaballs(void)
{
  metaballs_t* mb = &state.metaballs;

  WGPU_RELEASE_RESOURCE(RenderPipeline, mb->render_pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, mb->pipeline_layout);
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

      /* Recreate depth texture */
      recreate_depth_texture(wgpu_context);
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
    state.point_lights.data.count
      = (uint32_t)state.gui_settings.point_lights_count;
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

  /* Update camera */
  camera_controller_update(&state.camera_controller, state.delta_time);
  update_uniform_buffers(wgpu_context);

  /* Update metaballs simulation */
  update_metaballs_sim(state.delta_time);

  /* Render GUI */
  render_gui(wgpu_context, state.delta_time);

  /* Get current swap chain texture view */
  WGPUTextureView backbuffer_view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Main command encoder"),
                          });

  /* Compute pass - update metaballs field and run marching cubes */
  {
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(
      cmd_encoder, &(WGPUComputePassDescriptor){
                     .label = STRVIEW("Metaballs compute pass"),
                   });

    dispatch_metaballs_compute(compute_pass);

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
  }

  /* Main render pass */
  WGPURenderPassColorAttachment color_attachment = {
    .view       = backbuffer_view,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1f, 0.1f, 0.15f, 1.0f},
  };

  WGPURenderPassDepthStencilAttachment depth_attachment = {
    .view            = state.renderer.depth_texture_view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };

  WGPURenderPassDescriptor render_pass_desc = {
    .label                  = STRVIEW("Main render pass"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = &color_attachment,
    .depthStencilAttachment = &depth_attachment,
  };

  WGPURenderPassEncoder render_pass
    = wgpuCommandEncoderBeginRenderPass(cmd_encoder, &render_pass_desc);

  /* Render metaballs */
  render_metaballs(render_pass);

  wgpuRenderPassEncoderEnd(render_pass);
  wgpuRenderPassEncoderRelease(render_pass);

  /* Render imgui */
  imgui_overlay_render(wgpu_context);

  /* Submit commands */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_encoder);

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

  /* Cleanup metaballs */
  cleanup_metaballs();
  cleanup_metaballs_compute();

  /* Cleanup renderer */
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
  struct Metaballs {
    count: u32,
    balls: array<Metaball>,
  }

  struct Metaball {
    position: vec3<f32>,
    radius: f32,
    strength: f32,
    subtract: f32,
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

  @compute @workgroup_size(4, 4, 4)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let position = volume.min + (volume.step * vec3<f32>(global_id.xyz));

    var fieldValue: f32 = 0.0;
    for (var i: u32 = 0u; i < metaballs.count; i = i + 1u) {
      let ball = metaballs.balls[i];
      let d = distance(ball.position, position);
      fieldValue = fieldValue + ball.strength / (d * d);
    }

    let valueIndex = global_id.x +
                     (global_id.y * volume.size.x) +
                     (global_id.z * volume.size.x * volume.size.y);
    volume.values[valueIndex] = fieldValue;
  }
);

static const char* marching_cubes_compute_shader = CODE(
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

static const char* metaballs_vertex_shader = CODE(
  @group(0) @binding(0) var<uniform> projection: mat4x4<f32>;
  @group(0) @binding(1) var<uniform> view: mat4x4<f32>;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
  };

  @vertex
  fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = input.position;
    output.position = projection * view * vec4<f32>(world_pos, 1.0);
    output.world_pos = world_pos;
    output.normal = normalize(input.normal);
    return output;
  }
  );

  static const char* metaballs_fragment_shader = CODE(
  struct FragmentInput {
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
  };

  @fragment
  fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(input.normal);

    let ambient = 0.2;
    let diffuse = max(dot(normal, light_dir), 0.0);
    let lighting = ambient + diffuse * 0.8;

    let base_color = vec3<f32>(0.8, 0.3, 0.2);
    let final_color = base_color * lighting;

    return vec4<f32>(final_color, 1.0);
  }
);

static const char* bloom_blur_compute_shader = CODE(
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

static const char* bloom_pass_fragment_shader = CODE(
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

static const char* box_outline_vertex_shader = CODE(
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

static const char* box_outline_fragment_shader = CODE(
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

static const char* copy_pass_fragment_shader = CODE(
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

static const char* effect_vertex_shader = CODE(
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

static const char* ground_vertex_shader = CODE(
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

static const char* ground_fragment_shader = CODE(
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

static const char* ground_shadow_vertex_shader = CODE(
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

static const char* metaballs_shadow_vertex_shader = CODE(
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

static const char* particles_vertex_shader = CODE(
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

static const char* particles_fragment_shader = CODE(
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
    var dist = distance(input.uv, vec2(0.5), );
    if (dist > 0.5) {
      discard;
    }
    var output: Output;
    output.normal = vec4(0.0, 0.0, 0.0, 0.1);
    output.albedo = vec4(input.color, 1.0);
    return output;
  }
);

static const char* result_pass_fragment_shader = CODE(
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

  const PI = 3.141592653589793;

  @compute @workgroup_size(64, 1, 1)
  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    var index = GlobalInvocationID.x;
    if (index >= config.numLights) {
      return;
    }

    lightsBuffer.lights[index].position.x += lightsBuffer.lights[index].velocity.x * view.deltaTime;
    lightsBuffer.lights[index].position.z += lightsBuffer.lights[index].velocity.z * view.deltaTime;

    const size = 42.0;
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

static const char* deferred_pass_fragment_shader = CODE(
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
    var specular = numerator / denominator;

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
    var specular = numerator / vec3(denominator, denominator, denominator);

    var radiance = light.color * light.intensity;
    return (kD * surface.albedo.rgb / vec3(PI, PI, PI) + specular) * radiance * attenuation * NdotL;
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

    var projectedDepth = textureSample(spotLightDepthTexture, depthSampler, shadowPos.xy);

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
