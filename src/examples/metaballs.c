#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <string.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

/* Forward declare - not included directly due to header conflicts */

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Metaballs
 *
 * This example demonstrates real-time metaball rendering using marching cubes
 * on the CPU, with tri-planar texture mapping on the GPU. The metaballs are
 * animated blobs that merge and split, rendered with selectable textures
 * (lava, slime, water). An orbit camera allows interactive viewing.
 *
 * Ref:
 * https://github.com/toji/webgpu-metaballs
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Forward declarations for WGSL shaders (defined at bottom of file)
 * -------------------------------------------------------------------------- */
static const char* metaball_vertex_shader_wgsl;
static const char* metaball_fragment_shader_wgsl;
static const char* light_sprite_vertex_shader_wgsl;
static const char* light_sprite_fragment_shader_wgsl;
static const char* env_vertex_shader_wgsl;
static const char* env_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */
#define MAX_METABALLS 16
#define NUM_METABALLS 16
#define SAMPLE_COUNT 4
#define MAX_VERTEX_COUNT (65536u)
#define MAX_INDEX_COUNT (65536u * 3u)
#define MAX_LIGHTS 1024
#define MAX_ENV_PRIMITIVES 256
#define MAX_ENV_IMAGES 64
#define MAX_ENV_MATERIALS 64

/* -------------------------------------------------------------------------- *
 * Marching Cubes Tables
 * -------------------------------------------------------------------------- */
static const uint16_t mc_edge_table[256] = {
  // clang-format off
  0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
  0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f,
  0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
  0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
  0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5,
  0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
  0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
  0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
  0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
  0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc,
  0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
  0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
  0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055,
  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
  0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
  0x066, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
  0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
  0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
  0x13a, 0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
  0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09,
  0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
  0x203, 0x109, 0x000,
  // clang-format on
};

/* First value = tri count, followed by 15 index values (-1 = unused) */
static const int8_t mc_tri_table[256 * 16] = {
  // clang-format off
  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  8,
  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  1,  9,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  8,  3,  9,  8,  1,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 6,  9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,
  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, 3,  3,  11, 2,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  11, 2,  8,  11, 0,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  9,  0,  2,  3,  11, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1,
  -1, -1, 6,  3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, 9,  3,  9,  0,
  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, 6,  9,  8,  10, 10, 8,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  4,  7,  8,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 6,  4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 6,  0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, 6,  1,  2,
  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  3,  4,  7,  3,  0,
  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, 9,  9,  2,  10, 9,  0,  2,  8,  4,
  7,  -1, -1, -1, -1, -1, -1, 12, 2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,
  4,  -1, -1, -1, 6,  8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 9,  11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, 9,  9,
  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, 12, 4,  7,  11, 9,
  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, 9,  3,  10, 1,  3,  11, 10, 7,
  8,  4,  -1, -1, -1, -1, -1, -1, 12, 1,  11, 10, 1,  4,  11, 1,  0,  4,  7,
  11, 4,  -1, -1, -1, 12, 4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1,
  -1, -1, 9,  4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, 3,
  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  5,  4,
  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  5,  4,  1,  5,  0,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  5,  4,  8,  3,  5,  3,  1,  5,
  -1, -1, -1, -1, -1, -1, 6,  1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1,
  9,  5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, 12, 2,  10,
  5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, 6,  9,  5,  4,  2,  3,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  11, 2,  0,  8,  11, 4,  9,
  5,  -1, -1, -1, -1, -1, -1, 9,  0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1,
  -1, -1, -1, -1, 12, 2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1,
  -1, 9,  10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, 12, 4,
  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, 12, 5,  4,  0,  5,
  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, 9,  5,  4,  8,  5,  8,  10, 10,
  8,  11, -1, -1, -1, -1, -1, -1, 6,  9,  7,  8,  5,  7,  9,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1,
  -1, -1, 9,  0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, 6,
  1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  9,  7,  8,
  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, 12, 10, 1,  2,  9,  5,  0,
  5,  3,  0,  5,  7,  3,  -1, -1, -1, 12, 8,  0,  2,  8,  2,  5,  8,  5,  7,
  10, 5,  2,  -1, -1, -1, 9,  2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1,
  -1, -1, -1, 9,  7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1,
  12, 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, 12, 2,  3,
  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, 9,  11, 2,  1,  11, 1,
  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, 12, 9,  5,  8,  8,  5,  7,  10, 1,
  3,  10, 3,  11, -1, -1, -1, 15, 5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,
  10, 11, 10, 0,  15, 11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,
  0,  6,  11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10,
  6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  8,  3,  5,
  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  0,  1,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  8,  3,  1,  9,  8,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, 6,  1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, 9,
  9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, 12, 5,  9,  8,
  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, 6,  2,  3,  11, 10, 6,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 0,  8,  11, 2,  0,  10, 6,  5,
  -1, -1, -1, -1, -1, -1, 9,  0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1,
  -1, -1, -1, 12, 5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1,
  9,  6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,
  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, 12, 3,  11, 6,  0,  3,
  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, 9,  6,  5,  9,  6,  9,  11, 11, 9,
  8,  -1, -1, -1, -1, -1, -1, 6,  5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 9,  4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1,
  -1, 9,  1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, 12, 10,
  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, 9,  6,  1,  2,  6,
  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  5,  5,  2,  6,  3,
  0,  4,  3,  4,  7,  -1, -1, -1, 12, 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,
  2,  6,  -1, -1, -1, 15, 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,
  6,  9,  9,  3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, 12,
  5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, 12, 0,  1,  9,
  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, 15, 9,  2,  1,  9,  11, 2,
  9,  4,  11, 7,  11, 4,  5,  10, 6,  12, 8,  4,  7,  3,  11, 5,  3,  5,  1,
  5,  11, 6,  -1, -1, -1, 15, 5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,
  0,  4,  11, 15, 0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,
  12, 6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, 6,  10, 4,
  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  10, 6,  4,  9,
  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, 9,  10, 0,  1,  10, 6,  0,  6,  4,
  0,  -1, -1, -1, -1, -1, -1, 12, 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,
  10, -1, -1, -1, 9,  1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1,
  -1, 12, 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, 6,  0,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  3,  2,  8,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, 9,  10, 4,  9,  10, 6,  4,  11,
  2,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,  2,  2,  8,  11, 4,  9,  10, 4,
  10, 6,  -1, -1, -1, 12, 3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1,
  -1, -1, 15, 6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  12,
  9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, 15, 8,  11, 1,
  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  9,  3,  11, 6,  3,  6,  0,
  0,  6,  4,  -1, -1, -1, -1, -1, -1, 6,  6,  4,  8,  11, 6,  8,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1,
  -1, -1, -1, 12, 0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1,
  12, 10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, 9,  10, 6,
  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  6,  1,  6,
  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,  6,  9,  2,  9,  1,  6,  7,
  9,  0,  9,  3,  7,  3,  9,  9,  7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1,
  -1, -1, -1, -1, 6,  7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 12, 2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,
  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  15, 1,  8,  0,  1,
  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, 12, 11, 2,  1,  11, 1,  7,  10,
  6,  1,  6,  7,  1,  -1, -1, -1, 15, 8,  9,  6,  8,  6,  7,  9,  1,  6,  11,
  6,  3,  1,  3,  6,  6,  0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 12, 7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, 3,
  7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  7,  6,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  3,  0,  8,  11, 7,  6,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  1,  9,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, 6,  10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 9,  2,  9,
  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 12, 6,  11, 7,  2,  10,
  3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, 6,  7,  2,  3,  6,  2,  7,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 9,  7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1,
  -1, -1, -1, -1, 9,  2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1,
  -1, 12, 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, 9,  10,
  7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, 12, 10, 7,  6,  1,
  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, 12, 0,  3,  7,  0,  7,  10, 0,
  10, 9,  6,  10, 7,  -1, -1, -1, 9,  7,  6,  10, 7,  10, 8,  8,  10, 9,  -1,
  -1, -1, -1, -1, -1, 6,  6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, 9,
  8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, 12, 9,  4,  6,
  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, 9,  6,  8,  4,  6,  11, 8,
  2,  10, 1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 3,  0,  11, 0,  6,  11,
  0,  4,  6,  -1, -1, -1, 12, 4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,
  -1, -1, -1, 15, 10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,
  9,  8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, 6,  0,  4,
  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 1,  9,  0,  2,  3,
  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, 9,  1,  9,  4,  1,  4,  2,  2,  4,
  6,  -1, -1, -1, -1, -1, -1, 12, 8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10,
  1,  -1, -1, -1, 9,  10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1,
  -1, 15, 4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  6,  10,
  9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  4,  9,  5,  7,
  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  3,  4,  9,  5,  11,
  7,  6,  -1, -1, -1, -1, -1, -1, 9,  5,  0,  1,  5,  4,  0,  7,  6,  11, -1,
  -1, -1, -1, -1, -1, 12, 11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1,
  -1, -1, 9,  9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, 12,
  6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, 12, 7,  6,  11,
  5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, 15, 3,  4,  8,  3,  5,  4,
  3,  2,  5,  10, 5,  2,  11, 7,  6,  9,  7,  2,  3,  7,  6,  2,  5,  4,  9,
  -1, -1, -1, -1, -1, -1, 12, 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,
  -1, -1, -1, 12, 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1,
  15, 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  12, 9,  5,
  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, 15, 1,  6,  10, 1,  7,
  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  15, 4,  0,  10, 4,  10, 5,  0,  3,
  10, 6,  10, 7,  3,  7,  10, 12, 7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,
  10, -1, -1, -1, 9,  6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1,
  -1, 12, 3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, 12, 0,
  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, 9,  6,  11, 3,  6,
  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 9,  5,  11, 9,
  11, 8,  11, 5,  6,  -1, -1, -1, 15, 0,  11, 3,  0,  6,  11, 0,  9,  6,  5,
  6,  9,  1,  2,  10, 15, 11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,
  2,  5,  12, 6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, 12,
  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, 9,  9,  5,  6,
  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, 15, 1,  5,  8,  1,  8,  0,
  5,  6,  8,  3,  8,  2,  6,  2,  8,  6,  1,  5,  6,  2,  1,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 15, 1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,
  8,  9,  6,  12, 10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1,
  6,  0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10, 5,
  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  11, 5,  10, 7,  5,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 5,  10, 11, 7,  5,  8,  3,
  0,  -1, -1, -1, -1, -1, -1, 9,  5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1,
  -1, -1, -1, -1, 12, 10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1,
  -1, 9,  11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, 12, 0,
  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, 12, 9,  7,  5,  9,
  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, 15, 7,  5,  2,  7,  2,  11, 5,
  9,  2,  3,  2,  8,  9,  8,  2,  9,  2,  5,  10, 2,  3,  5,  3,  7,  5,  -1,
  -1, -1, -1, -1, -1, 12, 8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1,
  -1, -1, 12, 9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, 15,
  9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  6,  1,  3,  5,
  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  7,  0,  7,  1,
  1,  7,  5,  -1, -1, -1, -1, -1, -1, 9,  9,  0,  3,  9,  3,  5,  5,  3,  7,
  -1, -1, -1, -1, -1, -1, 6,  9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1,
  12, 5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, 12, 0,  1,
  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, 15, 10, 11, 4,  10, 4,
  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  12, 2,  5,  1,  2,  8,  5,  2,  11,
  8,  4,  5,  8,  -1, -1, -1, 15, 0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11,
  1,  5,  1,  11, 15, 0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,
  5,  6,  9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,
  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, 9,  5,  10, 2,  5,
  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, 15, 3,  10, 2,  3,  5,  10, 3,
  8,  5,  4,  5,  8,  0,  1,  9,  12, 5,  10, 2,  5,  2,  4,  1,  9,  2,  9,
  4,  2,  -1, -1, -1, 9,  8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1,
  -1, -1, 6,  0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,
  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, 3,  9,  4,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  11, 7,  4,  9,  11,
  9,  10, 11, -1, -1, -1, -1, -1, -1, 12, 0,  8,  3,  4,  9,  7,  9,  11, 7,
  9,  10, 11, -1, -1, -1, 12, 1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11,
  -1, -1, -1, 15, 3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,
  12, 4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, 15, 9,  7,
  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  9,  11, 7,  4,  11, 4,
  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, 12, 11, 7,  4,  11, 4,  2,  8,  3,
  4,  3,  2,  4,  -1, -1, -1, 12, 2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,
  9,  -1, -1, -1, 15, 9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,
  7,  15, 3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, 6,  1,
  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  9,  1,  4,
  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 4,  9,  1,  4,  1,  7,  0,
  8,  1,  8,  7,  1,  -1, -1, -1, 6,  4,  0,  3,  7,  4,  3,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 3,  4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 6,  9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, 9,  0,  1,  10,
  0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, 6,  3,  1,  10, 11, 3,  10,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  2,  11, 1,  11, 9,  9,  11, 8,
  -1, -1, -1, -1, -1, -1, 12, 3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,
  -1, -1, -1, 6,  0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  3,  3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,  3,
  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, 6,  9,  10, 2,  0,  9,
  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,  3,  8,  2,  8,  10, 0,  1,
  8,  1,  10, 8,  -1, -1, -1, 3,  1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 3,  0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,
  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  // clang-format on
};

/* -------------------------------------------------------------------------- *
 * Metaball types
 * -------------------------------------------------------------------------- */
typedef struct metaball_t {
  vec3 position;
  float radius;
  float strength;
  float subtract;
} metaball_t;

/* Isosurface volume */
typedef struct volume_t {
  float x_min, x_max, x_step;
  float y_min, y_max, y_step;
  float z_min, z_max, z_step;
  uint32_t width, height, depth;
  float* values;
} volume_t;

/* Metaball style enum */
typedef enum metaball_style_t {
  METABALL_STYLE_LAVA  = 0,
  METABALL_STYLE_SLIME = 1,
  METABALL_STYLE_WATER = 2,
  METABALL_STYLE_COUNT = 3,
} metaball_style_t;

/* Metaball resolution enum */
typedef enum metaball_resolution_t {
  METABALL_RES_LOW    = 0,
  METABALL_RES_MEDIUM = 1,
  METABALL_RES_HIGH   = 2,
  METABALL_RES_ULTRA  = 3,
  METABALL_RES_COUNT  = 4,
} metaball_resolution_t;

/* GPU light (matches WGSL Light struct: 32 bytes) */
typedef struct {
  float position[3];
  float range;
  float color[3];
  float intensity;
} gpu_light_t;

/* Light uniform buffer layout (matches WGSL GlobalLightUniforms) */
typedef struct {
  float ambient[3];
  uint32_t light_count;
  gpu_light_t lights[MAX_LIGHTS];
} light_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Environment model types
 * -------------------------------------------------------------------------- */

/* Interleaved vertex: position(3) + normal(3) + texcoord(2) = 32 bytes */
typedef struct {
  float position[3];
  float normal[3];
  float texcoord[2];
} env_vertex_t;

/* A single draw call */
typedef struct {
  uint32_t first_index;
  uint32_t index_count;
  int32_t material_index;
  mat4 model_matrix;
  WGPUBuffer model_buffer; /* per-primitive uniform: mat4 */
  WGPUBindGroup model_bind_group;
} env_primitive_t;

/* Material: base color factor + texture index */
typedef struct {
  float base_color_factor[4];
  int32_t base_color_texture_index; /* -1 = no texture */
  bool double_sided;
  WGPUBindGroup bind_group;
  WGPURenderPipeline pipeline; /* per-material pipeline (cull mode varies) */
} env_material_t;

/* Texture image loaded from GLB */
typedef struct {
  WGPUTexture texture;
  WGPUTextureView view;
} env_image_t;

/* Full environment model */
typedef struct {
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t vertex_count;
  uint32_t index_count;

  env_image_t images[MAX_ENV_IMAGES];
  uint32_t image_count;

  env_material_t materials[MAX_ENV_MATERIALS];
  uint32_t material_count;

  env_primitive_t primitives[MAX_ENV_PRIMITIVES];
  uint32_t primitive_count;

  gpu_light_t lights[64];
  uint32_t light_count;

  bool loaded;
} env_model_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */
static struct {
  /* Metaballs */
  metaball_t balls[MAX_METABALLS];
  uint32_t ball_count;

  /* Marching cubes volume */
  volume_t volume;
  float volume_values[64 * 64 * 64]; /* pre-allocated for max resolution */

  /* Mesh buffers (CPU side) */
  float positions[MAX_VERTEX_COUNT * 3];
  float normals[MAX_VERTEX_COUNT * 3];
  uint32_t indices[MAX_INDEX_COUNT];
  uint32_t vertex_count;
  uint32_t index_count;

  /* GPU buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer normal_buffer;
  WGPUBuffer index_buffer;
  WGPUBuffer uniform_buffer;
  WGPUBuffer lights_buffer;

  /* Lights */
  light_uniforms_t light_uniforms;
  float scene_light_intensities[64]; /* original intensities from env */
  uint32_t scene_light_count;

  /* Light sprites */
  WGPURenderPipeline light_sprite_pipeline;
  WGPUPipelineLayout light_sprite_pipeline_layout;

  /* Environment model */
  env_model_t env;
  WGPUBindGroupLayout env_material_bgl;
  WGPUBindGroupLayout env_model_bgl;
  WGPUPipelineLayout env_pipeline_layout;
  WGPURenderPipeline env_pipeline;
  WGPUSampler env_sampler;
  WGPUTexture env_default_texture;
  WGPUTextureView env_default_texture_view; /* 1x1 white fallback */

  /* Texture */
  wgpu_texture_t texture;
  uint8_t file_buffer[2048 * 2048 * 4];
  WGPUSampler sampler;

  /* Bind groups & layouts */
  WGPUBindGroupLayout frame_bgl;
  WGPUBindGroupLayout material_bgl;
  WGPUBindGroup frame_bind_group;
  WGPUBindGroup material_bind_group;

  /* Pipeline */
  WGPURenderPipeline pipeline;
  WGPUPipelineLayout pipeline_layout;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Camera (orbit) */
  struct {
    float orbit_x;  /* pitch angle */
    float orbit_y;  /* yaw angle */
    float distance; /* distance from target */
    vec3 target;    /* look-at target */
    mat4 view_matrix;
    mat4 projection_matrix;
    vec3 position;
    bool mouse_down;
    float last_mouse_x;
    float last_mouse_y;
  } camera;

  /* Uniforms */
  struct {
    mat4 projection;
    mat4 inv_projection;
    float output_size[2];
    float z_near;
    float z_far;
  } projection_uniforms;

  struct {
    mat4 view;
    vec3 camera_position;
    float time;
  } view_uniforms;

  /* Settings */
  struct {
    metaball_style_t style;
    metaball_resolution_t resolution;
    bool draw_metaballs;
    bool render_light_sprites;
    bool render_environment;
    bool environment_lights;
    bool metaball_lights;
  } settings;

  /* GUI */
  const char* style_names[METABALL_STYLE_COUNT];
  const char* resolution_names[METABALL_RES_COUNT];

  /* Timing */
  uint64_t last_frame_time;
  float elapsed_time;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Discard,
    .clearValue = {0.0, 0.0, 0.0, 0.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .camera = {
    .orbit_x  = (float)(GLM_PI * 0.1),
    .orbit_y  = 0.0f,
    .distance = 5.0f,
    .target   = {0.0f, 1.0f, 0.0f},
  },
  .settings = {
    .style                = METABALL_STYLE_LAVA,
    .resolution           = METABALL_RES_HIGH,
    .draw_metaballs       = true,
    .render_light_sprites = true,
    .render_environment   = true,
    .environment_lights   = true,
    .metaball_lights      = true,
  },
  .style_names      = {"Lava", "Slime", "Water"},
  .resolution_names = {"Low (0.2)", "Medium (0.1)", "High (0.075)", "Ultra (0.05)"},
};

/* -------------------------------------------------------------------------- *
 * Resolution step values
 * -------------------------------------------------------------------------- */
static const float resolution_steps[METABALL_RES_COUNT] = {
  0.2f,
  0.1f,
  0.075f,
  0.05f,
};

/* Texture paths per style */
static const char* texture_paths[METABALL_STYLE_COUNT] = {
  "assets/textures/lava.jpg",
  "assets/textures/slime.png",
  "assets/textures/water.jpg",
};

/* Light colors per style (from renderer.js setMetaballStyle) */
static const float metaball_light_colors[METABALL_STYLE_COUNT][3] = {
  {0.9f, 0.1f, 0.0f}, /* lava */
  {0.0f, 0.9f, 0.0f}, /* slime */
  {0.4f, 0.5f, 0.9f}, /* water */
};

/* -------------------------------------------------------------------------- *
 * Metaball animation
 * -------------------------------------------------------------------------- */
static void update_metaballs(float timestamp)
{
  state.ball_count = 0;
  const float t    = timestamp * 0.0005f;
  const float strength
    = 5.0f / ((sqrtf((float)NUM_METABALLS) - 1.0f) / 4.0f + 1.0f);
  const float subtract = 12.0f;

  for (int i = 0; i < NUM_METABALLS; ++i) {
    const float fi   = (float)i;
    metaball_t* ball = &state.balls[state.ball_count];
    ball->position[0]
      = cosf(fi + 1.12f * t * 0.21f * sinf(0.72f + 0.83f * fi)) * 0.5f;
    ball->position[1]
      = (sinf(fi + 1.26f * t * (1.03f + 0.5f * cosf(0.21f * fi))) + 1.0f)
        * 1.0f;
    ball->position[2]
      = cosf(fi + 1.32f * t * 0.1f * sinf(0.92f + 0.53f * fi)) * 0.5f;
    ball->radius   = sqrtf(strength / subtract);
    ball->strength = strength;
    ball->subtract = subtract;
    state.ball_count++;
  }
}

/* -------------------------------------------------------------------------- *
 * Isosurface evaluation
 * -------------------------------------------------------------------------- */
static float surface_func(float x, float y, float z)
{
  /* Floor geometry */
  if ((x * x + z * z < 1.1f) && y < 0.0f) {
    return 100.0f;
  }

  float result   = 0.0f;
  const vec3 pos = {x, y, z};
  for (uint32_t i = 0; i < state.ball_count; ++i) {
    const metaball_t* ball = &state.balls[i];
    const float dx         = pos[0] - ball->position[0];
    const float dy         = pos[1] - ball->position[1];
    const float dz         = pos[2] - ball->position[2];
    const float dist_sq    = dx * dx + dy * dy + dz * dz;
    const float val = ball->strength / (0.000001f + dist_sq) - ball->subtract;
    if (val > 0.0f) {
      result += val;
    }
  }
  return result;
}

/* -------------------------------------------------------------------------- *
 * Marching cubes volume setup
 * -------------------------------------------------------------------------- */
static void init_volume(float step)
{
  volume_t* vol = &state.volume;
  vol->x_min    = -1.05f;
  vol->x_max    = 1.05f;
  vol->x_step   = step;
  vol->y_min    = -0.1f;
  vol->y_max    = 2.5f;
  vol->y_step   = step;
  vol->z_min    = -1.05f;
  vol->z_max    = 1.1f;
  vol->z_step   = step;
  vol->width    = (uint32_t)((vol->x_max - vol->x_min) / vol->x_step) + 1;
  vol->height   = (uint32_t)((vol->y_max - vol->y_min) / vol->y_step) + 1;
  vol->depth    = (uint32_t)((vol->z_max - vol->z_min) / vol->z_step) + 1;
  vol->values   = state.volume_values;
}

static void update_volume(void)
{
  volume_t* vol   = &state.volume;
  uint32_t offset = 0;
  for (uint32_t k = 0; k < vol->depth; ++k) {
    const float z = vol->z_min + vol->z_step * (float)k;
    for (uint32_t j = 0; j < vol->height; ++j) {
      const float y = vol->y_min + vol->y_step * (float)j;
      for (uint32_t i = 0; i < vol->width; ++i) {
        const float x         = vol->x_min + vol->x_step * (float)i;
        vol->values[offset++] = surface_func(x, y, z);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Marching cubes mesh generation
 * -------------------------------------------------------------------------- */
static float value_at(uint32_t i, uint32_t j, uint32_t k)
{
  const volume_t* vol = &state.volume;
  return vol->values[i + j * vol->width + k * vol->width * vol->height];
}

static void compute_normal(vec3 out, uint32_t i, uint32_t j, uint32_t k)
{
  out[0] = value_at(i > 0 ? i - 1 : 0, j, k)
           - value_at(i + 1 < state.volume.width ? i + 1 : i, j, k);
  out[1] = value_at(i, j > 0 ? j - 1 : 0, k)
           - value_at(i, j + 1 < state.volume.height ? j + 1 : j, k);
  out[2] = value_at(i, j, k > 0 ? k - 1 : 0)
           - value_at(i, j, k + 1 < state.volume.depth ? k + 1 : k);
}

static void interp_x(uint32_t edge_idx, uint32_t* index_list,
                     uint32_t* vert_off, float threshold, uint32_t i,
                     uint32_t j, uint32_t k, float va, float vb)
{
  const volume_t* vol = &state.volume;
  const float mu      = (threshold - va) / (vb - va);
  const uint32_t vo   = *vert_off;
  const uint32_t off  = vo * 3;

  state.positions[off + 0]
    = vol->x_min + vol->x_step * (float)i + mu * vol->x_step;
  state.positions[off + 1] = vol->y_min + vol->y_step * (float)j;
  state.positions[off + 2] = vol->z_min + vol->z_step * (float)k;

  vec3 na, nb, lerped;
  compute_normal(na, i, j, k);
  compute_normal(nb, i + 1, j, k);
  glm_vec3_lerp(na, nb, mu, lerped);
  state.normals[off + 0] = lerped[0];
  state.normals[off + 1] = lerped[1];
  state.normals[off + 2] = lerped[2];

  index_list[edge_idx] = vo;
  (*vert_off)++;
}

static void interp_y(uint32_t edge_idx, uint32_t* index_list,
                     uint32_t* vert_off, float threshold, uint32_t i,
                     uint32_t j, uint32_t k, float va, float vb)
{
  const volume_t* vol = &state.volume;
  const float mu      = (threshold - va) / (vb - va);
  const uint32_t vo   = *vert_off;
  const uint32_t off  = vo * 3;

  state.positions[off + 0] = vol->x_min + vol->x_step * (float)i;
  state.positions[off + 1]
    = vol->y_min + vol->y_step * (float)j + mu * vol->y_step;
  state.positions[off + 2] = vol->z_min + vol->z_step * (float)k;

  vec3 na, nb, lerped;
  compute_normal(na, i, j, k);
  compute_normal(nb, i, j + 1, k);
  glm_vec3_lerp(na, nb, mu, lerped);
  state.normals[off + 0] = lerped[0];
  state.normals[off + 1] = lerped[1];
  state.normals[off + 2] = lerped[2];

  index_list[edge_idx] = vo;
  (*vert_off)++;
}

static void interp_z(uint32_t edge_idx, uint32_t* index_list,
                     uint32_t* vert_off, float threshold, uint32_t i,
                     uint32_t j, uint32_t k, float va, float vb)
{
  const volume_t* vol = &state.volume;
  const float mu      = (threshold - va) / (vb - va);
  const uint32_t vo   = *vert_off;
  const uint32_t off  = vo * 3;

  state.positions[off + 0] = vol->x_min + vol->x_step * (float)i;
  state.positions[off + 1] = vol->y_min + vol->y_step * (float)j;
  state.positions[off + 2]
    = vol->z_min + vol->z_step * (float)k + mu * vol->z_step;

  vec3 na, nb, lerped;
  compute_normal(na, i, j, k);
  compute_normal(nb, i, j, k + 1);
  glm_vec3_lerp(na, nb, mu, lerped);
  state.normals[off + 0] = lerped[0];
  state.normals[off + 1] = lerped[1];
  state.normals[off + 2] = lerped[2];

  index_list[edge_idx] = vo;
  (*vert_off)++;
}

static void generate_mesh(float threshold)
{
  const volume_t* vol    = &state.volume;
  uint32_t vertex_offset = 0;
  uint32_t index_offset  = 0;
  uint32_t index_list[12];
  float values[8];

  for (uint32_t k = 0; k < vol->depth - 1; ++k) {
    for (uint32_t j = 0; j < vol->height - 1; ++j) {
      for (uint32_t i = 0; i < vol->width - 1; ++i) {
        /* Evaluate corners */
        values[0] = value_at(i, j, k);
        values[1] = value_at(i + 1, j, k);
        values[2] = value_at(i + 1, j + 1, k);
        values[3] = value_at(i, j + 1, k);
        values[4] = value_at(i, j, k + 1);
        values[5] = value_at(i + 1, j, k + 1);
        values[6] = value_at(i + 1, j + 1, k + 1);
        values[7] = value_at(i, j + 1, k + 1);

        /* Cube index */
        uint32_t cube_index = 0;
        if (values[0] < threshold)
          cube_index |= 1;
        if (values[1] < threshold)
          cube_index |= 2;
        if (values[2] < threshold)
          cube_index |= 4;
        if (values[3] < threshold)
          cube_index |= 8;
        if (values[4] < threshold)
          cube_index |= 16;
        if (values[5] < threshold)
          cube_index |= 32;
        if (values[6] < threshold)
          cube_index |= 64;
        if (values[7] < threshold)
          cube_index |= 128;

        const uint16_t edges = mc_edge_table[cube_index];
        if (edges == 0)
          continue;

        /* Check vertex budget */
        if (vertex_offset + 12 >= MAX_VERTEX_COUNT)
          goto done;
        if (index_offset + 15 >= MAX_INDEX_COUNT)
          goto done;

        /* Interpolate vertices along edges */
        if (edges & 1)
          interp_x(0, index_list, &vertex_offset, threshold, i, j, k, values[0],
                   values[1]);
        if (edges & 2)
          interp_y(1, index_list, &vertex_offset, threshold, i + 1, j, k,
                   values[1], values[2]);
        if (edges & 4)
          interp_x(2, index_list, &vertex_offset, threshold, i, j + 1, k,
                   values[3], values[2]);
        if (edges & 8)
          interp_y(3, index_list, &vertex_offset, threshold, i, j, k, values[0],
                   values[3]);
        if (edges & 16)
          interp_x(4, index_list, &vertex_offset, threshold, i, j, k + 1,
                   values[4], values[5]);
        if (edges & 32)
          interp_y(5, index_list, &vertex_offset, threshold, i + 1, j, k + 1,
                   values[5], values[6]);
        if (edges & 64)
          interp_x(6, index_list, &vertex_offset, threshold, i, j + 1, k + 1,
                   values[7], values[6]);
        if (edges & 128)
          interp_y(7, index_list, &vertex_offset, threshold, i, j, k + 1,
                   values[4], values[7]);
        if (edges & 256)
          interp_z(8, index_list, &vertex_offset, threshold, i, j, k, values[0],
                   values[4]);
        if (edges & 512)
          interp_z(9, index_list, &vertex_offset, threshold, i + 1, j, k,
                   values[1], values[5]);
        if (edges & 1024)
          interp_z(10, index_list, &vertex_offset, threshold, i + 1, j + 1, k,
                   values[2], values[6]);
        if (edges & 2048)
          interp_z(11, index_list, &vertex_offset, threshold, i, j + 1, k,
                   values[3], values[7]);

        /* Record triangle indices */
        const uint32_t tri_offset = cube_index * 16;
        const int8_t tri_count    = mc_tri_table[tri_offset];
        for (int t = 0; t < tri_count; ++t) {
          const int8_t idx              = mc_tri_table[tri_offset + 1 + t];
          state.indices[index_offset++] = index_list[idx];
        }
      }
    }
  }

done:
  state.vertex_count = vertex_offset;
  state.index_count  = index_offset;
}

/* -------------------------------------------------------------------------- *
 * Orbit camera
 * -------------------------------------------------------------------------- */
static void camera_update_view_matrix(void)
{
  const float cx = cosf(state.camera.orbit_x);
  const float sx = sinf(state.camera.orbit_x);
  const float cy = cosf(state.camera.orbit_y);
  const float sy = sinf(state.camera.orbit_y);

  state.camera.position[0]
    = state.camera.target[0] + state.camera.distance * cx * sy;
  state.camera.position[1]
    = state.camera.target[1] + state.camera.distance * sx;
  state.camera.position[2]
    = state.camera.target[2] + state.camera.distance * cx * cy;

  glm_lookat(state.camera.position, state.camera.target,
             (vec3){0.0f, 1.0f, 0.0f}, state.camera.view_matrix);
}

static void camera_update_projection(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  const float fov    = (float)(GLM_PI * 0.5);
  const float z_near = 0.2f;
  const float z_far  = 100.0f;

  glm_perspective(fov, aspect, z_near, z_far, state.camera.projection_matrix);

  /* Store projection uniforms */
  glm_mat4_copy(state.camera.projection_matrix,
                state.projection_uniforms.projection);
  glm_mat4_inv(state.camera.projection_matrix,
               state.projection_uniforms.inv_projection);
  state.projection_uniforms.output_size[0] = (float)wgpu_context->width;
  state.projection_uniforms.output_size[1] = (float)wgpu_context->height;
  state.projection_uniforms.z_near         = z_near;
  state.projection_uniforms.z_far          = z_far;
}

/* -------------------------------------------------------------------------- *
 * Light system
 * -------------------------------------------------------------------------- */
static void update_lights(void)
{
  /* Start with scene lights (none for now, would come from glTF environment) */
  uint32_t light_index = state.scene_light_count;

  /* Attach a light to each metaball */
  if (state.settings.draw_metaballs) {
    const float* lc = metaball_light_colors[state.settings.style];
    for (uint32_t i = 0; i < state.ball_count && light_index < MAX_LIGHTS;
         ++i, ++light_index) {
      gpu_light_t* light     = &state.light_uniforms.lights[light_index];
      const metaball_t* ball = &state.balls[i];
      light->position[0]     = ball->position[0];
      light->position[1]     = ball->position[1];
      light->position[2]     = ball->position[2];
      light->color[0]        = lc[0];
      light->color[1]        = lc[1];
      light->color[2]        = lc[2];
      light->intensity       = state.settings.metaball_lights ? 4.0f : 0.0f;

      /* Compute light range from intensity */
      const float light_radius           = 0.05f;
      const float illumination_threshold = 0.001f;
      light->range
        = light_radius
          * (sqrtf(light->intensity / illumination_threshold) - 1.0f);
    }
  }

  /* Enable/disable scene lights */
  for (uint32_t i = 0; i < state.scene_light_count; ++i) {
    state.light_uniforms.lights[i].intensity
      = state.settings.environment_lights ? state.scene_light_intensities[i] :
                                            0.0f;
  }

  state.light_uniforms.light_count = light_index;
}

/* -------------------------------------------------------------------------- *
 * Texture loading
 * -------------------------------------------------------------------------- */

/* Create a 1x1 white fallback texture */
static void init_env_default_texture(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D size          = {1, 1, 1};
  WGPUTextureDescriptor desc = {
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = size,
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.env_default_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &desc);
  uint8_t white[4] = {255, 255, 255, 255};
  wgpu_image_to_texure(wgpu_context, state.env_default_texture, white, size, 4);
  state.env_default_texture_view = wgpuTextureCreateView(
    state.env_default_texture, &(WGPUTextureViewDescriptor){
                                 .format        = WGPUTextureFormat_RGBA8Unorm,
                                 .dimension     = WGPUTextureViewDimension_2D,
                                 .mipLevelCount = 1,
                                 .arrayLayerCount = 1,
                               });
}

/* -------------------------------------------------------------------------- *
 * Environment model loading (from GLB using cgltf)
 * -------------------------------------------------------------------------- */
static void load_environment_model(wgpu_context_t* wgpu_context)
{
  const char* glb_path = "assets/models/Dungeon/dungeon.glb";

  cgltf_options options = {0};
  cgltf_data* data      = NULL;
  cgltf_result result   = cgltf_parse_file(&options, glb_path, &data);
  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to parse GLB: %s (error %d)\n", glb_path, result);
    return;
  }

  result = cgltf_load_buffers(&options, data, glb_path);
  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to load GLB buffers: %s\n", glb_path);
    cgltf_free(data);
    return;
  }

  env_model_t* env = &state.env;
  memset(env, 0, sizeof(*env));

  /* --- Load images --- */
  env->image_count = (uint32_t)data->images_count;
  if (env->image_count > MAX_ENV_IMAGES) {
    env->image_count = MAX_ENV_IMAGES;
  }
  for (uint32_t i = 0; i < env->image_count; ++i) {
    cgltf_image* img = &data->images[i];
    void* raw_data   = NULL;
    size_t raw_size  = 0;

    if (img->buffer_view != NULL) {
      /* Image embedded in GLB binary buffer */
      raw_data
        = (uint8_t*)img->buffer_view->buffer->data + img->buffer_view->offset;
      raw_size = img->buffer_view->size;
    }

    if (raw_data && raw_size > 0) {
      int img_w, img_h, img_channels;
      stbi_uc* pixels
        = stbi_load_from_memory((const stbi_uc*)raw_data, (int)raw_size, &img_w,
                                &img_h, &img_channels, 4);
      if (pixels) {
        WGPUExtent3D tex_size = {
          .width              = (uint32_t)img_w,
          .height             = (uint32_t)img_h,
          .depthOrArrayLayers = 1,
        };
        WGPUTextureDescriptor tex_desc = {
          .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
          .dimension     = WGPUTextureDimension_2D,
          .size          = tex_size,
          .format        = WGPUTextureFormat_RGBA8Unorm,
          .mipLevelCount = 1,
          .sampleCount   = 1,
        };
        env->images[i].texture
          = wgpuDeviceCreateTexture(wgpu_context->device, &tex_desc);
        wgpu_image_to_texure(wgpu_context, env->images[i].texture, pixels,
                             tex_size, 4);
        env->images[i].view = wgpuTextureCreateView(
          env->images[i].texture, &(WGPUTextureViewDescriptor){
                                    .format    = WGPUTextureFormat_RGBA8Unorm,
                                    .dimension = WGPUTextureViewDimension_2D,
                                    .mipLevelCount   = 1,
                                    .arrayLayerCount = 1,
                                  });
        stbi_image_free(pixels);
      }
    }
  }

  /* --- Load materials --- */
  env->material_count = (uint32_t)data->materials_count;
  if (env->material_count > MAX_ENV_MATERIALS) {
    env->material_count = MAX_ENV_MATERIALS;
  }
  for (uint32_t i = 0; i < env->material_count; ++i) {
    cgltf_material* mat               = &data->materials[i];
    env_material_t* env_mat           = &env->materials[i];
    env_mat->double_sided             = mat->double_sided;
    env_mat->base_color_texture_index = -1;

    /* Default base color factor: white */
    env_mat->base_color_factor[0] = 1.0f;
    env_mat->base_color_factor[1] = 1.0f;
    env_mat->base_color_factor[2] = 1.0f;
    env_mat->base_color_factor[3] = 1.0f;

    if (mat->has_pbr_metallic_roughness) {
      cgltf_pbr_metallic_roughness* pbr = &mat->pbr_metallic_roughness;
      memcpy(env_mat->base_color_factor, pbr->base_color_factor,
             sizeof(float) * 4);
      if (pbr->base_color_texture.texture != NULL) {
        cgltf_texture* tex = pbr->base_color_texture.texture;
        if (tex->image != NULL) {
          env_mat->base_color_texture_index
            = (int32_t)(tex->image - data->images);
        }
      }
    }
  }

  /* --- Load meshes and primitives --- */
  /* Collect all vertices and indices into single buffers */
  env_vertex_t* all_vertices  = NULL;
  uint32_t total_vertex_count = 0;
  uint32_t* all_indices       = NULL;
  uint32_t total_index_count  = 0;

  const cgltf_scene* scene = data->scene ? data->scene : &data->scenes[0];

  for (cgltf_size ni = 0; ni < scene->nodes_count; ++ni) {
    cgltf_node* node = scene->nodes[ni];

    /* Skip light-only nodes */
    if (node->mesh == NULL) {
      continue;
    }

    /* Compute world matrix */
    mat4 world_matrix;
    cgltf_node_transform_world(node, (float*)world_matrix);

    cgltf_mesh* mesh = node->mesh;
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi) {
      cgltf_primitive* prim = &mesh->primitives[pi];

      if (prim->indices == NULL) {
        continue;
      }

      if (env->primitive_count >= MAX_ENV_PRIMITIVES) {
        break;
      }

      uint32_t index_start  = total_index_count;
      uint32_t vertex_start = total_vertex_count;

      /* --- Extract vertex attributes --- */
      float* buf_pos        = NULL;
      float* buf_normal     = NULL;
      float* buf_uv         = NULL;
      cgltf_size vert_count = 0;
      size_t pos_stride     = 0;
      size_t normal_stride  = 0;
      size_t uv_stride      = 0;

      for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
        cgltf_attribute* attr = &prim->attributes[ai];
        cgltf_accessor* acc   = attr->data;
        cgltf_buffer_view* bv = acc->buffer_view;
        uint8_t* base = (uint8_t*)bv->buffer->data + bv->offset + acc->offset;

        if (attr->type == cgltf_attribute_type_position) {
          buf_pos    = (float*)base;
          vert_count = acc->count;
          pos_stride = bv->stride > 0 ? bv->stride : (3 * sizeof(float));
        }
        else if (attr->type == cgltf_attribute_type_normal) {
          buf_normal    = (float*)base;
          normal_stride = bv->stride > 0 ? bv->stride : (3 * sizeof(float));
        }
        else if (attr->type == cgltf_attribute_type_texcoord) {
          buf_uv    = (float*)base;
          uv_stride = bv->stride > 0 ? bv->stride : (2 * sizeof(float));
        }
      }

      if (buf_pos == NULL || vert_count == 0) {
        continue;
      }

      total_vertex_count += (uint32_t)vert_count;
      all_vertices
        = realloc(all_vertices, total_vertex_count * sizeof(env_vertex_t));

      for (cgltf_size v = 0; v < vert_count; ++v) {
        env_vertex_t vert = {0};
        float* p          = (float*)((uint8_t*)buf_pos + v * pos_stride);
        memcpy(vert.position, p, sizeof(float) * 3);
        if (buf_normal) {
          float* n = (float*)((uint8_t*)buf_normal + v * normal_stride);
          memcpy(vert.normal, n, sizeof(float) * 3);
        }
        if (buf_uv) {
          float* u = (float*)((uint8_t*)buf_uv + v * uv_stride);
          memcpy(vert.texcoord, u, sizeof(float) * 2);
        }
        all_vertices[vertex_start + v] = vert;
      }

      /* --- Extract indices --- */
      cgltf_accessor* idx_acc = prim->indices;
      uint32_t prim_idx_count = (uint32_t)idx_acc->count;
      total_index_count += prim_idx_count;
      all_indices = realloc(all_indices, total_index_count * sizeof(uint32_t));

      for (cgltf_size ii = 0; ii < idx_acc->count; ++ii) {
        all_indices[index_start + ii]
          = (uint32_t)cgltf_accessor_read_index(idx_acc, ii) + vertex_start;
      }

      /* --- Record primitive --- */
      env_primitive_t* ep = &env->primitives[env->primitive_count];
      ep->first_index     = index_start;
      ep->index_count     = prim_idx_count;
      ep->material_index
        = prim->material ? (int32_t)(prim->material - data->materials) : 0;
      glm_mat4_copy(world_matrix, ep->model_matrix);
      env->primitive_count++;
    }
  }

  /* --- Extract lights (KHR_lights_punctual) --- */
  if (data->lights_count > 0) {
    for (cgltf_size ni = 0; ni < scene->nodes_count; ++ni) {
      cgltf_node* node = scene->nodes[ni];
      if (node->light == NULL) {
        continue;
      }
      if (env->light_count >= 64) {
        break;
      }

      cgltf_light* light = node->light;
      gpu_light_t* gl    = &env->lights[env->light_count];

      /* Get light world position from the node transform */
      mat4 world_matrix;
      cgltf_node_transform_world(node, (float*)world_matrix);
      gl->position[0] = world_matrix[3][0];
      gl->position[1] = world_matrix[3][1];
      gl->position[2] = world_matrix[3][2];

      gl->color[0] = light->color[0];
      gl->color[1] = light->color[1];
      gl->color[2] = light->color[2];

      /* Divide intensity by 4*PI (candela to internal intensity) */
      gl->intensity = light->intensity / (4.0f * (float)GLM_PI);

      /* Compute range from intensity if not explicitly set */
      if (light->range > 0.0f) {
        gl->range = light->range;
      }
      else {
        const float light_radius           = 0.05f;
        const float illumination_threshold = 0.001f;
        gl->range                          = light_radius
                    * (sqrtf(gl->intensity / illumination_threshold) - 1.0f);
      }

      env->light_count++;
    }
  }

  /* --- Create GPU vertex and index buffers --- */
  if (total_vertex_count > 0 && total_index_count > 0) {
    env->vertex_count = total_vertex_count;
    env->index_count  = total_index_count;

    env->vertex_buffer = wgpu_create_buffer_from_data(
      wgpu_context, all_vertices, total_vertex_count * sizeof(env_vertex_t),
      WGPUBufferUsage_Vertex);

    env->index_buffer = wgpu_create_buffer_from_data(
      wgpu_context, all_indices, total_index_count * sizeof(uint32_t),
      WGPUBufferUsage_Index);

    /* Per-primitive model matrix uniform buffers */
    for (uint32_t i = 0; i < env->primitive_count; ++i) {
      env_primitive_t* ep = &env->primitives[i];
      ep->model_buffer    = wgpu_create_buffer_from_data(
        wgpu_context, &ep->model_matrix, sizeof(mat4), WGPUBufferUsage_Uniform);
    }

    env->loaded = true;
  }

  if (all_vertices)
    free(all_vertices);
  if (all_indices)
    free(all_indices);
  cgltf_free(data);
}

/* -------------------------------------------------------------------------- *
 * Metaball texture loading
 * -------------------------------------------------------------------------- */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    fprintf(stderr, "Texture fetch failed: %d\n", response->error_code);
    return;
  }
  int img_width, img_height, num_channels;
  stbi_uc* pixels
    = stbi_load_from_memory(response->data.ptr, (int)response->data.size,
                            &img_width, &img_height, &num_channels, 4);
  if (pixels) {
    state.texture.desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_width,
        .height             = (uint32_t)img_height,
        .depthOrArrayLayers = 1,
      },
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .pixels    = {.ptr = pixels, .size = (size_t)(img_width * img_height * 4)},
      .is_dirty  = true,
    };
  }
}

static void load_texture(const char* path)
{
  sfetch_send(&(sfetch_request_t){
    .path     = path,
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  state.texture = wgpu_create_color_bars_texture(wgpu_context, NULL);
  load_texture(texture_paths[state.settings.style]);
}

static void init_sampler(wgpu_context_t* wgpu_context)
{
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
}

/* -------------------------------------------------------------------------- *
 * GPU buffers
 * -------------------------------------------------------------------------- */
static void init_gpu_buffers(wgpu_context_t* wgpu_context)
{
  const uint32_t vb_size = MAX_VERTEX_COUNT * 3 * sizeof(float);
  const uint32_t ib_size = MAX_INDEX_COUNT * sizeof(uint32_t);

  state.vertex_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Metaballs Vertex Buffer"),
      .size  = vb_size,
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
    });

  state.normal_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Metaballs Normal Buffer"),
      .size  = vb_size,
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
    });

  state.index_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Metaballs Index Buffer"),
      .size  = ib_size,
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
    });

  /* Uniform buffer: projection (144) + pad to 256 + view (80) = 336, round up
   */
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Metaballs Uniform Buffer"),
      .size  = 512,
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    });

  /* Lights storage buffer */
  state.lights_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Lights Storage Buffer"),
      .size  = sizeof(light_uniforms_t),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
    });
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts and bind groups
 * -------------------------------------------------------------------------- */
static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Frame bind group layout: projection + view uniforms + lights storage */
  state.frame_bgl = wgpuDeviceCreateBindGroupLayout(wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Frame BGL"),
      .entryCount = 3,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_ReadOnlyStorage},
        },
      },
    });

  /* Material bind group layout: sampler + texture */
  state.material_bgl = wgpuDeviceCreateBindGroupLayout(wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Material BGL"),
      .entryCount = 2,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .texture    = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
      },
    });
}

static void init_frame_bind_group(wgpu_context_t* wgpu_context)
{
  if (state.frame_bind_group) {
    wgpuBindGroupRelease(state.frame_bind_group);
  }
  state.frame_bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Frame Bind Group"),
      .layout     = state.frame_bgl,
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding = 0,
          .buffer  = state.uniform_buffer,
          .size    = 144,  /* ProjectionUniforms */
        },
        {
          .binding = 1,
          .buffer  = state.uniform_buffer,
          .offset  = 256,
          .size    = 80,  /* ViewUniforms */
        },
        {
          .binding = 2,
          .buffer  = state.lights_buffer,
          .size    = sizeof(light_uniforms_t),
        },
      },
    });
}

static void init_material_bind_group(wgpu_context_t* wgpu_context)
{
  if (state.material_bind_group) {
    wgpuBindGroupRelease(state.material_bind_group);
  }
  state.material_bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Material Bind Group"),
      .layout     = state.material_bgl,
      .entryCount = 2,
      .entries = (WGPUBindGroupEntry[]){
        {
          .binding  = 0,
          .sampler  = state.sampler,
        },
        {
          .binding     = 1,
          .textureView = state.texture.view,
        },
      },
    });
}

/* -------------------------------------------------------------------------- *
 * Render pipeline
 * -------------------------------------------------------------------------- */
static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = 2,
      .bindGroupLayouts = (WGPUBindGroupLayout[]){
        state.frame_bgl,
        state.material_bgl,
      },
    });

  /* Shader module */
  static char shader_source[8192];
  snprintf(shader_source, sizeof(shader_source), "%s\n%s",
           metaball_vertex_shader_wgsl, metaball_fragment_shader_wgsl);

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_source);

  /* Render pipeline */
  state.pipeline = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Metaball Pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module     = shader_module,
        .entryPoint = STRVIEW("vertexMain"),
        .bufferCount = 2,
        .buffers = (WGPUVertexBufferLayout[]){
          {
            .arrayStride    = 12,
            .stepMode       = WGPUVertexStepMode_Vertex,
            .attributeCount = 1,
            .attributes = &(WGPUVertexAttribute){
              .shaderLocation = 0,
              .format         = WGPUVertexFormat_Float32x3,
              .offset         = 0,
            },
          },
          {
            .arrayStride    = 12,
            .stepMode       = WGPUVertexStepMode_Vertex,
            .attributeCount = 1,
            .attributes = &(WGPUVertexAttribute){
              .shaderLocation = 1,
              .format         = WGPUVertexFormat_Float32x3,
              .offset         = 0,
            },
          },
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = true,
        .depthCompare        = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = SAMPLE_COUNT,
        .mask  = 0xFFFFFFFF,
      },
      .fragment = &(WGPUFragmentState){
        .module     = shader_module,
        .entryPoint = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  wgpuShaderModuleRelease(shader_module);
}

/* -------------------------------------------------------------------------- *
 * Light sprite pipeline
 * -------------------------------------------------------------------------- */
static void init_light_sprite_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout: only frame bind group (has projection + view + lights) */
  state.light_sprite_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = (WGPUBindGroupLayout[]){state.frame_bgl},
    });

  /* Shader modules */
  WGPUShaderModule vert_module = wgpu_create_shader_module(
    wgpu_context->device, light_sprite_vertex_shader_wgsl);
  WGPUShaderModule frag_module = wgpu_create_shader_module(
    wgpu_context->device, light_sprite_fragment_shader_wgsl);

  /* Render pipeline with additive blending */
  state.light_sprite_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Light Sprite Pipeline"),
      .layout = state.light_sprite_pipeline_layout,
      .vertex = {
        .module     = vert_module,
        .entryPoint = STRVIEW("vertexMain"),
      },
      .primitive = {
        .topology         = WGPUPrimitiveTopology_TriangleStrip,
        .stripIndexFormat  = WGPUIndexFormat_Uint32,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = false,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = SAMPLE_COUNT,
        .mask  = 0xFFFFFFFF,
      },
      .fragment = &(WGPUFragmentState){
        .module     = frag_module,
        .entryPoint = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
          .blend     = &(WGPUBlendState){
            .color = {
              .srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            },
            .alpha = {
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add,
            },
          },
        },
      },
    });

  wgpuShaderModuleRelease(vert_module);
  wgpuShaderModuleRelease(frag_module);
}

/* -------------------------------------------------------------------------- *
 * Environment pipeline and bind groups
 * -------------------------------------------------------------------------- */
static void init_environment_pipeline(wgpu_context_t* wgpu_context)
{
  if (!state.env.loaded) {
    return;
  }

  /* Environment sampler */
  state.env_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });

  /* Environment material bind group layout:
     binding 0: material uniform (base_color_factor vec4 = 16 bytes, pad to 48)
     binding 1: sampler
     binding 2: base color texture */
  state.env_material_bgl = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Env Material BGL"),
      .entryCount = 3,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 2,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
          },
        },
      },
    });

  /* Environment model bind group layout:
     binding 0: model matrix uniform (mat4 = 64 bytes) */
  state.env_model_bgl = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Env Model BGL"),
      .entryCount = 1,
      .entries = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex,
          .buffer     = {.type = WGPUBufferBindingType_Uniform},
        },
      },
    });

  /* Pipeline layout: frame @0, material @1, model @2 */
  state.env_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = 3,
      .bindGroupLayouts = (WGPUBindGroupLayout[]){
        state.frame_bgl,
        state.env_material_bgl,
        state.env_model_bgl,
      },
    });

  /* Compile env shaders */
  static char env_shader_buf[16384];
  snprintf(env_shader_buf, sizeof(env_shader_buf), "%s\n%s",
           env_vertex_shader_wgsl, env_fragment_shader_wgsl);
  WGPUShaderModule env_shader
    = wgpu_create_shader_module(wgpu_context->device, env_shader_buf);

  /* Create one pipeline (back-face culling) */
  state.env_pipeline = wgpuDeviceCreateRenderPipeline(wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Environment Pipeline"),
      .layout = state.env_pipeline_layout,
      .vertex = {
        .module      = env_shader,
        .entryPoint  = STRVIEW("vertexMain"),
        .bufferCount = 1,
        .buffers = &(WGPUVertexBufferLayout){
          .arrayStride    = sizeof(env_vertex_t),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 3,
          .attributes = (WGPUVertexAttribute[]){
            {.shaderLocation = 0,
             .format = WGPUVertexFormat_Float32x3,
             .offset = offsetof(env_vertex_t, position)},
            {.shaderLocation = 1,
             .format = WGPUVertexFormat_Float32x3,
             .offset = offsetof(env_vertex_t, normal)},
            {.shaderLocation = 2,
             .format = WGPUVertexFormat_Float32x2,
             .offset = offsetof(env_vertex_t, texcoord)},
          },
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = wgpu_context->depth_stencil_format,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = SAMPLE_COUNT,
        .mask  = 0xFFFFFFFF,
      },
      .fragment = &(WGPUFragmentState){
        .module      = env_shader,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
    });

  wgpuShaderModuleRelease(env_shader);

  /* Create per-material bind groups */
  for (uint32_t i = 0; i < state.env.material_count; ++i) {
    env_material_t* mat = &state.env.materials[i];

    /* Material uniform buffer: base_color_factor (vec4 = 16 bytes) */
    /* Pad to 48 bytes to match WGSL MaterialUniforms alignment */
    float mat_data[12] = {0};
    memcpy(mat_data, mat->base_color_factor, sizeof(float) * 4);

    WGPUBuffer mat_buffer = wgpu_create_buffer_from_data(
      wgpu_context, mat_data, sizeof(mat_data), WGPUBufferUsage_Uniform);

    /* Pick texture view: use base color texture if available, else default */
    WGPUTextureView tex_view = state.env_default_texture_view;
    if (mat->base_color_texture_index >= 0
        && (uint32_t)mat->base_color_texture_index < state.env.image_count) {
      WGPUTextureView img_view
        = state.env.images[mat->base_color_texture_index].view;
      if (img_view) {
        tex_view = img_view;
      }
    }

    mat->bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Env Material BG"),
        .layout     = state.env_material_bgl,
        .entryCount = 3,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .buffer = mat_buffer, .size = sizeof(mat_data)},
          {.binding = 1, .sampler = state.env_sampler},
          {.binding = 2, .textureView = tex_view},
        },
      });

    /* Store pipeline reference (same for all materials since we use CullBack)
     */
    mat->pipeline = state.env_pipeline;
  }

  /* Create per-primitive model bind groups */
  for (uint32_t i = 0; i < state.env.primitive_count; ++i) {
    env_primitive_t* prim = &state.env.primitives[i];
    prim->model_bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Env Model BG"),
        .layout     = state.env_model_bgl,
        .entryCount = 1,
        .entries = &(WGPUBindGroupEntry){
          .binding = 0,
          .buffer  = prim->model_buffer,
          .size    = sizeof(mat4),
        },
      });
  }

  /* Copy scene lights to the light uniforms */
  state.scene_light_count = state.env.light_count;
  for (uint32_t i = 0; i < state.env.light_count; ++i) {
    state.light_uniforms.lights[i]   = state.env.lights[i];
    state.scene_light_intensities[i] = state.env.lights[i].intensity;
  }
  state.light_uniforms.ambient[0] = 0.15f;
  state.light_uniforms.ambient[1] = 0.15f;
  state.light_uniforms.ambient[2] = 0.15f;
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer update
 * -------------------------------------------------------------------------- */
static void update_uniforms(wgpu_context_t* wgpu_context)
{
  /* Projection uniforms: mat4 projection + mat4 inverse + vec2 size + f32
   * zNear + f32 zFar = 144 bytes */
  camera_update_projection(wgpu_context);
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.projection_uniforms, 144);

  /* View uniforms: mat4 view + vec3 pos + f32 time = 80 bytes at offset 256 */
  camera_update_view_matrix();
  glm_mat4_copy(state.camera.view_matrix, state.view_uniforms.view);
  glm_vec3_copy(state.camera.position, state.view_uniforms.camera_position);
  state.view_uniforms.time = state.elapsed_time;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 256,
                       &state.view_uniforms, 80);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */
static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);
  igBegin("WebGPU Metaballs", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Style selector */
  int style = (int)state.settings.style;
  if (imgui_overlay_combo_box("Style", &style, state.style_names,
                              METABALL_STYLE_COUNT)) {
    state.settings.style = (metaball_style_t)style;
    load_texture(texture_paths[state.settings.style]);
  }

  /* Resolution selector */
  int resolution = (int)state.settings.resolution;
  if (imgui_overlay_combo_box("Resolution", &resolution, state.resolution_names,
                              METABALL_RES_COUNT)) {
    state.settings.resolution = (metaball_resolution_t)resolution;
    init_volume(resolution_steps[state.settings.resolution]);
  }

  /* Draw metaballs toggle */
  igCheckbox("Draw Metaballs", &state.settings.draw_metaballs);

  igSeparator();
  igText("Rendering Options");

  /* Light sprites toggle */
  igCheckbox("Render Light Sprites", &state.settings.render_light_sprites);

  /* Environment toggle */
  igCheckbox("Render Environment", &state.settings.render_environment);

  /* Light enable toggles */
  igCheckbox("Environment Lights", &state.settings.environment_lights);
  igCheckbox("Metaball Lights", &state.settings.metaball_lights);

  /* Stats */
  igSeparator();
  igText("Vertices: %u", state.vertex_count);
  igText("Indices: %u", state.index_count);
  igText("Triangles: %u", state.index_count / 3);
  igText("Lights: %u", state.light_uniforms.light_count);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */
static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  ImGuiIO* io            = igGetIO();
  bool imgui_wants_mouse = io->WantCaptureMouse;

  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN && !imgui_wants_mouse) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.camera.mouse_down   = true;
      state.camera.last_mouse_x = input_event->mouse_x;
      state.camera.last_mouse_y = input_event->mouse_y;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP
           && !imgui_wants_mouse) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.camera.mouse_down = false;
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && !imgui_wants_mouse) {
    if (state.camera.mouse_down) {
      float dx = input_event->mouse_x - state.camera.last_mouse_x;
      float dy = input_event->mouse_y - state.camera.last_mouse_y;
      state.camera.last_mouse_x = input_event->mouse_x;
      state.camera.last_mouse_y = input_event->mouse_y;
      state.camera.orbit_y += dx * 0.025f;
      state.camera.orbit_x += dy * 0.025f;

      /* Clamp pitch */
      if (state.camera.orbit_x > (float)(GLM_PI * 0.5))
        state.camera.orbit_x = (float)(GLM_PI * 0.5);
      if (state.camera.orbit_x < (float)(GLM_PI * -0.5))
        state.camera.orbit_x = (float)(GLM_PI * -0.5);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_SCROLL
           && !imgui_wants_mouse) {
    state.camera.distance -= input_event->scroll_y * 0.5f;
    if (state.camera.distance < 1.0f)
      state.camera.distance = 1.0f;
    if (state.camera.distance > 10.0f)
      state.camera.distance = 10.0f;
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */
static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 4,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });

    init_volume(resolution_steps[state.settings.resolution]);
    init_gpu_buffers(wgpu_context);
    init_sampler(wgpu_context);
    init_texture(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_frame_bind_group(wgpu_context);
    init_material_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
    init_light_sprite_pipeline(wgpu_context);

    /* Environment model */
    init_env_default_texture(wgpu_context);
    load_environment_model(wgpu_context);
    init_environment_pipeline(wgpu_context);

    imgui_overlay_init(wgpu_context);

    state.last_frame_time = stm_now();
    state.initialized     = true;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file requests */
  sfetch_dowork();

  /* Update texture if newly loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    init_material_bind_group(wgpu_context);
  }

  /* Timing */
  uint64_t now          = stm_now();
  float delta           = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;
  state.elapsed_time += delta * 1000.0f; /* ms */

  /* Update metaballs and mesh */
  if (state.settings.draw_metaballs) {
    update_metaballs(state.elapsed_time);
    update_volume();
    generate_mesh(40.0f);

    /* Upload mesh data to GPU */
    if (state.vertex_count > 0) {
      wgpuQueueWriteBuffer(wgpu_context->queue, state.vertex_buffer, 0,
                           state.positions,
                           state.vertex_count * 3 * sizeof(float));
      wgpuQueueWriteBuffer(wgpu_context->queue, state.normal_buffer, 0,
                           state.normals,
                           state.vertex_count * 3 * sizeof(float));
    }
    if (state.index_count > 0) {
      wgpuQueueWriteBuffer(wgpu_context->queue, state.index_buffer, 0,
                           state.indices, state.index_count * sizeof(uint32_t));
    }
  }

  /* Update lights (attach to metaballs + scene lights) */
  update_lights();
  wgpuQueueWriteBuffer(wgpu_context->queue, state.lights_buffer, 0,
                       &state.light_uniforms, sizeof(light_uniforms_t));

  /* Update uniforms */
  update_uniforms(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta);
  render_gui(wgpu_context);

  /* Begin render pass */
  state.color_attachment.view          = wgpu_context->msaa_view;
  state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view  = wgpu_context->depth_stencil_view;

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Draw environment model */
  if (state.settings.render_environment && state.env.loaded) {
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.frame_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.env.vertex_buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.env.index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    for (uint32_t i = 0; i < state.env.primitive_count; ++i) {
      env_primitive_t* prim = &state.env.primitives[i];
      env_material_t* mat   = &state.env.materials[prim->material_index];
      wgpuRenderPassEncoderSetPipeline(rpass_enc, mat->pipeline);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 1, mat->bind_group, 0, NULL);
      wgpuRenderPassEncoderSetBindGroup(rpass_enc, 2, prim->model_bind_group, 0,
                                        NULL);
      wgpuRenderPassEncoderDrawIndexed(rpass_enc, prim->index_count, 1,
                                       prim->first_index, 0, 0);
    }
  }

  /* Draw metaballs */
  if (state.settings.draw_metaballs && state.index_count > 0) {
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.frame_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 1, state.material_bind_group,
                                      0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertex_buffer, 0,
                                         state.vertex_count * 3
                                           * sizeof(float));
    wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 1, state.normal_buffer, 0,
                                         state.vertex_count * 3
                                           * sizeof(float));
    wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        state.index_count * sizeof(uint32_t));
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.index_count, 1, 0, 0, 0);
  }

  /* Draw light sprites (instanced billboards) */
  if (state.settings.render_light_sprites
      && state.light_uniforms.light_count > 0) {
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.light_sprite_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.frame_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(rpass_enc, 4, state.light_uniforms.light_count, 0,
                              0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  sfetch_shutdown();
  imgui_overlay_shutdown();
  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.normal_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.lights_buffer)
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.frame_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.material_bgl)
  WGPU_RELEASE_RESOURCE(BindGroup, state.frame_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.material_bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.light_sprite_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.light_sprite_pipeline_layout)

  /* Environment cleanup */
  if (state.env.loaded) {
    WGPU_RELEASE_RESOURCE(Buffer, state.env.vertex_buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.env.index_buffer)
    for (uint32_t i = 0; i < state.env.primitive_count; ++i) {
      WGPU_RELEASE_RESOURCE(Buffer, state.env.primitives[i].model_buffer)
      WGPU_RELEASE_RESOURCE(BindGroup, state.env.primitives[i].model_bind_group)
    }
    for (uint32_t i = 0; i < state.env.material_count; ++i) {
      WGPU_RELEASE_RESOURCE(BindGroup, state.env.materials[i].bind_group)
    }
    for (uint32_t i = 0; i < state.env.image_count; ++i) {
      WGPU_RELEASE_RESOURCE(TextureView, state.env.images[i].view)
      WGPU_RELEASE_RESOURCE(Texture, state.env.images[i].texture)
    }
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.env_default_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.env_default_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.env_sampler)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.env_material_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.env_model_bgl)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.env_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.env_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Metaballs",
    .sample_count   = SAMPLE_COUNT,
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
static const char* metaball_vertex_shader_wgsl = CODE(
  struct ProjectionUniforms {
    matrix : mat4x4f,
    inverseMatrix : mat4x4f,
    outputSize : vec2f,
    zNear : f32,
    zFar : f32,
  }
  @group(0) @binding(0) var<uniform> projection : ProjectionUniforms;

  struct ViewUniforms {
    matrix : mat4x4f,
    position : vec3f,
    time : f32,
  }
  @group(0) @binding(1) var<uniform> view : ViewUniforms;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
  }

  struct VertexOutput {
    @location(0) uvX : vec2f,
    @location(1) uvY : vec2f,
    @location(2) uvZ : vec2f,
    @location(3) normal : vec3f,
    @builtin(position) position : vec4f,
  }

  @vertex
  fn vertexMain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    let worldPosition = input.position;
    let flow = vec3f(sin(view.time * 0.0001), cos(view.time * 0.0004), sin(view.time * 0.00007));

    output.normal = input.normal;
    output.uvX = worldPosition.yz + flow.yz;
    output.uvY = worldPosition.xz + flow.xz;
    output.uvZ = worldPosition.xy + flow.xy;

    output.position = projection.matrix * view.matrix * vec4f(input.position, 1);
    return output;
  }
);

static const char* metaball_fragment_shader_wgsl = CODE(
  fn linearTosRGB(linear : vec3f) -> vec3f {
    if (all(linear <= vec3(0.0031308))) {
      return linear * 12.92;
    }
    return (pow(abs(linear), vec3(1.0/2.4)) * 1.055) - vec3(0.055);
  }

  @group(1) @binding(0) var baseSampler : sampler;
  @group(1) @binding(1) var baseTexture : texture_2d<f32>;

  struct FragInput {
    @location(0) uvX : vec2f,
    @location(1) uvY : vec2f,
    @location(2) uvZ : vec2f,
    @location(3) normal : vec3f,
  }

  @fragment
  fn fragmentMain(input : FragInput) -> @location(0) vec4f {
    let blending = normalize(max(abs(input.normal), vec3f(0.00001)));

    let xTex = textureSample(baseTexture, baseSampler, input.uvX);
    let yTex = textureSample(baseTexture, baseSampler, input.uvY);
    let zTex = textureSample(baseTexture, baseSampler, input.uvZ);

    let tex = xTex * blending.x + yTex * blending.y + zTex * blending.z;

    return vec4f(linearTosRGB(tex.rgb), 1);
  }
);

/* Light sprite shaders (from light-sprite.js) */

static const char* light_sprite_vertex_shader_wgsl = CODE(
  var<private> pos : array<vec2f, 4> = array<vec2f, 4>(
    vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, -1.0)
  );

  struct ProjectionUniforms {
    matrix : mat4x4f,
    inverseMatrix : mat4x4f,
    outputSize : vec2f,
    zNear : f32,
    zFar : f32,
  }
  @group(0) @binding(0) var<uniform> projection : ProjectionUniforms;

  struct ViewUniforms {
    matrix : mat4x4f,
    position : vec3f,
    time : f32,
  }
  @group(0) @binding(1) var<uniform> view : ViewUniforms;

  struct Light {
    position : vec3f,
    range : f32,
    color : vec3f,
    intensity : f32,
  }

  struct GlobalLightUniforms {
    ambient : vec3f,
    lightCount : u32,
    lights : array<Light>,
  }
  @group(0) @binding(2) var<storage> globalLights : GlobalLightUniforms;

  struct VertexInput {
    @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) instanceIndex : u32,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) localPos : vec2f,
    @location(1) color: vec3f,
  }

  @vertex
  fn vertexMain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;

    output.localPos = pos[input.vertexIndex];
    output.color = globalLights.lights[input.instanceIndex].color
                   * globalLights.lights[input.instanceIndex].intensity;
    let worldPos = vec3(output.localPos, 0.0)
                   * globalLights.lights[input.instanceIndex].range * 0.025;

    var bbModelViewMatrix : mat4x4f;
    bbModelViewMatrix[3] = vec4(
      globalLights.lights[input.instanceIndex].position, 1.0);
    bbModelViewMatrix = view.matrix * bbModelViewMatrix;
    bbModelViewMatrix[0][0] = 1.0;
    bbModelViewMatrix[0][1] = 0.0;
    bbModelViewMatrix[0][2] = 0.0;

    bbModelViewMatrix[1][0] = 0.0;
    bbModelViewMatrix[1][1] = 1.0;
    bbModelViewMatrix[1][2] = 0.0;

    bbModelViewMatrix[2][0] = 0.0;
    bbModelViewMatrix[2][1] = 0.0;
    bbModelViewMatrix[2][2] = 1.0;

    output.position = projection.matrix * bbModelViewMatrix
                      * vec4(worldPos, 1.0);
    return output;
  }
);

static const char* light_sprite_fragment_shader_wgsl = CODE(
  fn linearTosRGB(linear : vec3f) -> vec3f {
    if (all(linear <= vec3(0.0031308))) {
      return linear * 12.92;
    }
    return (pow(abs(linear), vec3(1.0/2.4)) * 1.055) - vec3(0.055);
  }

  struct FragmentInput {
    @location(0) localPos : vec2f,
    @location(1) color: vec3f,
  }

  @fragment
  fn fragmentMain(input : FragmentInput) -> @location(0) vec4f {
    let distToCenter = length(input.localPos);
    let fade = (1.0 - distToCenter) * (1.0 / (distToCenter * distToCenter));
    return vec4(linearTosRGB(input.color * fade), fade);
  }
);

/* -------------------------------------------------------------------------- *
 * Environment shaders (PBR-like with point light iteration)
 * -------------------------------------------------------------------------- */

static const char* env_vertex_shader_wgsl = CODE(
  struct ProjectionUniforms {
    matrix : mat4x4f,
    inverseMatrix : mat4x4f,
    outputSize : vec2f,
    zNear : f32,
    zFar : f32,
  }
  @group(0) @binding(0) var<uniform> projection : ProjectionUniforms;

  struct ViewUniforms {
    matrix : mat4x4f,
    position : vec3f,
    time : f32,
  }
  @group(0) @binding(1) var<uniform> view : ViewUniforms;

  struct ModelUniforms {
    matrix : mat4x4f,
  }
  @group(2) @binding(0) var<uniform> model : ModelUniforms;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal : vec3f,
    @location(2) texcoord : vec2f,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPos : vec3f,
    @location(1) normal : vec3f,
    @location(2) texcoord : vec2f,
  }

  @vertex
  fn vertexMain(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    let worldPosition = (model.matrix * vec4f(input.position, 1.0)).xyz;
    output.worldPos = worldPosition;
    output.normal = normalize((model.matrix * vec4f(input.normal, 0.0)).xyz);
    output.texcoord = input.texcoord;
    output.position = projection.matrix * view.matrix * vec4f(worldPosition, 1.0);
    return output;
  }
);

static const char* env_fragment_shader_wgsl = CODE(
  const PI = 3.14159265359;

  fn linearTosRGB(linear : vec3f) -> vec3f {
    if (all(linear <= vec3(0.0031308))) {
      return linear * 12.92;
    }
    return (pow(abs(linear), vec3(1.0/2.4)) * 1.055) - vec3(0.055);
  }

  fn sRGBToLinear(srgb : vec3f) -> vec3f {
    if (all(srgb <= vec3f(0.04045))) {
      return srgb / vec3f(12.92);
    }
    return pow((srgb + vec3f(0.055)) / vec3f(1.055), vec3f(2.4));
  }

  struct Light {
    position : vec3f,
    range : f32,
    color : vec3f,
    intensity : f32,
  }

  struct GlobalLightUniforms {
    ambient : vec3f,
    lightCount : u32,
    lights : array<Light>,
  }
  @group(0) @binding(2) var<storage> globalLights : GlobalLightUniforms;

  struct MaterialUniforms {
    baseColorFactor : vec4f,
    metallicRoughnessFactor : vec2f,
    emissiveFactor : vec3f,
  }
  @group(1) @binding(0) var<uniform> material : MaterialUniforms;
  @group(1) @binding(1) var materialSampler : sampler;
  @group(1) @binding(2) var baseColorTexture : texture_2d<f32>;

  struct FragInput {
    @location(0) worldPos : vec3f,
    @location(1) normal : vec3f,
    @location(2) texcoord : vec2f,
  }

  fn rangeAttenuation(range : f32, distance : f32) -> f32 {
    if (range <= 0.0) {
      return 1.0 / pow(distance, 2.0);
    }
    let s = distance / range;
    if (s > 1.0) {
      return 0.0;
    }
    let s2 = s * s;
    return (1.0 - s2) * (1.0 - s2) / (1.0 + 4.0 * s);
  }

  @fragment
  fn fragmentMain(input : FragInput) -> @location(0) vec4f {
    let baseColor = textureSample(baseColorTexture, materialSampler, input.texcoord);
    let albedo = sRGBToLinear(baseColor.rgb) * material.baseColorFactor.rgb;
    let N = normalize(input.normal);

    /* Accumulate point light contributions (simplified Lambert diffuse) */
    var Lo = vec3f(0.0);
    for (var i : u32 = 0u; i < globalLights.lightCount; i = i + 1u) {
      let light = globalLights.lights[i];
      let lightDir = light.position - input.worldPos;
      let distance = length(lightDir);
      let L = normalize(lightDir);
      let NdotL = max(dot(N, L), 0.0);

      let attenuation = rangeAttenuation(light.range, distance);
      let radiance = light.color * light.intensity * attenuation;

      Lo = Lo + (albedo / PI) * radiance * NdotL;
    }

    /* Ambient */
    let ambient = globalLights.ambient * albedo;
    let color = ambient + Lo;

    return vec4f(linearTosRGB(color), baseColor.a * material.baseColorFactor.a);
  }
);
// clang-format on
