#ifndef MESHES_H
#define MESHES_H

#include <stdint.h>

/* -------------------------------------------------------------------------- *
 * Cube mesh
 * -------------------------------------------------------------------------- */

typedef struct cube_mesh_t {
  uint64_t vertex_size;
  uint64_t position_offset;
  uint64_t color_offset;
  uint64_t uv_offset;
  uint64_t vertex_count;
  float vertex_array[360];
} cube_mesh_t;

void cube_mesh_init(cube_mesh_t* cube_mesh);

/* -------------------------------------------------------------------------- *
 * Indexed cube mesh
 * -------------------------------------------------------------------------- */

typedef struct indexed_cube_mesh_t {
  uint64_t vertex_count;
  uint64_t index_count;
  uint64_t color_count;
  float vertex_array[3 * 8];
  uint32_t index_array[2 * 3 * 6];
  uint8_t color_array[4 * 8];
} indexed_cube_mesh_t;

void indexed_cube_mesh_init(indexed_cube_mesh_t* cube_mesh);

/* -------------------------------------------------------------------------- *
 * Stanford Dragon
 * -------------------------------------------------------------------------- */

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#include <rply.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define POSITION_COUNT_RES_4 5205
#define CELL_COUNT_RES_4 11102
#define STANFORD_DRAGON_MESH_SCALE 500

typedef struct stanford_dragon_mesh_t {
  struct {
    float data[POSITION_COUNT_RES_4][3];
    uint64_t count; // number of vertices (should be 5205)
  } positions;
  struct {
    uint16_t data[CELL_COUNT_RES_4][3];
    uint64_t count; // number of faces (should be 11102)
  } triangles;      // triangles
  struct {
    float data[POSITION_COUNT_RES_4][3];
    uint64_t count; // number of normals (should be 5205)
  } normals;
  struct {
    float data[POSITION_COUNT_RES_4][2];
    uint64_t count; // number of uvs (should be 5205)
  } uvs;
} stanford_dragon_mesh_t;

/**
 * @brief Loads the 'stanford-dragon' PLY file (quality level 4).
 * @see https://github.com/hughsk/stanford-dragon
 *
 * Uses: ANSI C Library for PLY file format input and output
 * @see http://w3.impa.br/~diego/software/rply/
 */
int stanford_dragon_mesh_init(stanford_dragon_mesh_t* stanford_dragon_mesh);

typedef enum projected_plane_enum {
  ProjectedPlane_XY = 0,
  ProjectedPlane_XZ = 1,
  ProjectedPlane_YZ = 2,
} projected_plane_enum;

/**
 * @brief Computes surface normals.
 * @param stanford_dragon_mesh mesh object
 */
void stanford_dragon_mesh_compute_normals(
  stanford_dragon_mesh_t* stanford_dragon_mesh);

/**
 * @brief Computes some easy uvs for testing.
 * @param stanford_dragon_mesh mesh object
 * @param projected_plane plane to project to
 */
void stanford_dragon_mesh_compute_projected_plane_uvs(
  stanford_dragon_mesh_t* stanford_dragon_mesh,
  projected_plane_enum projected_plane);

#endif /* MESHES_H */
