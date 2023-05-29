#ifndef MESHES_H
#define MESHES_H

#include <stdint.h>

/* -------------------------------------------------------------------------- *
 * Plane mesh
 * -------------------------------------------------------------------------- */

#define MAX_PLANE_VERTEX_COUNT 1024 * 1024 * 4

typedef struct plane_vertex_t {
  float position[3];
  float normal[3];
  float uv[2];
} plane_vertex_t;

typedef struct plane_mesh_t {
  float width;
  float height;
  uint32_t rows;
  uint32_t columns;
  uint64_t vertex_count;
  uint64_t index_count;
  plane_vertex_t vertices[MAX_PLANE_VERTEX_COUNT];
  uint32_t indices[MAX_PLANE_VERTEX_COUNT * 6];
} plane_mesh_t;

typedef struct plane_mesh_init_options_t {
  float width;
  float height;
  uint32_t rows;
  uint32_t columns;
} plane_mesh_init_options_t;

void plane_mesh_init(plane_mesh_t* plane_mesh,
                     plane_mesh_init_options_t* options);

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
 * Sphere mesh
 * -------------------------------------------------------------------------- */

typedef struct sphere_mesh_t {
  struct {
    float* data;
    uint64_t size;
  } vertices;
  struct {
    uint16_t* data;
    uint64_t size;
  } indices;
} sphere_mesh_t;

typedef struct sphere_mesh_layout_t {
  uint32_t vertex_stride;
  uint32_t positions_offset;
  uint32_t normal_offset;
  uint32_t uv_offset;
} sphere_mesh_layout_t;

void sphere_mesh_layout_init(sphere_mesh_layout_t* sphere_layout);
void sphere_mesh_init(sphere_mesh_t* sphere_mesh, float radius,
                      uint32_t width_segments, uint32_t height_segments,
                      float randomness);
void sphere_mesh_destroy(sphere_mesh_t* sphere_mesh);

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
