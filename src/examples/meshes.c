#include "meshes.h"

#include <cglm/cglm.h>
#include <string.h>

#include "../core/log.h"
#include "../core/macro.h"

/* -------------------------------------------------------------------------- *
 * Cube mesh
 * -------------------------------------------------------------------------- */

void cube_mesh_init(cube_mesh_t* cube_mesh)
{
  (*cube_mesh) = (cube_mesh_t) {
    .vertex_size = 4 * 10, // Byte size of one cube vertex.
    .position_offset = 0,
    .color_offset = 4 * 4, // Byte offset of cube vertex color attribute.
    .uv_offset = 4 * 8,
    .vertex_count = 36,
    .vertex_array = {
      // float4 position, float4 color, float2 uv,
      1,  -1, 1,  1, 1, 0, 1, 1, 1, 1, //
      -1, -1, 1,  1, 0, 0, 1, 1, 0, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, //
      1,  -1, -1, 1, 1, 0, 0, 1, 1, 0, //
      1,  -1, 1,  1, 1, 0, 1, 1, 1, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, //

      1,  1,  1,  1, 1, 1, 1, 1, 1, 1, //
      1,  -1, 1,  1, 1, 0, 1, 1, 0, 1, //
      1,  -1, -1, 1, 1, 0, 0, 1, 0, 0, //
      1,  1,  -1, 1, 1, 1, 0, 1, 1, 0, //
      1,  1,  1,  1, 1, 1, 1, 1, 1, 1, //
      1,  -1, -1, 1, 1, 0, 0, 1, 0, 0, //

      -1, 1,  1,  1, 0, 1, 1, 1, 1, 1, //
      1,  1,  1,  1, 1, 1, 1, 1, 0, 1, //
      1,  1,  -1, 1, 1, 1, 0, 1, 0, 0, //
      -1, 1,  -1, 1, 0, 1, 0, 1, 1, 0, //
      -1, 1,  1,  1, 0, 1, 1, 1, 1, 1, //
      1,  1,  -1, 1, 1, 1, 0, 1, 0, 0, //

      -1, -1, 1,  1, 0, 0, 1, 1, 1, 1, //
      -1, 1,  1,  1, 0, 1, 1, 1, 0, 1, //
      -1, 1,  -1, 1, 0, 1, 0, 1, 0, 0, //
      -1, -1, -1, 1, 0, 0, 0, 1, 1, 0, //
      -1, -1, 1,  1, 0, 0, 1, 1, 1, 1, //
      -1, 1,  -1, 1, 0, 1, 0, 1, 0, 0, //

      1,  1,  1,  1, 1, 1, 1, 1, 1, 1, //
      -1, 1,  1,  1, 0, 1, 1, 1, 0, 1, //
      -1, -1, 1,  1, 0, 0, 1, 1, 0, 0, //
      -1, -1, 1,  1, 0, 0, 1, 1, 0, 0, //
      1,  -1, 1,  1, 1, 0, 1, 1, 1, 0, //
      1,  1,  1,  1, 1, 1, 1, 1, 1, 1, //

      1,  -1, -1, 1, 1, 0, 0, 1, 1, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1, 0, 1, //
      -1, 1,  -1, 1, 0, 1, 0, 1, 0, 0, //
      1,  1,  -1, 1, 1, 1, 0, 1, 1, 0, //
      1,  -1, -1, 1, 1, 0, 0, 1, 1, 1, //
      -1, 1,  -1, 1, 0, 1, 0, 1, 0, 0, //
    },
  };
}

/* -------------------------------------------------------------------------- *
 * Stanford Dragon
 * -------------------------------------------------------------------------- */

#define STANFORD_DRAGON_MESH_DEBUG_PRINT 0

#if STANFORD_DRAGON_MESH_DEBUG_PRINT
static void debug_print(stanford_dragon_mesh_t* stanford_dragon_mesh)
{
  ASSERT(stanford_dragon_mesh);
  if (stanford_dragon_mesh->positions.count == 0
      || stanford_dragon_mesh->triangles.count == 0) {
    return;
  }

  // Vertices and indices count
  printf("nvertices=%ld\nntriangles=%ld\n",
         stanford_dragon_mesh->positions.count,
         stanford_dragon_mesh->triangles.count);
  // Vertices data
  for (uint32_t i = 0; i < stanford_dragon_mesh->positions.count; ++i) {
    printf("%g ", stanford_dragon_mesh->positions.data[i][0]);
    printf("%g ", stanford_dragon_mesh->positions.data[i][1]);
    printf("%g\n", stanford_dragon_mesh->positions.data[i][2]);
  }
  // indices data
  for (uint32_t i = 0; i < stanford_dragon_mesh->triangles.count; ++i) {
    printf("%hu ", stanford_dragon_mesh->triangles.data[i][0]);
    printf("%hu ", stanford_dragon_mesh->triangles.data[i][1]);
    printf("%hu\n", stanford_dragon_mesh->triangles.data[i][2]);
  }
}
#endif

static int vertex_cb(p_ply_argument argument)
{
  // Vertex index
  long vertex_index;
  ply_get_argument_element(argument, NULL, &vertex_index);

  // x, y or z index
  long index_data;
  stanford_dragon_mesh_t* pdata;
  ply_get_argument_user_data(argument, (void**)&pdata, &index_data);
  ASSERT(pdata)

  // Get value
  float value = (float)ply_get_argument_value(argument);
  value *= STANFORD_DRAGON_MESH_SCALE;

  // Store value
  ASSERT((size_t)vertex_index < pdata->positions.count);
  ASSERT(index_data >= 0 && index_data < 3);
  pdata->positions.data[vertex_index][index_data] = value;

  return 1;
}

static int face_cb(p_ply_argument argument)
{
  // Face index
  long face_index;
  ply_get_argument_element(argument, NULL, &face_index);

  // Data pointer
  stanford_dragon_mesh_t* pdata;
  ply_get_argument_user_data(argument, (void**)&pdata, NULL);
  ASSERT(pdata);

  // Value
  long length, value_index;
  ply_get_argument_property(argument, NULL, &length, &value_index);
  if (value_index >= 0 && value_index < 3) {
    // Get value
    uint16_t value = (uint16_t)ply_get_argument_value(argument);

    // Store value
    ASSERT((size_t)face_index < pdata->triangles.count);
    pdata->triangles.data[face_index][value_index] = value;
  }

  return 1;
}

int stanford_dragon_mesh_init(stanford_dragon_mesh_t* stanford_dragon_mesh)
{
  ASSERT(stanford_dragon_mesh)

  p_ply ply = ply_open("meshes/dragon_vrip_res4.ply", NULL, 0, NULL);
  if (!ply) {
    return 1;
  }
  if (!ply_read_header(ply)) {
    return 1;
  }

  stanford_dragon_mesh->positions.count
    = ply_set_read_cb(ply, "vertex", "x", vertex_cb, stanford_dragon_mesh, 0);
  ply_set_read_cb(ply, "vertex", "y", vertex_cb, stanford_dragon_mesh, 1);
  ply_set_read_cb(ply, "vertex", "z", vertex_cb, stanford_dragon_mesh, 2);
  stanford_dragon_mesh->triangles.count = ply_set_read_cb(
    ply, "face", "vertex_indices", face_cb, stanford_dragon_mesh, 0);
  stanford_dragon_mesh->normals.count = stanford_dragon_mesh->positions.count;
  memset(stanford_dragon_mesh->normals.data, 0,
         sizeof(stanford_dragon_mesh->normals.data)); // Initialize to zero
  stanford_dragon_mesh->uvs.count = stanford_dragon_mesh->positions.count;
  memset(stanford_dragon_mesh->uvs.data, 0,
         sizeof(stanford_dragon_mesh->uvs.data)); // Initialize to zero
  if (!ply_read(ply)) {
    return 1;
  }
  ply_close(ply);

  ASSERT(stanford_dragon_mesh->positions.count == POSITION_COUNT_RES_4);
  ASSERT(stanford_dragon_mesh->triangles.count == CELL_COUNT_RES_4);

#if STANFORD_DRAGON_MESH_DEBUG_PRINT
  debug_print(stanford_dragon_mesh);
#endif

  // Compute surface normals
  stanford_dragon_mesh_compute_normals(stanford_dragon_mesh);

  // Compute some easy uvs for testing
  stanford_dragon_mesh_compute_projected_plane_uvs(stanford_dragon_mesh,
                                                   ProjectedPlane_XY);

  return 0;
}

void stanford_dragon_mesh_compute_normals(
  stanford_dragon_mesh_t* stanford_dragon_mesh)
{
  float(*positions)[3]          = stanford_dragon_mesh->positions.data;
  float(*normals)[3]            = stanford_dragon_mesh->normals.data;
  const uint64_t triangle_count = stanford_dragon_mesh->triangles.count;
  uint16_t* triangle            = NULL;
  vec3 *p0 = NULL, *p1 = NULL, *p2 = NULL;
  vec3 v0, v1, norm;
  uint16_t i0, i1, i2;
  for (uint64_t i = 0; i < triangle_count; ++i) {
    triangle = stanford_dragon_mesh->triangles.data[i];
    i0       = triangle[0];
    i1       = triangle[1];
    i2       = triangle[2];

    p0 = &positions[i0];
    p1 = &positions[i1];
    p2 = &positions[i2];

    glm_vec3_sub(*p1, *p0, v0);
    glm_vec3_sub(*p2, *p0, v1);

    glm_vec3_normalize(v0);
    glm_vec3_normalize(v1);
    glm_vec3_cross(v0, v1, norm);

    // Accumulate the normals.
    glm_vec3_add(normals[i0], norm, normals[i0]);
    glm_vec3_add(normals[i1], norm, normals[i1]);
    glm_vec3_add(normals[i2], norm, normals[i2]);
  }
  // Normalize accumulated normals.
  for (uint16_t i = 0; i < stanford_dragon_mesh->normals.count; ++i) {
    glm_vec3_normalize(normals[i]);
  }
}

static const uint32_t projected_plane2_ids[3][2] = {
  {0, 1}, // XY
  {0, 2}, // XZ
  {1, 2}, // YZ
};

void stanford_dragon_mesh_compute_projected_plane_uvs(
  stanford_dragon_mesh_t* stanford_dragon_mesh,
  projected_plane_enum projected_plane)
{
  const uint32_t* idxs = projected_plane2_ids[(uint32_t)projected_plane];
  float(*uvs)[2]       = stanford_dragon_mesh->uvs.data;
  float extent_min[2]  = {FLT_MAX, FLT_MAX};
  float extent_max[2]  = {FLT_MIN, FLT_MIN};
  vec3* pos            = NULL;
  for (uint64_t i = 0; i < stanford_dragon_mesh->positions.count; ++i) {
    // Simply project to the selected plane
    pos       = &stanford_dragon_mesh->positions.data[i];
    uvs[i][0] = (*pos)[idxs[0]];
    uvs[i][1] = (*pos)[idxs[1]];

    extent_min[0] = MIN((*pos)[idxs[0]], extent_min[0]);
    extent_min[1] = MIN((*pos)[idxs[1]], extent_min[1]);
    extent_max[0] = MAX((*pos)[idxs[0]], extent_max[0]);
    extent_max[1] = MAX((*pos)[idxs[1]], extent_max[1]);
  }
  vec2* uv = NULL;
  for (uint64_t i = 0; i < stanford_dragon_mesh->uvs.count; ++i) {
    uv       = &stanford_dragon_mesh->uvs.data[i];
    (*uv)[0] = ((*uv)[0] - extent_min[0]) / (extent_max[0] - extent_min[0]);
    (*uv)[1] = ((*uv)[1] - extent_min[1]) / (extent_max[1] - extent_min[1]);
  }
}
