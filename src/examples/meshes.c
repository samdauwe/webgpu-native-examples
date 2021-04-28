#include "meshes.h"

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
  if (stanford_dragon_mesh->vertices.count == 0
      || stanford_dragon_mesh->triangles.count == 0) {
    return;
  }

  // Vertices and indices count
  printf("nvertices=%ld\nntriangles=%ld\n",
         stanford_dragon_mesh->vertices.count,
         stanford_dragon_mesh->triangles.count);
  // Vertices data
  for (uint32_t i = 0; i < stanford_dragon_mesh->vertices.count; ++i) {
    printf("%g ", stanford_dragon_mesh->vertices.data[i][0]);
    printf("%g ", stanford_dragon_mesh->vertices.data[i][1]);
    printf("%g\n", stanford_dragon_mesh->vertices.data[i][2]);
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
  ASSERT((size_t)vertex_index < pdata->vertices.count);
  ASSERT(index_data >= 0 && index_data < 3);
  pdata->vertices.data[vertex_index][index_data] = value;

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

  stanford_dragon_mesh->vertices.count
    = ply_set_read_cb(ply, "vertex", "x", vertex_cb, stanford_dragon_mesh, 0);
  ply_set_read_cb(ply, "vertex", "y", vertex_cb, stanford_dragon_mesh, 1);
  ply_set_read_cb(ply, "vertex", "z", vertex_cb, stanford_dragon_mesh, 2);
  stanford_dragon_mesh->triangles.count = ply_set_read_cb(
    ply, "face", "vertex_indices", face_cb, stanford_dragon_mesh, 0);
  if (!ply_read(ply)) {
    return 1;
  }
  ply_close(ply);

  ASSERT(stanford_dragon_mesh->vertices.count == POSITION_COUNT_RES_4);
  ASSERT(stanford_dragon_mesh->triangles.count == CELL_COUNT_RES_4);

#if STANFORD_DRAGON_MESH_DEBUG_PRINT
  debug_print(stanford_dragon_mesh);
#endif

  return 0;
}
