#include "meshes.h"

#include <cJSON.h>
#include <cglm/cglm.h>
#include <string.h>

#include "../core/file.h"
#include "../core/macro.h"
#include "../core/math.h"

/* -------------------------------------------------------------------------- *
 * Plane mesh
 * -------------------------------------------------------------------------- */

void plane_mesh_generate_vertices(plane_mesh_t* plane_mesh)
{
  plane_mesh->vertex_count = 0;
  const float row_height   = plane_mesh->height / (float)plane_mesh->rows;
  const float col_width    = plane_mesh->width / (float)plane_mesh->columns;
  float x = 0.0f, y = 0.0f;
  for (uint32_t row = 0; row <= plane_mesh->rows; ++row) {
    y = row * row_height;

    for (uint32_t col = 0; col <= plane_mesh->columns; ++col) {
      x = col * col_width;

      plane_vertex_t* vertex = &plane_mesh->vertices[plane_mesh->vertex_count];
      {
        // Vertex position
        vertex->position[0] = x;
        vertex->position[1] = y;
        vertex->position[2] = 0.0f;

        // Vertex normal
        vertex->normal[0] = 0.0f;
        vertex->normal[1] = 0.0f;
        vertex->normal[2] = 1.0f;

        // Vertex uv
        vertex->uv[0] = col / plane_mesh->columns;
        vertex->uv[1] = 1 - row / plane_mesh->rows;
      }
      ++plane_mesh->vertex_count;
    }
  }
}

void plane_mesh_generate_indices(plane_mesh_t* plane_mesh)
{
  plane_mesh->index_count       = 0;
  const uint32_t columns_offset = plane_mesh->columns + 1;
  uint32_t left_bottom = 0, right_bottom = 0, left_up = 0, right_up = 0;
  for (uint32_t row = 0; row < plane_mesh->rows; ++row) {
    for (uint32_t col = 0; col < plane_mesh->columns; ++col) {
      left_bottom  = columns_offset * row + col;
      right_bottom = columns_offset * row + (col + 1);
      left_up      = columns_offset * (row + 1) + col;
      right_up     = columns_offset * (row + 1) + (col + 1);

      // CCW frontface
      plane_mesh->indices[plane_mesh->index_count++] = left_up;
      plane_mesh->indices[plane_mesh->index_count++] = left_bottom;
      plane_mesh->indices[plane_mesh->index_count++] = right_bottom;

      plane_mesh->indices[plane_mesh->index_count++] = right_up;
      plane_mesh->indices[plane_mesh->index_count++] = left_up;
      plane_mesh->indices[plane_mesh->index_count++] = right_bottom;
    }
  }
}

void plane_mesh_init(plane_mesh_t* plane_mesh,
                     plane_mesh_init_options_t* options)
{
  // Initialize dimensions
  plane_mesh->width   = options ? options->width : 1.0f;
  plane_mesh->height  = options ? options->height : 1.0f;
  plane_mesh->rows    = options ? options->rows : 1;
  plane_mesh->columns = options ? options->columns : 1;

  ASSERT((plane_mesh->rows + 1) * (plane_mesh->columns + 1)
         < MAX_PLANE_VERTEX_COUNT)

  // Generate vertices and indices
  plane_mesh_generate_vertices(plane_mesh);
  plane_mesh_generate_indices(plane_mesh);
}

/* -------------------------------------------------------------------------- *
 * Box mesh
 * -------------------------------------------------------------------------- */

void box_mesh_create_with_tangents(box_mesh_t* box_mesh, float width,
                                   float height, float depth)
{
  //    __________
  //   /         /|      y
  //  /   +y    / |      ^
  // /_________/  |      |
  // |         |+x|      +---> x
  // |   +z    |  |     /
  // |         | /     z
  // |_________|/
  //
  const uint8_t p_x = 0; /* +x */
  const uint8_t n_x = 1; /* -x */
  const uint8_t p_y = 2; /* +y */
  const uint8_t n_y = 3; /* -y */
  const uint8_t p_z = 4; /* +z */
  const uint8_t n_z = 5; /* -z */

  struct {
    uint8_t tangent;
    uint8_t bitangent;
    uint8_t normal;
  } faces[BOX_MESH_FACES_COUNT] = {
    [0] = { .tangent = n_z, .bitangent = p_y, .normal = p_x, },
    [1] = { .tangent = p_z, .bitangent = p_y, .normal = n_x, },
    [2] = { .tangent = p_x, .bitangent = n_z, .normal = p_y, },
    [3] = { .tangent = p_x, .bitangent = p_z, .normal = n_y, },
    [4] = { .tangent = p_x, .bitangent = p_y, .normal = p_z, },
    [5] = { .tangent = n_x, .bitangent = p_y, .normal = n_z, },
  };

  uint16_t vertices_per_side = BOX_MESH_VERTICES_PER_SIDE;
  box_mesh->vertex_count     = BOX_MESH_VERTICES_COUNT;
  box_mesh->index_count      = BOX_MESH_INDICES_COUNT;
  box_mesh->vertex_stride    = BOX_MESH_VERTEX_STRIDE;

  const float half_vecs[BOX_MESH_FACES_COUNT][3] = {
    {+width / 2.0f, 0.0f, 0.0f},  /* +x */
    {-width / 2.0f, 0.0f, 0.0f},  /* -x */
    {0.0f, +height / 2.0f, 0.0f}, /* +y */
    {0.0f, -height / 2.0f, 0.0f}, /* -y */
    {0.0f, 0.0f, +depth / 2.0f},  /* +z */
    {0.0f, 0.0f, -depth / 2.0f},  /* -z */
  };

  uint32_t vertex_offset = 0;
  uint32_t index_offset  = 0;
  for (uint8_t face_index = 0; face_index < ARRAY_SIZE(faces); ++face_index) {
    const float* tangent   = half_vecs[faces[face_index].tangent];
    const float* bitangent = half_vecs[faces[face_index].bitangent];
    const float* normal    = half_vecs[faces[face_index].normal];

    for (uint8_t u = 0; u < 2; ++u) {
      for (uint8_t v = 0; v < 2; ++v) {
        for (uint8_t i = 0; i < 3; ++i) {
          box_mesh->vertex_array[vertex_offset++]
            = normal[i] + (u == 0 ? -1 : 1) * tangent[i]
              + (v == 0 ? -1 : 1) * bitangent[i];
        }
        for (uint8_t i = 0; i < 3; i++) {
          box_mesh->vertex_array[vertex_offset++] = normal[i];
        }
        box_mesh->vertex_array[vertex_offset++] = u;
        box_mesh->vertex_array[vertex_offset++] = v;
        for (uint8_t i = 0; i < 3; ++i) {
          box_mesh->vertex_array[vertex_offset++] = tangent[i];
        }
        for (uint8_t i = 0; i < 3; i++) {
          box_mesh->vertex_array[vertex_offset++] = bitangent[i];
        }
      }
    }

    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 0;
    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 2;
    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 1;

    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 2;
    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 3;
    box_mesh->index_array[index_offset++] = face_index * vertices_per_side + 1;
  }
}

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
      1, -1, 1, 1,   1, 0, 1, 1,  0, 1, //
      -1, -1, 1, 1,  0, 0, 1, 1,  1, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1,  1, 0, //
      1, -1, -1, 1,  1, 0, 0, 1,  0, 0, //
      1, -1, 1, 1,   1, 0, 1, 1,  0, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1,  1, 0, //

      1, 1, 1, 1,    1, 1, 1, 1,  0, 1, //
      1, -1, 1, 1,   1, 0, 1, 1,  1, 1, //
      1, -1, -1, 1,  1, 0, 0, 1,  1, 0, //
      1, 1, -1, 1,   1, 1, 0, 1,  0, 0, //
      1, 1, 1, 1,    1, 1, 1, 1,  0, 1, //
      1, -1, -1, 1,  1, 0, 0, 1,  1, 0, //

      -1, 1, 1, 1,   0, 1, 1, 1,  0, 1, //
      1, 1, 1, 1,    1, 1, 1, 1,  1, 1, //
      1, 1, -1, 1,   1, 1, 0, 1,  1, 0, //
      -1, 1, -1, 1,  0, 1, 0, 1,  0, 0, //
      -1, 1, 1, 1,   0, 1, 1, 1,  0, 1, //
      1, 1, -1, 1,   1, 1, 0, 1,  1, 0, //

      -1, -1, 1, 1,  0, 0, 1, 1,  0, 1, //
      -1, 1, 1, 1,   0, 1, 1, 1,  1, 1, //
      -1, 1, -1, 1,  0, 1, 0, 1,  1, 0, //
      -1, -1, -1, 1, 0, 0, 0, 1,  0, 0, //
      -1, -1, 1, 1,  0, 0, 1, 1,  0, 1, //
      -1, 1, -1, 1,  0, 1, 0, 1,  1, 0, //

      1, 1, 1, 1,    1, 1, 1, 1,  0, 1, //
      -1, 1, 1, 1,   0, 1, 1, 1,  1, 1, //
      -1, -1, 1, 1,  0, 0, 1, 1,  1, 0, //
      -1, -1, 1, 1,  0, 0, 1, 1,  1, 0, //
      1, -1, 1, 1,   1, 0, 1, 1,  0, 0, //
      1, 1, 1, 1,    1, 1, 1, 1,  0, 1, //

      1, -1, -1, 1,  1, 0, 0, 1,  0, 1, //
      -1, -1, -1, 1, 0, 0, 0, 1,  1, 1, //
      -1, 1, -1, 1,  0, 1, 0, 1,  1, 0, //
      1, 1, -1, 1,   1, 1, 0, 1,  0, 0, //
      1, -1, -1, 1,  1, 0, 0, 1,  0, 1, //
      -1, 1, -1, 1,  0, 1, 0, 1,  1, 0, //
    },
  };
}

/* -------------------------------------------------------------------------- *
 * Indexed cube mesh
 * -------------------------------------------------------------------------- */

void indexed_cube_mesh_init(indexed_cube_mesh_t* cube_mesh)
{
  (*cube_mesh) = (indexed_cube_mesh_t) {
    .vertex_count = 8,
    .index_count = 2 * 3 * 6,
    .color_count = 8,
    .vertex_array = {
      -1.0f, -1.0f, -1.0f, // 0
       1.0f, -1.0f, -1.0f, // 1
       1.0f, -1.0f,  1.0f, // 2
      -1.0f, -1.0f,  1.0f, // 3
      -1.0f,  1.0f, -1.0f, // 4
       1.0f,  1.0f, -1.0f, // 5
       1.0f,  1.0f,  1.0f, // 6
      -1.0f,  1.0f,  1.0f, // 7
    },
    .index_array = {
      // BOTTOM
      0, 1, 2, /* */  0, 2, 3,
      // TOP
      4, 5, 6,  /* */  4, 6, 7,
      // FRONT
      3, 2, 6,  /* */  3, 6, 7,
      // BACK
      1, 0, 4,  /* */  1, 4, 5,
      // LEFT
      3, 0, 7,  /* */  0, 7, 4,
      // RIGHT
      2, 1, 6,  /* */  1, 6, 5,
    },
  };

  float* vertices = cube_mesh->vertex_array;
  uint8_t* colors = cube_mesh->color_array;
  float x = 0.0f, y = 0.0f, z = 0.0f;
  for (uint8_t i = 0; i < 8; ++i) {
    x = vertices[3 * i + 0];
    y = vertices[3 * i + 1];
    z = vertices[3 * i + 2];

    colors[4 * i + 0] = 255 * (x + 1) / 2;
    colors[4 * i + 1] = 255 * (y + 1) / 2;
    colors[4 * i + 2] = 255 * (z + 1) / 2;
    colors[4 * i + 3] = 255;
  }
}

/* -------------------------------------------------------------------------- *
 * Sphere mesh
 * -------------------------------------------------------------------------- */

void sphere_mesh_layout_init(sphere_mesh_layout_t* sphere_layout)
{
  sphere_layout->vertex_stride    = 8 * 4;
  sphere_layout->positions_offset = 0;
  sphere_layout->normal_offset    = 3 * 4;
  sphere_layout->uv_offset        = 6 * 4;
}

// Borrowed and simplified from:
// https://github.com/mrdoob/three.js/blob/master/src/geometries/SphereGeometry.js
void sphere_mesh_init(sphere_mesh_t* sphere_mesh, float radius,
                      uint32_t width_segments, uint32_t height_segments,
                      float randomness)
{
  width_segments  = MAX(3u, floor(width_segments));
  height_segments = MAX(2u, floor(height_segments));

  uint32_t vertices_count
    = (width_segments + 1) * (height_segments + 1) * (3 + 3 + 2);
  float* vertices = (float*)malloc(vertices_count * sizeof(float));

  uint32_t indices_count = width_segments * height_segments * 6;
  uint16_t* indices      = (uint16_t*)malloc(indices_count * sizeof(uint16_t));

  vec3 first_vertex = GLM_VEC3_ZERO_INIT;
  vec3 vertex       = GLM_VEC3_ZERO_INIT;
  vec3 normal       = GLM_VEC3_ZERO_INIT;

  uint32_t index      = 0;
  uint32_t grid_count = height_segments + 1;
  uint16_t** grid     = (uint16_t**)malloc(grid_count * sizeof(uint16_t*));

  /* Generate vertices, normals and uvs */
  uint32_t vc = 0, ic = 0, gc = 0;
  for (uint32_t iy = 0; iy <= height_segments; ++iy) {
    uint16_t* vertices_row
      = (uint16_t*)malloc((width_segments + 1) * sizeof(uint16_t));
    uint32_t vri  = 0;
    const float v = iy / (float)height_segments;

    // special case for the poles
    float u_offset = 0.0f;
    if (iy == 0) {
      u_offset = 0.5f / width_segments;
    }
    else if (iy == height_segments) {
      u_offset = -0.5f / width_segments;
    }

    for (uint32_t ix = 0; ix <= width_segments; ++ix) {
      const float u = ix / (float)width_segments;

      /* Poles should just use the same position all the way around. */
      if (ix == width_segments) {
        glm_vec3_copy(first_vertex, vertex);
      }
      else if (ix == 0 || (iy != 0 && iy != height_segments)) {
        const float rr
          = radius + (random_float() - 0.5f) * 2.0f * randomness * radius;

        /* vertex */
        vertex[0] = -rr * cos(u * PI * 2.0f) * sin(v * PI);
        vertex[1] = rr * cos(v * PI);
        vertex[2] = rr * sin(u * PI * 2.0f) * sin(v * PI);

        if (ix == 0) {
          glm_vec3_copy(vertex, first_vertex);
        }
      }
      vertices[vc++] = vertex[0];
      vertices[vc++] = vertex[1];
      vertices[vc++] = vertex[2];

      /* normal */
      glm_vec3_copy(vertex, normal);
      glm_vec3_normalize(normal);
      vertices[vc++] = normal[0];
      vertices[vc++] = normal[1];
      vertices[vc++] = normal[2];

      /* uv */
      vertices[vc++]      = u + u_offset;
      vertices[vc++]      = 1 - v;
      vertices_row[vri++] = index++;
    }

    grid[gc++] = vertices_row;
  }

  /* indices */
  uint16_t a = 0, b = 0, c = 0, d = 0;
  for (uint32_t iy = 0; iy < height_segments; ++iy) {
    for (uint32_t ix = 0; ix < width_segments; ++ix) {
      a = grid[iy][ix + 1];
      b = grid[iy][ix];
      c = grid[iy + 1][ix];
      d = grid[iy + 1][ix + 1];

      if (iy != 0) {
        indices[ic++] = a;
        indices[ic++] = b;
        indices[ic++] = d;
      }
      if (iy != height_segments - 1) {
        indices[ic++] = b;
        indices[ic++] = c;
        indices[ic++] = d;
      }
    }
  }

  /* Cleanup temporary grid */
  for (uint32_t gci = 0; gci < grid_count; ++gci) {
    free(grid[gci]);
  }
  free(grid);

  /* Sphere */
  memset(sphere_mesh, 0, sizeof(*sphere_mesh));
  sphere_mesh->vertices.data   = vertices;
  sphere_mesh->vertices.length = vc;
  sphere_mesh->indices.data    = indices;
  sphere_mesh->indices.length  = ic;
}

void sphere_mesh_destroy(sphere_mesh_t* sphere_mesh)
{
  if (sphere_mesh->vertices.data) {
    free(sphere_mesh->vertices.data);
  }
  if (sphere_mesh->indices.data) {
    free(sphere_mesh->indices.data);
  }
  memset(sphere_mesh, 0, sizeof(*sphere_mesh));
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
  ASSERT(pdata && (size_t)vertex_index < pdata->positions.count);
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
    ASSERT(pdata && (size_t)face_index < pdata->triangles.count);
    pdata->triangles.data[face_index][value_index] = value;
  }

  return 1;
}

int stanford_dragon_mesh_init(stanford_dragon_mesh_t* stanford_dragon_mesh)
{
  ASSERT(stanford_dragon_mesh)

  p_ply ply = ply_open("meshes/dragon_vrip_res4.ply", NULL, 0, NULL);
  if (!ply) {
    return EXIT_FAILURE;
  }
  if (!ply_read_header(ply)) {
    return EXIT_FAILURE;
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
    return EXIT_FAILURE;
  }
  ply_close(ply);

  ASSERT(stanford_dragon_mesh->positions.count
         == STANFORD_DRAGON_POSITION_COUNT_RES_4);
  ASSERT(stanford_dragon_mesh->triangles.count
         == STANFORD_DRAGON_CELL_COUNT_RES_4);

#if STANFORD_DRAGON_MESH_DEBUG_PRINT
  debug_print(stanford_dragon_mesh);
#endif

  // Compute surface normals
  stanford_dragon_mesh_compute_normals(stanford_dragon_mesh);

  // Compute some easy uvs for testing
  stanford_dragon_mesh_compute_projected_plane_uvs(stanford_dragon_mesh,
                                                   ProjectedPlane_XY);

  return EXIT_SUCCESS;
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

/* -------------------------------------------------------------------------- *
 * Utah teapot
 * -------------------------------------------------------------------------- */

int utah_teapot_mesh_init(utah_teapot_mesh_t* utah_teapot_mesh)
{
  ASSERT(utah_teapot_mesh);

  int res = EXIT_FAILURE;

  file_read_result_t file_read_result = {0};
  read_file("meshes/teapot.json", &file_read_result, true);
  const char* const json_data = (const char* const)file_read_result.data;

  const cJSON* position_array = NULL;
  const cJSON* position_item  = NULL;
  const cJSON* cell_array     = NULL;
  const cJSON* cell_item      = NULL;
  cJSON* model_json           = cJSON_Parse(json_data);
  if (model_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      fprintf(stderr, "Error before: %s\n", error_ptr);
    }
    goto load_json_end;
  }

  if (!cJSON_IsObject(model_json)
      || !cJSON_HasObjectItem(model_json, "positions")
      || !cJSON_HasObjectItem(model_json, "cells")) {
    fprintf(stderr,
            "Invalid mesh file, does not contain 'positions' array or 'cells' "
            "array\n");
    goto load_json_end;
  }

  /* Parse positions */
  {
    position_array = cJSON_GetObjectItemCaseSensitive(model_json, "positions");
    if (!cJSON_IsArray(position_array)) {
      fprintf(stderr, "Positions object item is not an array\n");
      goto load_json_end;
    }

    utah_teapot_mesh->positions.count = cJSON_GetArraySize(position_array);
    ASSERT(utah_teapot_mesh->positions.count == UTAH_TEAPOT_POSITION_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(position_item, position_array)
    {
      if (!(cJSON_GetArraySize(position_item) == 3)) {
        fprintf(stderr, "Position item is not an array of size 3\n");
        goto load_json_end;
      }
      for (uint32_t i = 0; i < 3; ++i) {
        utah_teapot_mesh->positions.data[c][i]
          = cJSON_GetArrayItem(position_item, i)->valuedouble;
      }
      c++;
    }
  }

  /* Parse cells */
  {
    cell_array = cJSON_GetObjectItemCaseSensitive(model_json, "cells");
    if (!cJSON_IsArray(cell_array)) {
      fprintf(stderr, "Cells object item is not an array\n");
      goto load_json_end;
    }

    utah_teapot_mesh->triangles.count = cJSON_GetArraySize(cell_array);
    ASSERT(utah_teapot_mesh->triangles.count == UTAH_TEAPOT_CELL_COUNT);

    uint32_t c = 0;
    cJSON_ArrayForEach(cell_item, cell_array)
    {
      if (!(cJSON_GetArraySize(cell_item) == 3)) {
        fprintf(stderr, "Cell item is not an array of size 3\n");
        goto load_json_end;
      }
      for (uint32_t i = 0; i < 3; ++i) {
        utah_teapot_mesh->triangles.data[c][i]
          = (uint16_t)cJSON_GetArrayItem(cell_item, i)->valueint;
      }
      c++;
    }
  }

  res = EXIT_SUCCESS;

load_json_end:
  cJSON_Delete(model_json);
  free(file_read_result.data);

  return res;
}

void utah_teapot_mesh_compute_normals(utah_teapot_mesh_t* utah_teapot_mesh)
{
  float(*positions)[3]          = utah_teapot_mesh->positions.data;
  float(*normals)[3]            = utah_teapot_mesh->normals.data;
  const uint64_t triangle_count = utah_teapot_mesh->triangles.count;
  uint16_t* triangle            = NULL;
  vec3 *p0 = NULL, *p1 = NULL, *p2 = NULL;
  vec3 v0, v1, norm;
  uint16_t i0, i1, i2;
  for (uint64_t i = 0; i < triangle_count; ++i) {
    triangle = utah_teapot_mesh->triangles.data[i];
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
  for (uint16_t i = 0; i < utah_teapot_mesh->normals.count; ++i) {
    glm_vec3_normalize(normals[i]);
  }
}
