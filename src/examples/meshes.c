#include "meshes.h"

#include <cJSON.h>
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#endif
#include <cglm/cglm.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#include <string.h>

#include "webgpu/wgpu_common.h"

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
 * Generic mesh functions
 * -------------------------------------------------------------------------- */

void mesh_create_renderable(WGPUDevice device, const mesh_t* mesh,
                            bool store_vertices, bool store_indices,
                            mesh_renderable_t* renderable)
{
  ASSERT(device != NULL);
  ASSERT(mesh != NULL);
  ASSERT(renderable != NULL);
  ASSERT(mesh->vertices != NULL);
  ASSERT(mesh->indices != NULL);

  // Define buffer usage
  WGPUBufferUsage vertex_buffer_usage
    = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
  if (store_vertices) {
    vertex_buffer_usage |= WGPUBufferUsage_Storage;
  }

  WGPUBufferUsage index_buffer_usage
    = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
  if (store_indices) {
    index_buffer_usage |= WGPUBufferUsage_Storage;
  }

  // Create vertex buffer
  WGPUBufferDescriptor vertex_buffer_desc = {
    .label            = STRVIEW("Mesh vertex buffer"),
    .usage            = vertex_buffer_usage,
    .size             = mesh->vertices_size,
    .mappedAtCreation = true,
  };
  renderable->vertex_buffer
    = wgpuDeviceCreateBuffer(device, &vertex_buffer_desc);
  ASSERT(renderable->vertex_buffer != NULL);

  // Copy vertex data to buffer
  void* vertex_mapping = wgpuBufferGetMappedRange(renderable->vertex_buffer, 0,
                                                  mesh->vertices_size);
  ASSERT(vertex_mapping != NULL);
  memcpy(vertex_mapping, mesh->vertices, mesh->vertices_size);
  wgpuBufferUnmap(renderable->vertex_buffer);

  // Create index buffer
  WGPUBufferDescriptor index_buffer_desc = {
    .label            = STRVIEW("Mesh index buffer"),
    .usage            = index_buffer_usage,
    .size             = mesh->indices_size,
    .mappedAtCreation = true,
  };
  renderable->index_buffer = wgpuDeviceCreateBuffer(device, &index_buffer_desc);
  ASSERT(renderable->index_buffer != NULL);

  // Copy index data to buffer
  void* index_mapping
    = wgpuBufferGetMappedRange(renderable->index_buffer, 0, mesh->indices_size);
  ASSERT(index_mapping != NULL);
  memcpy(index_mapping, mesh->indices, mesh->indices_size);
  wgpuBufferUnmap(renderable->index_buffer);

  // Set metadata
  renderable->index_count = mesh->indices_count;
  renderable->bind_group  = NULL;
}

void mesh_renderable_destroy(mesh_renderable_t* renderable)
{
  if (renderable == NULL) {
    return;
  }

  if (renderable->vertex_buffer != NULL) {
    wgpuBufferRelease(renderable->vertex_buffer);
    renderable->vertex_buffer = NULL;
  }

  if (renderable->index_buffer != NULL) {
    wgpuBufferRelease(renderable->index_buffer);
    renderable->index_buffer = NULL;
  }

  if (renderable->bind_group != NULL) {
    wgpuBindGroupRelease(renderable->bind_group);
    renderable->bind_group = NULL;
  }

  renderable->index_count = 0;
}

void mesh_get_position_at_index(const mesh_t* mesh, uint64_t index,
                                float out_pos[3])
{
  ASSERT(mesh != NULL);
  ASSERT(mesh->vertices != NULL);
  ASSERT(out_pos != NULL);

  // Position is at offset 0 in the vertex data
  const uint64_t byte_offset = index * mesh->vertex_stride;
  const float* vertex_data
    = (const float*)((const uint8_t*)mesh->vertices + byte_offset);

  out_pos[0] = vertex_data[0];
  out_pos[1] = vertex_data[1];
  out_pos[2] = vertex_data[2];
}

void mesh_get_normal_at_index(const mesh_t* mesh, uint64_t index,
                              float out_normal[3])
{
  ASSERT(mesh != NULL);
  ASSERT(mesh->vertices != NULL);
  ASSERT(out_normal != NULL);

  // Normal is at offset 3 * sizeof(float) in the vertex data
  const uint64_t byte_offset = index * mesh->vertex_stride + 3 * sizeof(float);
  const float* vertex_data
    = (const float*)((const uint8_t*)mesh->vertices + byte_offset);

  out_normal[0] = vertex_data[0];
  out_normal[1] = vertex_data[1];
  out_normal[2] = vertex_data[2];
}

void mesh_get_uv_at_index(const mesh_t* mesh, uint64_t index, float out_uv[2])
{
  ASSERT(mesh != NULL);
  ASSERT(mesh->vertices != NULL);
  ASSERT(out_uv != NULL);

  // UV is at offset 6 * sizeof(float) in the vertex data
  const uint64_t byte_offset = index * mesh->vertex_stride + 6 * sizeof(float);
  const float* vertex_data
    = (const float*)((const uint8_t*)mesh->vertices + byte_offset);

  out_uv[0] = vertex_data[0];
  out_uv[1] = vertex_data[1];
}

/* -------------------------------------------------------------------------- *
 * Primitive mesh functions
 * -------------------------------------------------------------------------- */

void primitive_vertex_data_init(primitive_vertex_data_t* data,
                                uint64_t vertex_count, uint64_t index_count)
{
  ASSERT(data != NULL);
  data->vertex_count = vertex_count;
  data->index_count  = index_count;
  data->vertices
    = (float*)malloc(vertex_count * PRIMITIVE_VERTEX_SIZE * sizeof(float));
  data->indices = (uint16_t*)malloc(index_count * sizeof(uint16_t));
  ASSERT(data->vertices != NULL);
  ASSERT(data->indices != NULL);
}

void primitive_vertex_data_destroy(primitive_vertex_data_t* data)
{
  if (data == NULL) {
    return;
  }
  if (data->vertices != NULL) {
    free(data->vertices);
    data->vertices = NULL;
  }
  if (data->indices != NULL) {
    free(data->indices);
    data->indices = NULL;
  }
  data->vertex_count = 0;
  data->index_count  = 0;
}

void primitive_to_mesh(const primitive_vertex_data_t* data, mesh_t* mesh)
{
  ASSERT(data != NULL);
  ASSERT(mesh != NULL);

  mesh->vertices       = data->vertices;
  mesh->indices        = data->indices;
  mesh->vertices_size  = data->vertex_count * PRIMITIVE_VERTEX_STRIDE;
  mesh->indices_size   = data->index_count * sizeof(uint16_t);
  mesh->indices_count  = data->index_count;
  mesh->vertex_stride  = PRIMITIVE_VERTEX_STRIDE;
  mesh->indices_uint32 = false;
}

/* --- Plane primitive --- */

void primitive_create_plane(const primitive_plane_options_t* options,
                            primitive_vertex_data_t* data)
{
  /* Default values */
  float width             = (options != NULL) ? options->width : 1.0f;
  float depth             = (options != NULL) ? options->depth : 1.0f;
  uint32_t subdivisions_w = (options != NULL) ? options->subdivisions_width : 1;
  uint32_t subdivisions_d = (options != NULL) ? options->subdivisions_depth : 1;

  uint64_t num_vertices = (subdivisions_w + 1) * (subdivisions_d + 1);
  uint64_t num_indices  = subdivisions_w * subdivisions_d * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  /* Generate vertices */
  uint64_t cursor = 0;
  for (uint32_t z = 0; z <= subdivisions_d; z++) {
    for (uint32_t x = 0; x <= subdivisions_w; x++) {
      float u = (float)x / (float)subdivisions_w;
      float v = (float)z / (float)subdivisions_d;

      /* Position */
      data->vertices[cursor++] = width * u - width * 0.5f;
      data->vertices[cursor++] = 0.0f;
      data->vertices[cursor++] = depth * v - depth * 0.5f;
      /* Normal */
      data->vertices[cursor++] = 0.0f;
      data->vertices[cursor++] = 1.0f;
      data->vertices[cursor++] = 0.0f;
      /* UV */
      data->vertices[cursor++] = u;
      data->vertices[cursor++] = v;
    }
  }

  /* Generate indices */
  uint32_t num_verts_across = subdivisions_w + 1;
  cursor                    = 0;
  for (uint32_t z = 0; z < subdivisions_d; z++) {
    for (uint32_t x = 0; x < subdivisions_w; x++) {
      /* Triangle 1 */
      data->indices[cursor++] = (z + 0) * num_verts_across + x;
      data->indices[cursor++] = (z + 1) * num_verts_across + x;
      data->indices[cursor++] = (z + 0) * num_verts_across + x + 1;
      /* Triangle 2 */
      data->indices[cursor++] = (z + 1) * num_verts_across + x;
      data->indices[cursor++] = (z + 1) * num_verts_across + x + 1;
      data->indices[cursor++] = (z + 0) * num_verts_across + x + 1;
    }
  }
}

/* --- Sphere primitive --- */

void primitive_create_sphere(const primitive_sphere_options_t* options,
                             primitive_vertex_data_t* data)
{
  /* Default values */
  float radius         = (options != NULL) ? options->radius : 1.0f;
  uint32_t subdiv_axis = (options != NULL) ? options->subdivisions_axis : 24;
  uint32_t subdiv_height
    = (options != NULL) ? options->subdivisions_height : 12;
  float start_lat  = (options != NULL) ? options->start_latitude : 0.0f;
  float end_lat    = (options != NULL) ? options->end_latitude : GLM_PI;
  float start_long = (options != NULL) ? options->start_longitude : 0.0f;
  float end_long   = (options != NULL) ? options->end_longitude : GLM_PI * 2.0f;

  ASSERT(subdiv_axis > 0 && subdiv_height > 0);

  float lat_range  = end_lat - start_lat;
  float long_range = end_long - start_long;

  uint64_t num_vertices = (subdiv_axis + 1) * (subdiv_height + 1);
  uint64_t num_indices  = subdiv_axis * subdiv_height * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  /* Generate vertices */
  uint64_t cursor = 0;
  for (uint32_t y = 0; y <= subdiv_height; y++) {
    for (uint32_t x = 0; x <= subdiv_axis; x++) {
      float u     = (float)x / (float)subdiv_axis;
      float v     = (float)y / (float)subdiv_height;
      float theta = long_range * u + start_long;
      float phi   = lat_range * v + start_lat;

      float sin_theta = sinf(theta);
      float cos_theta = cosf(theta);
      float sin_phi   = sinf(phi);
      float cos_phi   = cosf(phi);

      float ux = cos_theta * sin_phi;
      float uy = cos_phi;
      float uz = sin_theta * sin_phi;

      /* Position */
      data->vertices[cursor++] = radius * ux;
      data->vertices[cursor++] = radius * uy;
      data->vertices[cursor++] = radius * uz;
      /* Normal */
      data->vertices[cursor++] = ux;
      data->vertices[cursor++] = uy;
      data->vertices[cursor++] = uz;
      /* UV */
      data->vertices[cursor++] = 1.0f - u;
      data->vertices[cursor++] = v;
    }
  }

  /* Generate indices */
  uint32_t num_verts_around = subdiv_axis + 1;
  cursor                    = 0;
  for (uint32_t x = 0; x < subdiv_axis; x++) {
    for (uint32_t y = 0; y < subdiv_height; y++) {
      /* Triangle 1 */
      data->indices[cursor++] = (y + 0) * num_verts_around + x;
      data->indices[cursor++] = (y + 0) * num_verts_around + x + 1;
      data->indices[cursor++] = (y + 1) * num_verts_around + x;
      /* Triangle 2 */
      data->indices[cursor++] = (y + 1) * num_verts_around + x;
      data->indices[cursor++] = (y + 0) * num_verts_around + x + 1;
      data->indices[cursor++] = (y + 1) * num_verts_around + x + 1;
    }
  }
}

/* --- Cube primitive --- */

static const int32_t CUBE_FACE_INDICES[6][4] = {
  {3, 7, 5, 1}, /* right  */
  {6, 2, 0, 4}, /* left   */
  {6, 7, 3, 2}, /* top    */
  {0, 1, 5, 4}, /* bottom */
  {7, 6, 4, 5}, /* front  */
  {2, 3, 1, 0}, /* back   */
};

void primitive_create_cube(const primitive_cube_options_t* options,
                           primitive_vertex_data_t* data)
{
  float size = (options != NULL) ? options->size : 1.0f;
  float k    = size / 2.0f;

  float corner_vertices[8][3] = {
    {-k, -k, -k}, {+k, -k, -k}, {-k, +k, -k}, {+k, +k, -k},
    {-k, -k, +k}, {+k, -k, +k}, {-k, +k, +k}, {+k, +k, +k},
  };

  float face_normals[6][3] = {
    {+1, 0, 0}, {-1, 0, 0}, {0, +1, 0}, {0, -1, 0}, {0, 0, +1}, {0, 0, -1},
  };

  float uv_coords[4][2] = {
    {1, 0},
    {0, 0},
    {0, 1},
    {1, 1},
  };

  uint64_t num_vertices = 6 * 4;
  uint64_t num_indices  = 6 * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  uint64_t v_cursor = 0;
  uint64_t i_cursor = 0;

  for (int f = 0; f < 6; ++f) {
    const int32_t* face_idx = CUBE_FACE_INDICES[f];
    for (int v = 0; v < 4; ++v) {
      float* pos    = corner_vertices[face_idx[v]];
      float* normal = face_normals[f];
      float* uv     = uv_coords[v];

      /* Position */
      data->vertices[v_cursor++] = pos[0];
      data->vertices[v_cursor++] = pos[1];
      data->vertices[v_cursor++] = pos[2];
      /* Normal */
      data->vertices[v_cursor++] = normal[0];
      data->vertices[v_cursor++] = normal[1];
      data->vertices[v_cursor++] = normal[2];
      /* UV */
      data->vertices[v_cursor++] = uv[0];
      data->vertices[v_cursor++] = uv[1];
    }

    uint16_t offset           = 4 * f;
    data->indices[i_cursor++] = offset + 0;
    data->indices[i_cursor++] = offset + 1;
    data->indices[i_cursor++] = offset + 2;
    data->indices[i_cursor++] = offset + 0;
    data->indices[i_cursor++] = offset + 2;
    data->indices[i_cursor++] = offset + 3;
  }
}

/* --- Truncated Cone primitive --- */

void primitive_create_truncated_cone(
  const primitive_truncated_cone_options_t* options,
  primitive_vertex_data_t* data)
{
  /* Default values */
  float bottom_radius = (options != NULL) ? options->bottom_radius : 1.0f;
  float top_radius    = (options != NULL) ? options->top_radius : 0.0f;
  float height        = (options != NULL) ? options->height : 1.0f;
  uint32_t radial_sub = (options != NULL) ? options->radial_subdivisions : 24;
  uint32_t vert_sub   = (options != NULL) ? options->vertical_subdivisions : 1;
  bool top_cap        = (options != NULL) ? options->top_cap : true;
  bool bottom_cap     = (options != NULL) ? options->bottom_cap : true;

  ASSERT(radial_sub >= 3);
  ASSERT(vert_sub >= 1);

  int32_t extra         = (top_cap ? 2 : 0) + (bottom_cap ? 2 : 0);
  uint64_t num_vertices = (radial_sub + 1) * (vert_sub + 1 + extra);
  uint64_t num_indices  = radial_sub * (vert_sub + extra / 2) * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  uint32_t verts_around_edge = radial_sub + 1;
  float slant                = atan2f(bottom_radius - top_radius, height);
  float cos_slant            = cosf(slant);
  float sin_slant            = sinf(slant);

  int32_t start = top_cap ? -2 : 0;
  int32_t end   = (int32_t)vert_sub + (bottom_cap ? 2 : 0);

  uint64_t cursor = 0;
  for (int32_t yy = start; yy <= end; ++yy) {
    float v           = (float)yy / (float)vert_sub;
    float y           = height * v;
    float ring_radius = 0.0f;

    if (yy < 0) {
      y           = 0.0f;
      v           = 1.0f;
      ring_radius = bottom_radius;
    }
    else if (yy > (int32_t)vert_sub) {
      y           = height;
      v           = 1.0f;
      ring_radius = top_radius;
    }
    else {
      ring_radius
        = bottom_radius
          + (top_radius - bottom_radius) * ((float)yy / (float)vert_sub);
    }

    if (yy == -2 || yy == (int32_t)vert_sub + 2) {
      ring_radius = 0.0f;
      v           = 0.0f;
    }

    y -= height / 2.0f;

    for (uint32_t ii = 0; ii < verts_around_edge; ++ii) {
      float angle = (float)ii * GLM_PI * 2.0f / (float)radial_sub;
      float sin_a = sinf(angle);
      float cos_a = cosf(angle);

      /* Position */
      data->vertices[cursor++] = sin_a * ring_radius;
      data->vertices[cursor++] = y;
      data->vertices[cursor++] = cos_a * ring_radius;

      /* Normal */
      if (yy < 0) {
        data->vertices[cursor++] = 0.0f;
        data->vertices[cursor++] = -1.0f;
        data->vertices[cursor++] = 0.0f;
      }
      else if (yy > (int32_t)vert_sub) {
        data->vertices[cursor++] = 0.0f;
        data->vertices[cursor++] = 1.0f;
        data->vertices[cursor++] = 0.0f;
      }
      else if (ring_radius == 0.0f) {
        data->vertices[cursor++] = 0.0f;
        data->vertices[cursor++] = 0.0f;
        data->vertices[cursor++] = 0.0f;
      }
      else {
        data->vertices[cursor++] = sin_a * cos_slant;
        data->vertices[cursor++] = sin_slant;
        data->vertices[cursor++] = cos_a * cos_slant;
      }

      /* UV */
      data->vertices[cursor++] = (float)ii / (float)radial_sub;
      data->vertices[cursor++] = 1.0f - v;
    }
  }

  /* Generate indices */
  cursor = 0;
  for (int32_t yy = 0; yy < (int32_t)vert_sub + extra; ++yy) {
    if ((yy == 1 && top_cap)
        || (yy == (int32_t)vert_sub + extra - 2 && bottom_cap)) {
      continue;
    }
    for (uint32_t ii = 0; ii < radial_sub; ++ii) {
      data->indices[cursor++] = verts_around_edge * (yy + 0) + ii;
      data->indices[cursor++] = verts_around_edge * (yy + 0) + ii + 1;
      data->indices[cursor++] = verts_around_edge * (yy + 1) + ii + 1;

      data->indices[cursor++] = verts_around_edge * (yy + 0) + ii;
      data->indices[cursor++] = verts_around_edge * (yy + 1) + ii + 1;
      data->indices[cursor++] = verts_around_edge * (yy + 1) + ii;
    }
  }
}

/* --- Cylinder primitive --- */

void primitive_create_cylinder(const primitive_cylinder_options_t* options,
                               primitive_vertex_data_t* data)
{
  primitive_truncated_cone_options_t cone_opts = {
    .bottom_radius = (options != NULL) ? options->radius : 1.0f,
    .top_radius    = (options != NULL) ? options->radius : 1.0f,
    .height        = (options != NULL) ? options->height : 1.0f,
    .radial_subdivisions
    = (options != NULL) ? options->radial_subdivisions : 24,
    .vertical_subdivisions
    = (options != NULL) ? options->vertical_subdivisions : 1,
    .top_cap    = (options != NULL) ? options->top_cap : true,
    .bottom_cap = (options != NULL) ? options->bottom_cap : true,
  };
  primitive_create_truncated_cone(&cone_opts, data);
}

/* --- Torus primitive --- */

void primitive_create_torus(const primitive_torus_options_t* options,
                            primitive_vertex_data_t* data)
{
  /* Default values */
  float radius        = (options != NULL) ? options->radius : 1.0f;
  float thickness     = (options != NULL) ? options->thickness : 0.24f;
  uint32_t radial_sub = (options != NULL) ? options->radial_subdivisions : 24;
  uint32_t body_sub   = (options != NULL) ? options->body_subdivisions : 12;
  float start_angle   = (options != NULL) ? options->start_angle : 0.0f;
  float end_angle     = (options != NULL) ? options->end_angle : GLM_PI * 2.0f;

  ASSERT(radial_sub >= 3);
  ASSERT(body_sub >= 3);

  float range           = end_angle - start_angle;
  uint32_t radial_parts = radial_sub + 1;
  uint32_t body_parts   = body_sub + 1;
  uint64_t num_vertices = radial_parts * body_parts;
  uint64_t num_indices  = radial_sub * body_sub * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  /* Generate vertices */
  uint64_t cursor = 0;
  for (uint32_t slice = 0; slice < body_parts; ++slice) {
    float v           = (float)slice / (float)body_sub;
    float slice_angle = v * GLM_PI * 2.0f;
    float slice_sin   = sinf(slice_angle);
    float ring_radius = radius + slice_sin * thickness;
    float ny          = cosf(slice_angle);
    float y           = ny * thickness;

    for (uint32_t ring = 0; ring < radial_parts; ++ring) {
      float u          = (float)ring / (float)radial_sub;
      float ring_angle = start_angle + u * range;
      float x_sin      = sinf(ring_angle);
      float z_cos      = cosf(ring_angle);
      float x          = x_sin * ring_radius;
      float z          = z_cos * ring_radius;
      float nx         = x_sin * slice_sin;
      float nz         = z_cos * slice_sin;

      /* Position */
      data->vertices[cursor++] = x;
      data->vertices[cursor++] = y;
      data->vertices[cursor++] = z;
      /* Normal */
      data->vertices[cursor++] = nx;
      data->vertices[cursor++] = ny;
      data->vertices[cursor++] = nz;
      /* UV */
      data->vertices[cursor++] = u;
      data->vertices[cursor++] = 1.0f - v;
    }
  }

  /* Generate indices */
  cursor = 0;
  for (uint32_t slice = 0; slice < body_sub; ++slice) {
    for (uint32_t ring = 0; ring < radial_sub; ++ring) {
      uint32_t next_ring  = ring + 1;
      uint32_t next_slice = slice + 1;

      data->indices[cursor++] = radial_parts * slice + ring;
      data->indices[cursor++] = radial_parts * next_slice + ring;
      data->indices[cursor++] = radial_parts * slice + next_ring;

      data->indices[cursor++] = radial_parts * next_slice + ring;
      data->indices[cursor++] = radial_parts * next_slice + next_ring;
      data->indices[cursor++] = radial_parts * slice + next_ring;
    }
  }
}

/* --- Disc primitive --- */

void primitive_create_disc(const primitive_disc_options_t* options,
                           primitive_vertex_data_t* data)
{
  /* Default values */
  float radius       = (options != NULL) ? options->radius : 1.0f;
  uint32_t divisions = (options != NULL) ? options->divisions : 24;
  uint32_t stacks    = (options != NULL) ? options->stacks : 1;
  float inner_radius = (options != NULL) ? options->inner_radius : 0.0f;
  float stack_power  = (options != NULL) ? options->stack_power : 1.0f;

  ASSERT(divisions >= 3);

  uint64_t num_vertices = (divisions + 1) * (stacks + 1);
  uint64_t num_indices  = stacks * divisions * 6;

  primitive_vertex_data_init(data, num_vertices, num_indices);

  uint32_t first_index      = 0;
  float radius_span         = radius - inner_radius;
  uint32_t points_per_stack = divisions + 1;

  uint64_t v_cursor = 0;
  uint64_t i_cursor = 0;

  for (uint32_t stack = 0; stack <= stacks; ++stack) {
    float stack_radius
      = inner_radius
        + radius_span * powf((float)stack / (float)stacks, stack_power);

    for (uint32_t i = 0; i <= divisions; ++i) {
      float theta = (2.0f * GLM_PI * (float)i) / (float)divisions;
      float x     = stack_radius * cosf(theta);
      float z     = stack_radius * sinf(theta);

      /* Position */
      data->vertices[v_cursor++] = x;
      data->vertices[v_cursor++] = 0.0f;
      data->vertices[v_cursor++] = z;
      /* Normal */
      data->vertices[v_cursor++] = 0.0f;
      data->vertices[v_cursor++] = 1.0f;
      data->vertices[v_cursor++] = 0.0f;
      /* UV */
      data->vertices[v_cursor++] = 1.0f - (float)i / (float)divisions;
      data->vertices[v_cursor++] = (float)stack / (float)stacks;

      if (stack > 0 && i != divisions) {
        uint16_t a = first_index + (i + 1);
        uint16_t b = first_index + i;
        uint16_t c = first_index + i - points_per_stack;
        uint16_t d = first_index + (i + 1) - points_per_stack;

        data->indices[i_cursor++] = a;
        data->indices[i_cursor++] = b;
        data->indices[i_cursor++] = c;

        data->indices[i_cursor++] = a;
        data->indices[i_cursor++] = c;
        data->indices[i_cursor++] = d;
      }
    }
    first_index += divisions + 1;
  }
}

/* --- Utility functions --- */

void primitive_deindex(const primitive_vertex_data_t* src,
                       primitive_vertex_data_t* dst)
{
  ASSERT(src != NULL);
  ASSERT(dst != NULL);

  uint64_t num_elements = src->index_count;
  primitive_vertex_data_init(dst, num_elements, num_elements);

  for (uint64_t i = 0; i < num_elements; ++i) {
    uint64_t src_off = src->indices[i] * PRIMITIVE_VERTEX_SIZE;
    uint64_t dst_off = i * PRIMITIVE_VERTEX_SIZE;
    for (uint32_t j = 0; j < PRIMITIVE_VERTEX_SIZE; ++j) {
      dst->vertices[dst_off + j] = src->vertices[src_off + j];
    }
    dst->indices[i] = (uint16_t)i;
  }
}

void primitive_generate_triangle_normals(primitive_vertex_data_t* data)
{
  ASSERT(data != NULL);

  for (uint64_t ii = 0; ii < data->vertex_count; ii += 3) {
    float* v0 = &data->vertices[ii * PRIMITIVE_VERTEX_SIZE];
    float* v1 = &data->vertices[(ii + 1) * PRIMITIVE_VERTEX_SIZE];
    float* v2 = &data->vertices[(ii + 2) * PRIMITIVE_VERTEX_SIZE];

    /* Position vectors */
    vec3 p0 = {v0[0], v0[1], v0[2]};
    vec3 p1 = {v1[0], v1[1], v1[2]};
    vec3 p2 = {v2[0], v2[1], v2[2]};

    vec3 e0, e1, n;
    glm_vec3_sub(p0, p1, e0);
    glm_vec3_sub(p0, p2, e1);
    glm_vec3_normalize(e0);
    glm_vec3_normalize(e1);
    glm_vec3_cross(e0, e1, n);

    /* Set normals for all 3 vertices */
    v0[3] = n[0];
    v0[4] = n[1];
    v0[5] = n[2];
    v1[3] = n[0];
    v1[4] = n[1];
    v1[5] = n[2];
    v2[3] = n[0];
    v2[4] = n[1];
    v2[5] = n[2];
  }
}

void primitive_facet(const primitive_vertex_data_t* src,
                     primitive_vertex_data_t* dst)
{
  ASSERT(src != NULL);
  ASSERT(dst != NULL);

  primitive_deindex(src, dst);
  primitive_generate_triangle_normals(dst);
}

void primitive_reorient(primitive_vertex_data_t* data, const float* matrix)
{
  ASSERT(data != NULL);
  ASSERT(matrix != NULL);

  for (uint64_t i = 0; i < data->vertex_count; ++i) {
    float* v = &data->vertices[i * PRIMITIVE_VERTEX_SIZE];

    /* Transform position */
    vec4 pos = {v[0], v[1], v[2], 1.0f};
    vec4 result;
    glm_mat4_mulv((vec4*)matrix, pos, result);
    v[0] = result[0];
    v[1] = result[1];
    v[2] = result[2];

    /* Transform normal (upper 3x3 only) */
    vec3 norm = {v[3], v[4], v[5]};
    vec3 norm_result;
    mat3 upper;
    glm_mat4_pick3((vec4*)matrix, upper);
    glm_mat3_mulv(upper, norm, norm_result);
    v[3] = norm_result[0];
    v[4] = norm_result[1];
    v[5] = norm_result[2];
  }
}

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
  width_segments  = MAX(3u, (uint32_t)floor(width_segments));
  height_segments = MAX(2u, (uint32_t)floor(height_segments));

  /* Each vertex has: position(3) + normal(3) + uv(2) = 8 floats */
  uint32_t vertices_count = (width_segments + 1) * (height_segments + 1) * 8;
  float* vertices         = (float*)malloc(vertices_count * sizeof(float));
  if (!vertices) {
    return;
  }

  uint32_t indices_count = width_segments * height_segments * 6;
  uint16_t* indices      = (uint16_t*)malloc(indices_count * sizeof(uint16_t));
  if (!indices) {
    free(vertices);
    return;
  }

  vec3 first_vertex = GLM_VEC3_ZERO_INIT;
  vec3 vertex       = GLM_VEC3_ZERO_INIT;
  vec3 normal       = GLM_VEC3_ZERO_INIT;

  uint32_t index      = 0;
  uint32_t grid_count = height_segments + 1;
  uint16_t** grid     = (uint16_t**)malloc(grid_count * sizeof(uint16_t*));
  if (!grid) {
    free(vertices);
    free(indices);
    return;
  }

  /* Generate vertices, normals and uvs */
  uint32_t vc = 0, ic = 0, gc = 0;
  for (uint32_t iy = 0; iy <= height_segments; ++iy) {
    uint16_t* vertices_row
      = (uint16_t*)malloc((width_segments + 1) * sizeof(uint16_t));
    if (!vertices_row) {
      /* Cleanup on allocation failure */
      for (uint32_t gci = 0; gci < gc; ++gci) {
        free(grid[gci]);
      }
      free(grid);
      free(vertices);
      free(indices);
      return;
    }
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
  if (!sphere_mesh) {
    return;
  }

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

  p_ply ply = ply_open("assets/meshes/dragon_vrip_res4.ply", NULL, 0, NULL);
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
  float (*positions)[3]         = stanford_dragon_mesh->positions.data;
  float (*normals)[3]           = stanford_dragon_mesh->normals.data;
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
  float (*uvs)[2]      = stanford_dragon_mesh->uvs.data;
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

int utah_teapot_mesh_init(utah_teapot_mesh_t* utah_teapot_mesh,
                          const char* const utah_teapot_json)
{
  ASSERT(utah_teapot_mesh);

  int res = EXIT_FAILURE;

  const cJSON* position_array = NULL;
  const cJSON* position_item  = NULL;
  const cJSON* cell_array     = NULL;
  const cJSON* cell_item      = NULL;
  cJSON* model_json           = cJSON_Parse(utah_teapot_json);
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

  return res;
}

void utah_teapot_mesh_compute_normals(utah_teapot_mesh_t* utah_teapot_mesh)
{
  float (*positions)[3]         = utah_teapot_mesh->positions.data;
  float (*normals)[3]           = utah_teapot_mesh->normals.data;
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

/* -------------------------------------------------------------------------- *
 * Generic mesh utility functions
 * -------------------------------------------------------------------------- */

void compute_surface_normals(const float (*positions)[3],
                             uint64_t position_count,
                             const uint16_t (*triangles)[3],
                             uint64_t triangle_count, float (*out_normals)[3])
{
  ASSERT(positions != NULL);
  ASSERT(triangles != NULL);
  ASSERT(out_normals != NULL);

  // Initialize normals to zero
  for (uint64_t i = 0; i < position_count; ++i) {
    out_normals[i][0] = 0.0f;
    out_normals[i][1] = 0.0f;
    out_normals[i][2] = 0.0f;
  }

  // Compute normals for each triangle
  vec3 v0, v1, norm;
  for (uint64_t i = 0; i < triangle_count; ++i) {
    const uint16_t i0 = triangles[i][0];
    const uint16_t i1 = triangles[i][1];
    const uint16_t i2 = triangles[i][2];

    vec3* p0 = (vec3*)&positions[i0];
    vec3* p1 = (vec3*)&positions[i1];
    vec3* p2 = (vec3*)&positions[i2];

    // v0 = p1 - p0
    glm_vec3_sub(*p1, *p0, v0);
    // v1 = p2 - p0
    glm_vec3_sub(*p2, *p0, v1);

    // Normalize vectors
    glm_vec3_normalize(v0);
    glm_vec3_normalize(v1);

    // Compute cross product
    glm_vec3_cross(v0, v1, norm);

    // Accumulate the normals
    glm_vec3_add(out_normals[i0], norm, out_normals[i0]);
    glm_vec3_add(out_normals[i1], norm, out_normals[i1]);
    glm_vec3_add(out_normals[i2], norm, out_normals[i2]);
  }

  // Normalize accumulated normals
  for (uint64_t i = 0; i < position_count; ++i) {
    glm_vec3_normalize(out_normals[i]);
  }
}

void compute_projected_plane_uvs(const float (*positions)[3],
                                 uint64_t position_count,
                                 projected_plane_enum projected_plane,
                                 float (*out_uvs)[2])
{
  ASSERT(positions != NULL);
  ASSERT(out_uvs != NULL);

  const uint32_t* idxs = projected_plane2_ids[(uint32_t)projected_plane];
  float extent_min[2]  = {FLT_MAX, FLT_MAX};
  float extent_max[2]  = {FLT_MIN, FLT_MIN};

  // Initialize UVs to zero and compute extents
  for (uint64_t i = 0; i < position_count; ++i) {
    // Simply project to the selected plane
    out_uvs[i][0] = positions[i][idxs[0]];
    out_uvs[i][1] = positions[i][idxs[1]];

    extent_min[0] = MIN(positions[i][idxs[0]], extent_min[0]);
    extent_min[1] = MIN(positions[i][idxs[1]], extent_min[1]);
    extent_max[0] = MAX(positions[i][idxs[0]], extent_max[0]);
    extent_max[1] = MAX(positions[i][idxs[1]], extent_max[1]);
  }

  // Normalize UVs to [0, 1] range
  for (uint64_t i = 0; i < position_count; ++i) {
    out_uvs[i][0]
      = (out_uvs[i][0] - extent_min[0]) / (extent_max[0] - extent_min[0]);
    out_uvs[i][1]
      = (out_uvs[i][1] - extent_min[1]) / (extent_max[1] - extent_min[1]);
  }
}

/* -------------------------------------------------------------------------- *
 * Generate normals with max angle
 * -------------------------------------------------------------------------- */

/* Helper structure for vertex deduplication */
typedef struct vertex_key_t {
  float pos[3];
  float norm[3];
  uint32_t index;
  struct vertex_key_t* next; /* For hash collision chaining */
} vertex_key_t;

/* Simple hash function for vertex positions */
static uint32_t hash_vec3(const float v[3], uint32_t table_size)
{
  /* Simple FNV-1a hash */
  uint32_t hash        = 2166136261u;
  const uint8_t* bytes = (const uint8_t*)v;
  for (size_t i = 0; i < 3 * sizeof(float); ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash % table_size;
}

/* Check if two vec3 are equal */
static bool vec3_equals(const float a[3], const float b[3])
{
  return fabsf(a[0] - b[0]) < 1e-6f && fabsf(a[1] - b[1]) < 1e-6f
         && fabsf(a[2] - b[2]) < 1e-6f;
}

/* Helper structure for building dynamic arrays */
typedef struct dynamic_array_t {
  void* data;
  uint64_t count;
  uint64_t capacity;
  size_t element_size;
} dynamic_array_t;

static void dynamic_array_init(dynamic_array_t* arr, size_t element_size,
                               uint64_t initial_capacity)
{
  arr->element_size = element_size;
  arr->count        = 0;
  arr->capacity     = initial_capacity;
  arr->data         = malloc(arr->capacity * element_size);
}

static void dynamic_array_push(dynamic_array_t* arr, const void* element)
{
  if (arr->count >= arr->capacity) {
    arr->capacity *= 2;
    arr->data = realloc(arr->data, arr->capacity * arr->element_size);
  }
  memcpy((uint8_t*)arr->data + arr->count * arr->element_size, element,
         arr->element_size);
  arr->count++;
}

static void dynamic_array_destroy(dynamic_array_t* arr)
{
  if (arr->data) {
    free(arr->data);
  }
  memset(arr, 0, sizeof(*arr));
}

int generate_normals_with_max_angle(float max_angle,
                                    const float (*positions)[3],
                                    uint64_t position_count,
                                    const uint16_t (*triangles)[3],
                                    uint64_t triangle_count,
                                    generate_normals_result_t* out_result)
{
  ASSERT(positions != NULL);
  ASSERT(triangles != NULL);
  ASSERT(out_result != NULL);

  memset(out_result, 0, sizeof(*out_result));

  const float max_angle_cos = cosf(max_angle);

  /* Step 1: Compute face normals */
  vec3* face_normals = (vec3*)malloc(triangle_count * sizeof(vec3));
  if (!face_normals) {
    return -1;
  }

  for (uint64_t i = 0; i < triangle_count; ++i) {
    const uint16_t i0 = triangles[i][0];
    const uint16_t i1 = triangles[i][1];
    const uint16_t i2 = triangles[i][2];

    vec3* v1 = (vec3*)&positions[i0];
    vec3* v2 = (vec3*)&positions[i1];
    vec3* v3 = (vec3*)&positions[i2];

    vec3 edge1, edge2;
    glm_vec3_sub(*v2, *v1, edge1);
    glm_vec3_sub(*v3, *v1, edge2);
    glm_vec3_cross(edge1, edge2, face_normals[i]);
    glm_vec3_normalize(face_normals[i]);
  }

  /* Step 2: Build vertex index mapping (position -> shared index) */
  const uint32_t hash_table_size = position_count * 2 + 1;
  vertex_key_t** pos_hash_table
    = (vertex_key_t**)calloc(hash_table_size, sizeof(vertex_key_t*));
  if (!pos_hash_table) {
    free(face_normals);
    return -1;
  }

  uint32_t* vert_indices = (uint32_t*)malloc(position_count * sizeof(uint32_t));
  if (!vert_indices) {
    free(pos_hash_table);
    free(face_normals);
    return -1;
  }

  uint32_t shared_vert_count = 0;
  for (uint64_t i = 0; i < position_count; ++i) {
    const uint32_t hash = hash_vec3(positions[i], hash_table_size);

    /* Check if this position already exists */
    vertex_key_t* entry = pos_hash_table[hash];
    bool found          = false;
    while (entry) {
      if (vec3_equals(entry->pos, positions[i])) {
        vert_indices[i] = entry->index;
        found           = true;
        break;
      }
      entry = entry->next;
    }

    if (!found) {
      /* Add new entry */
      vertex_key_t* new_entry = (vertex_key_t*)malloc(sizeof(vertex_key_t));
      glm_vec3_copy((float*)positions[i], new_entry->pos);
      new_entry->index     = shared_vert_count++;
      new_entry->next      = pos_hash_table[hash];
      pos_hash_table[hash] = new_entry;
      vert_indices[i]      = new_entry->index;
    }
  }

  /* Step 3: Build face lists for each shared vertex */
  dynamic_array_t* vert_faces
    = (dynamic_array_t*)malloc(shared_vert_count * sizeof(dynamic_array_t));
  if (!vert_faces) {
    /* Cleanup */
    for (uint32_t i = 0; i < hash_table_size; ++i) {
      vertex_key_t* entry = pos_hash_table[i];
      while (entry) {
        vertex_key_t* next = entry->next;
        free(entry);
        entry = next;
      }
    }
    free(pos_hash_table);
    free(vert_indices);
    free(face_normals);
    return -1;
  }

  for (uint32_t i = 0; i < shared_vert_count; ++i) {
    dynamic_array_init(&vert_faces[i], sizeof(uint32_t), 8);
  }

  /* Associate faces with vertices */
  for (uint64_t i = 0; i < triangle_count; ++i) {
    for (uint32_t j = 0; j < 3; ++j) {
      const uint16_t vert_idx   = triangles[i][j];
      const uint32_t shared_idx = vert_indices[vert_idx];
      const uint32_t face_idx   = (uint32_t)i;
      dynamic_array_push(&vert_faces[shared_idx], &face_idx);
    }
  }

  /* Step 4: Generate new vertices with computed normals */
  dynamic_array_t new_positions, new_normals, new_triangles;
  dynamic_array_init(&new_positions, sizeof(vec3), position_count);
  dynamic_array_init(&new_normals, sizeof(vec3), position_count);
  dynamic_array_init(&new_triangles, sizeof(uint16_t) * 3, triangle_count);

  /* Hash table for deduplicating new vertices (position + normal) */
  vertex_key_t** vert_hash_table
    = (vertex_key_t**)calloc(hash_table_size, sizeof(vertex_key_t*));
  if (!vert_hash_table) {
    /* Cleanup */
    dynamic_array_destroy(&new_positions);
    dynamic_array_destroy(&new_normals);
    dynamic_array_destroy(&new_triangles);
    for (uint32_t i = 0; i < shared_vert_count; ++i) {
      dynamic_array_destroy(&vert_faces[i]);
    }
    free(vert_faces);
    for (uint32_t i = 0; i < hash_table_size; ++i) {
      vertex_key_t* entry = pos_hash_table[i];
      while (entry) {
        vertex_key_t* next = entry->next;
        free(entry);
        entry = next;
      }
    }
    free(pos_hash_table);
    free(vert_indices);
    free(face_normals);
    return -1;
  }

  /* Process each face */
  for (uint64_t face_idx = 0; face_idx < triangle_count; ++face_idx) {
    vec3* this_face_normal = (vec3*)&face_normals[face_idx];
    uint16_t new_triangle[3];

    /* Process each vertex of the face */
    for (uint32_t j = 0; j < 3; ++j) {
      const uint16_t vert_idx   = triangles[face_idx][j];
      const uint32_t shared_idx = vert_indices[vert_idx];
      const vec3* pos           = &positions[vert_idx];

      /* Compute normal by averaging faces within max_angle */
      vec3 norm                    = {0, 0, 0};
      const dynamic_array_t* faces = &vert_faces[shared_idx];
      const uint32_t* face_list    = (const uint32_t*)faces->data;

      for (uint64_t k = 0; k < faces->count; ++k) {
        const uint32_t other_face_idx = face_list[k];
        vec3* other_face_normal       = (vec3*)&face_normals[other_face_idx];

        /* Check angle */
        const float dot = glm_vec3_dot(*this_face_normal, *other_face_normal);
        if (dot > max_angle_cos) {
          glm_vec3_add(norm, *other_face_normal, norm);
        }
      }
      glm_vec3_normalize(norm);

      /* Find or create vertex with this position and normal */
      const uint32_t hash = hash_vec3(*pos, hash_table_size);
      vertex_key_t* entry = vert_hash_table[hash];
      bool found          = false;
      uint32_t new_idx    = 0;

      while (entry) {
        if (vec3_equals(entry->pos, *pos) && vec3_equals(entry->norm, norm)) {
          new_idx = entry->index;
          found   = true;
          break;
        }
        entry = entry->next;
      }

      if (!found) {
        /* Add new vertex */
        new_idx = (uint32_t)new_positions.count;
        dynamic_array_push(&new_positions, pos);
        dynamic_array_push(&new_normals, &norm);

        vertex_key_t* new_entry = (vertex_key_t*)malloc(sizeof(vertex_key_t));
        glm_vec3_copy((float*)*pos, new_entry->pos);
        glm_vec3_copy(norm, new_entry->norm);
        new_entry->index      = new_idx;
        new_entry->next       = vert_hash_table[hash];
        vert_hash_table[hash] = new_entry;
      }

      new_triangle[j] = (uint16_t)new_idx;
    }

    dynamic_array_push(&new_triangles, new_triangle);
  }

  /* Step 5: Copy results */
  out_result->position_count = new_positions.count;
  out_result->triangle_count = new_triangles.count;

  out_result->positions
    = (float*)malloc(new_positions.count * 3 * sizeof(float));
  out_result->normals = (float*)malloc(new_normals.count * 3 * sizeof(float));
  out_result->triangles
    = (uint16_t*)malloc(new_triangles.count * 3 * sizeof(uint16_t));

  if (!out_result->positions || !out_result->normals
      || !out_result->triangles) {
    generate_normals_result_destroy(out_result);
    /* Cleanup */
    for (uint32_t i = 0; i < hash_table_size; ++i) {
      vertex_key_t* entry = vert_hash_table[i];
      while (entry) {
        vertex_key_t* next = entry->next;
        free(entry);
        entry = next;
      }
    }
    free(vert_hash_table);
    dynamic_array_destroy(&new_positions);
    dynamic_array_destroy(&new_normals);
    dynamic_array_destroy(&new_triangles);
    for (uint32_t i = 0; i < shared_vert_count; ++i) {
      dynamic_array_destroy(&vert_faces[i]);
    }
    free(vert_faces);
    for (uint32_t i = 0; i < hash_table_size; ++i) {
      vertex_key_t* entry = pos_hash_table[i];
      while (entry) {
        vertex_key_t* next = entry->next;
        free(entry);
        entry = next;
      }
    }
    free(pos_hash_table);
    free(vert_indices);
    free(face_normals);
    return -1;
  }

  memcpy(out_result->positions, new_positions.data,
         new_positions.count * 3 * sizeof(float));
  memcpy(out_result->normals, new_normals.data,
         new_normals.count * 3 * sizeof(float));
  memcpy(out_result->triangles, new_triangles.data,
         new_triangles.count * 3 * sizeof(uint16_t));

  /* Cleanup */
  for (uint32_t i = 0; i < hash_table_size; ++i) {
    vertex_key_t* entry = vert_hash_table[i];
    while (entry) {
      vertex_key_t* next = entry->next;
      free(entry);
      entry = next;
    }
  }
  free(vert_hash_table);

  for (uint32_t i = 0; i < hash_table_size; ++i) {
    vertex_key_t* entry = pos_hash_table[i];
    while (entry) {
      vertex_key_t* next = entry->next;
      free(entry);
      entry = next;
    }
  }
  free(pos_hash_table);

  dynamic_array_destroy(&new_positions);
  dynamic_array_destroy(&new_normals);
  dynamic_array_destroy(&new_triangles);

  for (uint32_t i = 0; i < shared_vert_count; ++i) {
    dynamic_array_destroy(&vert_faces[i]);
  }
  free(vert_faces);
  free(vert_indices);
  free(face_normals);

  return 0;
}

void generate_normals_result_destroy(generate_normals_result_t* result)
{
  if (!result) {
    return;
  }

  if (result->positions) {
    free(result->positions);
    result->positions = NULL;
  }

  if (result->normals) {
    free(result->normals);
    result->normals = NULL;
  }

  if (result->triangles) {
    free(result->triangles);
    result->triangles = NULL;
  }

  result->position_count = 0;
  result->triangle_count = 0;
}
