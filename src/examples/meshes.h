#ifndef MESHES_H
#define MESHES_H

#include <stdbool.h>
#include <stdint.h>
#include <webgpu/webgpu.h>

/* -------------------------------------------------------------------------- *
 * Box mesh
 * -------------------------------------------------------------------------- */

#define BOX_MESH_FACES_COUNT (6)
#define BOX_MESH_VERTICES_PER_SIDE (4)
#define BOX_MESH_INDICES_PER_SIZE (6)
#define BOX_MESH_F32S_PER_VERTEX                                               \
  (14) // position : vec3f, tangent : vec3f, bitangent : vec3f, normal : vec3f,
       // uv :vec2f
#define BOX_MESH_VERTEX_STRIDE (BOX_MESH_F32S_PER_VERTEX * 4)
#define BOX_MESH_VERTICES_COUNT                                                \
  (BOX_MESH_FACES_COUNT * BOX_MESH_VERTICES_PER_SIDE * BOX_MESH_F32S_PER_VERTEX)
#define BOX_MESH_INDICES_COUNT                                                 \
  (BOX_MESH_FACES_COUNT * BOX_MESH_INDICES_PER_SIZE)

typedef struct box_mesh_t {
  uint64_t vertex_count;
  uint64_t index_count;
  float vertex_array[BOX_MESH_VERTICES_COUNT];
  uint16_t index_array[BOX_MESH_INDICES_COUNT];
  uint32_t vertex_stride;
} box_mesh_t;

/**
 * @brief Constructs a box mesh with the given dimensions.
 * The vertex buffer will have the following vertex fields (in the given order):
 *   position  : float32x3
 *   normal    : float32x3
 *   uv        : float32x2
 *   tangent   : float32x3
 *   bitangent : float32x3
 * @param width the width of the box
 * @param height the height of the box
 * @param depth the depth of the box
 * @returns the box mesh with tangent and bitangents.
 */
void box_mesh_create_with_tangents(box_mesh_t* box_mesh, float width,
                                   float height, float depth);

/* -------------------------------------------------------------------------- *
 * Cube mesh
 * -------------------------------------------------------------------------- */

typedef struct cube_mesh_t {
  uint64_t vertex_size; /* Byte size of one cube vertex. */
  uint64_t position_offset;
  uint64_t color_offset; /* Byte offset of cube vertex color attribute. */
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
 * Generic mesh structures and functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Generic mesh structure containing vertices, indices, and vertex stride
 * Matches the TypeScript Mesh interface
 */
typedef struct mesh_t {
  float* vertices;        /* Pointer to vertex data array */
  void* indices;          /* Pointer to index data (uint16 or uint32) */
  uint64_t vertices_size; /* Size of vertex data in bytes */
  uint64_t indices_size;  /* Size of index data in bytes */
  uint64_t indices_count; /* Number of indices */
  uint32_t vertex_stride; /* Number of bytes per vertex */
  bool indices_uint32;    /* true if indices are uint32, false for uint16 */
} mesh_t;

/**
 * @brief Renderable structure containing GPU buffers and metadata
 * Matches the TypeScript Renderable interface
 */
typedef struct mesh_renderable_t {
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint64_t index_count;
  WGPUBindGroup bind_group; /* Optional bind group */
} mesh_renderable_t;

/**
 * @brief Creates a mesh renderable from a mesh structure
 * @param device A valid GPUDevice
 * @param mesh An indexed triangle-list mesh
 * @param store_vertices Flag to allow vertex buffer as storage buffer
 * @param store_indices Flag to allow index buffer as storage buffer
 * @param renderable Output renderable structure
 */
void mesh_create_renderable(WGPUDevice device, const mesh_t* mesh,
                            bool store_vertices, bool store_indices,
                            mesh_renderable_t* renderable);

/**
 * @brief Destroys a mesh renderable and releases GPU resources
 * @param renderable The renderable to destroy
 */
void mesh_renderable_destroy(mesh_renderable_t* renderable);

/**
 * @brief Gets the position vector at a specific vertex index
 * @param mesh The mesh structure
 * @param index Vertex index
 * @param out_pos Output position vector [x, y, z]
 */
void mesh_get_position_at_index(const mesh_t* mesh, uint64_t index,
                                float out_pos[3]);

/**
 * @brief Gets the normal vector at a specific vertex index
 * @param mesh The mesh structure
 * @param index Vertex index
 * @param out_normal Output normal vector [x, y, z]
 */
void mesh_get_normal_at_index(const mesh_t* mesh, uint64_t index,
                              float out_normal[3]);

/**
 * @brief Gets the UV coordinates at a specific vertex index
 * @param mesh The mesh structure
 * @param index Vertex index
 * @param out_uv Output UV coordinates [u, v]
 */
void mesh_get_uv_at_index(const mesh_t* mesh, uint64_t index, float out_uv[2]);

/* -------------------------------------------------------------------------- *
 * Plane mesh
 * -------------------------------------------------------------------------- */

#define MAX_PLANE_VERTEX_COUNT (1024 * 1024 * 4)

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
 * Sphere mesh
 * -------------------------------------------------------------------------- */

typedef struct sphere_mesh_t {
  struct {
    float* data;
    uint64_t length;
  } vertices;
  struct {
    uint16_t* data;
    uint64_t length;
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

#define STANFORD_DRAGON_POSITION_COUNT_RES_4 (5205)
#define STANFORD_DRAGON_CELL_COUNT_RES_4 (11102)
#define STANFORD_DRAGON_MESH_SCALE (500)

typedef struct stanford_dragon_mesh_t {
  struct {
    float data[STANFORD_DRAGON_POSITION_COUNT_RES_4][3];
    uint64_t count; // number of vertices (should be 5205)
  } positions;
  struct {
    uint16_t data[STANFORD_DRAGON_CELL_COUNT_RES_4][3];
    uint64_t count; // number of faces (should be 11102)
  } triangles;      // triangles
  struct {
    float data[STANFORD_DRAGON_POSITION_COUNT_RES_4][3];
    uint64_t count; // number of normals (should be 5205)
  } normals;
  struct {
    float data[STANFORD_DRAGON_POSITION_COUNT_RES_4][2];
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

/* -------------------------------------------------------------------------- *
 * Utah teapot
 * -------------------------------------------------------------------------- */

#define UTAH_TEAPOT_POSITION_COUNT (792)
#define UTAH_TEAPOT_CELL_COUNT (992)

typedef struct utah_teapot_mesh_t {
  struct {
    float data[UTAH_TEAPOT_POSITION_COUNT][3];
    uint64_t count; // number of vertices (should be 5205)
  } positions;
  struct {
    uint16_t data[UTAH_TEAPOT_CELL_COUNT][3];
    uint64_t count; // number of faces (should be 11102)
  } triangles;      // triangles
  struct {
    float data[UTAH_TEAPOT_POSITION_COUNT][3];
    uint64_t count; // number of normals (should be 5205)
  } normals;
} utah_teapot_mesh_t;

/**
 * @brief Loads the 'Utah teapot' json file ().
 * @see https://github.com/mikolalysenko/teapot
 */
int utah_teapot_mesh_init(utah_teapot_mesh_t* utah_teapot_mesh,
                          const char* const utah_teapot_json);

/**
 * @brief Computes surface normals.
 * @param utah_teapot_mesh mesh object
 */
void utah_teapot_mesh_compute_normals(utah_teapot_mesh_t* utah_teapot_mesh);

/* -------------------------------------------------------------------------- *
 * Generic mesh utility functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Computes surface normals for a generic mesh
 * @param positions Array of position vectors [x, y, z]
 * @param position_count Number of positions
 * @param triangles Array of triangle indices [i0, i1, i2]
 * @param triangle_count Number of triangles
 * @param out_normals Output array for computed normals (must be pre-allocated)
 */
void compute_surface_normals(const float (*positions)[3],
                             uint64_t position_count,
                             const uint16_t (*triangles)[3],
                             uint64_t triangle_count, float (*out_normals)[3]);

/**
 * @brief Computes projected plane UVs for a generic mesh
 * @param positions Array of position vectors [x, y, z]
 * @param position_count Number of positions
 * @param projected_plane Plane to project onto (XY, XZ, or YZ)
 * @param out_uvs Output array for computed UVs (must be pre-allocated)
 */
void compute_projected_plane_uvs(const float (*positions)[3],
                                 uint64_t position_count,
                                 projected_plane_enum projected_plane,
                                 float (*out_uvs)[2]);

/**
 * @brief Result structure for generate_normals function
 */
typedef struct generate_normals_result_t {
  float* positions;        /* Newly generated positions array */
  float* normals;          /* Newly generated normals array */
  uint16_t* triangles;     /* Newly generated triangle indices */
  uint64_t position_count; /* Number of positions/normals */
  uint64_t triangle_count; /* Number of triangles */
} generate_normals_result_t;

/**
 * @brief Generates normals with smooth shading based on max angle
 * This function creates new vertices where needed to achieve proper shading
 * Only faces within max_angle of each other will share smoothed normals
 * @param max_angle Maximum angle in radians for normal smoothing
 * @param positions Input array of position vectors [x, y, z]
 * @param position_count Number of input positions
 * @param triangles Input array of triangle indices [i0, i1, i2]
 * @param triangle_count Number of input triangles
 * @param out_result Output structure with newly allocated data (caller must
 * free)
 * @return 0 on success, non-zero on failure
 */
int generate_normals_with_max_angle(float max_angle,
                                    const float (*positions)[3],
                                    uint64_t position_count,
                                    const uint16_t (*triangles)[3],
                                    uint64_t triangle_count,
                                    generate_normals_result_t* out_result);

/**
 * @brief Frees memory allocated by generate_normals_with_max_angle
 * @param result Result structure to free
 */
void generate_normals_result_destroy(generate_normals_result_t* result);

#endif /* MESHES_H */
