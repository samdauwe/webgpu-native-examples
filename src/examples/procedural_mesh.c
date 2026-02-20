#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#define CGLM_CLIPSPACE_INCLUDE_ALL
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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-conversion"
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#define PAR_SHAPES_IMPLEMENTATION
#include <par_shapes.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Procedural Mesh
 *
 * This WebGPU sample shows how to efficiently draw several procedurally
 * generated meshes:
 *  - All vertices and indices are stored in one large vertex/index buffer
 *  - Two bind groups are used - frame bind group and draw bind group
 *  - Simple physically-based shading is used
 *  - Single-pass wireframe rendering
 *  - Main drawing loop is optimized and changes just one dynamic offset before
 *    each draw call
 *
 * Ref:
 * https://github.com/michal-z/zig-gamedev/tree/main/samples/procedural_mesh_wgpu
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* common_shader_wgsl;
static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define ALIGNMENT 256u
#define MESH_COUNT 11u

/* -------------------------------------------------------------------------- *
 * Mesh generation using par_shapes
 *
 * @ref
 * https://github.com/michal-z/zig-gamedev/blob/main/libs/zmesh/src/Shape.zig
 * @see https://prideout.net/shapes
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    uint16_t* data;
    size_t len;
  } indices;
  struct {
    vec3* data;
    size_t len;
  } positions;
  struct {
    vec3* data;
    size_t len;
  } normals;
  par_shapes_mesh* handle;
} shape_t;

static void shape_deinit(shape_t* mesh)
{
  if (mesh->handle != NULL) {
    par_shapes_free_mesh(mesh->handle);
    mesh->handle = NULL;
  }
  mesh->indices.data   = NULL;
  mesh->indices.len    = 0;
  mesh->positions.data = NULL;
  mesh->positions.len  = 0;
  mesh->normals.data   = NULL;
  mesh->normals.len    = 0;
}

static shape_t init_shape(par_shapes_mesh* handle)
{
  return (shape_t){
    .indices.data   = (uint16_t*)&(handle->triangles[0]),
    .indices.len    = (size_t)(handle->ntriangles * 3),
    .positions.data = (vec3*)&(handle->points[0]),
    .positions.len  = (size_t)handle->npoints,
    .normals.data
    = (handle->normals != NULL) ? (vec3*)&(handle->normals[0]) : NULL,
    .normals.len = (handle->normals != NULL) ? (size_t)handle->npoints : 0,
    .handle      = handle,
  };
}

static void shape_invert(shape_t* shape, int face, int nfaces)
{
  par_shapes_invert(shape->handle, face, nfaces);
  *shape = init_shape(shape->handle);
}

static void shape_merge(shape_t* dst, shape_t* src)
{
  par_shapes_merge(dst->handle, src->handle);
  *dst = init_shape(dst->handle);
}

static void shape_rotate(shape_t* shape, float radians, float x, float y,
                         float z)
{
  float axis[3] = {x, y, z};
  par_shapes_rotate(shape->handle, radians, axis);
  *shape = init_shape(shape->handle);
}

static void shape_scale(shape_t* shape, float x, float y, float z)
{
  par_shapes_scale(shape->handle, x, y, z);
  *shape = init_shape(shape->handle);
}

static void shape_translate(shape_t* shape, float x, float y, float z)
{
  par_shapes_translate(shape->handle, x, y, z);
  *shape = init_shape(shape->handle);
}

static void shape_unweld(shape_t* shape)
{
  par_shapes_unweld(shape->handle, true);
  *shape = init_shape(shape->handle);
}

static void shape_compute_normals(shape_t* shape)
{
  par_shapes_compute_normals(shape->handle);
  *shape = init_shape(shape->handle);
}

static shape_t init_cylinder(int32_t slices, int32_t stacks)
{
  return init_shape(par_shapes_create_cylinder(slices, stacks));
}

static shape_t init_dodecahedron(void)
{
  return init_shape(par_shapes_create_dodecahedron());
}

static shape_t init_icosahedron(void)
{
  return init_shape(par_shapes_create_icosahedron());
}

static shape_t init_octahedron(void)
{
  return init_shape(par_shapes_create_octahedron());
}

static void terrain_generator(float const* uv, float* position, void* userdata)
{
  UNUSED_VAR(userdata);
  position[0] = uv[0];
  position[1] = 0.025f * random_float_min_max(uv[0], uv[1]);
  position[2] = uv[1];
}

static shape_t init_parametric(par_shapes_fn fn, int slices, int stacks,
                               void* userdata)
{
  return init_shape(par_shapes_create_parametric(fn, slices, stacks, userdata));
}

static shape_t init_parametric_disk(int32_t slices, int32_t stacks)
{
  return init_shape(par_shapes_create_parametric_disk(slices, stacks));
}

static shape_t init_parametric_sphere(int32_t slices, int32_t stacks)
{
  return init_shape(par_shapes_create_parametric_sphere(slices, stacks));
}

static shape_t init_rock(int32_t seed, int32_t subd)
{
  return init_shape(par_shapes_create_rock(seed, subd));
}

static shape_t init_subdivided_sphere(int32_t nsubd)
{
  return init_shape(par_shapes_create_subdivided_sphere(nsubd));
}

static shape_t init_tetrahedron(void)
{
  return init_shape(par_shapes_create_tetrahedron());
}

static shape_t init_torus(int32_t slices, int32_t stacks, float radius)
{
  return init_shape(par_shapes_create_torus(slices, stacks, radius));
}

static shape_t init_trefoil_knot(int32_t slices, int32_t stacks, float radius)
{
  return init_shape(par_shapes_create_trefoil_knot(slices, stacks, radius));
}

/* -------------------------------------------------------------------------- *
 * Procedural Mesh types
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 position;
  vec3 normal;
} vertex_t;

typedef struct {
  mat4 world_to_clip;
  vec3 camera_position;
  float _pad;
} frame_uniforms_t;

typedef struct {
  mat4 object_to_world;
  vec4 basecolor_roughness;
} draw_uniforms_t;

typedef struct {
  uint32_t index_offset;
  int32_t vertex_offset;
  uint32_t num_indices;
  uint32_t num_vertices;
} mesh_t;

typedef struct {
  uint32_t mesh_index;
  vec3 position;
  vec4 basecolor_roughness;
} drawable_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Geometry */
  uint32_t total_num_vertices;
  uint32_t total_num_indices;
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  /* Uniform buffers */
  struct {
    wgpu_buffer_t frame;
    struct {
      wgpu_buffer_t buffer;
      uint64_t model_size;
    } draw;
  } uniform_buffers;
  /* Bind groups and layouts */
  WGPUBindGroupLayout frame_bind_group_layout;
  WGPUBindGroupLayout draw_bind_group_layout;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroup frame_bind_group;
  WGPUBindGroup draw_bind_group;
  /* Render pipeline */
  WGPURenderPipeline pipeline;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Scene data */
  drawable_t drawables[MESH_COUNT];
  mesh_t meshes[MESH_COUNT];
  frame_uniforms_t frame_uniforms;
  struct {
    draw_uniforms_t data;
    uint8_t padding[ALIGNMENT - sizeof(draw_uniforms_t)];
  } draw_uniforms[MESH_COUNT];
  /* Camera */
  struct {
    vec3 eye;
    vec3 target;
    float fov;
    float near_plane;
    float far_plane;
    float orbit_angle;
    float orbit_speed;
    bool auto_rotate;
  } camera;
  /* GUI */
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
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
    .eye        = {0.0f, 4.0f, 0.0f},
    .target     = {0.0f, 0.0f, 3.0f},
    .fov        = 45.0f,
    .near_plane = 0.01f,
    .far_plane  = 200.0f,
    .orbit_angle = GLM_PI + 0.25f * GLM_PI,
    .orbit_speed = 0.0f,
    .auto_rotate = false,
  },
};

/* -------------------------------------------------------------------------- *
 * Mesh scene building
 * -------------------------------------------------------------------------- */

static void append_mesh(uint32_t mesh_index, shape_t* mesh, mesh_t* meshes,
                        uint16_t** meshes_indices, uint32_t* meshes_indices_len,
                        vec3** meshes_positions, uint32_t* meshes_positions_len,
                        vec3** meshes_normals, uint32_t* meshes_normals_len)
{
  meshes[mesh_index] = (mesh_t){
    .index_offset  = *meshes_indices_len,
    .vertex_offset = (int32_t)*meshes_positions_len,
    .num_indices   = (uint32_t)mesh->indices.len,
    .num_vertices  = (uint32_t)mesh->positions.len,
  };

  /* Indices */
  *meshes_indices_len += (uint32_t)mesh->indices.len;
  uint32_t idx_size = (*meshes_indices_len) * (uint32_t)sizeof(uint16_t);
  if (*meshes_indices == NULL) {
    *meshes_indices = (uint16_t*)malloc(idx_size);
  }
  else {
    *meshes_indices = (uint16_t*)realloc(*meshes_indices, idx_size);
  }
  memcpy(&(*meshes_indices)[*meshes_indices_len - (uint32_t)mesh->indices.len],
         mesh->indices.data, mesh->indices.len * sizeof(uint16_t));

  /* Positions */
  *meshes_positions_len += (uint32_t)mesh->positions.len;
  uint32_t pos_size = (*meshes_positions_len) * (uint32_t)sizeof(vec3);
  if (*meshes_positions == NULL) {
    *meshes_positions = (vec3*)malloc(pos_size);
  }
  else {
    *meshes_positions = (vec3*)realloc(*meshes_positions, pos_size);
  }
  memcpy(
    &(*meshes_positions)[*meshes_positions_len - (uint32_t)mesh->positions.len],
    mesh->positions.data, mesh->positions.len * sizeof(vec3));

  /* Normals */
  *meshes_normals_len += (uint32_t)mesh->normals.len;
  uint32_t norm_size = (*meshes_normals_len) * (uint32_t)sizeof(vec3);
  if (*meshes_normals == NULL) {
    *meshes_normals = (vec3*)malloc(norm_size);
  }
  else {
    *meshes_normals = (vec3*)realloc(*meshes_normals, norm_size);
  }
  memcpy(&(*meshes_normals)[*meshes_normals_len - (uint32_t)mesh->normals.len],
         mesh->normals.data, mesh->normals.len * sizeof(vec3));
}

/**
 * @brief Initialize a scene with parametric surfaces and other simple shapes.
 * @see https://prideout.net/shapes
 */
static void init_scene(drawable_t* drawables, mesh_t* meshes,
                       uint16_t** meshes_indices, uint32_t* meshes_indices_len,
                       vec3** meshes_positions, uint32_t* meshes_positions_len,
                       vec3** meshes_normals, uint32_t* meshes_normals_len)
{
  uint32_t mesh_index = 0;

  /* Trefoil knot */
  {
    shape_t mesh = init_trefoil_knot(10, 128, 0.8f);
    shape_rotate(&mesh, (float)GLM_PI_2, 1.0f, 0.0f, 0.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {0.0f, 1.0f, 0.0f},
      .basecolor_roughness = {0.0f, 0.7f, 0.0f, 0.6f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Parametric sphere */
  {
    shape_t mesh = init_parametric_sphere(20, 20);
    shape_rotate(&mesh, (float)GLM_PI_2, 1.0f, 0.0f, 0.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {3.0f, 1.0f, 0.0f},
      .basecolor_roughness = {0.7f, 0.0f, 0.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Icosahedron */
  {
    shape_t mesh = init_icosahedron();
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {-3.0f, 1.0f, 0.0f},
      .basecolor_roughness = {0.7f, 0.6f, 0.0f, 0.4f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Dodecahedron */
  {
    shape_t mesh = init_dodecahedron();
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {0.0f, 1.0f, 3.0f},
      .basecolor_roughness = {0.0f, 0.1f, 1.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Cylinder with top and bottom caps */
  {
    shape_t disk = init_parametric_disk(10, 2);
    shape_invert(&disk, 0, 0);
    shape_t cylinder = init_cylinder(10, 4);
    shape_merge(&cylinder, &disk);
    shape_translate(&cylinder, 0.0f, 0.0f, -1.0f);
    shape_invert(&disk, 0, 0);
    shape_merge(&cylinder, &disk);
    shape_scale(&cylinder, 0.5f, 0.5f, 2.0f);
    shape_rotate(&cylinder, (float)GLM_PI_2, 1.0f, 0.0f, 0.0f);
    shape_unweld(&cylinder);
    shape_compute_normals(&cylinder);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {-3.0f, 0.0f, 3.0f},
      .basecolor_roughness = {1.0f, 0.0f, 0.0f, 0.3f},
    };
    append_mesh(mesh_index, &cylinder, meshes, meshes_indices,
                meshes_indices_len, meshes_positions, meshes_positions_len,
                meshes_normals, meshes_normals_len);
    shape_deinit(&cylinder);
    shape_deinit(&disk);
  }
  /* Torus */
  {
    shape_t mesh = init_torus(10, 20, 0.2f);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {3.0f, 1.5f, 3.0f},
      .basecolor_roughness = {1.0f, 0.5f, 0.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Subdivided sphere */
  {
    shape_t mesh = init_subdivided_sphere(3);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {3.0f, 1.0f, 6.0f},
      .basecolor_roughness = {0.0f, 1.0f, 0.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Tetrahedron */
  {
    shape_t mesh = init_tetrahedron();
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {0.0f, 0.5f, 6.0f},
      .basecolor_roughness = {1.0f, 0.0f, 1.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Octahedron */
  {
    shape_t mesh = init_octahedron();
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {-3.0f, 1.0f, 6.0f},
      .basecolor_roughness = {0.2f, 0.0f, 1.0f, 0.2f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Rock */
  {
    shape_t mesh = init_rock(123, 4);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {-6.0f, 0.0f, 3.0f},
      .basecolor_roughness = {1.0f, 1.0f, 1.0f, 1.0f},
    };
    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&mesh);
  }
  /* Custom parametric (simple terrain) */
  {
    shape_t ground = init_parametric(terrain_generator, 40, 40, NULL);
    shape_translate(&ground, -0.5f, -0.0f, -0.5f);
    shape_invert(&ground, 0, 0);
    shape_scale(&ground, 20.0f, 20.0f, 20.0f);
    shape_compute_normals(&ground);
    mesh_index++;
    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .position            = {0.0f, 0.0f, 0.0f},
      .basecolor_roughness = {0.1f, 0.1f, 0.1f, 1.0f},
    };
    append_mesh(mesh_index, &ground, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);
    shape_deinit(&ground);
  }
}

/* -------------------------------------------------------------------------- *
 * Initialization functions
 * -------------------------------------------------------------------------- */

static void init_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  uint16_t* meshes_indices      = NULL;
  uint32_t meshes_indices_len   = 0;
  vec3* meshes_positions        = NULL;
  uint32_t meshes_positions_len = 0;
  vec3* meshes_normals          = NULL;
  uint32_t meshes_normals_len   = 0;

  init_scene(state.drawables, state.meshes, &meshes_indices,
             &meshes_indices_len, &meshes_positions, &meshes_positions_len,
             &meshes_normals, &meshes_normals_len);

  state.total_num_vertices = meshes_positions_len;
  state.total_num_indices  = meshes_indices_len;

  /* Create interleaved vertex buffer */
  {
    vertex_t* vertex_data
      = (vertex_t*)malloc(state.total_num_vertices * sizeof(vertex_t));
    for (uint32_t i = 0; i < meshes_positions_len; ++i) {
      glm_vec3_copy(meshes_positions[i], vertex_data[i].position);
      glm_vec3_copy(meshes_normals[i], vertex_data[i].normal);
    }
    state.vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Procedural mesh - Vertices buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = state.total_num_vertices * sizeof(vertex_t),
                      .initial.data = vertex_data,
                    });
    free(vertex_data);
  }

  /* Create index buffer */
  state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Procedural mesh - Indices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = state.total_num_indices * sizeof(uint16_t),
                    .initial.data = meshes_indices,
                  });

  /* Cleanup temporary buffers */
  free(meshes_indices);
  free(meshes_positions);
  free(meshes_normals);
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Frame uniform buffer */
  state.uniform_buffers.frame = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Procedural mesh - Frame uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(frame_uniforms_t),
                  });

  /* Draw uniform buffer (with alignment padding for dynamic offsets) */
  state.uniform_buffers.draw.model_size = sizeof(draw_uniforms_t);
  state.uniform_buffers.draw.buffer     = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                        .label = "Procedural mesh - Draw uniform buffer",
                        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                        .size  = sizeof(state.draw_uniforms),
                  });
}

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Frame bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(frame_uniforms_t),
        },
      },
    };
    state.frame_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Frame - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(state.frame_bind_group_layout != NULL);
  }

  /* Draw bind group layout (with dynamic offset) */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout){
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = state.uniform_buffers.draw.model_size,
        },
      },
    };
    state.draw_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = STRVIEW("Draw - Bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(state.draw_bind_group_layout != NULL);
  }
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    state.frame_bind_group_layout, /* Group 0 */
    state.draw_bind_group_layout,  /* Group 1 */
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Procedural mesh - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Merge common + vertex shader */
  size_t common_len = strlen(common_shader_wgsl);
  size_t vert_len   = strlen(vertex_shader_wgsl);
  size_t frag_len   = strlen(fragment_shader_wgsl);

  char* vert_full = (char*)malloc(common_len + vert_len + 2);
  memcpy(vert_full, common_shader_wgsl, common_len);
  vert_full[common_len] = '\n';
  memcpy(vert_full + common_len + 1, vertex_shader_wgsl, vert_len + 1);

  char* frag_full = (char*)malloc(common_len + frag_len + 2);
  memcpy(frag_full, common_shader_wgsl, common_len);
  frag_full[common_len] = '\n';
  memcpy(frag_full + common_len + 1, fragment_shader_wgsl, frag_len + 1);

  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vert_full);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, frag_full);

  free(vert_full);
  free(frag_full);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(mesh, sizeof(vertex_t),
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, position)),
                            /* Attribute location 1: Normal */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, normal)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Procedural mesh - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &mesh_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CW,
      .cullMode  = WGPUCullMode_Back,
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff,
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Frame bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffers.frame.buffer,
        .offset  = 0,
        .size    = state.uniform_buffers.frame.size,
      },
    };
    state.frame_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Frame - Bind group"),
                              .layout     = state.frame_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.frame_bind_group != NULL);
  }

  /* Draw bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry){
        .binding = 0,
        .buffer  = state.uniform_buffers.draw.buffer.buffer,
        .offset  = 0,
        .size    = state.uniform_buffers.draw.model_size,
      },
    };
    state.draw_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Draw - Bind group"),
                              .layout     = state.draw_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.draw_bind_group != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Update functions
 * -------------------------------------------------------------------------- */

static void update_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Compute camera eye position */
  vec3 eye;
  if (state.camera.auto_rotate) {
    state.camera.orbit_angle += state.camera.orbit_speed * (1.0f / 60.0f);
  }
  float r = 18.0f;
  eye[0]  = r * sinf(state.camera.orbit_angle);
  eye[1]  = state.camera.eye[1];
  eye[2]  = r * cosf(state.camera.orbit_angle);

  /* View matrix (left-handed to match WebGPU clip space) */
  mat4 view;
  glm_lookat_lh_zo(eye, state.camera.target, (vec3){0.0f, 1.0f, 0.0f}, view);

  /* Projection matrix (left-handed, zero-to-one depth for WebGPU) */
  mat4 proj;
  glm_perspective_lh_zo(glm_rad(state.camera.fov), aspect_ratio,
                        state.camera.near_plane, state.camera.far_plane, proj);

  /* Combined world_to_clip = projection * view */
  mat4 world_to_clip;
  glm_mat4_mul(proj, view, world_to_clip);

  /* Transpose for row-vector shader convention (vec * mat) */
  glm_mat4_transpose_to(world_to_clip, state.frame_uniforms.world_to_clip);

  /* Camera position for PBR lighting */
  glm_vec3_copy(eye, state.frame_uniforms.camera_position);

  /* Write to GPU */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.frame.buffer,
                       0, &state.frame_uniforms, sizeof(frame_uniforms_t));
}

static void update_draw_uniforms(wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < MESH_COUNT; ++i) {
    drawable_t* drawable = &state.drawables[i];
    mat4 model;
    glm_mat4_identity(model);
    glm_translate(model, drawable->position);

    draw_uniforms_t* du = &state.draw_uniforms[i].data;
    glm_mat4_transpose_to(model, du->object_to_world);
    glm_vec4_copy(drawable->basecolor_roughness, du->basecolor_roughness);
  }

  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.uniform_buffers.draw.buffer.buffer, 0,
                       state.draw_uniforms, sizeof(state.draw_uniforms));
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

  igBegin("Procedural Mesh", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Info", NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_text("Left Mouse + drag to orbit");
    imgui_overlay_text("Meshes: %u", MESH_COUNT);
    imgui_overlay_text("Total vertices: %u", state.total_num_vertices);
    imgui_overlay_text("Total indices: %u", state.total_num_indices);
  }

  if (igCollapsingHeaderBoolPtr("Camera", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_checkbox("Auto Rotate", &state.camera.auto_rotate);
    imgui_overlay_slider_float("Orbit Speed", &state.camera.orbit_speed, 0.0f,
                               2.0f, "%.2f");
    imgui_overlay_slider_float("FOV", &state.camera.fov, 10.0f, 90.0f, "%.0f");
    imgui_overlay_slider_float("Eye Y", &state.camera.eye[1], -10.0f, 10.0f,
                               "%.1f");
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Let ImGui handle mouse if it wants it */
  if (imgui_overlay_want_capture_mouse()) {
    return;
  }

  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
      && input_event->mouse_btn_pressed
      && input_event->mouse_button == BUTTON_LEFT) {
    state.camera.orbit_angle -= input_event->mouse_dx * 0.005f;
    state.camera.eye[1] += input_event->mouse_dy * 0.05f;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_SCROLL) {
    state.camera.fov -= input_event->scroll_y * 1.0f;
    state.camera.fov = MAX(10.0f, MIN(90.0f, state.camera.fov));
  }
}

/* -------------------------------------------------------------------------- *
 * Lifecycle callbacks
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_vertex_and_index_buffers(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_pipeline(wgpu_context);
    init_bind_groups(wgpu_context);
    update_draw_uniforms(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update uniforms */
  update_view_matrices(wgpu_context);

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Bind vertex buffer */
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertex_buffer.buffer,
                                       0, WGPU_WHOLE_SIZE);

  /* Bind index buffer */
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.index_buffer.buffer,
                                      WGPUIndexFormat_Uint16, 0,
                                      WGPU_WHOLE_SIZE);

  /* Bind pipeline */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);

  /* Set frame bind group */
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.frame_bind_group, 0, 0);

  /* Draw indexed geometries with dynamic offsets */
  for (uint32_t i = 0; i < MESH_COUNT; ++i) {
    uint32_t dynamic_offset = i * ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 1, state.draw_bind_group, 1,
                                      &dynamic_offset);
    wgpuRenderPassEncoderDrawIndexed(rpass_enc, state.meshes[i].num_indices, 1,
                                     state.meshes[i].index_offset,
                                     state.meshes[i].vertex_offset, 0);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
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

  imgui_overlay_shutdown();

  wgpu_destroy_buffer(&state.vertex_buffer);
  wgpu_destroy_buffer(&state.index_buffer);
  wgpu_destroy_buffer(&state.uniform_buffers.frame);
  wgpu_destroy_buffer(&state.uniform_buffers.draw.buffer);
  WGPU_RELEASE_RESOURCE(BindGroup, state.frame_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.draw_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.frame_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.draw_bind_group_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Procedural Mesh",
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
static const char* common_shader_wgsl = CODE(
  struct DrawUniforms {
    object_to_world: mat4x4<f32>,
    basecolor_roughness: vec4<f32>,
  }
  @group(1) @binding(0) var<uniform> draw_uniforms: DrawUniforms;

  struct FrameUniforms {
    world_to_clip: mat4x4<f32>,
    camera_position: vec3<f32>,
  }
  @group(0) @binding(0) var<uniform> frame_uniforms: FrameUniforms;
);

static const char* vertex_shader_wgsl = CODE(
  struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentrics: vec3<f32>,
  }
  @vertex fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
  ) -> VertexOut {
    var output: VertexOut;
    output.position_clip = vec4(position, 1.0) * draw_uniforms.object_to_world
                           * frame_uniforms.world_to_clip;
    output.position = (vec4(position, 1.0) * draw_uniforms.object_to_world).xyz;
    output.normal = normal * mat3x3(
      draw_uniforms.object_to_world[0].xyz,
      draw_uniforms.object_to_world[1].xyz,
      draw_uniforms.object_to_world[2].xyz,
    );
    let index = vertex_index % 3u;
    output.barycentrics = vec3(
      f32(index == 0u), f32(index == 1u), f32(index == 2u)
    );
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  const pi = 3.1415926;

  fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

  fn distributionGgx(n: vec3<f32>, h: vec3<f32>, alpha: f32) -> f32 {
    let alpha_sq = alpha * alpha;
    let n_dot_h = saturate(dot(n, h));
    let k = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / (pi * k * k);
  }

  fn geometrySchlickGgx(x: f32, k: f32) -> f32 {
    return x / (x * (1.0 - k) + k);
  }

  fn geometrySmith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>,
                   k: f32) -> f32 {
    let n_dot_v = saturate(dot(n, v));
    let n_dot_l = saturate(dot(n, l));
    return geometrySchlickGgx(n_dot_v, k) * geometrySchlickGgx(n_dot_l, k);
  }

  fn fresnelSchlick(h_dot_v: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.0, 1.0, 1.0) - f0) * pow(1.0 - h_dot_v, 5.0);
  }

  @fragment fn main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentrics: vec3<f32>,
  ) -> @location(0) vec4<f32> {
    let v = normalize(frame_uniforms.camera_position - position);
    let n = normalize(normal);

    let base_color = draw_uniforms.basecolor_roughness.xyz;
    let ao = 1.0;
    var roughness = draw_uniforms.basecolor_roughness.a;
    var metallic: f32;
    if (roughness < 0.0) { metallic = 1.0; } else { metallic = 0.0; }
    roughness = abs(roughness);

    let alpha = roughness * roughness;
    var k = alpha + 1.0;
    k = (k * k) / 8.0;
    var f0 = vec3(0.04);
    f0 = mix(f0, base_color, metallic);

    let light_positions = array<vec3<f32>, 4>(
      vec3(25.0, 15.0, 25.0),
      vec3(-25.0, 15.0, 25.0),
      vec3(25.0, 15.0, -25.0),
      vec3(-25.0, 15.0, -25.0),
    );
    let light_radiance = array<vec3<f32>, 4>(
      4.0 * vec3(0.0, 100.0, 250.0),
      8.0 * vec3(200.0, 150.0, 250.0),
      3.0 * vec3(200.0, 0.0, 0.0),
      9.0 * vec3(200.0, 150.0, 0.0),
    );

    var lo = vec3(0.0);
    for (var light_index: i32 = 0; light_index < 4;
         light_index = light_index + 1) {
      let lvec = light_positions[light_index] - position;
      let l = normalize(lvec);
      let h = normalize(l + v);
      let distance_sq = dot(lvec, lvec);
      let attenuation = 1.0 / distance_sq;
      let radiance = light_radiance[light_index] * attenuation;
      let f = fresnelSchlick(saturate(dot(h, v)), f0);
      let ndf = distributionGgx(n, h, alpha);
      let g = geometrySmith(n, v, l, k);
      let numerator = ndf * g * f;
      let denominator = 4.0 * saturate(dot(n, v)) * saturate(dot(n, l));
      let specular = numerator / max(denominator, 0.001);
      let ks = f;
      let kd = (vec3(1.0) - ks) * (1.0 - metallic);
      let n_dot_l = saturate(dot(n, l));
      lo = lo + (kd * base_color / pi + specular) * radiance * n_dot_l;
    }

    let ambient = vec3(0.03) * base_color * ao;
    var color = ambient + lo;
    color = color / (color + 1.0);
    color = pow(color, vec3(1.0 / 2.2));

    var barys = barycentrics;
    barys.z = 1.0 - barys.x - barys.y;
    let deltas = fwidth(barys);
    let smoothing = deltas * 1.0;
    let thickness = deltas * 0.25;
    barys = smoothstep(thickness, thickness + smoothing, barys);
    let min_bary = min(barys.x, min(barys.y, barys.z));
    return vec4(min_bary * color, 1.0);
  }
);
// clang-format on
