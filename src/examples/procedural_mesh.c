#include "example_base.h"

#include "../webgpu/imgui_overlay.h"

#define PAR_SHAPES_IMPLEMENTATION
#include "par_shapes.h"

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
 * Procedural Mesh example
 * -------------------------------------------------------------------------- */

#define ALIGNMENT 256u // 256-byte alignment
#define MESH_COUNT 11u

/* -------------------------------------------------------------------------- *
 * Mesh generation
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
  struct {
    vec2* data;
    size_t len;
  } texcoords;
  par_shapes_mesh* handle;
} shape_t;

static void shape_deinit(shape_t* mesh)
{
  if (mesh->indices.len > 0) {
    mesh->indices.data = NULL;
    mesh->indices.len  = 0;
  }
  if (mesh->positions.len > 0) {
    mesh->positions.data = NULL;
    mesh->positions.len  = 0;
  }
  if (mesh->normals.len > 0) {
    mesh->normals.data = NULL;
    mesh->normals.len  = 0;
  }
  if (mesh->texcoords.len > 0) {
    mesh->texcoords.data = NULL;
    mesh->texcoords.len  = 0;
  }
  if (mesh->handle != NULL) {
    par_shapes_free_mesh(mesh->handle);
    mesh->handle = NULL;
  }
}

static shape_t init_shape(par_shapes_mesh* handle)
{
  return (shape_t){
    .indices.data   = (uint16_t*)&(handle->triangles[0]),
    .indices.len    = handle->ntriangles * 3,
    .positions.data = (vec3*)&(handle->points[0]),
    .positions.len  = handle->npoints,
    .normals.data
    = (handle->normals != NULL) ? (vec3*)&(handle->normals[0]) : NULL,
    .normals.len = (handle->normals != NULL) ? handle->npoints : 0,
    .texcoords.data
    = (handle->tcoords != NULL) ? (vec2*)&(handle->tcoords[0]) : NULL,
    .texcoords.len = (handle->tcoords != NULL) ? handle->npoints : 0,
    .handle        = handle,
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
  par_shapes_rotate(shape->handle, radians, (float[]){x, y, z});
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
 * Procedural Mesh Example
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 position;
  vec3 normal;
} vertex_t;

typedef struct {
  mat4 projection_matrix;
  mat4 view_matrix;
} frame_uniforms_t;

typedef struct {
  mat4 model;
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

static struct {
  WGPUBindGroupLayout frame_bind_group_layout;
  WGPUBindGroupLayout draw_bind_group_layout;
  WGPUPipelineLayout pipeline_layout;

  WGPUBindGroup frame_bind_group;
  WGPUBindGroup draw_bind_group;
  WGPURenderPipeline pipeline;

  uint32_t total_num_vertices;
  uint32_t total_num_indices;

  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  struct {
    wgpu_buffer_t frame;
    struct {
      wgpu_buffer_t buffer;
      uint64_t model_size;
    } draw;
  } uniform_buffers;

  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;

  // Render pass descriptor for frame buffer writes
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;

  drawable_t drawables[MESH_COUNT];
  mesh_t meshes[MESH_COUNT];

  frame_uniforms_t frame_uniforms;

  struct {
    draw_uniforms_t data;
    uint8_t padding[176];
  } draw_uniforms[MESH_COUNT];

  // Other variables
  const char* example_title;
  bool prepared;
} demo_state = {
  .example_title = "Procedural Mesh",
  .prepared      = false,
};

static void setup_camera(wgpu_example_context_t* context)
{
  context->camera       = camera_create();
  context->camera->type = CameraType_LookAt;
  camera_set_position(context->camera, (vec3){1.0f, -1.0f, -18.0f});
  camera_set_rotation(context->camera, (vec3){3.0f * PI2, 0.0f, 0.0f});
  camera_set_rotation_speed(context->camera, 0.25f);
  camera_set_perspective(context->camera, 25.0f,
                         context->window_size.aspect_ratio, 0.01f, 200.0f);
}

static void append_mesh(uint32_t mesh_index, shape_t* mesh, mesh_t* meshes,
                        uint16_t** meshes_indices, uint32_t* meshes_indices_len,
                        vec3** meshes_positions, uint32_t* meshes_positions_len,
                        vec3** meshes_normals, uint32_t* meshes_normals_len)
{
  meshes[mesh_index] = (mesh_t){
    .index_offset  = *meshes_indices_len,
    .vertex_offset = (int32_t)*meshes_positions_len,
    .num_indices   = mesh->indices.len,
    .num_vertices  = mesh->positions.len,
  };

  // Indices
  *meshes_indices_len += mesh->indices.len;
  uint32_t meshes_indices_size
    = (*meshes_indices_len) * sizeof(**meshes_indices);
  if (*meshes_indices == NULL) {
    *meshes_indices = (uint16_t*)malloc(meshes_indices_size);
  }
  else {
    *meshes_indices = (uint16_t*)realloc(*meshes_indices, meshes_indices_size);
  }
  memcpy(&(*meshes_indices)[*meshes_indices_len - mesh->indices.len],
         mesh->indices.data, mesh->indices.len * sizeof(*mesh->indices.data));

  // Positions
  *meshes_positions_len += mesh->positions.len;
  uint32_t meshes_positions_size
    = (*meshes_positions_len) * sizeof(**meshes_positions);
  if (*meshes_positions == NULL) {
    *meshes_positions = (vec3*)malloc(meshes_positions_size);
  }
  else {
    *meshes_positions
      = (vec3*)realloc(*meshes_positions, meshes_positions_size);
  }
  memcpy(&(*meshes_positions)[*meshes_positions_len - mesh->positions.len],
         mesh->positions.data,
         mesh->positions.len * sizeof(*mesh->positions.data));

  // Normals
  *meshes_normals_len += mesh->normals.len;
  uint32_t meshes_normals_size
    = (*meshes_normals_len) * sizeof(**meshes_normals);
  if (*meshes_normals == NULL) {
    *meshes_normals = (vec3*)malloc(meshes_normals_size);
  }
  else {
    *meshes_normals = (vec3*)realloc(*meshes_normals, meshes_normals_size);
  }
  memcpy(&(*meshes_normals)[*meshes_normals_len - mesh->normals.len],
         mesh->normals.data, mesh->normals.len * sizeof(*mesh->normals.data));
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
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 0.0f, -0.5f, -1.0f);
    shape_rotate(&mesh, PI_2, 1.0f, 0.0f, 0.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.0f, 0.7f, 0.0f, 0.6f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Parametric sphere. */
  {
    shape_t mesh = init_parametric_sphere(20, 20);
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 3.0f, -0.5f, -1.0f);
    shape_rotate(&mesh, PI_2, 1.0f, 0.0f, 0.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.7f, 0.0f, 0.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Icosahedron. */
  {
    shape_t mesh = init_icosahedron();
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, -3.0f, 1.0f, -0.5f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.7f, 0.6f, 0.0f, 0.4f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Dodecahedron. */
  {
    shape_t mesh = init_dodecahedron();
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 0.0f, 1.0f, 3.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.0f, 0.1f, 1.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Cylinder with top and bottom caps. */
  {
    shape_t disk = init_parametric_disk(10, 2);
    shape_invert(&disk, 0, 0);

    shape_t cylinder = init_cylinder(10, 4);

    shape_merge(&cylinder, &disk);
    shape_translate(&cylinder, 0.0f, 0.0f, -1.0f);
    shape_invert(&disk, 0, 0);
    shape_merge(&cylinder, &disk);

    shape_scale(&cylinder, 0.5f, 0.5f, 2.0f);
    shape_rotate(&cylinder, PI_2, 1.0f, 0.0f, 0.0f);

    shape_scale(&cylinder, 0.5f, 0.5f, 0.5f);
    shape_translate(&cylinder, -3.0f, 1.0f, 3.0f);
    shape_unweld(&cylinder);
    shape_compute_normals(&cylinder);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {1.0f, 0.0f, 0.0f, 0.3f},
    };

    append_mesh(mesh_index, &cylinder, meshes, meshes_indices,
                meshes_indices_len, meshes_positions, meshes_positions_len,
                meshes_normals, meshes_normals_len);

    shape_deinit(&cylinder);
    shape_deinit(&disk);
  }
  /* Torus. */
  {
    shape_t mesh = init_torus(10, 20, 0.2f);
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 3.0f, 1.0f, 3.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {1.0f, 0.5f, 0.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Subdivided sphere. */
  {
    shape_t mesh = init_subdivided_sphere(3);
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 3.0f, 1.0f, 6.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.0f, 1.0f, 0.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Tetrahedron. */
  {
    shape_t mesh = init_tetrahedron();
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, 0.0f, 1.0f, 6.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {1.0f, 0.0f, 1.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Octahedron. */
  {
    shape_t mesh = init_octahedron();
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, -3.0f, 1.0f, 6.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.2f, 0.0f, 1.0f, 0.2f},
    };

    append_mesh(mesh_index, &mesh, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&mesh);
  }
  /* Rock. */
  {
    shape_t mesh = init_rock(123, 4);
    shape_scale(&mesh, 0.5f, 0.5f, 0.5f);
    shape_translate(&mesh, -6.0f, 1.0f, 3.0f);
    shape_unweld(&mesh);
    shape_compute_normals(&mesh);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
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
    shape_translate(&ground, -0.6f, 0.01f, -0.5f);
    shape_invert(&ground, 0, 0);
    shape_scale(&ground, 20.0f, 20.0f, 20.f);
    shape_unweld(&ground);
    shape_compute_normals(&ground);

    mesh_index++;

    drawables[mesh_index] = (drawable_t){
      .mesh_index          = mesh_index,
      .basecolor_roughness = {0.1f, 0.1f, 0.1f, 1.0f},
    };

    append_mesh(mesh_index, &ground, meshes, meshes_indices, meshes_indices_len,
                meshes_positions, meshes_positions_len, meshes_normals,
                meshes_normals_len);

    shape_deinit(&ground);
  }
}

static void prepare_vertex_and_index_buffer(wgpu_context_t* wgpu_context)
{
  uint16_t* meshes_indices      = NULL;
  uint32_t meshes_indices_len   = 0;
  vec3* meshes_positions        = NULL;
  uint32_t meshes_positions_len = 0;
  vec3* meshes_normals          = NULL;
  uint32_t meshes_normals_len   = 0;
  init_scene(demo_state.drawables, demo_state.meshes, &meshes_indices,
             &meshes_indices_len, &meshes_positions, &meshes_positions_len,
             &meshes_normals, &meshes_normals_len);

  demo_state.total_num_vertices = meshes_positions_len;
  demo_state.total_num_indices  = meshes_indices_len;

  // Create a vertex buffer
  {
    vertex_t* vertex_data
      = (vertex_t*)malloc(demo_state.total_num_vertices * sizeof(vertex_t));
    for (uint32_t i = 0; i < meshes_positions_len; ++i) {
      glm_vec3_copy(meshes_positions[i], vertex_data[i].position);
      glm_vec3_copy(meshes_normals[i], vertex_data[i].normal);
    }
    demo_state.vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = demo_state.total_num_vertices * sizeof(vertex_t),
                      .initial.data = vertex_data,
                    });
    free(vertex_data);
  }

  // Create a index buffer
  demo_state.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = demo_state.total_num_indices * sizeof(uint16_t),
                    .initial.data = meshes_indices,
                  });

  // Cleanup allocated memory
  if (meshes_indices != NULL) {
    free(meshes_indices);
  }
  if (meshes_positions != NULL) {
    free(meshes_positions);
  }
  if (meshes_normals != NULL) {
    free(meshes_normals);
  }
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Frame bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .minBindingSize =  sizeof(frame_uniforms_t),
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    demo_state.frame_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(demo_state.frame_bind_group_layout != NULL);
  }

  /* Draw bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize = demo_state.uniform_buffers.draw.model_size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    demo_state.draw_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(demo_state.draw_bind_group_layout != NULL);
  }
}

static void setup_render_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    demo_state.frame_bind_group_layout, // Group 0
    demo_state.draw_bind_group_layout,  // Group 1
  };
  demo_state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(demo_state.pipeline_layout != NULL);
}

static void prepare_rendering_pipeline(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera is rendered
  // in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth32Float,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(mesh, sizeof(vertex_t),
                            // Attribute location 0: Position
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, position)),
                            // Attribute location 1: Normal
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x3,
                                               offsetof(vertex_t, normal)))

  // Vertex state
  char* vertex_shader_wgsl_full
    = concat_strings(common_shader_wgsl, vertex_shader_wgsl, "\n");
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .wgsl_code.source = vertex_shader_wgsl_full,
                  .entry = "main",
                },
                .buffer_count = 1,
                .buffers = &mesh_vertex_buffer_layout,
              });
  free(vertex_shader_wgsl_full);

  // Fragment state
  char* fragment_shader_wgsl_full
    = concat_strings(common_shader_wgsl, fragment_shader_wgsl, "\n");
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .wgsl_code.source = fragment_shader_wgsl_full,
                  .entry = "main",
                },
                .target_count = 1,
                .targets = &color_target_state_desc,
              });
  free(fragment_shader_wgsl_full);

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  demo_state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "procedural_mesh_render_pipeline",
                            .layout       = demo_state.pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(demo_state.pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void update_frame_uniform_buffers(wgpu_example_context_t* context)
{
  // Update camera matrices
  camera_t* camera = context->camera;
  glm_mat4_copy(camera->matrices.perspective,
                demo_state.frame_uniforms.projection_matrix);
  glm_mat4_copy(camera->matrices.view, demo_state.frame_uniforms.view_matrix);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(
    context->wgpu_context, demo_state.uniform_buffers.frame.buffer, 0,
    &demo_state.frame_uniforms, demo_state.uniform_buffers.frame.size);
}

static void update_draw_uniform_buffers(wgpu_context_t* wgpu_context)
{
  // Update "object to world" xform
  for (uint32_t i = 0; i < MESH_COUNT; ++i) {
    drawable_t* drawable = &demo_state.drawables[i];
    vec3* drawable_pos   = &drawable->position;
    mat4 model           = {
      {1.0f, 0.0f, 0.0f, 0.0f}, //
      {0.0f, 1.0f, 0.0f, 0.0f}, //
      {0.0f, 0.0f, 1.0f, 0.0f}, //
      {(*drawable_pos)[0], (*drawable_pos)[1], (*drawable_pos)[2], 1.0f}, //
    };
    glm_mat4_transpose(model);

    draw_uniforms_t* draw_uniforms = &demo_state.draw_uniforms[i].data;
    glm_mat4_copy(model, draw_uniforms->model);
    glm_vec4_copy(drawable->basecolor_roughness,
                  draw_uniforms->basecolor_roughness);
  }

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(
    wgpu_context, demo_state.uniform_buffers.draw.buffer.buffer, 0,
    demo_state.draw_uniforms, demo_state.uniform_buffers.draw.buffer.size);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Create a frame uniform buffer
  demo_state.uniform_buffers.frame = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = sizeof(frame_uniforms_t),
    });

  // Create a draw uniform buffer
  demo_state.uniform_buffers.draw.model_size = sizeof(draw_uniforms_t);
  demo_state.uniform_buffers.draw.buffer.size
    = sizeof(demo_state.draw_uniforms);
  demo_state.uniform_buffers.draw.buffer = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
      .size  = demo_state.uniform_buffers.draw.buffer.size,
    });

  update_frame_uniform_buffers(context);
  update_draw_uniform_buffers(context->wgpu_context);
}

static void prepare_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Frame bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = demo_state.uniform_buffers.frame.buffer,
        .offset  = 0,
        .size    = demo_state.uniform_buffers.frame.size,
      },
    };
    demo_state.frame_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = demo_state.frame_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(demo_state.frame_bind_group != NULL);
  }

  /* Draw bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = demo_state.uniform_buffers.draw.buffer.buffer,
        .offset  = 0,
        .size    = demo_state.uniform_buffers.draw.model_size,
      },
    };
    demo_state.draw_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout     = demo_state.draw_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(demo_state.draw_bind_group != NULL);
  }
}

static void prepare_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  demo_state.depth_texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(demo_state.depth_texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  demo_state.depth_texture_view
    = wgpuTextureCreateView(demo_state.depth_texture, &texture_view_dec);
  ASSERT(demo_state.depth_texture_view != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  demo_state.render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // Assigned later
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Depth stencil attachment
  demo_state.render_pass.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view              = demo_state.depth_texture_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilClearValue = 1,
    };

  // Render pass descriptor
  demo_state.render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = demo_state.render_pass.color_attachments,
    .depthStencilAttachment = &demo_state.render_pass.depth_stencil_attachment,
  };
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_camera(context);
    prepare_vertex_and_index_buffer(context->wgpu_context);
    setup_bind_group_layouts(context->wgpu_context);
    setup_render_pipeline_layout(context->wgpu_context);
    prepare_rendering_pipeline(context->wgpu_context);
    prepare_uniform_buffers(context);
    prepare_bind_groups(context->wgpu_context);
    prepare_depth_texture(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    demo_state.prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Info")) {
    UNUSED_VAR(context);
    imgui_overlay_text("%s", "Left Mouse Btn + drag to rotate camera");
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  demo_state.render_pass.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &demo_state.render_pass.descriptor);

  // Bind vertex buffer (contains positions and normals)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       demo_state.vertex_buffer.buffer, 0,
                                       demo_state.vertex_buffer.size);

  // Bind index buffer
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, demo_state.index_buffer.buffer,
    WGPUIndexFormat_Uint16, 0, demo_state.index_buffer.size);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                   demo_state.pipeline);

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    demo_state.frame_bind_group, 0, 0);

  // Draw indexed geometries
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(demo_state.drawables); ++i) {
    uint32_t dynamic_offset = i * ALIGNMENT;
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                      demo_state.draw_bind_group, 1,
                                      &dynamic_offset);
    wgpuRenderPassEncoderDrawIndexed(
      wgpu_context->rpass_enc, demo_state.meshes[i].num_indices, 1,
      demo_state.meshes[i].index_offset, demo_state.meshes[i].vertex_offset, 0);
  }

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!demo_state.prepared) {
    return EXIT_FAILURE;
  }
  return example_draw(context);
}

static void example_on_view_changed(wgpu_example_context_t* context)
{
  update_frame_uniform_buffers(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  camera_release(context->camera);

  WGPU_RELEASE_RESOURCE(Buffer, demo_state.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.uniform_buffers.frame.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, demo_state.uniform_buffers.draw.buffer.buffer)

  WGPU_RELEASE_RESOURCE(Texture, demo_state.depth_texture)
  WGPU_RELEASE_RESOURCE(TextureView, demo_state.depth_texture_view)

  WGPU_RELEASE_RESOURCE(BindGroup, demo_state.draw_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, demo_state.frame_bind_group)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, demo_state.draw_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, demo_state.frame_bind_group_layout)

  WGPU_RELEASE_RESOURCE(PipelineLayout, demo_state.pipeline_layout)

  WGPU_RELEASE_RESOURCE(RenderPipeline, demo_state.pipeline)
}

void example_procedural_mesh(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = demo_state.example_title,
     .overlay = true,
     .vsync   = true,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
    .example_on_view_changed_func = &example_on_view_changed,
  });
  // clang-format on
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* common_shader_wgsl = CODE(
  struct FrameUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
  }
  @group(0) @binding(0) var<uniform> frame_uniforms: FrameUniforms;

  struct DrawUniforms {
    model: mat4x4<f32>,
    basecolor_roughness: vec4<f32>,
  }
  @group(1) @binding(0) var<uniform> draw_uniforms: DrawUniforms;
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
      let modelView = frame_uniforms.view * draw_uniforms.model;
      var output: VertexOut;
      output.position_clip = frame_uniforms.projection * modelView * vec4(position.xyz, 1.0);
      output.position = (modelView * vec4(position, 1.0)).xyz;
      output.normal = normal * mat3x3(
        draw_uniforms.model[0].xyz,
        draw_uniforms.model[1].xyz,
        draw_uniforms.model[2].xyz,
      );
      let index = vertex_index % 3u;
      output.barycentrics = vec3(f32(index == 0u), f32(index == 1u), f32(index == 2u));
      return output;
  }
  );

static const char* fragment_shader_wgsl = CODE(
  const pi = 3.1415926;

  fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

  // Trowbridge-Reitz GGX normal distribution function.
  fn distributionGgx(n: vec3<f32>, h: vec3<f32>, alpha: f32) -> f32 {
    let alpha_sq = alpha * alpha;
    let n_dot_h = saturate(dot(n, h));
    let k = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / (pi * k * k);
  }

  fn geometrySchlickGgx(x: f32, k: f32) -> f32 {
    return x / (x * (1.0 - k) + k);
  }

  fn geometrySmith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, k: f32) -> f32 {
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
    let v = normalize(position);
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
    for (var light_index: i32 = 0; light_index < 4; light_index = light_index + 1) {
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

    // wireframe
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
