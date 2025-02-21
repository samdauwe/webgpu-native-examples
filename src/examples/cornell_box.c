#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cornell Box
 *
 * A classic Cornell box, using a lightmap generated using software ray-tracing.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/cornell
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Shader store
 * -------------------------------------------------------------------------- */

#define PRESENTATION_FORMAT "bgra8unorm"

typedef struct {
  const char* filename;
  file_read_result_t read_result;
} shader_store_entry;

static struct {
  shader_store_entry common;
  shader_store_entry radiosity;
  shader_store_entry rasterizer;
  shader_store_entry raytracer;
  shader_store_entry tonemapper;
} shader_store = {
  .common.filename     = "shaders/cornell_box/common.wgsl",
  .radiosity.filename  = "shaders/cornell_box/radiosity.wgsl",
  .rasterizer.filename = "shaders/cornell_box/rasterizer.wgsl",
  .raytracer.filename  = "shaders/cornell_box/raytracer.wgsl",
  .tonemapper.filename = "shaders/cornell_box/tonemapper.wgsl",
};

static void initialize_shader_store_entry(shader_store_entry* entry)
{
  read_file(entry->filename, &entry->read_result, true);
  log_debug("Read file: %s, size: %d bytes\n", entry->filename,
            entry->read_result.size);
  ASSERT(entry->read_result.size > 0);
}

static void create_shader_store(void)
{
  initialize_shader_store_entry(&shader_store.common);
  initialize_shader_store_entry(&shader_store.radiosity);
  initialize_shader_store_entry(&shader_store.rasterizer);
  initialize_shader_store_entry(&shader_store.raytracer);
  initialize_shader_store_entry(&shader_store.tonemapper);
}

static void destroy_shader_store_entry(shader_store_entry* entry)
{
  if (entry->read_result.size > 0) {
    free(entry->read_result.data);
    entry->read_result.size = 0;
  }
}

static void destroy_shader_store(void)
{
  destroy_shader_store_entry(&shader_store.common);
  destroy_shader_store_entry(&shader_store.radiosity);
  destroy_shader_store_entry(&shader_store.rasterizer);
  destroy_shader_store_entry(&shader_store.raytracer);
  destroy_shader_store_entry(&shader_store.tonemapper);
}

static void concat_shader_store_entries(shader_store_entry* e1,
                                        shader_store_entry* e2, char** dst)
{
  uint32_t total_size = e1->read_result.size + 1 + e2->read_result.size + 1;
  *dst                = malloc(total_size);
  sprintf(*dst, "%s\n%s%c", e1->read_result.data, e2->read_result.data, '\0');
}

/* -------------------------------------------------------------------------- *
 * Common holds the shared WGSL between the shaders, including the common
 * uniform buffer.
 * -------------------------------------------------------------------------- */

typedef struct {
  /* The common uniform buffer bind group and layout */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
  } uniforms;
  wgpu_context_t* wgpu_context;
  wgpu_buffer_t uniform_buffer;
  struct {
    mat4 view_matrix;
    mat4 mvp;
    mat4 inv_mvp;
    mat4 projection_matrix;
  } ubo_vs;
  uint64_t frame;
} common_t;

static void common_init_defaults(common_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void common_create(common_t* this, wgpu_context_t* wgpu_context,
                          wgpu_buffer_t* quads)
{
  common_init_defaults(this);

  this->wgpu_context = wgpu_context;

  /* Uniform buffer */
  this->uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Common - Uniform buffer",
                    .size  = 0 +     /*         */
                            4 * 16 + /* mvp     */
                            4 * 16 + /* inv_mvp */
                            4 * 4,   /* seed    */
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                  });

  /* Uniforms bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Common uniforms */
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->uniform_buffer.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Quads */
        .binding = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type              = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize   = quads->size,
        },
        .sampler = {0},
      },
    };
    this->uniforms.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Common - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->uniforms.bind_group_layout != NULL)
  }

  /* Uniforms bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Common uniforms */
        .binding = 0,
        .buffer  = this->uniform_buffer.buffer,
        .offset  = 0,
        .size    = this->uniform_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Quads */
        .binding = 1,
        .buffer  = quads->buffer,
        .offset  = 0,
        .size    = quads->size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Common - Bind group",
      .layout     = this->uniforms.bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->uniforms.bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->uniforms.bind_group != NULL)
  }
}

static void common_destroy(common_t* this)
{
  wgpu_destroy_buffer(&this->uniform_buffer);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->uniforms.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->uniforms.bind_group)
}

typedef struct {
  bool rotate_camera;
  float aspect;
} common_update_params_t;

/* Updates the uniform buffer data */
static void common_update(common_t* this, common_update_params_t* params)
{
  glm_mat4_identity(this->ubo_vs.view_matrix);
  glm_mat4_identity(this->ubo_vs.mvp);
  glm_mat4_identity(this->ubo_vs.inv_mvp);

  glm_mat4_identity(this->ubo_vs.projection_matrix);
  glm_perspective(PI2 / 8.0f, params->aspect, 0.5f, 100.0f,
                  this->ubo_vs.projection_matrix);

  const float view_rotation
    = params->rotate_camera ? this->frame / 1000.0f : 0.0f;

  glm_lookat(
    (vec3){sin(view_rotation) * 15.0f, 5.0f, cos(view_rotation) * 15.0f},
    (vec3){0.0f, 5.0f, 0.0f}, (vec3){0.0f, 1.0f, 0.0f},
    this->ubo_vs.view_matrix);
  glm_mat4_mul(this->ubo_vs.projection_matrix, this->ubo_vs.view_matrix,
               this->ubo_vs.mvp);
  glm_mat4_inv(this->ubo_vs.mvp, this->ubo_vs.inv_mvp);

  float uniform_data_f32[36] = {0};
  uint8_t i                  = 0;
  for (uint8_t r = 0; r < 4; ++r) {
    for (uint8_t c = 0; c < 4; ++c) {
      uniform_data_f32[i++] = this->ubo_vs.mvp[r][c];
    }
  }
  for (uint8_t r = 0; r < 4; ++r) {
    for (uint8_t c = 0; c < 4; ++c) {
      uniform_data_f32[i++] = this->ubo_vs.inv_mvp[r][c];
    }
  }
  const float mult     = (float)0xffffffff;
  uniform_data_f32[32] = (uint32_t)(mult * random_float());
  uniform_data_f32[33] = (uint32_t)(mult * random_float());
  uniform_data_f32[34] = (uint32_t)(mult * random_float());

  wgpu_queue_write_buffer(this->wgpu_context, this->uniform_buffer.buffer, 0,
                          uniform_data_f32, sizeof(uniform_data_f32));

  this->frame++;
}

/* -------------------------------------------------------------------------- *
 * Scene holds the cornell-box scene information.
 * -------------------------------------------------------------------------- */

#define SCENE_QUADS_LENGTH 19u
#define SCENE_QUAD_STRIDE (16 * 4)
#define SCENE_QUAD_VERTEX_STRIDE (4 * 10)

typedef struct {
  vec3 center;
  vec3 right;
  vec3 up;
  vec3 color;
  float emissive;
} quad_t;

typedef enum {
  QuadType_Convex,
  QuadType_Concave,
} quad_type_t;

/**
 * @brief Calculates the length of a vec3
 *
 * @param {ReadonlyVec3} a vector to calculate length of
 * @returns {Number} length of a
 */
static float vec3_len(vec3 v)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];

  return sqrt(x * x + y * y + z * z);
}

/**
 * @brief Calculates the squared length of a vec3
 *
 * @param {ReadonlyVec3} a vector to calculate squared length of
 * @returns {Number} squared length of a
 */
static float vec3_sqr_len(vec3 v)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];

  return x * x + y * y + z * z;
}

static void vec3_sign(vec3 v, quad_type_t type, vec3* dst)
{
  glm_vec3_copy(v, *dst);
  if (type == QuadType_Convex) {
    glm_vec3_negate(*dst);
  }
}

static void reciprocal(vec3 v, vec3* dst)
{
  const float s = 1.0f / vec3_sqr_len(v);
  glm_vec3_mul((vec3){s, s, s}, v, *dst);
}

//      ─────────┐
//     ╱  +Y    ╱│
//    ┌────────┐ │
//    │        │+X
//    │   +Z   │ │
//    │        │╱
//    └────────┘
typedef enum {
  CubeFace_Positive_X,
  CubeFace_Positive_Y,
  CubeFace_Positive_Z,
  CubeFace_Negative_X,
  CubeFace_Negative_Y,
  CubeFace_Negative_Z,
} cube_face_t;

typedef struct {
  vec3 center;
  float width;
  float height;
  float depth;
  float rotation;
  vec3* color;
  uint8_t color_count;
  quad_type_t type;
} box_params_t;

static void create_box(box_params_t* params, quad_t* quads)
{
  //      ─────────┐
  //     ╱  +Y    ╱│
  //    ┌────────┐ │        y
  //    │        │+X        ^
  //    │   +Z   │ │        │ -z
  //    │        │╱         │╱
  //    └────────┘          └─────> x
  vec3 x = {
    cos(params->rotation) * (params->width / 2.0f), /* x */
    0.0f,                                           /* y */
    sin(params->rotation) * (params->depth / 2.0f)  /* z */
  };
  vec3 y = {0.0f, params->height / 2.0f, 0.0f};
  vec3 z = {
    sin(params->rotation) * (params->width / 2.0f), /* x */
    0.0f,                                           /* y */
    -cos(params->rotation) * (params->depth / 2.0f) /* z */
  };
  vec3 colors[6] = {0};
  for (uint8_t i = 0; i < 6; ++i) {
    glm_vec3_copy(params->color[MIN(i, params->color_count - 1)], colors[i]);
  }

  /* Box faces */
  {
    /* PositiveX */
    quad_t* quad = &quads[0];
    glm_vec3_add(params->center, x, quad->center);
    vec3 vec3_tmp = GLM_VEC3_ZERO_INIT;
    glm_vec3_negate_to(z, vec3_tmp);
    vec3_sign(vec3_tmp, params->type, &quad->right);
    glm_vec3_copy(y, quad->up);
    glm_vec3_copy(colors[CubeFace_Positive_X], quad->color);

    /* PositiveY */
    quad = &quads[1];
    glm_vec3_add(params->center, y, quad->center);
    vec3_sign(x, params->type, &quad->right);
    glm_vec3_negate_to(z, quad->up);
    glm_vec3_copy(colors[CubeFace_Positive_Y], quad->color);

    /* PositiveZ */
    quad = &quads[2];
    glm_vec3_add(params->center, z, quad->center);
    vec3_sign(x, params->type, &quad->right);
    glm_vec3_copy(y, quad->up);
    glm_vec3_copy(colors[CubeFace_Positive_Z], quad->color);

    /* NegativeX */
    quad = &quads[3];
    glm_vec3_sub(params->center, x, quad->center);
    vec3_sign(z, params->type, &quad->right);
    glm_vec3_copy(y, quad->up);
    glm_vec3_copy(colors[CubeFace_Negative_X], quad->color);

    /* NegativeY */
    quad = &quads[4];
    glm_vec3_sub(params->center, y, quad->center);
    vec3_sign(x, params->type, &quad->right);
    glm_vec3_copy(z, quad->up);
    glm_vec3_copy(colors[CubeFace_Negative_Y], quad->color);

    /* NegativeZ */
    quad = &quads[5];
    glm_vec3_sub(params->center, z, quad->center);
    glm_vec3_negate_to(x, vec3_tmp);
    vec3_sign(vec3_tmp, params->type, &quad->right);
    glm_vec3_copy(y, quad->up);
    glm_vec3_copy(colors[CubeFace_Negative_Z], quad->color);
  }
}

static quad_t light = {
  .center   = {0.0f, 9.95f, 0.0f},
  .right    = {1.0f, 0.0f, 0.0f},
  .up       = {0.0f, 0.0f, 1.0f},
  .color    = {5.0f, 5.0f, 5.0f},
  .emissive = 1.0f,
};

typedef struct {
  uint32_t vertex_count;
  uint32_t index_count;
  wgpu_buffer_t vertices;
  wgpu_buffer_t indices;
  WGPUVertexBufferLayout vertex_buffer_layout;
  WGPUVertexAttribute vertex_buffer_layout_attributes[3];
  wgpu_buffer_t quad_buffer;
  quad_t quads[SCENE_QUADS_LENGTH];
  uint32_t quads_length;
  vec3 light_center;
  float light_width;
  float light_height;
} scene_t;

static void scene_init_defaults(scene_t* this)
{
  memset(this, 0, sizeof(*this));

  this->quads_length = SCENE_QUADS_LENGTH;

  glm_vec3_copy(light.center, this->light_center);
  this->light_width  = vec3_len(light.right) * 2.0f;
  this->light_height = vec3_len(light.up) * 2.0f;
}

static void scene_create(scene_t* this, wgpu_context_t* wgpu_context)
{
  scene_init_defaults(this);

  /* Quads */
  {
    /* Box 1 - quads */
    {
      vec3 color_array[6] = {
        {0.0f, 0.5f, 0.0f}, /* PositiveX */
        {0.5f, 0.5f, 0.5f}, /* PositiveY */
        {0.5f, 0.5f, 0.5f}, /* PositiveZ */
        {0.5f, 0.0f, 0.0f}, /* NegativeX */
        {0.5f, 0.5f, 0.5f}, /* NegativeY */
        {0.5f, 0.5f, 0.5f}, /* NegativeZ */
      };
      create_box(
        &(box_params_t){
          .center      = {0.0f, 5.0f, 0.0f},
          .width       = 10.0f,
          .height      = 10.0f,
          .depth       = 10.0f,
          .rotation    = 0.0f,
          .color       = color_array,
          .color_count = (uint32_t)ARRAY_SIZE(color_array),
          .type        = QuadType_Concave,
        },
        &this->quads[0]);
    }

    /* Box 2 - quads */
    {
      vec3 color_array[1] = {{0.8f, 0.8f, 0.8f}};
      create_box(
        &(box_params_t){
          .center      = {1.5f, 1.5f, 1.0f},
          .width       = 3.0f,
          .height      = 3.0f,
          .depth       = 3.0f,
          .rotation    = 0.3f,
          .color       = color_array,
          .color_count = (uint32_t)ARRAY_SIZE(color_array),
          .type        = QuadType_Convex,
        },
        &this->quads[6]);
    }

    /* Box 3 - quads */
    {
      vec3 color_array[1] = {{0.8f, 0.8f, 0.8f}};
      create_box(
        &(box_params_t){
          .center      = {-2.0f, 3.0f, -2.0f},
          .width       = 3.0f,
          .height      = 6.0f,
          .depth       = 3.0f,
          .rotation    = -0.4f,
          .color       = color_array,
          .color_count = (uint32_t)ARRAY_SIZE(color_array),
          .type        = QuadType_Convex,
        },
        &this->quads[12]);
    }

    /* Light quad */
    memcpy(&this->quads[18], &light, sizeof(light));
  }

  /* Quad buffer */
  {
    float quad_data[SCENE_QUAD_STRIDE * SCENE_QUADS_LENGTH]          = {0};
    float vertex_data[SCENE_QUADS_LENGTH * SCENE_QUAD_VERTEX_STRIDE] = {0};
    uint16_t index_data[SCENE_QUADS_LENGTH * 9] = {0}; /* TODO: 6? */
    uint32_t quad_data_offset                   = 0;
    uint32_t vertex_data_offset                 = 0;
    uint32_t index_data_offset                  = 0;
    for (uint32_t quad_idx = 0; quad_idx < SCENE_QUADS_LENGTH; ++quad_idx) {
      quad_t* quad = &this->quads[quad_idx];
      vec3 normal  = GLM_VEC3_ZERO_INIT;
      glm_vec3_cross(quad->right, quad->up, normal);
      glm_vec3_normalize(normal);
      quad_data[quad_data_offset++] = normal[0];
      quad_data[quad_data_offset++] = normal[1];
      quad_data[quad_data_offset++] = normal[2];
      quad_data[quad_data_offset++] = -glm_vec3_dot(normal, quad->center);

      vec3 inv_right = GLM_VEC3_ZERO_INIT;
      reciprocal(quad->right, &inv_right);
      quad_data[quad_data_offset++] = inv_right[0];
      quad_data[quad_data_offset++] = inv_right[1];
      quad_data[quad_data_offset++] = inv_right[2];
      quad_data[quad_data_offset++] = -glm_vec3_dot(inv_right, quad->center);

      vec3 inv_up = GLM_VEC3_ZERO_INIT;
      reciprocal(quad->up, &inv_up);
      quad_data[quad_data_offset++] = inv_up[0];
      quad_data[quad_data_offset++] = inv_up[1];
      quad_data[quad_data_offset++] = inv_up[2];
      quad_data[quad_data_offset++] = -glm_vec3_dot(inv_up, quad->center);

      quad_data[quad_data_offset++] = quad->color[0];
      quad_data[quad_data_offset++] = quad->color[1];
      quad_data[quad_data_offset++] = quad->color[2];
      quad_data[quad_data_offset++] = quad->emissive;

      // a ----- b
      // |       |
      // |   m   |
      // |       |
      // c ----- d
      vec3 a = GLM_VEC3_ZERO_INIT;
      glm_vec3_sub(quad->center, quad->right, a);
      glm_vec3_add(a, quad->up, a);
      vec3 b = GLM_VEC3_ZERO_INIT;
      glm_vec3_add(quad->center, quad->right, b);
      glm_vec3_add(b, quad->up, b);
      vec3 c = GLM_VEC3_ZERO_INIT;
      glm_vec3_sub(quad->center, quad->right, c);
      glm_vec3_sub(c, quad->up, c);
      vec3 d = GLM_VEC3_ZERO_INIT;
      glm_vec3_add(quad->center, quad->right, d);
      glm_vec3_sub(d, quad->up, d);

      vertex_data[vertex_data_offset++] = a[0];
      vertex_data[vertex_data_offset++] = a[1];
      vertex_data[vertex_data_offset++] = a[2];
      vertex_data[vertex_data_offset++] = 1;
      vertex_data[vertex_data_offset++] = 0; /* uv.x */
      vertex_data[vertex_data_offset++] = 1; /* uv.y */
      vertex_data[vertex_data_offset++] = quad_idx;
      vertex_data[vertex_data_offset++] = quad->color[0] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[1] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[2] * quad->emissive;

      vertex_data[vertex_data_offset++] = b[0];
      vertex_data[vertex_data_offset++] = b[1];
      vertex_data[vertex_data_offset++] = b[2];
      vertex_data[vertex_data_offset++] = 1;
      vertex_data[vertex_data_offset++] = 1; /* uv.x */
      vertex_data[vertex_data_offset++] = 1; /* uv.y */
      vertex_data[vertex_data_offset++] = quad_idx;
      vertex_data[vertex_data_offset++] = quad->color[0] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[1] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[2] * quad->emissive;

      vertex_data[vertex_data_offset++] = c[0];
      vertex_data[vertex_data_offset++] = c[1];
      vertex_data[vertex_data_offset++] = c[2];
      vertex_data[vertex_data_offset++] = 1;
      vertex_data[vertex_data_offset++] = 0; /* uv.x */
      vertex_data[vertex_data_offset++] = 0; /* uv.y */
      vertex_data[vertex_data_offset++] = quad_idx;
      vertex_data[vertex_data_offset++] = quad->color[0] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[1] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[2] * quad->emissive;

      vertex_data[vertex_data_offset++] = d[0];
      vertex_data[vertex_data_offset++] = d[1];
      vertex_data[vertex_data_offset++] = d[2];
      vertex_data[vertex_data_offset++] = 1;
      vertex_data[vertex_data_offset++] = 1; /* uv.x */
      vertex_data[vertex_data_offset++] = 0; /* uv.y */
      vertex_data[vertex_data_offset++] = quad_idx;
      vertex_data[vertex_data_offset++] = quad->color[0] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[1] * quad->emissive;
      vertex_data[vertex_data_offset++] = quad->color[2] * quad->emissive;

      index_data[index_data_offset++] = this->vertex_count + 0; /* a */
      index_data[index_data_offset++] = this->vertex_count + 2; /* c */
      index_data[index_data_offset++] = this->vertex_count + 1; /* b */
      index_data[index_data_offset++] = this->vertex_count + 1; /* b */
      index_data[index_data_offset++] = this->vertex_count + 2; /* c */
      index_data[index_data_offset++] = this->vertex_count + 3; /* d */
      this->index_count += 6;
      this->vertex_count += 4;
    }

    /* Quads storage buffer */
    this->quad_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label        = "Scene quad - Storage buffer",
                      .size         = SCENE_QUAD_STRIDE * SCENE_QUADS_LENGTH,
                      .usage        = WGPUBufferUsage_Storage,
                      .initial.data = quad_data,
                    });

    /* Quads vertices buffer */
    this->vertices
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Scene - Vertices buffer",
                                           .size  = sizeof(vertex_data),
                                           .usage = WGPUBufferUsage_Vertex,
                                           .initial.data = vertex_data,
                                         });

    /* Quads indices buffer */
    this->indices
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Scene - Indices buffer",
                                           .size  = sizeof(index_data),
                                           .usage = WGPUBufferUsage_Index,
                                           .initial.data = index_data,
                                         });
  }

  /* Vertex buffer layout */
  {
    WGPUVertexAttribute attributes[3] = {
      [0] = (WGPUVertexAttribute) {
        // position
        .shaderLocation = 0,
        .offset         = 0 * 4,
        .format         = WGPUVertexFormat_Float32x4,
      },
      [1] = (WGPUVertexAttribute) {
        // uv
        .shaderLocation = 1,
        .offset         = 4 * 4,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [2] = (WGPUVertexAttribute) {
        // color
        .shaderLocation = 2,
        .offset         = 7 * 4,
        .format         = WGPUVertexFormat_Float32x3,
      },
    };
    memcpy(&this->vertex_buffer_layout_attributes[0], attributes,
           sizeof(attributes));
    this->vertex_buffer_layout = (WGPUVertexBufferLayout){
      .arrayStride    = SCENE_QUAD_VERTEX_STRIDE,
      .attributeCount = (uint32_t)ARRAY_SIZE(attributes),
      .attributes     = this->vertex_buffer_layout_attributes,
    };
  }
}

static void scene_destroy(scene_t* this)
{
  wgpu_destroy_buffer(&this->vertices);
  wgpu_destroy_buffer(&this->indices);
  wgpu_destroy_buffer(&this->quad_buffer);
}

/* --------------------------------------------------------------------------
 * Radiosity computes lightmaps, calculated by software raytracing of light in
 * the scene.
 * -------------------------------------------------------------------------- */

typedef struct {
  /* The output lightmap format and dimensions */
  WGPUTextureFormat lightmap_format;
  uint32_t lightmap_width;
  uint32_t lightmap_height;
  /* The output lightmap. */
  texture_t lightmap;
  uint32_t lightmap_depth_or_array_layers;
  // Number of photons emitted per workgroup.
  // This is equal to the workgroup size (one photon per invocation)
  uint32_t photons_per_workgroup;
  // Number of radiosity workgroups dispatched per frame.
  uint32_t workgroups_per_frame;
  uint32_t photons_per_frame;
  // Maximum value that can be added to the 'accumulation' buffer, per
  // photon, across all texels.
  uint32_t photon_energy;
  /* The total number of lightmap texels for all quads. */
  uint32_t total_lightmap_texels;
  uint32_t accumulation_to_lightmap_workgroup_size_x;
  uint32_t accumulation_to_lightmap_workgroup_size_y;
  wgpu_context_t* wgpu_context;
  common_t* common;
  scene_t* scene;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline radiosity_pipeline;
  WGPUComputePipeline accumulation_to_lightmap_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  wgpu_buffer_t accumulation_buffer;
  wgpu_buffer_t uniform_buffer;
  /* The 'accumulation' buffer average value */
  float accumulation_mean;
  // The maximum value of 'accumulationAverage' before all values in
  // 'accumulation' are reduced to avoid integer overflows.
  uint32_t accumulation_mean_max;
} radiosity_t;

static void radiosity_init_defaults(radiosity_t* this)
{
  memset(this, 0, sizeof(*this));

  this->lightmap_format = WGPUTextureFormat_RGBA16Float;
  this->lightmap_width  = 256;
  this->lightmap_height = 256;

  this->photons_per_workgroup = 256;
  this->workgroups_per_frame  = 1024;
  this->photons_per_frame
    = this->photons_per_workgroup * this->workgroups_per_frame;

  this->photon_energy = 100000;

  this->accumulation_to_lightmap_workgroup_size_x = 16;
  this->accumulation_to_lightmap_workgroup_size_y = 16;

  this->accumulation_mean     = 0.0f;
  this->accumulation_mean_max = 0x10000000;
}

static void radiosity_create(radiosity_t* this, wgpu_context_t* wgpu_context,
                             common_t* common, scene_t* scene)
{
  radiosity_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->scene        = scene;

  /* Lightmap */
  {
    /* Texture */
    WGPUTextureDescriptor texture_desc = {
      .label         = "Radiosity lightmap - Texture",
      .size          = (WGPUExtent3D) {
        .width              = this->lightmap_width,
        .height             = this->lightmap_height,
        .depthOrArrayLayers = this->scene->quads_length,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = this->lightmap_format,
      .usage
      = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
    };
    this->lightmap.texture
      = wgpuDeviceCreateTexture(this->wgpu_context->device, &texture_desc);
    ASSERT(this->lightmap.texture != NULL);
    this->lightmap_depth_or_array_layers = this->scene->quads_length;

    /* Texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Radiosity lightmap - Texture view",
      .dimension       = WGPUTextureViewDimension_2DArray,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = this->lightmap_depth_or_array_layers,
      .aspect          = WGPUTextureAspect_All,
    };
    this->lightmap.view
      = wgpuTextureCreateView(this->lightmap.texture, &texture_view_dec);
    ASSERT(this->lightmap.view != NULL);

    /* Texture sampler */
    WGPUSamplerDescriptor sampler_desc = {
      .label         = "Radiosity lightmap - Texture sampler",
      .addressModeU  = WGPUAddressMode_ClampToEdge,
      .addressModeV  = WGPUAddressMode_ClampToEdge,
      .addressModeW  = WGPUAddressMode_ClampToEdge,
      .minFilter     = WGPUFilterMode_Linear,
      .magFilter     = WGPUFilterMode_Linear,
      .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
      .lodMinClamp   = 0.0f,
      .lodMaxClamp   = 1.0f,
      .maxAnisotropy = 1,
    };
    this->lightmap.sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
    ASSERT(this->lightmap.sampler != NULL);
  }

  /* Accumulation buffer */
  {
    this->accumulation_buffer = wgpu_create_buffer(
      this->wgpu_context, &(wgpu_buffer_desc_t){
                            .label = "Radiosity accumulation - Storage buffer",
                            .size = this->lightmap_width * this->lightmap_height
                                    * this->scene->quads_length * 16,
                            .usage = WGPUBufferUsage_Storage,
                          });
    this->total_lightmap_texels = this->lightmap_width * this->lightmap_height
                                  * this->scene->quads_length;
  }

  /* Uniform buffer */
  {
    this->uniform_buffer = wgpu_create_buffer(
      this->wgpu_context,
      &(wgpu_buffer_desc_t){
        .label = "Radiosity - Uniform buffer",
        .size  = 8 * 4, /* 8 x f32 */
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      });
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: accumulation buffer */
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Storage,
          .minBindingSize = this->accumulation_buffer.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: lightmap */
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access = WGPUStorageTextureAccess_WriteOnly,
          .format = this->lightmap_format,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: radiosity_uniforms */
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->uniform_buffer.size,
        },
        .sampler = {0},
      }
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Radiosity - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: accumulation buffer */
        .binding = 0,
        .buffer  = this->accumulation_buffer.buffer,
        .offset  = 0,
        .size    = this->accumulation_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: lightmap */
        .binding     = 1,
        .textureView = this->lightmap.view,
      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: radiosity_uniforms */
        .binding = 2,
        .buffer  = this->uniform_buffer.buffer,
        .offset  = 0,
        .size    = this->uniform_buffer.size,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Radiosity - Bind group",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Radiosity accumulate - Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Compute shader */
  char* wgsl_code = {0};
  concat_shader_store_entries(&shader_store.common, &shader_store.radiosity,
                              &wgsl_code);

  /* Radiosity compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "PhotonsPerWorkgroup",
        .value = this->photons_per_workgroup,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "PhotonEnergy",
        .value = this->photon_energy,
      },
    };

    /* Compute shader */
    wgpu_shader_t radiosity_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      /* Compute shader WGSL */
                      .label           = "Radiosity - Compute shader",
                      .wgsl_code       = {wgsl_code},
                      .entry           = "radiosity",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->radiosity_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Radiosity radiosity - Compute pipeline",
        .layout  = this->pipeline_layout,
        .compute = radiosity_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&radiosity_comp_shader);
  }

  /* Accumulation to lightmap compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "AccumulationToLightmapWorkgroupSizeX",
        .value = this->accumulation_to_lightmap_workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "AccumulationToLightmapWorkgroupSizeY",
        .value = this->accumulation_to_lightmap_workgroup_size_y,
      },
    };

    /* Compute shader */
    wgpu_shader_t accumulation_to_lightmap_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      /* Compute shader WGSL */
                      .label     = "Accumulation to lightmap - Compute shader",
                      .wgsl_code = {wgsl_code},
                      .entry     = "accumulation_to_lightmap",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->accumulation_to_lightmap_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label  = "Radiosity accumulation to lightmap - Compute pipeline",
        .layout = this->pipeline_layout,
        .compute
        = accumulation_to_lightmap_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&accumulation_to_lightmap_comp_shader);
  }

  /* Cleanup */
  free(wgsl_code);
}

static void radiosity_destroy(radiosity_t* this)
{
  wgpu_destroy_texture(&this->lightmap);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->radiosity_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline,
                        this->accumulation_to_lightmap_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, this->accumulation_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffer.buffer)
}

static void radiosity_run(radiosity_t* this, WGPUCommandEncoder command_encoder)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  /* Calculate the new mean value for the accumulation buffer */
  this->accumulation_mean += (this->photons_per_frame * this->photon_energy)
                             / (float)this->total_lightmap_texels;

  // Calculate the 'accumulation' -> 'lightmap' scale factor from
  // 'accumulationMean'
  const float accumulation_to_lightmap_scale = 1.0f / this->accumulation_mean;
  // If 'accumulationMean' is greater than 'kAccumulationMeanMax', then
  // reduce the 'accumulation' buffer values to prevent u32 overflow.
  const float accumulation_buffer_scale
    = this->accumulation_mean > 2 * this->accumulation_mean_max ? 0.5f : 1.0f;
  this->accumulation_mean *= accumulation_buffer_scale;

  /* Update the radiosity uniform buffer data. */
  const float uniform_data_f32[8] = {
    accumulation_to_lightmap_scale, /* accumulation_to_lightmap_scale */
    accumulation_buffer_scale,      /* accumulation_buffer_scale      */
    this->scene->light_width,       /* light_width                    */
    this->scene->light_height,      /* light_height                   */
    this->scene->light_center[0],   /* light_center x                 */
    this->scene->light_center[1],   /* light_center y                 */
    this->scene->light_center[2],   /* light_center z                 */
  };
  wgpu_queue_write_buffer(wgpu_context, this->uniform_buffer.buffer, 0,
                          &uniform_data_f32[0], sizeof(uniform_data_f32));

  /* Dispatch the radiosity workgroups */
  wgpu_context->cpass_enc
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
  wgpuComputePassEncoderSetBindGroup(
    wgpu_context->cpass_enc, 0, this->common->uniforms.bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                     this->bind_group, 0, NULL);
  wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                    this->radiosity_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(wgpu_context->cpass_enc,
                                           this->workgroups_per_frame, 1, 1);

  /* Then copy the 'accumulation' data to 'lightmap' */
  wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc,
                                    this->accumulation_to_lightmap_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    wgpu_context->cpass_enc,
    ceil(this->lightmap_width
         / (float)this->accumulation_to_lightmap_workgroup_size_x),
    ceil(this->lightmap_height
         / (float)this->accumulation_to_lightmap_workgroup_size_y),
    this->lightmap_depth_or_array_layers);
  wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
}

/* --------------------------------------------------------------------------
 * Rasterizer renders the scene using a regular raserization graphics pipeline.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  scene_t* scene;
  texture_t depth_texture;
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } render_pass;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
} rasterizer_t;

static void rasterizer_init_defaults(rasterizer_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void rasterizer_create(rasterizer_t* this, wgpu_context_t* wgpu_context,
                              common_t* common, scene_t* scene,
                              radiosity_t* radiosity, texture_t* frame_buffer)
{
  rasterizer_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->scene        = scene;

  /* Depth texture */
  {
    /* Create the texture */
    WGPUExtent3D texture_extent = {
      .width              = frame_buffer->size.width,
      .height             = frame_buffer->size.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "Rasterizer renderer - Depth texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    this->depth_texture.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->depth_texture.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "Rasterizer renderer - Depth texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->depth_texture.view
      = wgpuTextureCreateView(this->depth_texture.texture, &texture_view_dec);
    ASSERT(this->depth_texture.view != NULL);
  }

  /* Render pass */
  {
    /* Color attachment */
    this->render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = frame_buffer->view,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
    };

    /* Depth-stencil attachment */
    this->render_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = this->depth_texture.view,
        .depthClearValue = 1.0f,
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
      };

    /* Render pass descriptor */
    this->render_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = "Rasterizer renderer - Render pass descriptor",
      .colorAttachmentCount   = 1,
      .colorAttachments       = this->render_pass.color_attachments,
      .depthStencilAttachment = &this->render_pass.depth_stencil_attachment,
    };
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: lightmap */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout) {
          .type  = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = "Rasterizer renderer - Bind group layout",
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: lightmap */
        .binding = 0,
        .textureView  = radiosity->lightmap.view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: sampler */
        .binding     = 1,
        .sampler = radiosity->lightmap.sampler,

      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Rasterizer renderer - Bind group",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Rasterizer - Renderer pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Rasterizer render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state_desc = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    };

    /* Color target state */
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
      .format    = frame_buffer->format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state_desc
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24Plus,
        .depth_write_enabled = true,
      });
    depth_stencil_state_desc.depthCompare = WGPUCompareFunction_Less;

    /* Shader code */
    char* wgsl_code = {0};
    concat_shader_store_entries(&shader_store.common, &shader_store.rasterizer,
                                &wgsl_code);

    /* Vertex state */
    WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
      wgpu_context, &(wgpu_vertex_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        /* Vertex shader WGSL */
        .label     = "Rasterizer renderer - Vertex module",
        .wgsl_code  = {wgsl_code},
        .entry     = "vs_main",
      },
      .buffer_count = 1,
      .buffers      = &this->scene->vertex_buffer_layout,
    });

    /* Fragment state */
    WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
      wgpu_context, &(wgpu_fragment_state_t){
      .shader_desc = (wgpu_shader_desc_t){
        /* Fragment shader WGSL */
        .label     = "Rasterizer renderer - Fragment module",
        .wgsl_code  = {wgsl_code},
        .entry     = "fs_main",
      },
      .target_count = 1,
      .targets = &color_target_state_desc,
    });

    /* Multisample state */
    WGPUMultisampleState multisample_state_desc
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    /* Create rendering pipeline using the specified states */
    this->pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device, &(WGPURenderPipelineDescriptor){
                              .label        = "Rasterizer - Renderer pipeline",
                              .layout       = this->pipeline_layout,
                              .primitive    = primitive_state_desc,
                              .vertex       = vertex_state_desc,
                              .fragment     = &fragment_state_desc,
                              .depthStencil = &depth_stencil_state_desc,
                              .multisample  = multisample_state_desc,
                            });
    ASSERT(this->pipeline != NULL);

    // Shader modules are no longer needed once the graphics pipeline has
    // been created
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
    free(wgsl_code);
  }
}

static void rasterizer_destroy(rasterizer_t* this)
{
  wgpu_destroy_texture(&this->depth_texture);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void rasterizer_run(rasterizer_t* this,
                           WGPUCommandEncoder command_encoder)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &this->render_pass.descriptor);
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, this->pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->scene->vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->scene->indices.buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    this->common->uniforms.bind_group, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                    this->bind_group, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->scene->index_count, 1, 0, 0, 0);
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
}

/* --------------------------------------------------------------------------
 * Raytracer renders the scene using a software ray-tracing compute pipeline.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  texture_t* frame_buffer;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
} raytracer_t;

static void raytracer_init_defaults(raytracer_t* this)
{
  memset(this, 0, sizeof(*this));

  this->workgroup_size_x = 16;
  this->workgroup_size_y = 16;
}

static void raytracer_create(raytracer_t* this, wgpu_context_t* wgpu_context,
                             common_t* common, radiosity_t* radiosity,
                             texture_t* frame_buffer)
{
  raytracer_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->frame_buffer = frame_buffer;

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: lightmap */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout) {
          .type  = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        /* Binding 2: framebuffer */
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access = WGPUStorageTextureAccess_WriteOnly,
          .format = radiosity->lightmap_format,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Raytracer - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: lightmap */
        .binding = 0,
        .textureView  = radiosity->lightmap.view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: sampler */
        .binding     = 1,
        .sampler = radiosity->lightmap.sampler,

      },
      [2] = (WGPUBindGroupEntry) {
        /* Binding 2: framebuffer */
        .binding = 2,
        .textureView = frame_buffer->view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Renderer - Bind group",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->common->uniforms.bind_group_layout, /* Group 0 */
      this->bind_group_layout,                  /* Group 1 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Raytracer - Compute pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Raytracer compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeX",
        .value = this->workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeY",
        .value = this->workgroup_size_y,
      },
    };

    /* Compute shader */
    char* wgsl_code = {0};
    concat_shader_store_entries(&shader_store.common, &shader_store.raytracer,
                                &wgsl_code);
    wgpu_shader_t raytracer_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      /* Compute shader WGSL */
                      .label           = "Raytracer - Compute shader",
                      .wgsl_code       = {wgsl_code},
                      .entry           = "main",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Raytracer - Compute pipeline",
        .layout  = this->pipeline_layout,
        .compute = raytracer_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&raytracer_comp_shader);
    free(wgsl_code);
  }
}

static void raytracer_destroy(raytracer_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void raytracer_run(raytracer_t* this, WGPUCommandEncoder command_encoder)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  wgpu_context->cpass_enc
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
  wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc, this->pipeline);
  wgpuComputePassEncoderSetBindGroup(
    wgpu_context->cpass_enc, 0, this->common->uniforms.bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 1,
                                     this->bind_group, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    wgpu_context->cpass_enc,
    ceil(this->frame_buffer->size.width / (float)this->workgroup_size_x),
    ceil(this->frame_buffer->size.height / (float)this->workgroup_size_y), 1);
  wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
}

/* --------------------------------------------------------------------------
 * Tonemapper implements a tonemapper to convert a linear-light framebuffer to a
 * gamma-correct, tonemapped framebuffer used for presentation.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  common_t* common;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPUComputePipeline pipeline;
  texture_t* input;
  texture_t* output;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
} tonemapper_t;

static void tonemapper_init_defaults(tonemapper_t* this)
{
  memset(this, 0, sizeof(*this));

  this->workgroup_size_x = 16;
  this->workgroup_size_y = 16;
}

static void tonemapper_create(tonemapper_t* this, wgpu_context_t* wgpu_context,
                              common_t* common, texture_t* input,
                              texture_t* output)
{
  tonemapper_init_defaults(this);

  this->wgpu_context = wgpu_context;
  this->common       = common;
  this->input        = input;
  this->output       = output;

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: input
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        // Binding 1: output
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = {
          .access        = WGPUStorageTextureAccess_WriteOnly,
          .format        = output->format,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Tonemapper - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        // Binding 0: input
        .binding     = 0,
        .textureView = this->input->view,
      },
      [1] = (WGPUBindGroupEntry) {
        // Binding 1: output
        .binding     = 1,
        .textureView = output->view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      this->wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = "Tonemapper - Bind group",
        .layout     = this->bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      });
    ASSERT(this->bind_group != NULL);
  }

  /* Compute pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->bind_group_layout, /* Group 0 */
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "Tonemap - Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Tonemap compute pipeline */
  {
    /* Constants */
    WGPUConstantEntry constant_entries[2] = {
      [0] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeX",
        .value = this->workgroup_size_x,
      },
      [1] = (WGPUConstantEntry) {
        .key   = "WorkgroupSizeY",
        .value = this->workgroup_size_y,
      },
    };

    /* Compute shader */
    char* wgsl_code = {0};
    concat_shader_store_entries(&shader_store.common, &shader_store.tonemapper,
                                &wgsl_code);
    wgpu_shader_t tonemapper_comp_shader = wgpu_shader_create(
      wgpu_context, &(wgpu_shader_desc_t){
                      // Compute shader WGSL
                      .label           = "Tonemapper - Compute shader",
                      .wgsl_code       = {wgsl_code},
                      .entry           = "main",
                      .constants.count = (uint32_t)ARRAY_SIZE(constant_entries),
                      .constants.entries = constant_entries,
                    });

    /* Compute pipeline*/
    this->pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "Tonemap - Compute pipeline",
        .layout  = this->pipeline_layout,
        .compute = tonemapper_comp_shader.programmable_stage_descriptor,
      });

    /* Cleanup */
    wgpu_shader_release(&tonemapper_comp_shader);
    free(wgsl_code);
  }
}

static void tonemapper_destroy(tonemapper_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void tonemapper_run(tonemapper_t* this,
                           WGPUCommandEncoder command_encoder)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  wgpu_context->cpass_enc
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
  wgpuComputePassEncoderSetBindGroup(wgpu_context->cpass_enc, 0,
                                     this->bind_group, 0, NULL);
  wgpuComputePassEncoderSetPipeline(wgpu_context->cpass_enc, this->pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(
    wgpu_context->cpass_enc,
    ceil(this->input->size.width / (float)this->workgroup_size_x),
    ceil(this->input->size.height / (float)this->workgroup_size_y), 1);
  wgpuComputePassEncoderEnd(wgpu_context->cpass_enc);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, wgpu_context->cpass_enc)
}

/* --------------------------------------------------------------------------
 * Result renderer.
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  struct {
    WGPURenderPassColorAttachment color_att_descriptors[1];
    WGPURenderPassDescriptor descriptor;
  } render_pass;
} result_renderer_t;

static void result_renderer_init_defaults(result_renderer_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void result_renderer_create(result_renderer_t* this,
                                   wgpu_context_t* wgpu_context,
                                   texture_t* texture)
{
  result_renderer_init_defaults(this);

  this->wgpu_context = wgpu_context;

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0: Fragment shader image view */
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout) {
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      }
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Quad - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        /* Binding 0: Fragment shader image sampler */
        .binding     = 0,
        .textureView = texture->view,
      },
      [1] = (WGPUBindGroupEntry) {
        /* Binding 1: Fragment shader image sampler */
        .binding = 1,
        .sampler = texture->sampler,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "Quad - Bind group",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Pipeline layout */
  {
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = "Quad - Render pipeline layout",
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &this->bind_group_layout,
                            });
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pass */
  {
    /* Color attachment */
    this->render_pass.color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
        .view       = NULL, /* Assigned later */
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 0.0f,
        },
    };

    /* Depth attachment */
    wgpu_setup_deph_stencil(wgpu_context, NULL);

    /* Render pass descriptor */
    this->render_pass.descriptor = (WGPURenderPassDescriptor){
      .colorAttachmentCount   = 1,
      .colorAttachments       = this->render_pass.color_att_descriptors,
      .depthStencilAttachment = &wgpu_context->depth_stencil.att_desc,
    };
  }

  /* Render pipeline */
  {
    /* Primitive state */
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    /* Color target state */
    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = wgpu_context->swap_chain.format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    /* Depth stencil state */
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depth_write_enabled = true,
      });

    /* Vertex state */
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Vertex shader SPIR-V */
                .label = "Quad - Vertex shader",
                .file  = "shaders/cornell_box/quad.vert.spv",
              },
              .buffer_count = 0,
              .buffers      = NULL,
            });

    /* Fragment state */
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Fragment shader SPIR-V */
                .label = "Quad - Fragment shader",
                .file  = "shaders/cornell_box/quad.frag.spv",
              },
              .target_count = 1,
              .targets      = &color_target_state,
            });

    /* Multisample state */
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    /* Render pipeline description */
    WGPURenderPipelineDescriptor pipeline_desc = {
      .label        = "Quad  - Render pipeline",
      .layout       = this->pipeline_layout,
      .primitive    = primitive_state,
      .depthStencil = &depth_stencil_state,
      .vertex       = vertex_state,
      .fragment     = &fragment_state,
      .multisample  = multisample_state,
    };
    this->pipeline
      = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
    ASSERT(this->pipeline != NULL);

    /* Partial cleanup */
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void result_renderer_destroy(result_renderer_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void result_renderer_run(result_renderer_t* this,
                                WGPUCommandEncoder command_encoder,
                                WGPUTextureView frame_buffer)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  // Set target frame buffer
  this->render_pass.color_att_descriptors[0].view = frame_buffer;

  // Create render pass encoder for encoding drawing commands
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &this->render_pass.descriptor);

  // Set viewport
  wgpuRenderPassEncoderSetViewport(
    wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
    (float)wgpu_context->surface.height, 0.0f, 1.0f);

  // Set scissor rectangle
  wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                      wgpu_context->surface.width,
                                      wgpu_context->surface.height);

  // Draw textured quad
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    this->bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, 3, 1, 0, 0);

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
}

/* --------------------------------------------------------------------------
 * Cornell box example.
 * -------------------------------------------------------------------------- */

// Example structs
static struct {
  struct {
    texture_t input;
    texture_t output;
  } frame_buffer;
  scene_t scene;
  common_t common;
  radiosity_t radiosity;
  rasterizer_t rasterizer;
  raytracer_t raytracer;
  tonemapper_t tonemapper;
  result_renderer_t result_renderer;
} example = {0};

// GUI
typedef enum {
  Renderer_Rasterizer,
  Renderer_Raytracer,
} renderer_t;

static struct {
  renderer_t renderer;
  bool rotate_camera;
} example_parms = {
  .renderer      = Renderer_Rasterizer,
  .rotate_camera = true,
};

static const char* renderer_names[2] = {"Rasterizer", "Raytracer"};

// Other variables
static const char* example_title = "Cornell box";
static bool prepared             = false;

static void create_frame_buffer(wgpu_context_t* wgpu_context,
                                texture_t* frame_buffer,
                                WGPUTextureFormat format)
{
  // Create the texture
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "Framebuffer - Texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = format,
    .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_StorageBinding
             | WGPUTextureUsage_TextureBinding,
  };
  *frame_buffer = (texture_t){
    .size.width              = texture_extent.width,
    .size.height             = texture_extent.height,
    .size.depthOrArrayLayers = texture_extent.depthOrArrayLayers,
    .mip_level_count         = texture_desc.mipLevelCount,
    .format                  = texture_desc.format,
    .dimension               = texture_desc.dimension,
    .texture = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc),
  };
  ASSERT(frame_buffer->texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "Framebuffer - Texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  frame_buffer->view
    = wgpuTextureCreateView(frame_buffer->texture, &texture_view_dec);
  ASSERT(frame_buffer->view != NULL);

  // Texture sampler
  WGPUSamplerDescriptor sampler_desc = {
    .label         = "Framebuffer - Texture sampler",
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  frame_buffer->sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  ASSERT(frame_buffer->sampler != NULL);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    create_shader_store();
    create_frame_buffer(context->wgpu_context, &example.frame_buffer.input,
                        WGPUTextureFormat_RGBA16Float);
    create_frame_buffer(context->wgpu_context, &example.frame_buffer.output,
                        WGPUTextureFormat_BGRA8Unorm);
    scene_create(&example.scene, context->wgpu_context);
    common_create(&example.common, context->wgpu_context,
                  &example.scene.quad_buffer);
    radiosity_create(&example.radiosity, context->wgpu_context, &example.common,
                     &example.scene);
    rasterizer_create(&example.rasterizer, context->wgpu_context,
                      &example.common, &example.scene, &example.radiosity,
                      &example.frame_buffer.input);
    raytracer_create(&example.raytracer, context->wgpu_context, &example.common,
                     &example.radiosity, &example.frame_buffer.input);
    tonemapper_create(&example.tonemapper, context->wgpu_context,
                      &example.common, &example.frame_buffer.input,
                      &example.frame_buffer.output);
    result_renderer_create(&example.result_renderer, context->wgpu_context,
                           &example.frame_buffer.output);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    int32_t current_renderer_index
      = (example_parms.renderer == Renderer_Rasterizer) ? 0 : 1;
    if (imgui_overlay_combo_box(context->imgui_overlay, "Renderer",
                                &current_renderer_index, renderer_names, 2)) {
      example_parms.renderer = (current_renderer_index == 0) ?
                                 Renderer_Rasterizer :
                                 Renderer_Raytracer;
    }
    imgui_overlay_checkBox(context->imgui_overlay, "Rotate Camera",
                           &example_parms.rotate_camera);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  WGPUTextureView frame_buffer = wgpu_context->swap_chain.frame_buffer;
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Update uniforms */
  common_update(&example.common,
                &(common_update_params_t){
                  .rotate_camera = example_parms.rotate_camera,
                  .aspect        = wgpu_context->surface.width
                            / (float)wgpu_context->surface.height,
                });

  /* Software raytracing */
  radiosity_run(&example.radiosity, wgpu_context->cmd_enc);

  switch (example_parms.renderer) {
    case Renderer_Rasterizer: {
      rasterizer_run(&example.rasterizer, wgpu_context->cmd_enc);
    } break;
    case Renderer_Raytracer: {
      raytracer_run(&example.raytracer, wgpu_context->cmd_enc);
    } break;
  }

  /* Tone mapping */
  tonemapper_run(&example.tonemapper, wgpu_context->cmd_enc);

  /* Render result */
  result_renderer_run(&example.result_renderer, wgpu_context->cmd_enc,
                      frame_buffer);

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  /* Prepare frame */
  prepare_frame(context);

  /* Command buffer to be submitted to the queue */
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  /* Submit command buffer to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  if (!prepared) {
    return EXIT_FAILURE;
  }

  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  destroy_shader_store();
  wgpu_destroy_texture(&example.frame_buffer.input);
  wgpu_destroy_texture(&example.frame_buffer.output);
  result_renderer_destroy(&example.result_renderer);
  tonemapper_destroy(&example.tonemapper);
  raytracer_destroy(&example.raytracer);
  rasterizer_destroy(&example.rasterizer);
  radiosity_destroy(&example.radiosity);
  common_destroy(&example.common);
  scene_destroy(&example.scene);
}

void example_cornell_box(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
