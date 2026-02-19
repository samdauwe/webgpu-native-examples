#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <string.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Post-processing
 *
 * This example shows how to use a post-processing effect to blend between two
 * scenes. Two instanced scenes (cubes and spheres) are rendered to offscreen
 * framebuffers, then blended together on a fullscreen quad using a cutoff mask
 * texture for transition effects.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-dojo/tree/master/src/examples/postprocessing-01
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* instanced_vertex_shader_wgsl;
static const char* instanced_fragment_shader_wgsl;
static const char* quad_vertex_shader_wgsl;
static const char* quad_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define INSTANCES_COUNT 500u
#define WORLD_SIZE_X 20u
#define WORLD_SIZE_Y 20u
#define WORLD_SIZE_Z 20u
#define PP_EPSILON 0.00001f

/* -------------------------------------------------------------------------- *
 * Base transform class to handle vectors and matrices
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/main/src/core/transform.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 position;
  vec3 rotation;
  vec3 scale;
  mat4 model_matrix;
  bool should_update;
} transform_t;

static void transform_set_rotation(transform_t* transform, vec3 rotation)
{
  glm_vec3_copy(rotation, transform->rotation);
  transform->should_update = true;
}

static void transform_update_model_matrix(transform_t* transform)
{
  glm_mat4_identity(transform->model_matrix);
  glm_translate(transform->model_matrix, transform->position);
  glm_rotate(transform->model_matrix, transform->rotation[0],
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(transform->model_matrix, transform->rotation[1],
             (vec3){0.0f, 1.0f, 0.0f});
  glm_rotate(transform->model_matrix, transform->rotation[2],
             (vec3){0.0f, 0.0f, 1.0f});
  glm_scale(transform->model_matrix, transform->scale);
  transform->should_update = false;
}

static void transform_init(transform_t* transform)
{
  glm_vec3_zero(transform->position);
  glm_vec3_zero(transform->rotation);
  glm_vec3_one(transform->scale);
  transform->should_update = true;
}

/* -------------------------------------------------------------------------- *
 * Orthographic Camera
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 up;
  float left, right, top, bottom, near, far;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 view_matrix;
} orthographic_camera_t;

static void orthographic_camera_update_view_matrix(orthographic_camera_t* cam)
{
  glm_lookat(cam->position, cam->look_at_position, cam->up, cam->view_matrix);
}

static void
orthographic_camera_update_projection_matrix(orthographic_camera_t* cam)
{
  glm_ortho(cam->left, cam->right, cam->bottom, cam->top, cam->near, cam->far,
            cam->projection_matrix);
}

static void orthographic_camera_init(orthographic_camera_t* cam, float left,
                                     float right, float top, float bottom,
                                     float near_val, float far_val)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, cam->up);
  cam->left   = left;
  cam->right  = right;
  cam->top    = top;
  cam->bottom = bottom;
  cam->near   = near_val;
  cam->far    = far_val;
  glm_vec3_zero(cam->position);
  glm_vec3_zero(cam->look_at_position);
  glm_mat4_zero(cam->projection_matrix);
  glm_mat4_zero(cam->view_matrix);
  orthographic_camera_update_projection_matrix(cam);
}

/* -------------------------------------------------------------------------- *
 * Perspective Camera
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 up;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 view_matrix;
  float field_of_view;
  float aspect;
  float near, far;
} perspective_camera_t;

static void perspective_camera_update_view_matrix(perspective_camera_t* cam)
{
  glm_lookat(cam->position, cam->look_at_position, cam->up, cam->view_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* cam)
{
  glm_perspective(cam->field_of_view, cam->aspect, cam->near, cam->far,
                  cam->projection_matrix);
}

static void perspective_camera_init(perspective_camera_t* cam, float fov,
                                    float aspect, float near_val, float far_val)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, cam->up);
  glm_vec3_zero(cam->position);
  glm_vec3_zero(cam->look_at_position);
  glm_mat4_zero(cam->projection_matrix);
  glm_mat4_zero(cam->view_matrix);
  cam->field_of_view = fov;
  cam->aspect        = aspect;
  cam->near          = near_val;
  cam->far           = far_val;
  perspective_camera_update_projection_matrix(cam);
}

/* -------------------------------------------------------------------------- *
 * Geometry helpers
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    float* data;
    uint32_t data_size;
    uint32_t count;
  } positions;
  struct {
    float* data;
    uint32_t data_size;
    uint32_t count;
  } normals;
  struct {
    float* data;
    uint32_t data_size;
    uint32_t count;
  } uvs;
  struct {
    uint32_t* data;
    uint32_t data_size;
    uint32_t count;
  } indices;
} geometry_t;

typedef struct {
  float model_matrix_data[INSTANCES_COUNT * 16];
  float normal_matrix_data[INSTANCES_COUNT * 16];
} instanced_geometry_t;

static void geometry_destroy(geometry_t* geo)
{
  if (geo->positions.data) {
    free(geo->positions.data);
    geo->positions.data = NULL;
  }
  if (geo->normals.data) {
    free(geo->normals.data);
    geo->normals.data = NULL;
  }
  if (geo->uvs.data) {
    free(geo->uvs.data);
    geo->uvs.data = NULL;
  }
  if (geo->indices.data) {
    free(geo->indices.data);
    geo->indices.data = NULL;
  }
}

/* Build a plane face for box/plane geometry */
static void build_plane(float* vertices, float* normal, float* uv,
                        uint32_t* indices, int32_t width, int32_t height,
                        int32_t depth, uint32_t w_segs, uint32_t h_segs,
                        uint32_t u, uint32_t v, uint32_t w, int32_t u_dir,
                        int32_t v_dir, uint32_t i, uint32_t ii)
{
  const uint32_t io = i;
  const float seg_w = (float)width / (float)w_segs;
  const float seg_h = (float)height / (float)h_segs;
  uint32_t a = 0, b = 0, c = 0, d = 0;
  float x = 0.0f, y = 0.0f;

  for (uint32_t iy = 0; iy <= h_segs; ++iy) {
    y = iy * seg_h - height / 2.0f;
    for (uint32_t ix = 0; ix <= w_segs; ++ix, ++i) {
      x = ix * seg_w - width / 2.0f;

      vertices[i * 3 + u] = x * u_dir;
      vertices[i * 3 + v] = y * v_dir;
      vertices[i * 3 + w] = depth / 2.0f;

      normal[i * 3 + u] = 0.0f;
      normal[i * 3 + v] = 0.0f;
      normal[i * 3 + w] = depth >= 0 ? 1.0f : -1.0f;

      uv[i * 2]     = (float)ix / (float)w_segs;
      uv[i * 2 + 1] = 1.0f - (float)iy / (float)h_segs;

      if (iy == h_segs || ix == w_segs) {
        continue;
      }

      a = io + ix + iy * (w_segs + 1);
      b = io + ix + (iy + 1) * (w_segs + 1);
      c = io + ix + (iy + 1) * (w_segs + 1) + 1;
      d = io + ix + iy * (w_segs + 1) + 1;

      indices[ii * 6]     = a;
      indices[ii * 6 + 1] = b;
      indices[ii * 6 + 2] = d;
      indices[ii * 6 + 3] = b;
      indices[ii * 6 + 4] = c;
      indices[ii * 6 + 5] = d;
      ++ii;
    }
  }
}

/* Generate a plane geometry */
static void create_plane(geometry_t* plane, uint32_t width, uint32_t height,
                         uint32_t w_segs, uint32_t h_segs)
{
  const uint32_t num         = (w_segs + 1) * (h_segs + 1);
  const uint32_t num_indices = w_segs * h_segs * 6;

  plane->positions.count     = num;
  plane->normals.count       = num;
  plane->uvs.count           = num;
  plane->indices.count       = num_indices;
  plane->positions.data_size = num * 3 * sizeof(float);
  plane->normals.data_size   = num * 3 * sizeof(float);
  plane->uvs.data_size       = num * 2 * sizeof(float);
  plane->indices.data_size   = num_indices * sizeof(uint32_t);

  plane->positions.data = (float*)malloc(plane->positions.data_size);
  plane->normals.data   = (float*)malloc(plane->normals.data_size);
  plane->uvs.data       = (float*)malloc(plane->uvs.data_size);
  plane->indices.data   = (uint32_t*)malloc(plane->indices.data_size);

  build_plane(plane->positions.data, plane->normals.data, plane->uvs.data,
              plane->indices.data, (int32_t)width, (int32_t)height, 0, w_segs,
              h_segs, 0, 1, 2, 1, -1, 0, 0);
}

/* Generate a box geometry */
static void create_box(geometry_t* box)
{
  const uint32_t w_segs = 1, h_segs = 1, d_segs = 1;
  const uint32_t width = 1, height = 1, depth = 1;
  const uint32_t num = (w_segs + 1) * (h_segs + 1) * 2
                       + (w_segs + 1) * (d_segs + 1) * 2
                       + (h_segs + 1) * (d_segs + 1) * 2;
  const uint32_t num_indices
    = (w_segs * h_segs * 2 + w_segs * d_segs * 2 + h_segs * d_segs * 2) * 6;

  box->positions.count     = num;
  box->normals.count       = num;
  box->uvs.count           = num;
  box->indices.count       = num_indices;
  box->positions.data_size = num * 3 * sizeof(float);
  box->normals.data_size   = num * 3 * sizeof(float);
  box->uvs.data_size       = num * 2 * sizeof(float);
  box->indices.data_size   = num_indices * sizeof(uint32_t);

  box->positions.data = (float*)malloc(box->positions.data_size);
  box->normals.data   = (float*)malloc(box->normals.data_size);
  box->uvs.data       = (float*)malloc(box->uvs.data_size);
  box->indices.data   = (uint32_t*)malloc(box->indices.data_size);

  uint32_t i = 0, ii = 0;

  /* RIGHT */
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, depth, height, width, d_segs, h_segs, 2, 1, 0,
              -1, -1, i, ii);
  /* LEFT */
  i += (d_segs + 1) * (h_segs + 1);
  ii += d_segs * h_segs;
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, depth, height, -((int32_t)width), d_segs,
              h_segs, 2, 1, 0, 1, -1, i, ii);
  /* TOP */
  i += (d_segs + 1) * (h_segs + 1);
  ii += d_segs * h_segs;
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, width, depth, height, d_segs, h_segs, 0, 2, 1,
              1, 1, i, ii);
  /* BOTTOM */
  i += (w_segs + 1) * (d_segs + 1);
  ii += w_segs * d_segs;
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, width, depth, -((int32_t)height), d_segs,
              h_segs, 0, 2, 1, 1, -1, i, ii);
  /* BACK */
  i += (w_segs + 1) * (d_segs + 1);
  ii += w_segs * d_segs;
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, width, height, -((int32_t)depth), w_segs,
              h_segs, 0, 1, 2, -1, -1, i, ii);
  /* FRONT */
  i += (w_segs + 1) * (h_segs + 1);
  ii += w_segs * h_segs;
  build_plane(box->positions.data, box->normals.data, box->uvs.data,
              box->indices.data, width, height, depth, w_segs, h_segs, 0, 1, 2,
              1, -1, i, ii);
}

/* Generate a sphere geometry */
static void create_sphere(geometry_t* sphere)
{
  const float radius    = 0.5f;
  const uint32_t w_segs = 16u;
  const uint32_t h_segs = (uint32_t)ceilf(w_segs * 0.5f);
  const float p_start   = 0.0f;
  const float p_length  = PI2;
  const float t_start   = 0.0f;
  const float t_length  = PI;
  const float te        = t_start + t_length;

  const uint32_t num         = (w_segs + 1) * (h_segs + 1);
  const uint32_t num_indices = w_segs * h_segs * 6;

  sphere->positions.count     = num;
  sphere->normals.count       = num;
  sphere->uvs.count           = num;
  sphere->indices.count       = num_indices;
  sphere->positions.data_size = num * 3 * sizeof(float);
  sphere->normals.data_size   = num * 3 * sizeof(float);
  sphere->uvs.data_size       = num * 2 * sizeof(float);
  sphere->indices.data_size   = num_indices * sizeof(uint32_t);

  sphere->positions.data = (float*)malloc(sphere->positions.data_size);
  sphere->normals.data   = (float*)malloc(sphere->normals.data_size);
  sphere->uvs.data       = (float*)malloc(sphere->uvs.data_size);
  sphere->indices.data   = (uint32_t*)malloc(sphere->indices.data_size);

  uint32_t vi    = 0;
  uint32_t iv    = 0;
  uint32_t ii    = 0;
  uint32_t* grid = (uint32_t*)malloc(num * sizeof(uint32_t));
  vec3 n         = GLM_VEC3_ZERO_INIT;

  for (uint32_t iy = 0; iy <= h_segs; ++iy) {
    float vf = iy / (float)h_segs;
    for (uint32_t ix = 0; ix <= w_segs; ++ix, ++vi) {
      float uf = ix / (float)w_segs;
      float x  = -radius * cosf(p_start + uf * p_length)
                * sinf(t_start + vf * t_length);
      float y = radius * cosf(t_start + vf * t_length);
      float z = radius * sinf(p_start + uf * p_length)
                * sinf(t_start + vf * t_length);

      sphere->positions.data[vi * 3]     = x;
      sphere->positions.data[vi * 3 + 1] = y;
      sphere->positions.data[vi * 3 + 2] = z;

      glm_vec3_copy((vec3){x, y, z}, n);
      glm_vec3_normalize(n);
      sphere->normals.data[vi * 3]     = n[0];
      sphere->normals.data[vi * 3 + 1] = n[1];
      sphere->normals.data[vi * 3 + 2] = n[2];

      sphere->uvs.data[vi * 2]     = uf;
      sphere->uvs.data[vi * 2 + 1] = 1.0f - vf;

      grid[(iy * (w_segs + 1)) + ix] = iv++;
    }
  }

  uint32_t a = 0, b = 0, c = 0, d = 0;
  for (uint32_t iy = 0; iy < h_segs; ++iy) {
    for (uint32_t ix = 0; ix < w_segs; ++ix) {
      a = grid[(iy * (w_segs + 1)) + (ix + 1)];
      b = grid[(iy * (w_segs + 1)) + ix];
      c = grid[((iy + 1) * (w_segs + 1)) + ix];
      d = grid[((iy + 1) * (w_segs + 1)) + (ix + 1)];

      if (iy != 0 || t_start > 0) {
        sphere->indices.data[ii * 3]     = a;
        sphere->indices.data[ii * 3 + 1] = b;
        sphere->indices.data[ii * 3 + 2] = d;
        ++ii;
      }
      if (iy != h_segs - 1 || te < PI) {
        sphere->indices.data[ii * 3]     = b;
        sphere->indices.data[ii * 3 + 1] = c;
        sphere->indices.data[ii * 3 + 2] = d;
        ++ii;
      }
    }
  }

  sphere->indices.count = ii * 3;
  free(grid);
}

/* -------------------------------------------------------------------------- *
 * GPU buffer helpers
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_buffer_t vertices;
  wgpu_buffer_t normals;
  wgpu_buffer_t uvs;
  wgpu_buffer_t indices;
} geometry_gpu_buffers_t;

typedef struct {
  wgpu_buffer_t model_matrix;
  wgpu_buffer_t normal_matrix;
} instanced_gpu_buffers_t;

typedef void (*instance_callback_t)(vec3* position, vec3* rotation, vec3* scale,
                                    bool scale_uniformly);

static void get_rand_position_scale_rotation(vec3* position, vec3* rotation,
                                             vec3* scale, bool scale_uniformly)
{
  (*position)[0] = (random_float() * 2.0f - 1.0f) * WORLD_SIZE_X;
  (*position)[1] = (random_float() * 2.0f - 1.0f) * WORLD_SIZE_Y;
  (*position)[2] = (random_float() * 2.0f - 1.0f) * WORLD_SIZE_Z;

  (*rotation)[0] = random_float() * PI2;
  (*rotation)[1] = random_float() * PI2;
  (*rotation)[2] = random_float() * PI2;

  (*scale)[0] = random_float() + 0.25f;
  (*scale)[1] = scale_uniformly ? (*scale)[0] : random_float() + 0.25f;
  (*scale)[2] = scale_uniformly ? (*scale)[0] : random_float() + 0.25f;
}

static void generate_instance_matrices(instanced_geometry_t* inst,
                                       instance_callback_t callback,
                                       bool scale_uniformly)
{
  const uint32_t count = ARRAY_SIZE(inst->model_matrix_data) / 16;
  mat4 model           = GLM_MAT4_ZERO_INIT;
  mat4 norm            = GLM_MAT4_ZERO_INIT;
  vec3 pos = GLM_VEC3_ZERO_INIT, rot = GLM_VEC3_ZERO_INIT,
       scl = GLM_VEC3_ZERO_INIT;

  for (uint32_t idx = 0; idx < count; ++idx) {
    uint32_t off = idx * 16;
    callback(&pos, &rot, &scl, scale_uniformly);

    glm_mat4_identity(model);
    glm_translate(model, pos);
    glm_rotate(model, rot[0], (vec3){1.0f, 0.0f, 0.0f});
    glm_rotate(model, rot[1], (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(model, rot[2], (vec3){0.0f, 0.0f, 1.0f});
    glm_scale(model, scl);

    glm_mat4_inv(model, norm);
    glm_mat4_transpose(norm);

    for (uint32_t n = 0; n < 16; ++n) {
      uint32_t r                        = n / 4;
      uint32_t c                        = n % 4;
      inst->model_matrix_data[off + n]  = model[r][c];
      inst->normal_matrix_data[off + n] = norm[r][c];
    }
  }
}

static void create_geo_gpu_buffers(wgpu_context_t* ctx, geometry_t* geo,
                                   geometry_gpu_buffers_t* buf)
{
  buf->vertices = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Vertices buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size         = geo->positions.data_size,
           .initial.data = geo->positions.data,
         });
  buf->normals = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Normals buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size         = geo->normals.data_size,
           .initial.data = geo->normals.data,
         });
  buf->uvs = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "UVs buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size         = geo->uvs.data_size,
           .initial.data = geo->uvs.data,
         });
  buf->indices = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Indices buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
           .size         = geo->indices.data_size,
           .count        = geo->indices.count,
           .initial.data = geo->indices.data,
         });
}

static void create_inst_gpu_buffers(wgpu_context_t* ctx,
                                    instanced_geometry_t* inst,
                                    instanced_gpu_buffers_t* buf)
{
  buf->model_matrix = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Instance model matrix buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size         = sizeof(inst->model_matrix_data),
           .initial.data = inst->model_matrix_data,
         });
  buf->normal_matrix = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Instance normal matrix buffer",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
           .size         = sizeof(inst->normal_matrix_data),
           .initial.data = inst->normal_matrix_data,
         });
}

/* -------------------------------------------------------------------------- *
 * Render pass helper
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */

static struct {
  /* Options / GUI state */
  struct {
    bool animatable;
    float tween_factor;
    float tween_factor_target;
    vec3 light_position;
    vec3 base_colors[2];
  } options;

  /* Cameras */
  perspective_camera_t persp_camera;
  orthographic_camera_t ortho_camera;

  /* Transform for fullscreen quad */
  transform_t quad_transform;

  /* Geometries (CPU data) */
  geometry_t quad_geo;
  geometry_t cube_geo;
  geometry_t sphere_geo;

  /* Instanced geometry data */
  instanced_geometry_t instanced_cube;
  instanced_geometry_t instanced_sphere;

  /* GPU buffers */
  geometry_gpu_buffers_t quad_buf;
  geometry_gpu_buffers_t cube_buf;
  geometry_gpu_buffers_t sphere_buf;
  instanced_gpu_buffers_t inst_cube_buf;
  instanced_gpu_buffers_t inst_sphere_buf;

  /* Uniform buffers */
  struct {
    wgpu_buffer_t persp_camera;
    wgpu_buffer_t ortho_camera;
    wgpu_buffer_t quad_transform;
    wgpu_buffer_t quad_tween_factor;
    wgpu_buffer_t light_position;
    wgpu_buffer_t base_colors[2];
  } uniforms;

  /* Textures */
  struct {
    WGPUTexture post_fx0_tex;
    WGPUTextureView post_fx0_view;
    WGPUTexture post_fx1_tex;
    WGPUTextureView post_fx1_view;
    WGPUSampler sampler;
    wgpu_texture_t cutoff_mask;
  } textures;

  /* Offscreen framebuffer */
  struct {
    WGPUTexture color_tex;
    WGPUTextureView color_view;
    WGPUTexture depth_tex;
    WGPUTextureView depth_view;
  } offscreen;

  /* Bind groups */
  struct {
    WGPUBindGroup persp_camera;
    WGPUBindGroup ortho_camera;
    WGPUBindGroup quad_transform;
    WGPUBindGroup quad_sampler;
    WGPUBindGroup quad_tween;
    WGPUBindGroup light_position;
    WGPUBindGroup base_colors[2];
  } bind_groups;

  /* Pipelines */
  WGPURenderPipeline quad_pipeline;
  WGPURenderPipeline scene_pipeline;

  /* Render passes */
  render_pass_t scene_rpass;
  render_pass_t postfx_rpass;

  /* Timing */
  float old_time;
  float last_tween_change_time;
  uint64_t last_frame_time;

  /* Async texture loading buffer */
  uint8_t file_buffer[2048 * 2048 * 4];

  WGPUBool initialized;
} state = {
  .options = {
    .animatable          = true,
    .tween_factor        = 0.0f,
    .tween_factor_target = 0.0f,
    .light_position      = {0.5f, 0.5f, 0.50f},
    .base_colors[0]      = {0.3f, 0.6f, 0.70f},
    .base_colors[1]      = {1.0f, 0.2f, 0.25f},
  },
};

/* -------------------------------------------------------------------------- *
 * Setup cameras
 * -------------------------------------------------------------------------- */

static void setup_cameras(wgpu_context_t* ctx)
{
  const float w = (float)ctx->width;
  const float h = (float)ctx->height;

  perspective_camera_init(&state.persp_camera, (45.0f * PI) / 180.0f, w / h,
                          0.1f, 100.0f);

  orthographic_camera_t* cam = &state.ortho_camera;
  orthographic_camera_init(cam, -w / 2.0f, w / 2.0f, h / 2.0f, -h / 2.0f, 0.1f,
                           3.0f);
  glm_vec3_copy((vec3){0.0f, 0.0f, 2.0f}, cam->position);
  glm_vec3_copy(GLM_VEC3_ZERO, cam->look_at_position);
  orthographic_camera_update_projection_matrix(cam);
  orthographic_camera_update_view_matrix(cam);
}

/* -------------------------------------------------------------------------- *
 * Prepare geometries
 * -------------------------------------------------------------------------- */

static void prepare_geometries(wgpu_context_t* ctx)
{
  /* Fullscreen quad */
  create_plane(&state.quad_geo, (uint32_t)ctx->width, (uint32_t)ctx->height, 1,
               1);
  create_geo_gpu_buffers(ctx, &state.quad_geo, &state.quad_buf);

  /* Cube */
  create_box(&state.cube_geo);
  create_geo_gpu_buffers(ctx, &state.cube_geo, &state.cube_buf);

  /* Instanced cube matrices */
  generate_instance_matrices(&state.instanced_cube,
                             get_rand_position_scale_rotation, false);
  create_inst_gpu_buffers(ctx, &state.instanced_cube, &state.inst_cube_buf);

  /* Sphere */
  create_sphere(&state.sphere_geo);
  create_geo_gpu_buffers(ctx, &state.sphere_geo, &state.sphere_buf);

  /* Instanced sphere matrices */
  generate_instance_matrices(&state.instanced_sphere,
                             get_rand_position_scale_rotation, false);
  create_inst_gpu_buffers(ctx, &state.instanced_sphere, &state.inst_sphere_buf);
}

/* -------------------------------------------------------------------------- *
 * Prepare uniform buffers
 * -------------------------------------------------------------------------- */

static void prepare_uniform_buffers(wgpu_context_t* ctx)
{
  transform_init(&state.quad_transform);

  state.uniforms.persp_camera = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label = "Perspective camera uniform",
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size  = 16 * 2 * sizeof(float),
         });

  state.uniforms.ortho_camera = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label = "Orthographic camera uniform",
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size  = 16 * 2 * sizeof(float),
         });

  state.uniforms.quad_transform = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label = "Quad transform uniform",
           .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size  = 16 * sizeof(float),
         });

  state.uniforms.quad_tween_factor = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Tween factor uniform",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size         = sizeof(float),
           .initial.data = &state.options.tween_factor,
         });

  state.uniforms.light_position = wgpu_create_buffer(
    ctx, &(wgpu_buffer_desc_t){
           .label        = "Light position uniform",
           .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
           .size         = 16,
           .initial.data = state.options.light_position,
         });

  for (uint32_t i = 0; i < 2; ++i) {
    state.uniforms.base_colors[i] = wgpu_create_buffer(
      ctx, &(wgpu_buffer_desc_t){
             .label        = "Base color uniform",
             .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
             .size         = 16,
             .initial.data = state.options.base_colors[i],
           });
  }
}

/* -------------------------------------------------------------------------- *
 * Prepare offscreen framebuffer
 * -------------------------------------------------------------------------- */

static void prepare_offscreen_framebuffer(wgpu_context_t* ctx)
{
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.color_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.depth_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.depth_view)

  WGPUExtent3D size = {
    .width              = (uint32_t)ctx->width,
    .height             = (uint32_t)ctx->height,
    .depthOrArrayLayers = 1,
  };

  /* Color */
  {
    WGPUTextureDescriptor desc = {
      .label         = STRVIEW("Offscreen color texture"),
      .size          = size,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = ctx->render_format,
      .usage         = WGPUTextureUsage_RenderAttachment
               | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc,
    };
    state.offscreen.color_tex = wgpuDeviceCreateTexture(ctx->device, &desc);
    ASSERT(state.offscreen.color_tex != NULL);

    state.offscreen.color_view = wgpuTextureCreateView(
      state.offscreen.color_tex, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("Offscreen color view"),
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .format    = desc.format,
                                   .baseMipLevel    = 0,
                                   .mipLevelCount   = 1,
                                   .baseArrayLayer  = 0,
                                   .arrayLayerCount = 1,
                                 });
    ASSERT(state.offscreen.color_view != NULL);
  }

  /* Depth stencil */
  {
    WGPUTextureDescriptor desc = {
      .label         = STRVIEW("Offscreen depth texture"),
      .size          = size,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    state.offscreen.depth_tex = wgpuDeviceCreateTexture(ctx->device, &desc);
    ASSERT(state.offscreen.depth_tex != NULL);

    state.offscreen.depth_view = wgpuTextureCreateView(
      state.offscreen.depth_tex, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("Offscreen depth view"),
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .format    = desc.format,
                                   .baseMipLevel    = 0,
                                   .mipLevelCount   = 1,
                                   .baseArrayLayer  = 0,
                                   .arrayLayerCount = 1,
                                   .aspect          = WGPUTextureAspect_All,
                                 });
    ASSERT(state.offscreen.depth_view != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Prepare post-processing textures
 * -------------------------------------------------------------------------- */

static void prepare_post_fx_textures(wgpu_context_t* ctx)
{
  WGPU_RELEASE_RESOURCE(Texture, state.textures.post_fx0_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.post_fx0_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.post_fx1_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.post_fx1_view)

  WGPUExtent3D size = {
    .width              = (uint32_t)ctx->width,
    .height             = (uint32_t)ctx->height,
    .depthOrArrayLayers = 1,
  };

  /* Post-fx0 */
  {
    WGPUTextureDescriptor desc = {
      .label     = STRVIEW("Post-fx0 texture"),
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = size,
      .format    = ctx->render_format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.textures.post_fx0_tex = wgpuDeviceCreateTexture(ctx->device, &desc);
    ASSERT(state.textures.post_fx0_tex);

    state.textures.post_fx0_view = wgpuTextureCreateView(
      state.textures.post_fx0_tex, &(WGPUTextureViewDescriptor){
                                     .label     = STRVIEW("Post-fx0 view"),
                                     .format    = desc.format,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                   });
    ASSERT(state.textures.post_fx0_view);
  }

  /* Post-fx1 */
  {
    WGPUTextureDescriptor desc = {
      .label     = STRVIEW("Post-fx1 texture"),
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = size,
      .format    = ctx->render_format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.textures.post_fx1_tex = wgpuDeviceCreateTexture(ctx->device, &desc);
    ASSERT(state.textures.post_fx1_tex);

    state.textures.post_fx1_view = wgpuTextureCreateView(
      state.textures.post_fx1_tex, &(WGPUTextureViewDescriptor){
                                     .label     = STRVIEW("Post-fx1 view"),
                                     .format    = desc.format,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                   });
    ASSERT(state.textures.post_fx1_view);
  }

  /* Sampler (only create once) */
  if (!state.textures.sampler) {
    state.textures.sampler = wgpuDeviceCreateSampler(
      ctx->device, &(WGPUSamplerDescriptor){
                     .label         = STRVIEW("Post-fx sampler"),
                     .addressModeU  = WGPUAddressMode_Repeat,
                     .addressModeV  = WGPUAddressMode_Repeat,
                     .addressModeW  = WGPUAddressMode_Repeat,
                     .magFilter     = WGPUFilterMode_Linear,
                     .minFilter     = WGPUFilterMode_Linear,
                     .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                     .lodMinClamp   = 0.0f,
                     .lodMaxClamp   = 1.0f,
                     .maxAnisotropy = 1,
                   });
    ASSERT(state.textures.sampler);
  }
}

/* -------------------------------------------------------------------------- *
 * Async cutoff mask texture loading via sokol_fetch
 * -------------------------------------------------------------------------- */

static void cutoff_mask_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    return;
  }

  int w = 0, h = 0, ch = 0;
  stbi_uc* pixels = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &w, &h, &ch, 4);
  if (pixels) {
    wgpu_texture_t* tex = *(wgpu_texture_t**)response->user_data;
    tex->desc           = (wgpu_texture_desc_t){
                .extent
      = {.width = (uint32_t)w, .height = (uint32_t)h, .depthOrArrayLayers = 1},
                .format = WGPUTextureFormat_RGBA8Unorm,
                .pixels = {.ptr = pixels, .size = (size_t)(w * h * 4)},
    };
    tex->desc.is_dirty = true;
  }
}

static void prepare_cutoff_mask_texture(wgpu_context_t* ctx)
{
  state.textures.cutoff_mask = wgpu_create_color_bars_texture(ctx, NULL);

  wgpu_texture_t* tex_ptr = &state.textures.cutoff_mask;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/transition2.png",
    .callback  = cutoff_mask_fetch_cb,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {.ptr = &tex_ptr, .size = sizeof(wgpu_texture_t*)},
  });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void prepare_quad_pipeline(wgpu_context_t* ctx)
{
  WGPUShaderModule vs
    = wgpu_create_shader_module(ctx->device, quad_vertex_shader_wgsl);
  WGPUShaderModule fs
    = wgpu_create_shader_module(ctx->device, quad_fragment_shader_wgsl);

  WGPUVertexAttribute pos_attr
    = {.shaderLocation = 0, .offset = 0, .format = WGPUVertexFormat_Float32x3};
  WGPUVertexAttribute uv_attr
    = {.shaderLocation = 1, .offset = 0, .format = WGPUVertexFormat_Float32x2};

  WGPUVertexBufferLayout vbl[2] = {
    {.arrayStride    = 12,
     .stepMode       = WGPUVertexStepMode_Vertex,
     .attributeCount = 1,
     .attributes     = &pos_attr},
    {.arrayStride    = 8,
     .stepMode       = WGPUVertexStepMode_Vertex,
     .attributeCount = 1,
     .attributes     = &uv_attr},
  };

  WGPUBlendState blend = wgpu_create_blend_state(true);

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("Fullscreen quad pipeline"),
    .vertex = {.module      = vs,
               .entryPoint  = STRVIEW("main"),
               .bufferCount = 2,
               .buffers     = vbl},
    .fragment
    = &(WGPUFragmentState){.module      = fs,
                           .entryPoint  = STRVIEW("main"),
                           .targetCount = 1,
                           .targets     = &(
                             WGPUColorTargetState){.format = ctx->render_format,
                                                       .blend  = &blend,
                                                       .writeMask
                                                       = WGPUColorWriteMask_All}},
    .primitive   = {.topology  = WGPUPrimitiveTopology_TriangleList,
                    .frontFace = WGPUFrontFace_CW,
                    .cullMode  = WGPUCullMode_Back},
    .multisample = {.count = 1, .mask = 0xffffffff},
  };

  state.quad_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  ASSERT(state.quad_pipeline != NULL);

  wgpuShaderModuleRelease(vs);
  wgpuShaderModuleRelease(fs);
}

static void prepare_scene_pipeline(wgpu_context_t* ctx)
{
  WGPUShaderModule vs
    = wgpu_create_shader_module(ctx->device, instanced_vertex_shader_wgsl);
  WGPUShaderModule fs
    = wgpu_create_shader_module(ctx->device, instanced_fragment_shader_wgsl);

  WGPUVertexAttribute a_pos
    = {.shaderLocation = 0, .offset = 0, .format = WGPUVertexFormat_Float32x3};
  WGPUVertexAttribute a_norm
    = {.shaderLocation = 1, .offset = 0, .format = WGPUVertexFormat_Float32x3};

  WGPUVertexAttribute a_model[4] = {
    {.shaderLocation = 2, .offset = 0, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 3, .offset = 16, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 4, .offset = 32, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 5, .offset = 48, .format = WGPUVertexFormat_Float32x4},
  };
  WGPUVertexAttribute a_nmat[4] = {
    {.shaderLocation = 6, .offset = 0, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 7, .offset = 16, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 8, .offset = 32, .format = WGPUVertexFormat_Float32x4},
    {.shaderLocation = 9, .offset = 48, .format = WGPUVertexFormat_Float32x4},
  };

  WGPUVertexBufferLayout vbl[4] = {
    {.arrayStride    = 12,
     .stepMode       = WGPUVertexStepMode_Vertex,
     .attributeCount = 1,
     .attributes     = &a_pos},
    {.arrayStride    = 12,
     .stepMode       = WGPUVertexStepMode_Vertex,
     .attributeCount = 1,
     .attributes     = &a_norm},
    {.arrayStride    = 64,
     .stepMode       = WGPUVertexStepMode_Instance,
     .attributeCount = 4,
     .attributes     = a_model},
    {.arrayStride    = 64,
     .stepMode       = WGPUVertexStepMode_Instance,
     .attributeCount = 4,
     .attributes     = a_nmat},
  };

  WGPUDepthStencilState ds
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24PlusStencil8,
      .depth_write_enabled = true,
    });
  ds.depthCompare = WGPUCompareFunction_Less;

  WGPUBlendState blend = wgpu_create_blend_state(true);

  WGPURenderPipelineDescriptor desc = {
    .label  = STRVIEW("Scene meshes pipeline"),
    .vertex = {.module      = vs,
               .entryPoint  = STRVIEW("main"),
               .bufferCount = 4,
               .buffers     = vbl},
    .fragment
    = &(WGPUFragmentState){.module      = fs,
                           .entryPoint  = STRVIEW("main"),
                           .targetCount = 1,
                           .targets     = &(
                             WGPUColorTargetState){.format = ctx->render_format,
                                                       .blend  = &blend,
                                                       .writeMask
                                                       = WGPUColorWriteMask_All}},
    .primitive    = {.topology  = WGPUPrimitiveTopology_TriangleList,
                     .frontFace = WGPUFrontFace_CCW,
                     .cullMode  = WGPUCullMode_None},
    .depthStencil = &ds,
    .multisample  = {.count = 1, .mask = 0xffffffff},
  };

  state.scene_pipeline = wgpuDeviceCreateRenderPipeline(ctx->device, &desc);
  ASSERT(state.scene_pipeline != NULL);

  wgpuShaderModuleRelease(vs);
  wgpuShaderModuleRelease(fs);
}

/* -------------------------------------------------------------------------- *
 * Setup bind groups
 * -------------------------------------------------------------------------- */

static void setup_bind_groups(wgpu_context_t* ctx)
{
  /* Perspective camera */
  state.bind_groups.persp_camera = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Persp camera bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.scene_pipeline, 0),
      .entryCount = 1,
      .entries
      = &(WGPUBindGroupEntry){.binding = 0,
                              .buffer  = state.uniforms.persp_camera.buffer,
                              .offset  = 0,
                              .size    = state.uniforms.persp_camera.size},
    });

  /* Orthographic camera */
  state.bind_groups.ortho_camera = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Ortho camera bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.quad_pipeline, 0),
      .entryCount = 1,
      .entries
      = &(WGPUBindGroupEntry){.binding = 0,
                              .buffer  = state.uniforms.ortho_camera.buffer,
                              .offset  = 0,
                              .size    = state.uniforms.ortho_camera.size},
    });

  /* Quad transform */
  state.bind_groups.quad_transform = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Quad transform bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.quad_pipeline, 1),
      .entryCount = 1,
      .entries
      = &(WGPUBindGroupEntry){.binding = 0,
                              .buffer  = state.uniforms.quad_transform.buffer,
                              .offset  = 0,
                              .size    = state.uniforms.quad_transform.size},
    });

  /* Quad sampler */
  WGPUBindGroupEntry sampler_entries[4] = {
    {.binding = 0, .sampler = state.textures.sampler},
    {.binding = 1, .textureView = state.textures.post_fx0_view},
    {.binding = 2, .textureView = state.textures.post_fx1_view},
    {.binding = 3, .textureView = state.textures.cutoff_mask.view},
  };
  state.bind_groups.quad_sampler = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Quad sampler bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.quad_pipeline, 2),
      .entryCount = 4,
      .entries    = sampler_entries,
    });

  /* Quad tween */
  state.bind_groups.quad_tween = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Quad tween bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.quad_pipeline, 3),
      .entryCount = 1,
      .entries
      = &(WGPUBindGroupEntry){.binding = 0,
                              .buffer = state.uniforms.quad_tween_factor.buffer,
                              .offset = 0,
                              .size   = state.uniforms.quad_tween_factor.size},
    });

  /* Light position */
  state.bind_groups.light_position = wgpuDeviceCreateBindGroup(
    ctx->device,
    &(WGPUBindGroupDescriptor){
      .label  = STRVIEW("Light position bind group"),
      .layout = wgpuRenderPipelineGetBindGroupLayout(state.scene_pipeline, 1),
      .entryCount = 1,
      .entries
      = &(WGPUBindGroupEntry){.binding = 0,
                              .buffer  = state.uniforms.light_position.buffer,
                              .offset  = 0,
                              .size    = state.uniforms.light_position.size},
    });

  /* Base colors */
  for (uint32_t i = 0; i < 2; ++i) {
    state.bind_groups.base_colors[i] = wgpuDeviceCreateBindGroup(
      ctx->device,
      &(WGPUBindGroupDescriptor){
        .label  = STRVIEW("Base color bind group"),
        .layout = wgpuRenderPipelineGetBindGroupLayout(state.scene_pipeline, 2),
        .entryCount = 1,
        .entries
        = &(WGPUBindGroupEntry){.binding = 0,
                                .buffer  = state.uniforms.base_colors[i].buffer,
                                .offset  = 0,
                                .size    = state.uniforms.base_colors[i].size},
      });
  }
}

/* -------------------------------------------------------------------------- *
 * Setup render passes
 * -------------------------------------------------------------------------- */

static void setup_render_passes(void)
{
  /* Scene render pass */
  state.scene_rpass.color_attachments[0] = (WGPURenderPassColorAttachment){
    .view       = NULL,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
  };
  state.scene_rpass.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view              = NULL,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };
  state.scene_rpass.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Scene render pass"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = state.scene_rpass.color_attachments,
    .depthStencilAttachment = &state.scene_rpass.depth_stencil_attachment,
  };

  /* Post-fx render pass */
  state.postfx_rpass.color_attachments[0] = (WGPURenderPassColorAttachment){
    .view       = NULL,
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1f, 0.1f, 0.1f, 1.0f},
  };
  state.postfx_rpass.descriptor = (WGPURenderPassDescriptor){
    .label                  = STRVIEW("Post-fx render pass"),
    .colorAttachmentCount   = 1,
    .colorAttachments       = state.postfx_rpass.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

/* -------------------------------------------------------------------------- *
 * Update uniforms
 * -------------------------------------------------------------------------- */

static void update_tween_factor(wgpu_context_t* ctx)
{
  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.quad_tween_factor.buffer, 0,
                       &state.options.tween_factor, sizeof(float));
}

static void update_uniforms(wgpu_context_t* ctx, float ts, float dt)
{
  if (ts - state.last_tween_change_time > 4.0f) {
    state.options.tween_factor_target
      = fabsf(state.options.tween_factor_target - 1.0f) < PP_EPSILON ? 0.0f :
                                                                       1.0f;
    state.last_tween_change_time = ts;
  }

  /* Update perspective camera */
  glm_vec3_copy((vec3){cosf(ts * 0.2f) * WORLD_SIZE_X, 0.0f,
                       sinf(ts * 0.2f) * WORLD_SIZE_Z},
                state.persp_camera.position);
  glm_vec3_copy(GLM_VEC3_ZERO, state.persp_camera.look_at_position);
  perspective_camera_update_projection_matrix(&state.persp_camera);
  perspective_camera_update_view_matrix(&state.persp_camera);

  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.persp_camera.buffer, 0,
                       state.persp_camera.projection_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.persp_camera.buffer,
                       16 * sizeof(float), state.persp_camera.view_matrix,
                       sizeof(mat4));

  /* Update orthographic camera */
  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.ortho_camera.buffer, 0,
                       state.ortho_camera.projection_matrix, sizeof(mat4));
  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.ortho_camera.buffer,
                       16 * sizeof(float), state.ortho_camera.view_matrix,
                       sizeof(mat4));

  /* Update quad transform */
  transform_set_rotation(&state.quad_transform, (vec3){PI, 0.0f, 0.0f});
  transform_update_model_matrix(&state.quad_transform);
  wgpuQueueWriteBuffer(ctx->queue, state.uniforms.quad_transform.buffer, 0,
                       state.quad_transform.model_matrix, sizeof(mat4));

  /* Update tween factor */
  if (state.options.animatable) {
    state.options.tween_factor
      += (state.options.tween_factor_target - state.options.tween_factor)
         * (dt * 2.0f);
  }
  update_tween_factor(ctx);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* ctx)
{
  UNUSED_VAR(ctx);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Post-processing Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  imgui_overlay_checkbox("Animatable", &state.options.animatable);
  imgui_overlay_slider_float("Tween Factor", &state.options.tween_factor, 0.0f,
                             1.0f, "%.2f");

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* ctx,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(ctx, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    prepare_offscreen_framebuffer(ctx);
    prepare_post_fx_textures(ctx);
    setup_cameras(ctx);

    /* Recreate quad geometry for new dimensions */
    geometry_destroy(&state.quad_geo);
    WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.vertices.buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.normals.buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.uvs.buffer)
    WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.indices.buffer)
    create_plane(&state.quad_geo, (uint32_t)ctx->width, (uint32_t)ctx->height,
                 1, 1);
    create_geo_gpu_buffers(ctx, &state.quad_geo, &state.quad_buf);

    /* Recreate bind groups that reference resized textures */
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.persp_camera)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.ortho_camera)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_transform)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_sampler)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_tween)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.light_position)
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.base_colors[0])
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.base_colors[1])
    setup_bind_groups(ctx);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* ctx)
{
  if (!ctx) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 1,
    .num_channels = 1,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  setup_cameras(ctx);
  prepare_geometries(ctx);
  prepare_uniform_buffers(ctx);
  prepare_offscreen_framebuffer(ctx);
  prepare_post_fx_textures(ctx);
  prepare_cutoff_mask_texture(ctx);
  prepare_quad_pipeline(ctx);
  prepare_scene_pipeline(ctx);
  setup_bind_groups(ctx);
  setup_render_passes();
  imgui_overlay_init(ctx);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* ctx)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update cutoff mask texture when loaded */
  if (state.textures.cutoff_mask.desc.is_dirty) {
    wgpu_recreate_texture(ctx, &state.textures.cutoff_mask);
    FREE_TEXTURE_PIXELS(state.textures.cutoff_mask);

    /* Recreate quad sampler bind group with new texture */
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_sampler)
    WGPUBindGroupEntry entries[4] = {
      {.binding = 0, .sampler = state.textures.sampler},
      {.binding = 1, .textureView = state.textures.post_fx0_view},
      {.binding = 2, .textureView = state.textures.post_fx1_view},
      {.binding = 3, .textureView = state.textures.cutoff_mask.view},
    };
    state.bind_groups.quad_sampler = wgpuDeviceCreateBindGroup(
      ctx->device,
      &(WGPUBindGroupDescriptor){
        .label  = STRVIEW("Quad sampler bind group"),
        .layout = wgpuRenderPipelineGetBindGroupLayout(state.quad_pipeline, 2),
        .entryCount = 4,
        .entries    = entries,
      });
  }

  /* Timing */
  const float ts = (float)stm_sec(stm_now());
  const float dt = ts - state.old_time;
  state.old_time = ts;

  update_uniforms(ctx, ts, dt);

  /* ImGui */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  float delta           = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;
  imgui_overlay_new_frame(ctx, delta);
  render_gui(ctx);

  WGPUDevice device          = ctx->device;
  WGPUQueue queue            = ctx->queue;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* --- Render instanced cubes to offscreen --- */
  state.scene_rpass.color_attachments[0].view     = state.offscreen.color_view;
  state.scene_rpass.depth_stencil_attachment.view = state.offscreen.depth_view;
  state.scene_rpass.color_attachments[0].clearValue
    = (WGPUColor){0.1f, 0.1f, 0.1f, 1.0f};
  {
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.scene_rpass.descriptor);
    wgpuRenderPassEncoderSetPipeline(rp, state.scene_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bind_groups.persp_camera, 0,
                                      0);
    wgpuRenderPassEncoderSetBindGroup(rp, 1, state.bind_groups.light_position,
                                      0, 0);
    wgpuRenderPassEncoderSetBindGroup(rp, 2, state.bind_groups.base_colors[0],
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.cube_buf.vertices.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 1, state.cube_buf.normals.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 2, state.inst_cube_buf.model_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 3, state.inst_cube_buf.normal_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, state.cube_buf.indices.buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, state.cube_geo.indices.count,
                                     INSTANCES_COUNT, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Copy offscreen to post-fx0 texture */
  {
    WGPUTexelCopyTextureInfo src = {.texture = state.offscreen.color_tex};
    WGPUTexelCopyTextureInfo dst = {.texture = state.textures.post_fx0_tex};
    WGPUExtent3D sz              = {.width              = (uint32_t)ctx->width,
                                    .height             = (uint32_t)ctx->height,
                                    .depthOrArrayLayers = 1};
    wgpuCommandEncoderCopyTextureToTexture(cmd_enc, &src, &dst, &sz);
  }

  /* --- Render instanced spheres to offscreen --- */
  state.scene_rpass.color_attachments[0].clearValue
    = (WGPUColor){0.225f, 0.225f, 0.225f, 1.0f};
  {
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.scene_rpass.descriptor);
    wgpuRenderPassEncoderSetPipeline(rp, state.scene_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bind_groups.persp_camera, 0,
                                      0);
    wgpuRenderPassEncoderSetBindGroup(rp, 1, state.bind_groups.light_position,
                                      0, 0);
    wgpuRenderPassEncoderSetBindGroup(rp, 2, state.bind_groups.base_colors[1],
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 0, state.sphere_buf.vertices.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 1, state.sphere_buf.normals.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 2, state.inst_sphere_buf.model_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 3, state.inst_sphere_buf.normal_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, state.sphere_buf.indices.buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, state.sphere_geo.indices.count,
                                     INSTANCES_COUNT, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Copy offscreen to post-fx1 texture */
  {
    WGPUTexelCopyTextureInfo src = {.texture = state.offscreen.color_tex};
    WGPUTexelCopyTextureInfo dst = {.texture = state.textures.post_fx1_tex};
    WGPUExtent3D sz              = {.width              = (uint32_t)ctx->width,
                                    .height             = (uint32_t)ctx->height,
                                    .depthOrArrayLayers = 1};
    wgpuCommandEncoderCopyTextureToTexture(cmd_enc, &src, &dst, &sz);
  }

  /* --- Render post-fx fullscreen quad to swapchain --- */
  state.postfx_rpass.color_attachments[0].view = ctx->swapchain_view;
  {
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.postfx_rpass.descriptor);
    wgpuRenderPassEncoderSetPipeline(rp, state.quad_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bind_groups.ortho_camera, 0,
                                      0);
    wgpuRenderPassEncoderSetBindGroup(rp, 1, state.bind_groups.quad_transform,
                                      0, 0);
    wgpuRenderPassEncoderSetBindGroup(rp, 2, state.bind_groups.quad_sampler, 0,
                                      0);
    wgpuRenderPassEncoderSetBindGroup(rp, 3, state.bind_groups.quad_tween, 0,
                                      0);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, state.quad_buf.vertices.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 1, state.quad_buf.uvs.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, state.quad_buf.indices.buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, state.quad_geo.indices.count, 1, 0, 0,
                                     0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(ctx);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* ctx)
{
  UNUSED_VAR(ctx);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Geometry CPU data */
  geometry_destroy(&state.quad_geo);
  geometry_destroy(&state.cube_geo);
  geometry_destroy(&state.sphere_geo);

  /* GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.normals.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.uvs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad_buf.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buf.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buf.normals.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buf.uvs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_buf.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_buf.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_buf.normals.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_buf.uvs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_buf.indices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_cube_buf.model_matrix.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_cube_buf.normal_matrix.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_sphere_buf.model_matrix.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.inst_sphere_buf.normal_matrix.buffer)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.persp_camera.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.ortho_camera.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.quad_transform.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.quad_tween_factor.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.light_position.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.base_colors[0].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniforms.base_colors[1].buffer)

  /* Textures */
  WGPU_RELEASE_RESOURCE(Texture, state.textures.post_fx0_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.post_fx0_view)
  WGPU_RELEASE_RESOURCE(Texture, state.textures.post_fx1_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.textures.post_fx1_view)
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)
  wgpu_destroy_texture(&state.textures.cutoff_mask);

  /* Offscreen framebuffer */
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.color_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.depth_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.depth_view)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.persp_camera)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.ortho_camera)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_transform)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_sampler)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.quad_tween)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.light_position)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.base_colors[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.base_colors[1])

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.quad_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.scene_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Post-processing",
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

/* Instanced vertex shader */
// clang-format off
static const char* instanced_vertex_shader_wgsl = CODE(
  struct Camera {
    projectionMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>
  }

  @group(0) @binding(0)
  var<uniform> camera: Camera;

  struct Input {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instanceModelMatrix0: vec4<f32>,
    @location(3) instanceModelMatrix1: vec4<f32>,
    @location(4) instanceModelMatrix2: vec4<f32>,
    @location(5) instanceModelMatrix3: vec4<f32>,
    @location(6) instanceNormalMatrix0: vec4<f32>,
    @location(7) instanceNormalMatrix1: vec4<f32>,
    @location(8) instanceNormalMatrix2: vec4<f32>,
    @location(9) instanceNormalMatrix3: vec4<f32>
  }

  struct Output {
    @builtin(position) Position: vec4<f32>,
    @location(0) normal: vec4<f32>,
    @location(1) pos: vec4<f32>
  }

  @vertex
  fn main(input: Input) -> Output {
    var output: Output;

    var instanceModelMatrix: mat4x4<f32> = mat4x4<f32>(
      input.instanceModelMatrix0,
      input.instanceModelMatrix1,
      input.instanceModelMatrix2,
      input.instanceModelMatrix3
    );

    var instanceModelInverseTransposeMatrix: mat4x4<f32> = mat4x4<f32>(
      input.instanceNormalMatrix0,
      input.instanceNormalMatrix1,
      input.instanceNormalMatrix2,
      input.instanceNormalMatrix3
    );

    var worldPosition: vec4<f32> = instanceModelMatrix * input.position;

    output.Position = camera.projectionMatrix *
                      camera.viewMatrix *
                      worldPosition;

    output.normal = instanceModelInverseTransposeMatrix * vec4<f32>(input.normal, 0.0);
    output.pos = worldPosition;

    return output;
  }
);
// clang-format on

/* Instanced fragment shader */
// clang-format off
static const char* instanced_fragment_shader_wgsl = CODE(
  struct Lighting {
    position: vec3<f32>
  }

  @group(1) @binding(0)
  var<uniform> lighting: Lighting;

  struct Material {
    baseColor: vec3<f32>
  }

  @group(2) @binding(0)
  var<uniform> material: Material;

  struct Input {
    @location(0) normal: vec4<f32>,
    @location(1) pos: vec4<f32>
  }

  @fragment
  fn main(input: Input) -> @location(0) vec4<f32> {
    var normal: vec3<f32> = normalize(input.normal.rgb);
    var lightColor: vec3<f32> = vec3<f32>(1.0);

    var ambientFactor: f32 = 0.1;
    var ambientLight: vec3<f32> = lightColor * ambientFactor;

    var lightDirection: vec3<f32> = normalize(lighting.position - input.pos.rgb);
    var diffuseStrength: f32 = max(dot(normal, lightDirection), 0.0);
    var diffuseLight: vec3<f32> = lightColor * diffuseStrength;

    var finalLight: vec3<f32> = diffuseLight + ambientLight;

    return vec4<f32>(material.baseColor * finalLight, 1.0);
  }
);
// clang-format on

/* Quad vertex shader */
// clang-format off
static const char* quad_vertex_shader_wgsl = CODE(
  struct Camera {
    projectionMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>
  }

  @group(0) @binding(0)
  var<uniform> camera: Camera;

  struct Transform {
    modelMatrix: mat4x4<f32>
  }

  @group(1) @binding(0)
  var<uniform> transform: Transform;

  struct Input {
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>
  }

  struct Output {
    @builtin(position) Position: vec4<f32>,
    @location(0) uv: vec2<f32>
  }

  @vertex
  fn main(input: Input) -> Output {
    var output: Output;

    output.Position = camera.projectionMatrix *
                      camera.viewMatrix *
                      transform.modelMatrix *
                      input.position;

    output.uv = input.uv;

    return output;
  }
);
// clang-format on

/* Quad fragment shader */
// clang-format off
static const char* quad_fragment_shader_wgsl = CODE(
  struct Input {
    @location(0) uv: vec2<f32>
  }

  @group(2) @binding(0) var mySampler: sampler;
  @group(2) @binding(1) var postFX0Texture: texture_2d<f32>;
  @group(2) @binding(2) var postFX1Texture: texture_2d<f32>;
  @group(2) @binding(3) var cutOffTexture: texture_2d<f32>;

  struct Tween {
    factor: f32
  }

  @group(3) @binding(0)
  var<uniform> tween: Tween;

  @fragment
  fn main(input: Input) -> @location(0) vec4<f32> {
    var result0: vec4<f32> = textureSample(postFX0Texture, mySampler, input.uv);
    var result1: vec4<f32> = textureSample(postFX1Texture, mySampler, input.uv);

    var cutoffResult: vec4<f32> = textureSample(cutOffTexture, mySampler, input.uv);

    var mixFactor: f32 = step(tween.factor * 1.05, cutoffResult.r);

    return mix(result0, result1, mixFactor);
  }
);
// clang-format on
