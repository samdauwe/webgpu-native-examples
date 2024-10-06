#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Post-processing
 *
 * This example shows how to use a post-processing effect to blend between two
 * scenes.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-dojo/tree/master/src/examples/postprocessing-01
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Global variables
 * -------------------------------------------------------------------------- */

#define INSTANCES_COUNT 500u
#define WORLD_SIZE_X 20u
#define WORLD_SIZE_Y 20u
#define WORLD_SIZE_Z 20u

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

static void transform_copy_from_matrix(transform_t* transform, mat4 matrix)
{
  glm_mat4_copy(matrix, transform->model_matrix);
  transform->should_update = false;
}

/**
 * @brief Sets position
 */
static void transform_set_position(transform_t* transform, vec3 position)
{
  glm_vec3_copy(position, transform->position);
  transform->should_update = true;
}

/**
 * @brief Sets scale
 */
static void transform_set_scale(transform_t* transform, vec3 scale)
{
  glm_vec3_copy(scale, transform->scale);
  transform->should_update = true;
}

/**
 * @brief Sets rotation
 */
static void transform_set_rotation(transform_t* transform, vec3 rotation)
{
  glm_vec3_copy(rotation, transform->rotation);
  transform->should_update = true;
}

/**
 * @brief Update model matrix with scale, rotation and translation.
 */
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
  UNUSED_FUNCTION(transform_copy_from_matrix);
  UNUSED_FUNCTION(transform_set_position);
  UNUSED_FUNCTION(transform_set_scale);

  glm_vec3_zero(transform->position);
  glm_vec3_zero(transform->rotation);
  glm_vec3_one(transform->scale);
  transform->should_update = true;
}

/* -------------------------------------------------------------------------- *
 * Orthographic Camera
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/main/src/camera/orthographic-camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 UP_VECTOR;
  float left;
  float right;
  float top;
  float bottom;
  float near;
  float far;
  float zoom;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 view_matrix;
} orthographic_camera_t;

static void orthographic_camera_set_position(orthographic_camera_t* camera,
                                             vec3 target)
{
  glm_vec3_copy(target, camera->position);
}

static void
orthographic_camera_update_view_matrix(orthographic_camera_t* camera)
{
  glm_lookat(camera->position,         /* eye    */
             camera->look_at_position, /* center */
             camera->UP_VECTOR,        /* up     */
             camera->view_matrix       /* dest   */
  );
}

static void
orthographic_camera_update_projection_matrix(orthographic_camera_t* camera)
{
  glm_ortho(camera->left,             /* left   */
            camera->right,            /* right  */
            camera->bottom,           /* bottom */
            camera->top,              /* top    */
            camera->near,             /* nearZ  */
            camera->far,              // farZ   */
            camera->projection_matrix /* dest   */
  );
}

static void orthographic_camera_look_at(orthographic_camera_t* camera,
                                        vec3 target)
{
  glm_vec3_copy(target, camera->look_at_position);
  orthographic_camera_update_view_matrix(camera);
}

static void orthographic_camera_init_defaults(orthographic_camera_t* camera)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, camera->UP_VECTOR);

  camera->left   = -1.0f;
  camera->right  = 1.0f;
  camera->top    = 1.0f;
  camera->bottom = -1.0f;
  camera->near   = 0.1f;
  camera->far    = 2000.0f;
  camera->zoom   = 1.0f;

  glm_vec3_zero(camera->position);
  glm_vec3_zero(camera->look_at_position);
  glm_mat4_zero(camera->projection_matrix);
  glm_mat4_zero(camera->view_matrix);
}

static void orthographic_camera_init(orthographic_camera_t* camera, float left,
                                     float right, float top, float bottom,
                                     float near, float far)
{
  orthographic_camera_init_defaults(camera);

  camera->left   = left;
  camera->right  = right;
  camera->top    = top;
  camera->bottom = bottom;

  camera->near = near;
  camera->far  = far;

  orthographic_camera_update_projection_matrix(camera);
}

/* -------------------------------------------------------------------------- *
 * Perspective Camera
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/main/src/camera/perspective-camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 UP_VECTOR;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 view_matrix;
  float zoom;
  float field_of_view;
  float aspect;
  float near;
  float far;
} perspective_camera_t;

static void perspective_camera_set_position(perspective_camera_t* camera,
                                            vec3 target)
{
  glm_vec3_copy(target, camera->position);
}

static void perspective_camera_update_view_matrix(perspective_camera_t* camera)
{
  glm_lookat(camera->position,         /* eye    */
             camera->look_at_position, /* center */
             camera->UP_VECTOR,        /* up     */
             camera->view_matrix       /* dest   */
  );
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* camera)
{
  glm_perspective(camera->field_of_view,    /* fovy   */
                  camera->aspect,           /* aspect */
                  camera->near,             /* nearZ  */
                  camera->far,              /* farZ   */
                  camera->projection_matrix /* dest   */
  );
}

static void perspective_camera_look_at(perspective_camera_t* camera,
                                       vec3 target)
{
  glm_vec3_copy(target, camera->look_at_position);
  perspective_camera_update_view_matrix(camera);
}

static void perspective_camera_init_defaults(perspective_camera_t* camera)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, camera->UP_VECTOR);
  glm_vec3_zero(camera->position);
  glm_vec3_zero(camera->look_at_position);
  glm_mat4_zero(camera->projection_matrix);
  glm_mat4_zero(camera->view_matrix);

  camera->zoom = 1.0f;
}

static void perspective_camera_init(perspective_camera_t* camera,
                                    float field_of_view, float aspect,
                                    float near, float far)
{
  perspective_camera_init_defaults(camera);

  camera->field_of_view = field_of_view;
  camera->aspect        = aspect;
  camera->near          = near;
  camera->far           = far;

  perspective_camera_update_projection_matrix(camera);
}

/* -------------------------------------------------------------------------- *
 * Geometry class
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    float* data;
    size_t data_size;
    size_t count;
  } positions;
  struct {
    float* data;
    size_t data_size;
    size_t count;
  } normals;
  struct {
    float* data;
    size_t data_size;
    size_t count;
  } uvs;
  struct {
    uint32_t* data;
    size_t data_size;
    size_t count;
  } indices;
} geometry_t;

typedef struct {
  float model_matrix_data[INSTANCES_COUNT * 16];
  float normal_matrix_data[INSTANCES_COUNT * 16];
} instanced_geometry_t;

static void geometry_destroy(geometry_t* geometry)
{
  if ((geometry->positions.data != NULL) && geometry->positions.data_size > 0) {
    free(geometry->positions.data);
    geometry->positions.data      = NULL;
    geometry->positions.data_size = 0;
    geometry->positions.count     = 0;
  }
  if ((geometry->normals.data != NULL) && geometry->normals.data_size > 0) {
    free(geometry->normals.data);
    geometry->normals.data      = NULL;
    geometry->normals.data_size = 0;
    geometry->normals.count     = 0;
  }
  if ((geometry->uvs.data != NULL) && geometry->uvs.data_size > 0) {
    free(geometry->uvs.data);
    geometry->uvs.data      = NULL;
    geometry->uvs.data_size = 0;
    geometry->uvs.count     = 0;
  }
  if ((geometry->indices.data != NULL) && geometry->indices.data_size > 0) {
    free(geometry->indices.data);
    geometry->indices.data      = NULL;
    geometry->indices.data_size = 0;
    geometry->indices.count     = 0;
  }
}

/* -------------------------------------------------------------------------- *
 * Plane Geometry
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/0f865ca0d47f9d0e1fd527ee6f30a6ade32edcd7/src/geometry-utils/create-plane.ts
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/0f865ca0d47f9d0e1fd527ee6f30a6ade32edcd7/src/geometry-utils/build-plane.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t width_segments;
  uint32_t height_segments;
} plane_desc_t;

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

      normal[i * 3 + u] = 0.f;
      normal[i * 3 + v] = 0.f;
      normal[i * 3 + w] = depth >= 0 ? 1.f : -1.f;

      uv[i * 2]     = ix / w_segs;
      uv[i * 2 + 1] = 1 - iy / h_segs;

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

/**
 * @brief Generates geometry data for a quad.
 * @param plane plane geometry
 * @param plane_desc params
 * @return pointer to the generated geometry data
 */
static geometry_t* create_plane(geometry_t* plane, plane_desc_t* plane_desc)
{
  const uint32_t width  = (plane_desc != NULL) ? plane_desc->width : 1;
  const uint32_t height = (plane_desc != NULL) ? plane_desc->height : 1;

  const uint32_t w_segs = (plane_desc != NULL) ? plane_desc->width_segments : 1;
  const uint32_t h_segs
    = (plane_desc != NULL) ? plane_desc->height_segments : 1;

  // Determine length of arrays
  const uint32_t num         = (w_segs + 1) * (h_segs + 1);
  const uint32_t num_indices = w_segs * h_segs * 6;

  // Set array sizes
  plane->positions.count = num;
  plane->normals.count   = num;
  plane->uvs.count       = num;
  plane->indices.count   = num_indices;

  // Set array size (in bytes)
  plane->positions.data_size = num * 3 * sizeof(float);
  plane->normals.data_size   = num * 3 * sizeof(float);
  plane->uvs.data_size       = num * 2 * sizeof(float);
  plane->indices.data_size   = num_indices * sizeof(uint32_t);

  // Generate empty arrays once
  plane->positions.data = (float*)malloc(plane->positions.data_size);
  plane->normals.data   = (float*)malloc(plane->normals.data_size);
  plane->uvs.data       = (float*)malloc(plane->uvs.data_size);
  plane->indices.data   = (uint32_t*)malloc(plane->indices.data_size);

  build_plane(plane->positions.data, /* vertices */
              plane->normals.data,   /* normal   */
              plane->uvs.data,       /* uv       */
              plane->indices.data,   /* indices  */
              width,                 /* width    */
              height,                /* height   */
              0,                     /* depth    */
              w_segs,                /* w_segs   */
              h_segs,                /* h_segs   */
              0,                     /* u        */
              1,                     /* v        */
              2,                     /* w        */
              1,                     /* u_dir    */
              -1,                    /* v_dir    */
              0,                     /* i        */
              0                      /* ii       */
  );

  return plane;
}

/* -------------------------------------------------------------------------- *
 * Box Geometry
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/0f865ca0d47f9d0e1fd527ee6f30a6ade32edcd7/src/geometry-utils/create-box.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t width_segments;
  uint32_t height_segments;
  uint32_t depth_segments;
  bool separate_faces;
} box_desc_t;

/**
 * @brief Generates geometry data for a box.
 * @param box box geometry
 * @param box_desc params
 * @return pointer to the generated geometry data
 */
static geometry_t* create_box(geometry_t* box, box_desc_t* box_desc)
{
  const uint32_t width  = (box_desc != NULL) ? box_desc->width : 1;
  const uint32_t height = (box_desc != NULL) ? box_desc->height : 1;
  const uint32_t depth  = (box_desc != NULL) ? box_desc->depth : 1;

  const uint32_t w_segs = (box_desc != NULL) ? box_desc->width_segments : 1;
  const uint32_t h_segs = (box_desc != NULL) ? box_desc->height_segments : 1;
  const uint32_t d_segs = (box_desc != NULL) ? box_desc->depth_segments : 1;

  const uint32_t num = (w_segs + 1) * (h_segs + 1) * 2 + //
                       (w_segs + 1) * (d_segs + 1) * 2 + //
                       (h_segs + 1) * (d_segs + 1) * 2;
  const uint32_t num_indices
    = (w_segs * h_segs * 2 + w_segs * d_segs * 2 + h_segs * d_segs * 2) * 6;

  // Set array count
  box->positions.count = num;
  box->normals.count   = num;
  box->uvs.count       = num;
  box->indices.count   = num_indices;

  // Set array size (in bytes)
  box->positions.data_size = num * 3 * sizeof(float);
  box->normals.data_size   = num * 3 * sizeof(float);
  box->uvs.data_size       = num * 2 * sizeof(float);
  box->indices.data_size   = num_indices * sizeof(uint32_t);

  // Generate empty arrays once
  box->positions.data = (float*)malloc(box->positions.data_size);
  box->normals.data   = (float*)malloc(box->normals.data_size);
  box->uvs.data       = (float*)malloc(box->uvs.data_size);
  box->indices.data   = (uint32_t*)malloc(box->indices.data_size);

  uint32_t i = 0, ii = 0;

  /* RIGHT */
  {
    build_plane(box->positions.data, // vertices
                box->normals.data,   // normal
                box->uvs.data,       // uv
                box->indices.data,   // indices
                depth,               // width
                height,              // height
                width,               // depth
                d_segs,              // w_segs
                h_segs,              // h_segs
                2,                   // u
                1,                   // v
                0,                   // w
                -1,                  // u_dir
                -1,                  // v_dir
                i,                   // i
                ii                   // ii
    );
  }

  /* LEFT */
  {
    build_plane(box->positions.data,                // vertices
                box->normals.data,                  // normal
                box->uvs.data,                      // uv
                box->indices.data,                  // indices
                depth,                              // width
                height,                             // height
                -((int)width),                      // depth
                d_segs,                             // w_segs
                h_segs,                             // h_segs
                2,                                  // u
                1,                                  // v
                0,                                  // w
                1,                                  // u_dir
                -1,                                 // v_dir
                (i += (d_segs + 1) * (h_segs + 1)), // i
                (ii += d_segs * h_segs)             // ii
    );
  }

  /* TOP */
  {
    build_plane(box->positions.data,                // vertices
                box->normals.data,                  // normal
                box->uvs.data,                      // uv
                box->indices.data,                  // indices
                width,                              // width
                depth,                              // height
                height,                             // depth
                d_segs,                             // w_segs
                h_segs,                             // h_segs
                0,                                  // u
                2,                                  // v
                1,                                  // w
                1,                                  // u_dir
                1,                                  // v_dir
                (i += (d_segs + 1) * (h_segs + 1)), // i
                (ii += d_segs * h_segs)             // ii
    );
  }

  /* BOTTOM */
  {
    build_plane(box->positions.data,                // vertices
                box->normals.data,                  // normal
                box->uvs.data,                      // uv
                box->indices.data,                  // indices
                width,                              // width
                depth,                              // height
                -((int)height),                     // depth
                d_segs,                             // w_segs
                h_segs,                             // h_segs
                0,                                  // u
                2,                                  // v
                1,                                  // w
                1,                                  // u_dir
                -1,                                 // v_dir
                (i += (w_segs + 1) * (d_segs + 1)), // i
                (ii += w_segs * d_segs)             // ii
    );
  }

  /* BACK */
  {
    build_plane(box->positions.data,                // vertices
                box->normals.data,                  // normal
                box->uvs.data,                      // uv
                box->indices.data,                  // indices
                width,                              // width
                height,                             // height
                -((int)depth),                      // depth
                w_segs,                             // w_segs
                h_segs,                             // h_segs
                0,                                  // u
                1,                                  // v
                2,                                  // w
                -1,                                 // u_dir
                -1,                                 // v_dir
                (i += (w_segs + 1) * (d_segs + 1)), // i
                (ii += w_segs * d_segs)             // ii
    );
  }

  /* FRONT */
  {
    build_plane(box->positions.data,                // vertices
                box->normals.data,                  // normal
                box->uvs.data,                      // uv
                box->indices.data,                  // indices
                width,                              // width
                height,                             // height
                depth,                              // depth
                w_segs,                             // w_segs
                h_segs,                             // h_segs
                0,                                  // u
                1,                                  // v
                2,                                  // w
                1,                                  // u_dir
                -1,                                 // v_dir
                (i += (w_segs + 1) * (h_segs + 1)), // i
                (ii += w_segs * h_segs)             // ii
    );
  }

  return box;
}

/* -------------------------------------------------------------------------- *
 * Sphere Geometry
 *
 * Ref:
 * https://github.com/gnikoloff/hwoa-rang-gl/blob/0f865ca0d47f9d0e1fd527ee6f30a6ade32edcd7/src/geometry-utils/create-sphere.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float radius;
  uint32_t width_segments;
  uint32_t height_segments;
  float phi_start;
  float phi_length;
  float theta_start;
  float theta_length;
} sphere_desc_t;

/**
 * @brief Generates geometry data for a sphere.
 * @param sphere sphere geometry
 * @param sphere_desc sphere creation parameters
 * @return pointer to the generated geometry data
 */
static geometry_t* create_sphere(geometry_t* sphere, sphere_desc_t* sphere_desc)
{
  const float radius = (sphere_desc != NULL) ? sphere_desc->radius : 0.5f;
  const uint32_t w_segs
    = (sphere_desc != NULL) ? sphere_desc->width_segments : 16u;
  const uint32_t h_segs = (sphere_desc != NULL) ? sphere_desc->height_segments :
                                                  (uint32_t)ceil(w_segs * 0.5f);
  const float p_start   = (sphere_desc != NULL) ? sphere_desc->phi_start : 0.0f;
  const float p_length  = (sphere_desc != NULL) ? sphere_desc->phi_length : PI2;
  const float t_start = (sphere_desc != NULL) ? sphere_desc->theta_start : 0.0f;
  const float t_length = (sphere_desc != NULL) ? sphere_desc->theta_length : PI;

  const uint32_t num         = (w_segs + 1) * (h_segs + 1);
  const uint32_t num_indices = w_segs * h_segs * 6;

  /* Set array count */
  sphere->positions.count = num;
  sphere->normals.count   = num;
  sphere->uvs.count       = num;
  sphere->indices.count   = num_indices;

  /* Set array size (in bytes) */
  sphere->positions.data_size = num * 3 * sizeof(float);
  sphere->normals.data_size   = num * 3 * sizeof(float);
  sphere->uvs.data_size       = num * 2 * sizeof(float);
  sphere->indices.data_size   = num_indices * sizeof(uint32_t);

  /* Generate empty arrays once */
  sphere->positions.data = (float*)malloc(sphere->positions.data_size);
  sphere->normals.data   = (float*)malloc(sphere->normals.data_size);
  sphere->uvs.data       = (float*)malloc(sphere->uvs.data_size);
  sphere->indices.data   = (uint32_t*)malloc(sphere->indices.data_size);

  uint32_t i        = 0;
  uint32_t iv       = 0;
  uint32_t ii       = 0;
  const uint32_t te = t_start + t_length;
  uint32_t* grid    = (uint32_t*)malloc(num * sizeof(uint32_t));

  vec3 n = GLM_VEC3_ZERO_INIT;

  float v = 0.0f, u = 0.0f, x = 0.0f, y = 0.0f, z = 0.0f;
  for (uint32_t iy = 0; iy <= h_segs; ++iy) {
    v = iy / (float)h_segs;
    for (uint32_t ix = 0; ix <= w_segs; ++ix, ++i) {
      u = ix / (float)w_segs;
      x = -radius *                     //
          cos(p_start + u * p_length) * //
          sin(t_start + v * t_length);
      y = radius * cos(t_start + v * t_length);
      z = radius * sin(p_start + u * p_length) * sin(t_start + v * t_length);

      sphere->positions.data[i * 3]     = x;
      sphere->positions.data[i * 3 + 1] = y;
      sphere->positions.data[i * 3 + 2] = z;

      glm_vec3_copy((vec3){x, y, z}, n);
      glm_vec3_normalize(n);

      sphere->normals.data[i * 3]     = n[0];
      sphere->normals.data[i * 3 + 1] = n[1];
      sphere->normals.data[i * 3 + 2] = n[2];

      sphere->uvs.data[i * 2]     = u;
      sphere->uvs.data[i * 2 + 1] = 1 - v;

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

  free(grid);

  return sphere;
}

/* -------------------------------------------------------------------------- *
 * Helper Functions
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-dojo/blob/master/src/examples/postprocessing-01/helpers.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_buffer_t vertices;
  wgpu_buffer_t normals;
  wgpu_buffer_t uvs;
  wgpu_buffer_t indices;
} geometry_gpu_buffers_t;

static void
geometry_gpu_buffers_destroy(geometry_gpu_buffers_t* geometry_gpu_buffers)
{
  WGPU_RELEASE_RESOURCE(Buffer, geometry_gpu_buffers->vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, geometry_gpu_buffers->normals.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, geometry_gpu_buffers->uvs.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, geometry_gpu_buffers->indices.buffer)
}

typedef struct {
  wgpu_buffer_t model_matrix;
  wgpu_buffer_t normal_matrix;
} instanced_geometry_gpu_buffers_t;

static void instanced_geometry_gpu_buffers_destroy(
  instanced_geometry_gpu_buffers_t* instanced_geometry_gpu_buffers)
{
  WGPU_RELEASE_RESOURCE(Buffer,
                        instanced_geometry_gpu_buffers->model_matrix.buffer)
  WGPU_RELEASE_RESOURCE(Buffer,
                        instanced_geometry_gpu_buffers->normal_matrix.buffer)
}

typedef void (*generate_instance_on_item_callback)(vec3* position,
                                                   vec3* rotation, vec3* scale,
                                                   bool scale_uniformly);

static instanced_geometry_t*
generate_instance_matrices(instanced_geometry_t* instanced_model,
                           generate_instance_on_item_callback on_item_callback,
                           bool scale_uniformly)
{
  uint32_t count = ARRAY_SIZE(instanced_model->model_matrix_data) / 16;
  vec3 instance_move_vector  = GLM_VEC3_ZERO_INIT;
  mat4 instance_model_matrix = GLM_MAT4_ZERO_INIT,
       normal_matrix         = GLM_MAT4_ZERO_INIT;
  vec3 position = GLM_VEC3_ZERO_INIT, rotation = GLM_VEC3_ZERO_INIT,
       scale = GLM_VEC3_ZERO_INIT;
  uint32_t r = 0, c = 0;
  for (uint32_t i = 0; i < count * 16; i += 16) {
    on_item_callback(&position, &rotation, &scale, scale_uniformly);

    glm_mat4_identity(instance_model_matrix);
    glm_vec3_copy((vec3){position[0], position[1], position[2]},
                  instance_move_vector);

    glm_translate(instance_model_matrix, instance_move_vector);
    glm_rotate(instance_model_matrix, rotation[0], (vec3){1.0f, 0.0f, 0.0f});
    glm_rotate(instance_model_matrix, rotation[1], (vec3){0.0f, 1.0f, 0.0f});
    glm_rotate(instance_model_matrix, rotation[2], (vec3){0.0f, 0.0f, 1.0f});

    glm_vec3_copy((vec3){scale[0], scale[1], scale[2]}, instance_move_vector);
    glm_scale(instance_model_matrix, instance_move_vector);

    glm_mat4_inv(instance_model_matrix, normal_matrix);
    glm_mat4_transpose(normal_matrix);

    for (uint32_t n = 0; n < 16; ++n) {
      r = n / 4;
      c = n % 4;

      instanced_model->model_matrix_data[i + n]  = instance_model_matrix[r][c];
      instanced_model->normal_matrix_data[i + n] = normal_matrix[r][c];
    }
  }

  return instanced_model;
}

static void
generate_gpu_buffers_from_geometry(wgpu_context_t* wgpu_context,
                                   geometry_t* geometry,
                                   geometry_gpu_buffers_t* gpu_buffers)
{
  /* Vertices */
  gpu_buffers->vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Vertices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = geometry->positions.data_size,
                    .initial.data = geometry->positions.data,
                  });

  /* Normals */
  gpu_buffers->normals = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Normals buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = geometry->normals.data_size,
                    .initial.data = geometry->normals.data,
                  });

  /* UVs */
  gpu_buffers->uvs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "UVs buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = geometry->uvs.data_size,
                    .initial.data = geometry->uvs.data,
                  });

  /* Indices */
  uint32_t index_count
    = geometry->indices.data_size / sizeof(geometry->indices.data[0]);
  gpu_buffers->indices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Indices buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = geometry->indices.data_size,
                    .count = index_count,
                    .initial.data = geometry->indices.data,
                  });
}

static void generate_gpu_buffers_from_instanced_geometry(
  wgpu_context_t* wgpu_context, instanced_geometry_t* geometry,
  instanced_geometry_gpu_buffers_t* gpu_buffers)
{
  /* Instance model matrix */
  gpu_buffers->model_matrix = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Instance - Model matrix",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(geometry->model_matrix_data),
                    .initial.data = geometry->model_matrix_data,
                  });

  /* Instance model matrix */
  gpu_buffers->normal_matrix = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Instance - Model matrix",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(geometry->normal_matrix_data),
                    .initial.data = geometry->normal_matrix_data,
                  });
}

/* -------------------------------------------------------------------------- *
 * Post-processing example
 * -------------------------------------------------------------------------- */

static struct {
  bool animatable;
  float tween_factor;
  float tween_factor_target;
  vec3 light_position;
  vec3 base_colors[2];
} options = {
  .animatable          = true,
  .tween_factor        = 0.0f,
  .tween_factor_target = 0.0f,
  .light_position      = {0.5f, 0.5f, 0.50f},
  .base_colors[0]      = {0.3f, 0.6f, 0.70f},
  .base_colors[1]      = {1.0f, 0.2f, 0.25f},
};

static transform_t quad_transform = {0};

static struct {
  perspective_camera_t perspective_camera;
  orthographic_camera_t orthographic_camera;
} cameras = {0};

static struct {
  geometry_t quad;
  geometry_t cube;
  geometry_t sphere;
} geometries = {0};

static struct {
  instanced_geometry_t cube;
  instanced_geometry_t sphere;
} instanced_geometries = {0};

static struct {
  geometry_gpu_buffers_t quad;
  geometry_gpu_buffers_t cube;
  geometry_gpu_buffers_t sphere;
  instanced_geometry_gpu_buffers_t instanced_cube;
  instanced_geometry_gpu_buffers_t instanced_sphere;
} vertex_buffers = {0};

static struct {
  wgpu_buffer_t persp_camera, ortho_camera, quad_transform, quad_tween_factor,
    light_position, base_colors[2];
} uniform_buffers = {0};

static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } post_fx0, post_fx1;
  WGPUSampler post_fx_sampler;
  texture_t cutoff_mask;
} textures = {0};

/* Framebuffer for offscreen rendering */
static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
  } color, depth_stencil;
} offscreen_framebuffer = {0};

static struct {
  WGPUBindGroup persp_camera;
  WGPUBindGroup ortho_camera;
  WGPUBindGroup quad_transform;
  WGPUBindGroup quad_sampler;
  WGPUBindGroup quad_tween;
  WGPUBindGroup light_position;
  WGPUBindGroup base_colors[2];
} bind_groups = {0};

static struct {
  WGPURenderPipeline fullscreen_quad;
  WGPURenderPipeline scene_meshes;
} pipelines = {0};

/* Render pass descriptor for frame buffer writes */
typedef struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass_t;

static struct {
  render_pass_t scene_render;
  render_pass_t post_fx;
} render_passes = {0};

/* Holds time info in seconds units */
static struct {
  float old_time;
  float last_tween_factor_target_change_time;
} time_info = {0};

/* Other variables */
static const char* example_title = "Post-processing";
static bool prepared             = false;

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

// Set up cameras for the demo
static void setup_cameras(wgpu_example_context_t* context)
{
  const float surface_width  = (float)context->wgpu_context->surface.width;
  const float surface_height = (float)context->wgpu_context->surface.height;

  /* Perspective camera */
  {
    perspective_camera_init(&cameras.perspective_camera, (45.0f * PI) / 180.0f,
                            surface_width / surface_height, 0.1f, 100.0f);
  }

  /* Orthographic camera */
  {
    orthographic_camera_t* camera = &cameras.orthographic_camera;

    orthographic_camera_init(camera, -surface_width / 2.0f,
                             surface_width / 2.0f, surface_height / 2.0f,
                             -surface_height / 2.0f, 0.1f, 3.0f);
    orthographic_camera_set_position(camera, (vec3){0.0f, 0.0f, 2.0f});
    orthographic_camera_look_at(camera, GLM_VEC3_ZERO);
    orthographic_camera_update_projection_matrix(camera);
    orthographic_camera_update_view_matrix(camera);
  }
}

/* Prepare geometries */
static void prepare_geometries(wgpu_context_t* wgpu_context)
{
  /* Prepare fullscreen quad gpu buffers */
  generate_gpu_buffers_from_geometry(
    wgpu_context,
    create_plane(&geometries.quad,
                 &(plane_desc_t){
                   .width           = wgpu_context->surface.width,
                   .height          = wgpu_context->surface.height,
                   .width_segments  = 1u,
                   .height_segments = 1u,
                 }),
    &vertex_buffers.quad);

  /* Prepare cube gpu buffers */
  generate_gpu_buffers_from_geometry(
    wgpu_context, create_box(&geometries.cube, NULL), &vertex_buffers.cube);

  /* Prepare instanced cube model matrices as gpu buffers */
  generate_gpu_buffers_from_instanced_geometry(
    wgpu_context,
    generate_instance_matrices(&instanced_geometries.cube,
                               get_rand_position_scale_rotation, false),
    &vertex_buffers.instanced_cube);

  /* Prepare sphere gpu buffers */
  generate_gpu_buffers_from_geometry(wgpu_context,
                                     create_sphere(&geometries.sphere, NULL),
                                     &vertex_buffers.sphere);

  /* Prepare instanced sphere model matrices as gpu buffers */
  generate_gpu_buffers_from_instanced_geometry(
    wgpu_context,
    generate_instance_matrices(&instanced_geometries.sphere,
                               get_rand_position_scale_rotation, false),
    &vertex_buffers.instanced_sphere);
}

static void update_tween_factor(wgpu_context_t* wgpu_context)
{
  wgpu_queue_write_buffer(wgpu_context,
                          uniform_buffers.quad_tween_factor.buffer, 0,
                          &options.tween_factor, sizeof(float));
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  const float ts     = context->frame.timestamp_millis / 1000.0f;
  const float dt     = ts - time_info.old_time;
  time_info.old_time = ts;

  if (ts - time_info.last_tween_factor_target_change_time > 4.0f) {
    options.tween_factor_target
      = fabs(options.tween_factor_target - 1.0f) < EPSILON ? 0 : 1;
    time_info.last_tween_factor_target_change_time = ts;
  }

  /* Write perspective camera projection and view matrix to uniform block */
  perspective_camera_set_position(&cameras.perspective_camera,
                                  (vec3){
                                    cos(ts * 0.2f) * WORLD_SIZE_X, /* x */
                                    0.0f,                          /* y */
                                    sin(ts * 0.2f) * WORLD_SIZE_Z, /* z */
                                  });
  perspective_camera_look_at(&cameras.perspective_camera, GLM_VEC3_ZERO);
  perspective_camera_update_projection_matrix(&cameras.perspective_camera);
  perspective_camera_update_view_matrix(&cameras.perspective_camera);
  wgpu_queue_write_buffer(context->wgpu_context,
                          uniform_buffers.persp_camera.buffer, 0,
                          cameras.perspective_camera.projection_matrix,
                          sizeof(cameras.perspective_camera.projection_matrix));
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.persp_camera.buffer,
    16 * sizeof(float), cameras.perspective_camera.view_matrix,
    sizeof(cameras.perspective_camera.view_matrix));

  /* Write ortho camera projection and view matrix to uniform block */
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.ortho_camera.buffer, 0,
    cameras.orthographic_camera.projection_matrix,
    sizeof(cameras.orthographic_camera.projection_matrix));
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.ortho_camera.buffer,
    16 * sizeof(float), cameras.orthographic_camera.view_matrix,
    sizeof(cameras.orthographic_camera.view_matrix));

  /* Write fullscreen quad model matrix to uniform block */
  transform_set_rotation(&quad_transform, (vec3){PI, 0.0f, 0.0f});
  transform_update_model_matrix(&quad_transform);
  wgpu_queue_write_buffer(
    context->wgpu_context, uniform_buffers.quad_transform.buffer, 0,
    quad_transform.model_matrix, sizeof(quad_transform.model_matrix));

  /* Write tween factor */
  if (options.animatable) {
    options.tween_factor
      += (options.tween_factor_target - options.tween_factor) * (dt * 2.0f);
  }
  update_tween_factor(context->wgpu_context);
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Perspective camera uniform block */
  uniform_buffers.persp_camera = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Perspective camera - Uniform block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 16 * 2 * sizeof(float),
                  });

  /* Orthographic camera uniform block */
  uniform_buffers.ortho_camera = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Orthographic camera - Uniform block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 16 * 2 * sizeof(float),
                  });

  /* Fullscreen quad transform uniform block */
  transform_init(&quad_transform);
  uniform_buffers.quad_transform = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fullscreen quad transform - Uniform block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 16 * sizeof(float),
                  });

  /* Tween factor uniform block */
  uniform_buffers.quad_tween_factor = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Tween factor - Uniform block",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(options.tween_factor),
                    .initial.data = &options.tween_factor,
                  });

  /* Instanced scenes light position as a typed 32 bit array */
  uniform_buffers.light_position = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Instanced scenes light position",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = 16,
                    .initial.data = options.light_position,
                  });

  /* Instanced scenes base colors */
  for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(options.base_colors); ++i) {
    uniform_buffers.base_colors[i] = wgpu_create_buffer(
      wgpu_context,
      &(wgpu_buffer_desc_t){
        .label        = "Instanced scenes base colors",
        .usage        = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
        .size         = 16,
        .initial.data = options.base_colors[i],
      });
  }
}

static void prepare_offscreen_framebuffer(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D texture_extent = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };

  /* Color attachment */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen framebuffer - Color attachment texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = wgpu_context->swap_chain.format,
      .usage         = WGPUTextureUsage_RenderAttachment
               | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc,
    };
    offscreen_framebuffer.color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_framebuffer.color.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label          = "Offscreen framebuffer - Color attachment texture view",
      .dimension      = WGPUTextureViewDimension_2D,
      .format         = texture_desc.format,
      .baseMipLevel   = 0,
      .mipLevelCount  = 1,
      .baseArrayLayer = 0,
      .arrayLayerCount = 1,
    };
    offscreen_framebuffer.color.texture_view = wgpuTextureCreateView(
      offscreen_framebuffer.color.texture, &texture_view_dec);
    ASSERT(offscreen_framebuffer.color.texture_view != NULL);
  }

  /* Depth stencil attachment */
  {
    WGPUTextureDescriptor texture_desc = {
      .label         = "Offscreen framebuffer - Depth attachment texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
    };
    offscreen_framebuffer.depth_stencil.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(offscreen_framebuffer.depth_stencil.texture != NULL);

    /* Create the texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label          = "Offscreen framebuffer - Depth attachment texture view",
      .dimension      = WGPUTextureViewDimension_2D,
      .format         = texture_desc.format,
      .baseMipLevel   = 0,
      .mipLevelCount  = 1,
      .baseArrayLayer = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    offscreen_framebuffer.depth_stencil.texture_view = wgpuTextureCreateView(
      offscreen_framebuffer.depth_stencil.texture, &texture_view_dec);
    ASSERT(offscreen_framebuffer.depth_stencil.texture_view != NULL);
  }
}

/* Set up texture and sampler needed for postprocessing */
static void prepare_textures(wgpu_context_t* wgpu_context)
{
  WGPUExtent3D texture_size = {
    .width              = wgpu_context->surface.width,
    .height             = wgpu_context->surface.height,
    .depthOrArrayLayers = 1,
  };

  /* Post-fx0 texture and view */
  {
    /* Create texture */
    WGPUTextureDescriptor texture_desc = {
      .label     = "Post-fx0 texture",
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = texture_size,
      .format    = wgpu_context->swap_chain.format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    textures.post_fx0.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.post_fx0.texture);

    /* Create texture view */
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Post-fx0 texture view",
      .format          = texture_desc.format,
      .dimension       = WGPUTextureViewDimension_2D,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    textures.post_fx0.view
      = wgpuTextureCreateView(textures.post_fx0.texture, &texture_view_desc);
    ASSERT(textures.post_fx0.view);
  }

  /* Post-fx1 texture and view */
  {
    /* Create texture */
    WGPUTextureDescriptor texture_desc = {
      .label     = "Post-fx1 texture",
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = texture_size,
      .format    = wgpu_context->swap_chain.format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    textures.post_fx1.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.post_fx1.texture);

    /* Create texture view */
    WGPUTextureViewDescriptor texture_view_desc = {
      .label           = "Post-fx1 texture view",
      .format          = texture_desc.format,
      .dimension       = WGPUTextureViewDimension_2D,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    textures.post_fx1.view
      = wgpuTextureCreateView(textures.post_fx1.texture, &texture_view_desc);
    ASSERT(textures.post_fx1.view);
  }

  /* Texture sampler */
  {
    WGPUSamplerDescriptor sampler_desc = {
      .label         = "Post-fx1 texture sampler",
      .addressModeU  = WGPUAddressMode_Repeat,
      .addressModeV  = WGPUAddressMode_Repeat,
      .addressModeW  = WGPUAddressMode_Repeat,
      .magFilter     = WGPUFilterMode_Linear,
      .minFilter     = WGPUFilterMode_Linear,
      .mipmapFilter  = WGPUMipmapFilterMode_Linear,
      .lodMinClamp   = 0.0f,
      .lodMaxClamp   = 1.0f,
      .maxAnisotropy = 1,
      .compare       = WGPUCompareFunction_Undefined,
    };
    textures.post_fx_sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
    ASSERT(textures.post_fx_sampler);
  }

  /* Cutoff mask transition texture */
  {
    const char* file     = "textures/transition2.png";
    textures.cutoff_mask = wgpu_create_texture_from_file(
      wgpu_context, file,
      &(struct wgpu_texture_load_options_t){
        .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding
                 | WGPUTextureUsage_RenderAttachment,
      });
  }
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Perspective camera bind group */
  {
    bind_groups.persp_camera = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
       .label  = "Perspective camera - Bind group",
       .layout = wgpuRenderPipelineGetBindGroupLayout(pipelines.scene_meshes, 0),
       .entryCount = 1,
       .entries    = &(WGPUBindGroupEntry) {
         .binding = 0,
         .buffer  = uniform_buffers.persp_camera.buffer,
         .offset  = 0,
         .size    = uniform_buffers.persp_camera.size,
       },
     }
    );
    ASSERT(bind_groups.persp_camera != NULL);
  }

  /* Orthographic camera bind group */
  {
    bind_groups.ortho_camera = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
       .label  = "Orthographic camera - Bind group",
       .layout = wgpuRenderPipelineGetBindGroupLayout(pipelines.fullscreen_quad,
                                                      0),
       .entryCount = 1,
       .entries    = &(WGPUBindGroupEntry) {
         .binding = 0,
         .buffer  = uniform_buffers.ortho_camera.buffer,
         .offset  = 0,
         .size    = uniform_buffers.ortho_camera.size,
       },
     }
    );
    ASSERT(bind_groups.ortho_camera != NULL);
  }

  /* Fullscreen quad transform uniform block */
  {
    bind_groups.quad_transform = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
       .label  = "Fullscreen quad transform - Uniform block",
       .layout = wgpuRenderPipelineGetBindGroupLayout(pipelines.fullscreen_quad,
                                                      1),
       .entryCount = 1,
       .entries    = &(WGPUBindGroupEntry) {
         .binding = 0,
         .buffer  = uniform_buffers.quad_transform.buffer,
         .offset  = 0,
         .size    = uniform_buffers.quad_transform.size,
       },
     }
    );
    ASSERT(bind_groups.quad_transform != NULL);
  }

  /* Quad sampler uniform bind group */
  {
    WGPUBindGroupEntry bg_entries[4] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = textures.post_fx_sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .textureView = textures.post_fx0.view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .textureView = textures.post_fx1.view,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .textureView = textures.cutoff_mask.view,
      },
    };
    bind_groups.quad_sampler = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "Quad sampler uniform - Bind group",
                              .layout = wgpuRenderPipelineGetBindGroupLayout(
                                pipelines.fullscreen_quad, 2),
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(bind_groups.quad_sampler != NULL);
  }

  /* Quad tween uniform bind group */
  {
    bind_groups.quad_tween = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
       .label  = "Quad tween uniform - Bind group",
       .layout = wgpuRenderPipelineGetBindGroupLayout(
         pipelines.fullscreen_quad, 3),
       .entryCount = 1,
       .entries    = &(WGPUBindGroupEntry) {
         .binding = 0,
         .buffer  = uniform_buffers.quad_tween_factor.buffer,
         .offset  = 0,
         .size    = uniform_buffers.quad_tween_factor.size,
       },
     }
    );
    ASSERT(bind_groups.quad_tween != NULL);
  }

  /* Light position uniform bind group */
  {
    bind_groups.light_position = wgpuDeviceCreateBindGroup(
          wgpu_context->device,
          &(WGPUBindGroupDescriptor) {
           .label  = "Light position uniform - Bind group",
           .layout = wgpuRenderPipelineGetBindGroupLayout(
             pipelines.scene_meshes, 1),
           .entryCount = 1,
           .entries    = &(WGPUBindGroupEntry) {
             .binding = 0,
             .buffer  = uniform_buffers.light_position.buffer,
             .offset  = 0,
             .size    = uniform_buffers.light_position.size,
           },
         }
        );
    ASSERT(bind_groups.light_position != NULL)
  }

  /* Base color uniform bind groups */
  {
    uint32_t base_color_cnt = (uint32_t)ARRAY_SIZE(uniform_buffers.base_colors);
    for (uint32_t i = 0; i < base_color_cnt; ++i) {
      bind_groups.base_colors[i] = wgpuDeviceCreateBindGroup(
            wgpu_context->device,
            &(WGPUBindGroupDescriptor) {
             .label  = "Base color uniform - Bind group",
             .layout = wgpuRenderPipelineGetBindGroupLayout(
               pipelines.scene_meshes, i + 1),
             .entryCount = 1,
             .entries    = &(WGPUBindGroupEntry) {
               .binding = 0,
               .buffer  = uniform_buffers.base_colors[i].buffer,
               .offset  = 0,
               .size    = uniform_buffers.base_colors[i].size,
             },
           }
          );
      ASSERT(bind_groups.base_colors[i] != NULL);
    }
  }
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Instanced scene render pass descriptor */
  {
    /* Color attachments */
    render_passes.scene_render.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = NULL, /* Assigned later */
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

    /* Depth attachment */
    render_passes.scene_render.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = offscreen_framebuffer.depth_stencil.texture_view,
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };

    /* Render pass descriptor */
    render_passes.scene_render.descriptor = (WGPURenderPassDescriptor){
      .label                = "Instanced scene - Render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = render_passes.scene_render.color_attachments,
      .depthStencilAttachment
      = &render_passes.scene_render.depth_stencil_attachment};
  }

  /* Postfx fullscreen quad render pass descriptor */
  {
    /* Color attachments */
    render_passes.post_fx.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = NULL,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.1f,
          .g = 0.1f,
          .b = 0.1f,
          .a = 1.0f,
        },
      };

    /* Render pass descriptor */
    render_passes.post_fx.descriptor = (WGPURenderPassDescriptor){
      .label                = "Postfx fullscreen quad - Render pass descriptor",
      .colorAttachmentCount = 1,
      .colorAttachments     = render_passes.post_fx.color_attachments,
      .depthStencilAttachment = NULL,
    };
  }
}

/* Fullscreen quad pipeline */
static void prepare_fullscreen_quad_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CW,
    .cullMode  = WGPUCullMode_Back,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Vertex buffer layout */
  WGPUVertexBufferLayout quad_vertex_buffer_layouts[2] = {0};
  {
    WGPUVertexAttribute attribute = {
      /* Shader location 0 : position attribute */
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    quad_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
      .arrayStride    = 3 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }
  {
    WGPUVertexAttribute attribute = {
      /* Shader location 1 : uv attribute */
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x2,
    };
    quad_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
      .arrayStride    = 2 * sizeof(float),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &attribute,
    };
  }

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Vertex shader WGSL */
                .label = "Quad shader - Vertex shader",
                .file  = "shaders/post_processing/quad-shader.vert.wgsl",
                .entry = "main"
              },
              .buffer_count = (uint32_t)ARRAY_SIZE(quad_vertex_buffer_layouts),
              .buffers      = quad_vertex_buffer_layouts,
            });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Fragment shader WGSL */
                .label = "Quad shader - Fragment shader",
                .file  = "shaders/post_processing/quad-shader.frag.wgsl",
                .entry = "main"
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

  /* Create rendering pipeline using the specified states */
  pipelines.fullscreen_quad = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label       = "Fullscreen quad - Render pipeline",
                            .primitive   = primitive_state,
                            .vertex      = vertex_state,
                            .fragment    = &fragment_state,
                            .multisample = multisample_state,
                          });
  ASSERT(pipelines.fullscreen_quad != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

/* Instanced meshes pipeline */
static void prepare_instanced_meshes_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
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
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Vertex buffer layout */
  WGPUVertexBufferLayout instanced_meshes_vertex_buffer_layouts[4] = {0};

  WGPUVertexAttribute attribute_0 = {
    /* Shader location 0 : position attribute */
    .shaderLocation = 0,
    .offset         = 0 * sizeof(float),
    .format         = WGPUVertexFormat_Float32x3,
  };
  instanced_meshes_vertex_buffer_layouts[0] = (WGPUVertexBufferLayout){
    .arrayStride    = 3 * sizeof(float),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = &attribute_0,
  };

  WGPUVertexAttribute attribute_1 = {
    /* Shader location 1 : normal attribute */
    .shaderLocation = 1,
    .offset         = 0,
    .format         = WGPUVertexFormat_Float32x3,
  };
  instanced_meshes_vertex_buffer_layouts[1] = (WGPUVertexBufferLayout){
    .arrayStride    = 3 * sizeof(float),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 1,
    .attributes     = &attribute_1,
  };

  // We need to pass the mat4x4<f32> instance world matrix as 4 vec4<f32>()
  // components It will occupy 4 input slots
  WGPUVertexAttribute attributes_2[4] = {
       [0] = (WGPUVertexAttribute) {
         .shaderLocation = 2,
         .offset         = 0 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [1] = (WGPUVertexAttribute) {
         .shaderLocation = 3,
         .offset         = 4 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [2] = (WGPUVertexAttribute) {
         .shaderLocation = 4,
         .offset         = 8 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [3] = (WGPUVertexAttribute) {
         .shaderLocation = 5,
         .offset         = 12 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       }
     };
  instanced_meshes_vertex_buffer_layouts[2] = (WGPUVertexBufferLayout){
    .arrayStride    = 16 * sizeof(float),
    .stepMode       = WGPUVertexStepMode_Instance,
    .attributeCount = (uint32_t)ARRAY_SIZE(attributes_2),
    .attributes     = attributes_2,
  };

  // We need to pass the mat4x4<f32> instance normal matrix as 4 vec4<f32>()
  // components It will occupy 4 input slots
  WGPUVertexAttribute attributes_3[4] = {
       [0] = (WGPUVertexAttribute) {
         .shaderLocation = 6,
         .offset         = 0 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [1] = (WGPUVertexAttribute) {
         .shaderLocation = 7,
         .offset         = 4 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [2] = (WGPUVertexAttribute) {
         .shaderLocation = 8,
         .offset         = 8 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       },
       [3] = (WGPUVertexAttribute) {
         .shaderLocation = 9,
         .offset         = 12 * sizeof(float),
         .format         = WGPUVertexFormat_Float32x4,
       }
     };
  instanced_meshes_vertex_buffer_layouts[3] = (WGPUVertexBufferLayout){
    .arrayStride    = 16 * sizeof(float),
    .stepMode       = WGPUVertexStepMode_Instance,
    .attributeCount = (uint32_t)ARRAY_SIZE(attributes_3),
    .attributes     = attributes_3,
  };

  /* Vertex state */
  const uint32_t buffer_count
    = (uint32_t)ARRAY_SIZE(instanced_meshes_vertex_buffer_layouts);
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
              wgpu_context, &(wgpu_vertex_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Vertex shader WGSL */
                .label = "Instanced-shader - Vertex shader",
                .file  = "shaders/post_processing/instanced-shader.vert.wgsl",
                .entry = "main"
              },
              .buffer_count = buffer_count,
              .buffers      = instanced_meshes_vertex_buffer_layouts,
            });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
              wgpu_context, &(wgpu_fragment_state_t){
              .shader_desc = (wgpu_shader_desc_t){
                /* Fragment shader WGSL */
                .label = "Instanced-shader - Fragment shader",
                .file  = "shaders/post_processing/instanced-shader.frag.wgsl",
                .entry = "main"
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

  /* Create rendering pipeline using the specified states */
  pipelines.scene_meshes = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Scene meshes - Render pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipelines.scene_meshes != NULL);

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Animatable",
                           &options.animatable);
    if (imgui_overlay_slider_float(context->imgui_overlay, "Tween Factor",
                                   &options.tween_factor, 0.0f, 1.0f, "%.1f")) {
      update_tween_factor(context->wgpu_context);
    }
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    setup_cameras(context);
    prepare_geometries(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    prepare_offscreen_framebuffer(context->wgpu_context);
    prepare_textures(context->wgpu_context);
    prepare_fullscreen_quad_pipeline(context->wgpu_context);
    prepare_instanced_meshes_pipeline(context->wgpu_context);
    setup_bind_groups(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Set target frame buffer */
  render_passes.scene_render.color_attachments[0].view
    = offscreen_framebuffer.color.texture_view;

  /* Render instanced cubes scene to default swapchainTexture */
  {
    render_passes.scene_render.color_attachments[0].clearValue = (WGPUColor){
      .r = 0.1f,
      .g = 0.1f,
      .b = 0.1f,
      .a = 1.0f,
    };

    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_passes.scene_render.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.scene_meshes);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.persp_camera, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                      bind_groups.light_position, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 2,
                                      bind_groups.base_colors[0], 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         vertex_buffers.cube.vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         vertex_buffers.cube.normals.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 2,
      vertex_buffers.instanced_cube.model_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 3,
      vertex_buffers.instanced_cube.normal_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, vertex_buffers.cube.indices.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     geometries.cube.indices.count,
                                     INSTANCES_COUNT, 0, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /**
   * Copy offscreen texture to another texture that will be outputted on the
   * fullscreen quad
   */
  {
    wgpuCommandEncoderCopyTextureToTexture(
      wgpu_context->cmd_enc,
      // source
      &(WGPUImageCopyTexture){
        .texture  = offscreen_framebuffer.color.texture,
        .mipLevel = 0,
      },
      // destination
      &(WGPUImageCopyTexture){
        .texture  = textures.post_fx0.texture,
        .mipLevel = 0,
      },
      // copySize
      &(WGPUExtent3D){
        .width              = wgpu_context->surface.width,
        .height             = wgpu_context->surface.height,
        .depthOrArrayLayers = 1,
      });
  }

  /**
   * Render instanced spheres scene to default swapchainTexture
   */
  {
    render_passes.scene_render.color_attachments[0].clearValue = (WGPUColor){
      .r = 0.225f,
      .g = 0.225f,
      .b = 0.225f,
      .a = 1.0f,
    };

    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_passes.scene_render.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.scene_meshes);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.persp_camera, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                      bind_groups.light_position, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 2,
                                      bind_groups.base_colors[1], 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         vertex_buffers.sphere.vertices.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         vertex_buffers.sphere.normals.buffer,
                                         0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 2,
      vertex_buffers.instanced_sphere.model_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 3,
      vertex_buffers.instanced_sphere.model_matrix.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, vertex_buffers.sphere.indices.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     geometries.sphere.indices.count,
                                     INSTANCES_COUNT, 0, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /**
   * Copy offscreen texture to another texture that will be outputted on the
   * fullscreen quad
   */
  {
    wgpuCommandEncoderCopyTextureToTexture(
      wgpu_context->cmd_enc,
      // source
      &(WGPUImageCopyTexture){
        .texture  = offscreen_framebuffer.color.texture,
        .mipLevel = 0,
      },
      // destination
      &(WGPUImageCopyTexture){
        .texture  = textures.post_fx1.texture,
        .mipLevel = 0,
      },
      // copySize
      &(WGPUExtent3D){
        .width              = wgpu_context->surface.width,
        .height             = wgpu_context->surface.height,
        .depthOrArrayLayers = 1,
      });
  }

  /* Set target frame buffer */
  render_passes.post_fx.color_attachments[0].view
    = wgpu_context->swap_chain.frame_buffer;

  /**
   * Render postfx fullscreen quad to screen
   */
  {
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_passes.post_fx.descriptor);
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc,
                                     pipelines.fullscreen_quad);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      bind_groups.ortho_camera, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 1,
                                      bind_groups.quad_transform, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 2,
                                      bind_groups.quad_sampler, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 3,
                                      bind_groups.quad_tween, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                         vertex_buffers.quad.vertices.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                         vertex_buffers.quad.uvs.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, vertex_buffers.quad.indices.buffer,
      WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     geometries.quad.indices.count, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
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

  /* Submit to queue */
  submit_command_buffers(context);

  /* Submit frame */
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  update_uniform_buffers(context);
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  geometry_destroy(&geometries.quad);
  geometry_destroy(&geometries.cube);
  geometry_destroy(&geometries.sphere);

  geometry_gpu_buffers_destroy(&vertex_buffers.quad);
  geometry_gpu_buffers_destroy(&vertex_buffers.cube);
  geometry_gpu_buffers_destroy(&vertex_buffers.sphere);
  instanced_geometry_gpu_buffers_destroy(&vertex_buffers.instanced_cube);
  instanced_geometry_gpu_buffers_destroy(&vertex_buffers.instanced_sphere);

  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.persp_camera.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.ortho_camera.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.quad_transform.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.quad_tween_factor.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.light_position.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.base_colors[0].buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffers.base_colors[1].buffer)

  WGPU_RELEASE_RESOURCE(Texture, textures.post_fx0.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.post_fx0.view)
  WGPU_RELEASE_RESOURCE(Texture, textures.post_fx1.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.post_fx1.view)
  WGPU_RELEASE_RESOURCE(Sampler, textures.post_fx_sampler)
  wgpu_destroy_texture(&textures.cutoff_mask);

  WGPU_RELEASE_RESOURCE(Texture, offscreen_framebuffer.color.texture)
  WGPU_RELEASE_RESOURCE(TextureView, offscreen_framebuffer.color.texture_view)
  WGPU_RELEASE_RESOURCE(Texture, offscreen_framebuffer.depth_stencil.texture)
  WGPU_RELEASE_RESOURCE(TextureView,
                        offscreen_framebuffer.depth_stencil.texture_view)

  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.persp_camera)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.ortho_camera)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.quad_transform)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.quad_sampler)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.quad_tween)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.light_position)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.base_colors[0])
  WGPU_RELEASE_RESOURCE(BindGroup, bind_groups.base_colors[1])

  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.fullscreen_quad)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipelines.scene_meshes)
}

void example_post_processing(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title   = example_title,
     .overlay = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy
  });
  // clang-format on
}
