#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/texture.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Metaballs
 *
 * WebGPU demo featuring marching cubes and bloom post-processing via compute
 * shaders, physically based shading, deferred rendering, gamma correction and
 * shadow mapping.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Protocol
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/protocol.ts
 * -------------------------------------------------------------------------- */

typedef enum quality_settings_enum {
  QualitySettings_Low    = 0,
  QualitySettings_Medium = 1,
  QualitySettings_High   = 2,
} quality_settings_enum;

typedef struct {
  bool bloom_toggle;
  uint32_t shadow_res;
  uint32_t point_lights_count;
  float output_scale;
  bool update_metaballs;
} quality_option_t;

/* -------------------------------------------------------------------------- *
 * Settings
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/settings.ts
 * -------------------------------------------------------------------------- */

static const quality_option_t QUALITIES[3] = {
  [QualitySettings_Low] = (quality_option_t) {
    .bloom_toggle       = false,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 1.0f,
    .update_metaballs   = false,
  },
  [QualitySettings_Medium] = (quality_option_t) {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 0.8f,
    .update_metaballs   = true,
   },
  [QualitySettings_High] = (quality_option_t) {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 128,
    .output_scale       = 1.0f,
    .update_metaballs   = true,
  },
};

static quality_settings_enum _quality = QualitySettings_Low;

static quality_option_t settings_get_quality_level()
{
  return QUALITIES[_quality];
}

static quality_settings_enum settings_get_quality()
{
  return _quality;
}

static void settings_set_quality(quality_settings_enum v)
{
  _quality = v;
}

/* -------------------------------------------------------------------------- *
 * Orthographic Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/orthographic-camera.ts
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
  glm_lookat(camera->position,         // eye
             camera->look_at_position, // center
             camera->UP_VECTOR,        // up
             camera->view_matrix       // dest
  );
}

static void
orthographic_camera_update_projection_matrix(orthographic_camera_t* camera)
{
  glm_ortho(camera->left,             // left
            camera->right,            // right
            camera->bottom,           // bottom
            camera->top,              // top
            camera->near,             // nearZ
            camera->far,              // farZ
            camera->projection_matrix // dest
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
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/perspective-camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 UP_VECTOR;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 projection_inv_matrix;
  mat4 view_matrix;
  mat4 view_inv_matrix;
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
  glm_lookat(camera->position,         // eye
             camera->look_at_position, // center
             camera->UP_VECTOR,        // up
             camera->view_matrix       // dest
  );
  glm_mat4_inv(camera->view_matrix, camera->view_inv_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* camera)
{
  glm_perspective(camera->field_of_view,    // fovy
                  camera->aspect,           // aspect
                  camera->near,             // nearZ
                  camera->far,              // farZ
                  camera->projection_matrix // dest
  );
  glm_mat4_inv(camera->projection_matrix, camera->projection_inv_matrix);
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
  glm_mat4_zero(camera->projection_inv_matrix);
  glm_mat4_zero(camera->view_matrix);
  glm_mat4_zero(camera->view_inv_matrix);

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
 * Camera Controller
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/camera-controller.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float value;
  float damping;
} damped_action_t;

static void damped_action_add_force(damped_action_t* action, float force)
{
  action->value += force;
}

/**
 * @brief Stops the damping.
 */
static void damped_action_stop(damped_action_t* action)
{
  action->value = 0.0f;
}

/**
 * @brief Updates the damping and calls.
 */
static float damped_action_update(damped_action_t* action)
{
  const bool is_active = action->value * action->value > 0.000001f;
  if (is_active) {
    action->value *= action->damping;
  }
  else {
    damped_action_stop(action);
  }
  return action->value;
}

static void damped_action_init_defaults(damped_action_t* action)
{
  action->value   = 0.0f;
  action->damping = 0.5f;
}

static void damped_action_init(damped_action_t* action)
{
  damped_action_init_defaults(action);
}

/* -------------------------------------------------------------------------- *
 * Cube Geometry
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/geometry/create-box.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    float* data[18 * 6];
    size_t data_size;
    size_t count;
  } positions;
  struct {
    float data[18 * 6];
    size_t data_size;
    size_t count;
  } normals;
  struct {
    float* data[12 * 6];
    size_t data_size;
    size_t count;
  } uvs;
} cube_geometry_t;

static void create_cube(cube_geometry_t* cube, vec3 dimensions)
{
  /* Cube vertex positions */
  vec3 position
    = {-dimensions[0] / 2.0f, -dimensions[1] / 2.0f, -dimensions[2] / 2.0f};
  float x      = position[0];
  float y      = position[1];
  float z      = position[2];
  float width  = dimensions[0];
  float height = dimensions[1];
  float depth  = dimensions[2];

  vec3 fbl = {x, y, z + depth};
  vec3 fbr = {x + width, y, z + depth};
  vec3 ftl = {x, y + height, z + depth};
  vec3 ftr = {x + width, y + height, z + depth};
  vec3 bbl = {x, y, z};
  vec3 bbr = {x + width, y, z};
  vec3 btl = {x, y + height, z};
  vec3 btr = {x + width, y + height, z};

  const float positions[18 * 6] = {
    // clang-format off
    /* front */
    fbl[0],
    fbl[1],
    fbl[2],
    fbr[0],
    fbr[1],
    fbr[2],
    ftl[0],
    ftl[1],
    ftl[2],
    ftl[0],
    ftl[1],
    ftl[2],
    fbr[0],
    fbr[1],
    fbr[2],
    ftr[0],
    ftr[1],
    ftr[2],

    /* right */
    fbr[0],
    fbr[1],
    fbr[2],
    bbr[0],
    bbr[1],
    bbr[2],
    ftr[0],
    ftr[1],
    ftr[2],
    ftr[0],
    ftr[1],
    ftr[2],
    bbr[0],
    bbr[1],
    bbr[2],
    btr[0],
    btr[1],
    btr[2],

    /* back */
    fbr[0],
    bbr[1],
    bbr[2],
    bbl[0],
    bbl[1],
    bbl[2],
    btr[0],
    btr[1],
    btr[2],
    btr[0],
    btr[1],
    btr[2],
    bbl[0],
    bbl[1],
    bbl[2],
    btl[0],
    btl[1],
    btl[2],

    /* left */
    bbl[0],
    bbl[1],
    bbl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    btl[0],
    btl[1],
    btl[2],
    btl[0],
    btl[1],
    btl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    ftl[0],
    ftl[1],
    ftl[2],

    /* top */
    ftl[0],
    ftl[1],
    ftl[2],
    ftr[0],
    ftr[1],
    ftr[2],
    btl[0],
    btl[1],
    btl[2],
    btl[0],
    btl[1],
    btl[2],
    ftr[0],
    ftr[1],
    ftr[2],
    btr[0],
    btr[1],
    btr[2],

    /* bottom */
    bbl[0],
    bbl[1],
    bbl[2],
    bbr[0],
    bbr[1],
    bbr[2],
    fbl[0],
    fbl[1],
    fbl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    bbr[0],
    bbr[1],
    bbr[2],
    fbr[0],
    fbr[1],
    fbr[2],
    // clang-format on
  };
  memcpy(cube->positions.data, positions, sizeof(positions));
  cube->positions.data_size = sizeof(positions);
  cube->positions.count     = (size_t)ARRAY_SIZE(positions);

  /* Cube vertex normals */
  static const float normals[18 * 6] = {
    0,  0,  1,  0,  0,  1,  0,  0,  1,
    0,  0,  1,  0,  0,  1,  0,  0,  1, /* front */

    1,  0,  0,  1,  0,  0,  1,  0,  0,
    1,  0,  0,  1,  0,  0,  1,  0,  0, /* right */

    0,  0,  -1, 0,  0,  -1, 0,  0,  -1,
    0,  0,  -1, 0,  0,  -1, 0,  0,  -1, /* back */

    -1, 0,  0,  -1, 0,  0,  -1, 0,  0,
    -1, 0,  0,  -1, 0,  0,  -1, 0,  0, /* left */

    0,  1,  0,  0,  1,  0,  0,  1,  0,
    0,  1,  0,  0,  1,  0,  0,  1,  0, /* top */

    0,  -1, 0,  0,  -1, 0,  0,  -1, 0,
    0,  -1, 0,  0,  -1, 0,  0,  -1, 0, /* bottom */
  };
  memcpy(cube->normals.data, normals, sizeof(normals));
  cube->normals.data_size = sizeof(normals);
  cube->normals.count     = (size_t)ARRAY_SIZE(normals);

  /* Cube vertex uvs */
  static const float uvs[12 * 6] = {
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* front */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* right */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* back */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* left */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* top */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* bottom */
  };
  memcpy(cube->uvs.data, uvs, sizeof(uvs));
  cube->uvs.data_size = sizeof(uvs);
  cube->uvs.count     = (size_t)ARRAY_SIZE(uvs);
}

/* -------------------------------------------------------------------------- *
 * Marching Cubes
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/geometry/marching-cubes.ts
 * -------------------------------------------------------------------------- */

static const uint16_t MARCHING_CUBES_EDGE_TABLE[] = {
  0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
  0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f,
  0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
  0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
  0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5,
  0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
  0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
  0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
  0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
  0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc,
  0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
  0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
  0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055,
  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
  0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
  0x066, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
  0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
  0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
  0x13a, 0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
  0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09,
  0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
  0x203, 0x109, 0x000,
};

// Each set of 16 values corresponds to one of the entries in the
// MarchingCubesEdgeTable above. The first value is the number of valid indices
// for this entry. The following 15 values are the triangle indices for this
// entry, with -1 representing "no index".
static const int8_t MARCHING_CUBES_TRI_TABLE[] = {
  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  8,
  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  1,  9,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  8,  3,  9,  8,  1,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 6,  9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,
  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, 3,  3,  11, 2,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  11, 2,  8,  11, 0,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  9,  0,  2,  3,  11, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1,
  -1, -1, 6,  3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, 9,  3,  9,  0,
  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, 6,  9,  8,  10, 10, 8,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  4,  7,  8,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 6,  4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 6,  0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, 6,  1,  2,
  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  3,  4,  7,  3,  0,
  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, 9,  9,  2,  10, 9,  0,  2,  8,  4,
  7,  -1, -1, -1, -1, -1, -1, 12, 2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,
  4,  -1, -1, -1, 6,  8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 9,  11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, 9,  9,
  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, 12, 4,  7,  11, 9,
  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, 9,  3,  10, 1,  3,  11, 10, 7,
  8,  4,  -1, -1, -1, -1, -1, -1, 12, 1,  11, 10, 1,  4,  11, 1,  0,  4,  7,
  11, 4,  -1, -1, -1, 12, 4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1,
  -1, -1, 9,  4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, 3,
  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  5,  4,
  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  5,  4,  1,  5,  0,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  5,  4,  8,  3,  5,  3,  1,  5,
  -1, -1, -1, -1, -1, -1, 6,  1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1,
  9,  5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, 12, 2,  10,
  5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, 6,  9,  5,  4,  2,  3,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  11, 2,  0,  8,  11, 4,  9,
  5,  -1, -1, -1, -1, -1, -1, 9,  0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1,
  -1, -1, -1, -1, 12, 2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1,
  -1, 9,  10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, 12, 4,
  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, 12, 5,  4,  0,  5,
  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, 9,  5,  4,  8,  5,  8,  10, 10,
  8,  11, -1, -1, -1, -1, -1, -1, 6,  9,  7,  8,  5,  7,  9,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1,
  -1, -1, 9,  0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, 6,
  1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  9,  7,  8,
  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, 12, 10, 1,  2,  9,  5,  0,
  5,  3,  0,  5,  7,  3,  -1, -1, -1, 12, 8,  0,  2,  8,  2,  5,  8,  5,  7,
  10, 5,  2,  -1, -1, -1, 9,  2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1,
  -1, -1, -1, 9,  7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1,
  12, 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, 12, 2,  3,
  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, 9,  11, 2,  1,  11, 1,
  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, 12, 9,  5,  8,  8,  5,  7,  10, 1,
  3,  10, 3,  11, -1, -1, -1, 15, 5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,
  10, 11, 10, 0,  15, 11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,
  0,  6,  11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10,
  6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  8,  3,  5,
  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  0,  1,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  8,  3,  1,  9,  8,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, 6,  1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, 9,
  9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, 12, 5,  9,  8,
  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, 6,  2,  3,  11, 10, 6,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 0,  8,  11, 2,  0,  10, 6,  5,
  -1, -1, -1, -1, -1, -1, 9,  0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1,
  -1, -1, -1, 12, 5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1,
  9,  6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,
  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, 12, 3,  11, 6,  0,  3,
  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, 9,  6,  5,  9,  6,  9,  11, 11, 9,
  8,  -1, -1, -1, -1, -1, -1, 6,  5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 9,  4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1,
  -1, 9,  1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, 12, 10,
  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, 9,  6,  1,  2,  6,
  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  5,  5,  2,  6,  3,
  0,  4,  3,  4,  7,  -1, -1, -1, 12, 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,
  2,  6,  -1, -1, -1, 15, 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,
  6,  9,  9,  3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, 12,
  5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, 12, 0,  1,  9,
  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, 15, 9,  2,  1,  9,  11, 2,
  9,  4,  11, 7,  11, 4,  5,  10, 6,  12, 8,  4,  7,  3,  11, 5,  3,  5,  1,
  5,  11, 6,  -1, -1, -1, 15, 5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,
  0,  4,  11, 15, 0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,
  12, 6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, 6,  10, 4,
  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  10, 6,  4,  9,
  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, 9,  10, 0,  1,  10, 6,  0,  6,  4,
  0,  -1, -1, -1, -1, -1, -1, 12, 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,
  10, -1, -1, -1, 9,  1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1,
  -1, 12, 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, 6,  0,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  3,  2,  8,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, 9,  10, 4,  9,  10, 6,  4,  11,
  2,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,  2,  2,  8,  11, 4,  9,  10, 4,
  10, 6,  -1, -1, -1, 12, 3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1,
  -1, -1, 15, 6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  12,
  9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, 15, 8,  11, 1,
  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  9,  3,  11, 6,  3,  6,  0,
  0,  6,  4,  -1, -1, -1, -1, -1, -1, 6,  6,  4,  8,  11, 6,  8,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1,
  -1, -1, -1, 12, 0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1,
  12, 10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, 9,  10, 6,
  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  6,  1,  6,
  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,  6,  9,  2,  9,  1,  6,  7,
  9,  0,  9,  3,  7,  3,  9,  9,  7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1,
  -1, -1, -1, -1, 6,  7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 12, 2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,
  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  15, 1,  8,  0,  1,
  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, 12, 11, 2,  1,  11, 1,  7,  10,
  6,  1,  6,  7,  1,  -1, -1, -1, 15, 8,  9,  6,  8,  6,  7,  9,  1,  6,  11,
  6,  3,  1,  3,  6,  6,  0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 12, 7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, 3,
  7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  7,  6,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  3,  0,  8,  11, 7,  6,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  1,  9,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, 6,  10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 9,  2,  9,
  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 12, 6,  11, 7,  2,  10,
  3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, 6,  7,  2,  3,  6,  2,  7,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 9,  7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1,
  -1, -1, -1, -1, 9,  2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1,
  -1, 12, 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, 9,  10,
  7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, 12, 10, 7,  6,  1,
  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, 12, 0,  3,  7,  0,  7,  10, 0,
  10, 9,  6,  10, 7,  -1, -1, -1, 9,  7,  6,  10, 7,  10, 8,  8,  10, 9,  -1,
  -1, -1, -1, -1, -1, 6,  6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, 9,
  8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, 12, 9,  4,  6,
  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, 9,  6,  8,  4,  6,  11, 8,
  2,  10, 1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 3,  0,  11, 0,  6,  11,
  0,  4,  6,  -1, -1, -1, 12, 4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,
  -1, -1, -1, 15, 10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,
  9,  8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, 6,  0,  4,
  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 1,  9,  0,  2,  3,
  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, 9,  1,  9,  4,  1,  4,  2,  2,  4,
  6,  -1, -1, -1, -1, -1, -1, 12, 8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10,
  1,  -1, -1, -1, 9,  10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1,
  -1, 15, 4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  6,  10,
  9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  4,  9,  5,  7,
  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  3,  4,  9,  5,  11,
  7,  6,  -1, -1, -1, -1, -1, -1, 9,  5,  0,  1,  5,  4,  0,  7,  6,  11, -1,
  -1, -1, -1, -1, -1, 12, 11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1,
  -1, -1, 9,  9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, 12,
  6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, 12, 7,  6,  11,
  5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, 15, 3,  4,  8,  3,  5,  4,
  3,  2,  5,  10, 5,  2,  11, 7,  6,  9,  7,  2,  3,  7,  6,  2,  5,  4,  9,
  -1, -1, -1, -1, -1, -1, 12, 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,
  -1, -1, -1, 12, 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1,
  15, 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  12, 9,  5,
  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, 15, 1,  6,  10, 1,  7,
  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  15, 4,  0,  10, 4,  10, 5,  0,  3,
  10, 6,  10, 7,  3,  7,  10, 12, 7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,
  10, -1, -1, -1, 9,  6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1,
  -1, 12, 3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, 12, 0,
  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, 9,  6,  11, 3,  6,
  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 9,  5,  11, 9,
  11, 8,  11, 5,  6,  -1, -1, -1, 15, 0,  11, 3,  0,  6,  11, 0,  9,  6,  5,
  6,  9,  1,  2,  10, 15, 11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,
  2,  5,  12, 6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, 12,
  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, 9,  9,  5,  6,
  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, 15, 1,  5,  8,  1,  8,  0,
  5,  6,  8,  3,  8,  2,  6,  2,  8,  6,  1,  5,  6,  2,  1,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 15, 1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,
  8,  9,  6,  12, 10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1,
  6,  0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10, 5,
  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  11, 5,  10, 7,  5,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 5,  10, 11, 7,  5,  8,  3,
  0,  -1, -1, -1, -1, -1, -1, 9,  5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1,
  -1, -1, -1, -1, 12, 10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1,
  -1, 9,  11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, 12, 0,
  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, 12, 9,  7,  5,  9,
  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, 15, 7,  5,  2,  7,  2,  11, 5,
  9,  2,  3,  2,  8,  9,  8,  2,  9,  2,  5,  10, 2,  3,  5,  3,  7,  5,  -1,
  -1, -1, -1, -1, -1, 12, 8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1,
  -1, -1, 12, 9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, 15,
  9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  6,  1,  3,  5,
  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  7,  0,  7,  1,
  1,  7,  5,  -1, -1, -1, -1, -1, -1, 9,  9,  0,  3,  9,  3,  5,  5,  3,  7,
  -1, -1, -1, -1, -1, -1, 6,  9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1,
  12, 5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, 12, 0,  1,
  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, 15, 10, 11, 4,  10, 4,
  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  12, 2,  5,  1,  2,  8,  5,  2,  11,
  8,  4,  5,  8,  -1, -1, -1, 15, 0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11,
  1,  5,  1,  11, 15, 0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,
  5,  6,  9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,
  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, 9,  5,  10, 2,  5,
  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, 15, 3,  10, 2,  3,  5,  10, 3,
  8,  5,  4,  5,  8,  0,  1,  9,  12, 5,  10, 2,  5,  2,  4,  1,  9,  2,  9,
  4,  2,  -1, -1, -1, 9,  8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1,
  -1, -1, 6,  0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,
  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, 3,  9,  4,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  11, 7,  4,  9,  11,
  9,  10, 11, -1, -1, -1, -1, -1, -1, 12, 0,  8,  3,  4,  9,  7,  9,  11, 7,
  9,  10, 11, -1, -1, -1, 12, 1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11,
  -1, -1, -1, 15, 3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,
  12, 4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, 15, 9,  7,
  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  9,  11, 7,  4,  11, 4,
  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, 12, 11, 7,  4,  11, 4,  2,  8,  3,
  4,  3,  2,  4,  -1, -1, -1, 12, 2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,
  9,  -1, -1, -1, 15, 9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,
  7,  15, 3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, 6,  1,
  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  9,  1,  4,
  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 4,  9,  1,  4,  1,  7,  0,
  8,  1,  8,  7,  1,  -1, -1, -1, 6,  4,  0,  3,  7,  4,  3,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 3,  4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 6,  9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, 9,  0,  1,  10,
  0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, 6,  3,  1,  10, 11, 3,  10,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  2,  11, 1,  11, 9,  9,  11, 8,
  -1, -1, -1, -1, -1, -1, 12, 3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,
  -1, -1, -1, 6,  0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  3,  3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,  3,
  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, 6,  9,  10, 2,  0,  9,
  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,  3,  8,  2,  8,  10, 0,  1,
  8,  1,  10, 8,  -1, -1, -1, 3,  1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 3,  0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,
  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

/* -------------------------------------------------------------------------- *
 * Clamp
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/helpers/clamp.ts
 * -------------------------------------------------------------------------- */

static float clamp(float num, float min, float max)
{
  return MIN(MAX(num, min), max);
}

/* -------------------------------------------------------------------------- *
 * Deg2Rad
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/helpers/deg-to-rad.ts
 * -------------------------------------------------------------------------- */

static float deg_to_rad(float deg)
{
  return (deg * PI) / 180.0f;
}
