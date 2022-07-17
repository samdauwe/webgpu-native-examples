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
 * Constants
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/constants.ts
 * -------------------------------------------------------------------------- */

#define MAX_METABALLS 256u
#define MAX_POINT_LIGHTS_COUNT 256u

static const WGPUTextureFormat DEPTH_FORMAT = WGPUTextureFormat_Depth24Plus;

static const uint32_t SHADOW_MAP_SIZE = 128;

static const uint32_t METABALLS_COMPUTE_WORKGROUP_SIZE[3] = {4, 4, 4};

static const vec4 BACKGROUND_COLOR = {0.1f, 0.1f, 0.1f, 1.0f};

/* -------------------------------------------------------------------------- *
 * Protocol
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/protocol.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float x_min;
  float y_min;
  float z_min;
  float x_step;
  float y_step;
  float z_step;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  float iso_level;
} ivolume_settings_t;

typedef struct {
  float x;
  float y;
  float z;
  float vx;
  float vy;
  float vz;
  float speed;
} imetaball_pos_t;

typedef struct {
  const char* fragment_shader;
  struct {
    WGPUBindGroupLayout* items;
    uint32_t item_count;
  } bind_group_layouts;
  struct {
    WGPUBindGroup* items;
    uint32_t item_count;
  } bind_groups;
  WGPUTextureFormat presentation_format;
  const char* label;
} iscreen_effect_t;

typedef struct {
  vec3 position;
  vec3 direction;
  vec3 color;
  float cut_off;
  float outer_cut_off;
  float intensity;
} ispot_light_t;

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

static void orthographic_camera_set_position(orthographic_camera_t* this,
                                             vec3 target)
{
  glm_vec3_copy(target, this->position);
}

static void orthographic_camera_update_view_matrix(orthographic_camera_t* this)
{
  glm_lookat(this->position,         // eye
             this->look_at_position, // center
             this->UP_VECTOR,        // up
             this->view_matrix       // dest
  );
}

static void
orthographic_camera_update_projection_matrix(orthographic_camera_t* this)
{
  glm_ortho(this->left,             // left
            this->right,            // right
            this->bottom,           // bottom
            this->top,              // top
            this->near,             // nearZ
            this->far,              // farZ
            this->projection_matrix // dest
  );
}

static void orthographic_camera_look_at(orthographic_camera_t* this,
                                        vec3 target)
{
  glm_vec3_copy(target, this->look_at_position);
  orthographic_camera_update_view_matrix(this);
}

static void orthographic_camera_init_defaults(orthographic_camera_t* this)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->UP_VECTOR);

  this->left   = -1.0f;
  this->right  = 1.0f;
  this->top    = 1.0f;
  this->bottom = -1.0f;
  this->near   = 0.1f;
  this->far    = 2000.0f;
  this->zoom   = 1.0f;

  glm_vec3_zero(this->position);
  glm_vec3_zero(this->look_at_position);
  glm_mat4_zero(this->projection_matrix);
  glm_mat4_zero(this->view_matrix);
}

static void orthographic_camera_init(orthographic_camera_t* this, float left,
                                     float right, float top, float bottom,
                                     float near, float far)
{
  orthographic_camera_init_defaults(this);

  this->left   = left;
  this->right  = right;
  this->top    = top;
  this->bottom = bottom;

  this->near = near;
  this->far  = far;

  orthographic_camera_update_projection_matrix(this);
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

static void perspective_camera_set_position(perspective_camera_t* this,
                                            vec3 target)
{
  glm_vec3_copy(target, this->position);
}

static void perspective_camera_update_view_matrix(perspective_camera_t* this)
{
  glm_lookat(this->position,         // eye
             this->look_at_position, // center
             this->UP_VECTOR,        // up
             this->view_matrix       // dest
  );
  glm_mat4_inv(this->view_matrix, this->view_inv_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* this)
{
  glm_perspective(this->field_of_view,    // fovy
                  this->aspect,           // aspect
                  this->near,             // nearZ
                  this->far,              // farZ
                  this->projection_matrix // dest
  );
  glm_mat4_inv(this->projection_matrix, this->projection_inv_matrix);
}

static void perspective_camera_look_at(perspective_camera_t* this, vec3 target)
{
  glm_vec3_copy(target, this->look_at_position);
  perspective_camera_update_view_matrix(this);
}

static void perspective_camera_init_defaults(perspective_camera_t* this)
{
  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->UP_VECTOR);
  glm_vec3_zero(this->position);
  glm_vec3_zero(this->look_at_position);
  glm_mat4_zero(this->projection_matrix);
  glm_mat4_zero(this->projection_inv_matrix);
  glm_mat4_zero(this->view_matrix);
  glm_mat4_zero(this->view_inv_matrix);

  this->zoom = 1.0f;
}

static void perspective_camera_init(perspective_camera_t* this,
                                    float field_of_view, float aspect,
                                    float near, float far)
{
  perspective_camera_init_defaults(this);

  this->field_of_view = field_of_view;
  this->aspect        = aspect;
  this->near          = near;
  this->far           = far;

  perspective_camera_update_projection_matrix(this);
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

static void damped_action_add_force(damped_action_t* this, float force)
{
  this->value += force;
}

/**
 * @brief Stops the damping.
 */
static void damped_action_stop(damped_action_t* this)
{
  this->value = 0.0f;
}

/**
 * @brief Updates the damping and calls.
 */
static float damped_action_update(damped_action_t* this)
{
  const bool is_active = this->value * this->value > 0.000001f;
  if (is_active) {
    this->value *= this->damping;
  }
  else {
    damped_action_stop(this);
  }
  return this->value;
}

static void damped_action_init_defaults(damped_action_t* this)
{
  this->value   = 0.0f;
  this->damping = 0.5f;
}

static void damped_action_init(damped_action_t* this)
{
  damped_action_init_defaults(this);
}

/* -------------------------------------------------------------------------- *
 * WebGPU Renderer
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/webgpu-renderer.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUExtent3D output_size;
  struct {
    WGPUBindGroupLayout frame;
  } bind_groups_layouts;
  struct {
    WGPUBindGroup frame;
  } bind_groups;
  struct {
    struct {
      WGPUTexture texture;
      WGPUTextureView view;
    } depth_texture;
  } textures;
} webgpu_renderer_t;

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
 * Metaballs Compute
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/compute/metaballs.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;

  ivolume_settings_t volume;
  imetaball_pos_t ball_positions[MAX_METABALLS];

  wgpu_buffer_t tables_buffer;
  wgpu_buffer_t metaball_buffer;
  wgpu_buffer_t volume_buffer;
  wgpu_buffer_t indirect_render_buffer;

  WGPUComputePipeline compute_metaballs_pipeline;
  WGPUComputePipeline compute_marching_cubes_pipeline;

  WGPUBindGroup compute_metaballs_bind_group;
  WGPUBindGroup compute_marching_cubes_bind_group;

  uint32_t indirect_render_array[9];
  uint8_t* metaball_array;
  uint32_t* metaball_array_header;
  float* metaball_array_balls;

  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t normal_buffer;
  wgpu_buffer_t index_buffer;

  uint32_t index_count;

  float strength;
  float strength_target;
  float subtract;
  float subtract_target;

  bool has_calced_once;
} metaballs_compute_t;

static bool metaballs_compute_is_ready(metaballs_compute_t* this)
{
  return (this->compute_metaballs_pipeline != NULL)
         && (this->compute_marching_cubes_pipeline != NULL);
}

static void metaballs_compute_init(metaballs_compute_t* this)
{
  {
    /* Compute shader */
    wgpu_shader_t comp_shader = wgpu_shader_create(
      this->renderer->wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .label = "metaballs isosurface compute shader",
        .file  = "shaders/compute_metaballs/metaball_field_compute_source.wgsl",
        .entry = "main",
      });

    /* Create pipeline */
    this->compute_metaballs_pipeline = wgpuDeviceCreateComputePipeline(
      this->renderer->wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = comp_shader.programmable_stage_descriptor,
      });

    /* Partial clean-up */
    wgpu_shader_release(&comp_shader);
  }

  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->metaball_buffer.buffer,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->volume_buffer.buffer,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .layout = wgpuComputePipelineGetBindGroupLayout(
        this->compute_marching_cubes_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    this->compute_metaballs_bind_group = wgpuDeviceCreateBindGroup(
      this->renderer->wgpu_context->device, &bg_desc);
    ASSERT(this->compute_metaballs_bind_group != NULL);
  }

  {
    /* Compute shader */
    wgpu_shader_t comp_shader = wgpu_shader_create(
      this->renderer->wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .label = "marching cubes computer shader",
        .file  = "shaders/compute_metaballs/marching_cubes_compute_source.wgsl",
        .entry = "main",
      });

    /* Create pipeline */
    this->compute_marching_cubes_pipeline = wgpuDeviceCreateComputePipeline(
      this->renderer->wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = comp_shader.programmable_stage_descriptor,
      });

    /* Partial clean-up */
    wgpu_shader_release(&comp_shader);
  }

  {
    WGPUBindGroupEntry bg_entries[6] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->tables_buffer.buffer,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->volume_buffer.buffer,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->vertex_buffer.buffer,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer  = this->normal_buffer.buffer,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .buffer  = this->index_buffer.buffer,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding = 5,
        .buffer  = this->indirect_render_buffer.buffer,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .layout = wgpuComputePipelineGetBindGroupLayout(
        this->compute_marching_cubes_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    this->compute_marching_cubes_bind_group = wgpuDeviceCreateBindGroup(
      this->renderer->wgpu_context->device, &bg_desc);
    ASSERT(this->compute_marching_cubes_bind_group != NULL);
  }
}

static void metaballs_compute_create(metaballs_compute_t* this,
                                     webgpu_renderer_t* renderer,
                                     ivolume_settings_t volume)
{
  memcpy(&this->volume, &volume, sizeof(ivolume_settings_t));
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  {
    size_t table_size
      = ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE)
        + ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE) * sizeof(int32_t);
    int32_t* tables_array = (int32_t*)malloc(table_size);

    size_t j = 0;
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE); ++i) {
      tables_array[j++] = MARCHING_CUBES_EDGE_TABLE[i];
    }
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE); ++i) {
      tables_array[j++] = MARCHING_CUBES_EDGE_TABLE[i];
    }

    this->tables_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "metaballs table buffer",
                                           .usage = WGPUBufferUsage_Storage,
                                           .size  = table_size,
                                           .initial.data = tables_array,
                                         });

    free(tables_array);
  }

  const uint32_t marching_cube_cells
    = (volume.width - 1) * (volume.height - 1) * (volume.depth - 1);
  const size_t vertex_buffer_size
    = sizeof(float) * 3 * 12 * marching_cube_cells;
  const size_t index_buffer_size = sizeof(uint32_t) * 15 * marching_cube_cells;

  this->vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs vertex buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  this->normal_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs normal buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  this->index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs index buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
                    .size  = index_buffer_size,
                  });
  this->index_count = index_buffer_size / sizeof(uint32_t);

  this->indirect_render_array[0] = 500;
  this->indirect_render_buffer   = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "metaballs indirect draw buffer",
                      .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect
                             | WGPUBufferUsage_CopyDst,
                      .size = sizeof(this->indirect_render_array),
                  });

  for (uint32_t i = 0; i < MAX_METABALLS; ++i) {
    this->ball_positions[i].x     = (random_float() * 2 - 1) * volume.x_min;
    this->ball_positions[i].y     = (random_float() * 2 - 1) * volume.y_min;
    this->ball_positions[i].z     = (random_float() * 2 - 1) * volume.z_min;
    this->ball_positions[i].vx    = random_float() * 1000;
    this->ball_positions[i].vy    = (random_float() * 2 - 1) * 10;
    this->ball_positions[i].vz    = random_float() * 1000;
    this->ball_positions[i].speed = random_float() * 2 + 0.3f;
  }

  metaballs_compute_init(this);
}

static void metaballs_compute_rearrange(metaballs_compute_t* this)
{
  this->subtract_target = 3.0f + random_float() * 3.0f;
  this->strength_target = 3.0f + random_float() * 3.0f;
}

static metaballs_compute_t*
metaballs_compute_update_sim(metaballs_compute_t* this,
                             WGPUComputePassEncoder compute_pass, float time,
                             float time_delta)
{
  UNUSED_VAR(time);

  if (!metaballs_compute_is_ready(this)) {
    return this;
  }

  this->subtract += (this->subtract_target - this->subtract) * time_delta * 4;
  this->strength += (this->strength_target - this->strength) * time_delta * 4;

  const uint32_t numblobs = MAX_METABALLS;

  this->metaball_array_header[0] = MAX_METABALLS;

  for (uint32_t i = 0; i < MAX_METABALLS; i++) {
    imetaball_pos_t* pos = &this->ball_positions[i];

    pos->vx += -pos->x * pos->speed * 20.0f;
    pos->vy += -pos->y * pos->speed * 20.0f;
    pos->vz += -pos->z * pos->speed * 20.0f;

    pos->x += pos->vx * pos->speed * time_delta * 0.0001f;
    pos->y += pos->vy * pos->speed * time_delta * 0.0001f;
    pos->z += pos->vz * pos->speed * time_delta * 0.0001f;

    const float padding = 0.9f;
    const float width   = fabs(this->volume.x_min) - padding;
    const float height  = fabs(this->volume.y_min) - padding;
    const float depth   = fabs(this->volume.z_min) - padding;

    if (pos->x > width) {
      pos->x = width;
      pos->vx *= -1.0f;
    }
    else if (pos->x < -width) {
      pos->x = -width;
      pos->vx *= -1.0f;
    }

    if (pos->y > height) {
      pos->y = height;
      pos->vy *= -1.0f;
    }
    else if (pos->y < -height) {
      pos->y = -height;
      pos->vy *= -1.0f;
    }

    if (pos->z > depth) {
      pos->z = depth;
      pos->vz *= -1.0f;
    }
    else if (pos->z < -depth) {
      pos->z = -depth;
      pos->vz *= -1.0f;
    }
  }

  for (uint32_t i = 0; i < numblobs; i++) {
    imetaball_pos_t* position              = &this->ball_positions[i];
    const uint32_t offset                  = i * 8;
    this->metaball_array_balls[offset]     = position->x;
    this->metaball_array_balls[offset + 1] = position->y;
    this->metaball_array_balls[offset + 2] = position->z;
    this->metaball_array_balls[offset + 3]
      = sqrt(this->strength / this->subtract);
    this->metaball_array_balls[offset + 4] = this->strength;
    this->metaball_array_balls[offset + 5] = this->subtract;
  }

  const uint32_t dispatch_size[3] = {
    this->volume.width / METABALLS_COMPUTE_WORKGROUP_SIZE[0],  //
    this->volume.height / METABALLS_COMPUTE_WORKGROUP_SIZE[1], //
    this->volume.depth / METABALLS_COMPUTE_WORKGROUP_SIZE[2],  //
  };

  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->metaball_buffer.buffer, 0,
                          &this->metaball_array, sizeof(this->metaball_array));
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->indirect_render_buffer.buffer, 0,
    &this->indirect_render_array, sizeof(this->indirect_render_array));

  /* Update metaballs */
  if (this->compute_metaballs_pipeline) {
    wgpuComputePassEncoderSetPipeline(compute_pass,
                                      this->compute_metaballs_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 0, this->compute_metaballs_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass, dispatch_size[0], dispatch_size[1], dispatch_size[2]);
  }

  /* Update marching cubes */
  if (this->compute_marching_cubes_pipeline) {
    wgpuComputePassEncoderSetPipeline(compute_pass,
                                      this->compute_marching_cubes_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 0, this->compute_marching_cubes_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass, dispatch_size[0], dispatch_size[1], dispatch_size[2]);
  }

  this->has_calced_once = true;

  return this;
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

static void create_cube(cube_geometry_t* this, vec3 dimensions)
{
  memset(this, 0, sizeof(*this));

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
  memcpy(this->positions.data, positions, sizeof(positions));
  this->positions.data_size = sizeof(positions);
  this->positions.count     = (size_t)ARRAY_SIZE(positions);

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
  memcpy(this->normals.data, normals, sizeof(normals));
  this->normals.data_size = sizeof(normals);
  this->normals.count     = (size_t)ARRAY_SIZE(normals);

  /* Cube vertex uvs */
  static const float uvs[12 * 6] = {
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* front */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* right */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* back */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* left */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* top */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* bottom */
  };
  memcpy(this->uvs.data, uvs, sizeof(uvs));
  this->uvs.data_size = sizeof(uvs);
  this->uvs.count     = (size_t)ARRAY_SIZE(uvs);
}

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

/* -------------------------------------------------------------------------- *
 * Point Lights
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/lighting/point-lights.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUBindGroupLayout lights_buffer_compute_bind_group_layout;
  WGPUBindGroup lights_buffer_compute_bind_group;
  WGPUPipelineLayout update_compute_pipeline_layout;
  WGPUComputePipeline update_compute_pipeline;
  wgpu_buffer_t lights_buffer;
  wgpu_buffer_t lights_config_uniform_buffer;
} point_lights_t;

static bool point_lights_is_ready(point_lights_t* this)
{
  return this->update_compute_pipeline != NULL;
}

static void point_lights_set_lights_count(point_lights_t* this, uint32_t v)
{
  const uint32_t lights_count_array[1] = {v};

  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->lights_config_uniform_buffer.buffer, 0,
                          lights_count_array, sizeof(lights_count_array));
}

static void point_lights_init(point_lights_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->lights_buffer_compute_bind_group_layout, // Group 0
      this->renderer->bind_groups_layouts.frame,     // Group 1
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "point light update compute pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->update_compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->update_compute_pipeline_layout != NULL);
  }

  /* Compute pipeline */
  {
    wgpu_shader_t comp_shader = wgpu_shader_create(
      wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .file = "shaders/compute_metaballs/UpdatePointLightsComputeShader.wgsl",
        .entry = "main",
      });
    this->update_compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "point light update compute pipeline",
        .layout  = this->update_compute_pipeline_layout,
        .compute = comp_shader.programmable_stage_descriptor,
      });
    wgpu_shader_release(&comp_shader);
  }
}

static void point_lights_create(point_lights_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Lights uniform buffer */
  {
    const uint32_t lights_data_stride = 16;
    float lights_data[lights_data_stride * MAX_POINT_LIGHTS_COUNT];
    float x, y, z, vel_x, vel_y, vel_z, r, g, b, radius, intensity;
    uint32_t offset;
    for (uint32_t i = 0; i < MAX_POINT_LIGHTS_COUNT; ++i) {
      offset = lights_data_stride * i;

      x = (random_float() * 2 - 1) * 20;
      y = -2.0f;
      z = (random_float() * 2 - 1) * 20;

      vel_x = random_float() * 4 - 2;
      vel_y = random_float() * 4 - 2;
      vel_z = random_float() * 4 - 2;

      r = random_float();
      g = random_float();
      b = random_float();

      radius    = 5 + random_float() * 3;
      intensity = 10 + random_float() * 10;

      // position
      lights_data[offset++] = x;
      lights_data[offset++] = y;
      lights_data[offset++] = z;
      // velocity
      lights_data[offset++] = vel_x;
      lights_data[offset++] = vel_y;
      lights_data[offset++] = vel_z;
      // color
      lights_data[offset++] = r;
      lights_data[offset++] = g;
      lights_data[offset++] = b;
      // radius
      lights_data[offset++] = radius;
      // intensity
      lights_data[offset++] = intensity;
    }
    this->lights_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_Storage,
                                           .size  = sizeof(lights_data),
                                           .initial.data = lights_data,
                                         });
  }

  /* Lights config uniform buffer */
  {
    const uint32_t lights_config_arr[1]
      = {settings_get_quality_level().point_lights_count};
    this->lights_config_uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = sizeof(lights_config_arr),
                                           .initial.data = lights_config_arr,
                                         });
  }

  /* Lights buffer compute bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
           .type           = WGPUBufferBindingType_Storage,
           .minBindingSize = this->lights_buffer.size,
         },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
             .type           = WGPUBufferBindingType_Storage,
             .minBindingSize = this->lights_config_uniform_buffer.size,
         },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "lights update compute bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->lights_buffer_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->lights_buffer_compute_bind_group_layout != NULL);
  }

  /* Lights buffer compute bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->lights_buffer.buffer,
        .size    = this->lights_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->lights_config_uniform_buffer.buffer,
        .size    = this->lights_config_uniform_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .layout     = this->lights_buffer_compute_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->lights_buffer_compute_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->lights_buffer_compute_bind_group != NULL);
  }
}

static void point_lights_destroy(point_lights_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        this->lights_buffer_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->lights_buffer_compute_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->update_compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->update_compute_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_config_uniform_buffer.buffer)
}

static point_lights_t*
point_lights_update_sim(point_lights_t* this,
                        WGPUComputePassEncoder compute_pass)
{
  if (!point_lights_is_ready(this)) {
    return this;
  }
  wgpuComputePassEncoderSetPipeline(compute_pass,
                                    this->update_compute_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->lights_buffer_compute_bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->renderer->bind_groups.frame, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(settings_get_quality_level().point_lights_count / 64.0f), 1,
    1);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Spot Light
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/lighting/spot-light.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  perspective_camera_t camera;

  vec3 _position;
  vec3 _direction;
  vec3 _color;
  float _cut_off;
  float _outer_cut_off;
  float _intensity;

  wgpu_buffer_t light_info_ubo;
  wgpu_buffer_t projection_ubo;
  wgpu_buffer_t view_ubo;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;
  struct {
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  struct {
    WGPUBindGroupLayout ubos;
    WGPUBindGroupLayout depth_texture;
  } bind_group_layouts;

  struct {
    WGPUBindGroup ubos;
    WGPUBindGroup depth_texture;
  } bind_groups;
} spot_light_t;

static void spot_light_get_position(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_position, *dest);
}

static void spot_light_set_position(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_position);
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 0, v, sizeof(vec3));

  perspective_camera_set_position(&this->camera,
                                  (vec3){-v[0] * 15, v[1], -v[2] * 15});
  perspective_camera_update_view_matrix(&this->camera);

  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->view_ubo.buffer,
                          0, this->camera.view_matrix, sizeof(mat4));
  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->view_ubo.buffer,
                          16 * sizeof(float), this->camera.view_inv_matrix,
                          sizeof(mat4));
}

static void spot_light_get_direction(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_direction, *dest);
}

static void spot_light_set_direction(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_direction);
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 4 * sizeof(float), v,
                          sizeof(vec3));

  perspective_camera_look_at(&this->camera, (vec3){v[0], v[1], v[2]});

  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->view_ubo.buffer,
                          0, this->camera.view_matrix, sizeof(mat4));
  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->view_ubo.buffer,
                          16 * sizeof(float), this->camera.view_inv_matrix,
                          sizeof(mat4));
}

static void spot_light_get_color(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_color, *dest);
}

static void spot_light_set_color(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_color);
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 8 * sizeof(float), v,
                          sizeof(vec3));
}

static float spot_light_get_cut_off(spot_light_t* this)
{
  return this->_cut_off;
}

static void spot_light_set_cut_off(spot_light_t* this, float v)
{
  this->_cut_off               = v;
  const float cut_off_array[1] = {cosf(v)};
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 11 * sizeof(float),
                          cut_off_array, sizeof(cut_off_array));
}

static float spot_light_get_outer_cut_off(spot_light_t* this)
{
  return this->_outer_cut_off;
}

static void spot_light_set_outer_cut_off(spot_light_t* this, float v)
{
  this->_outer_cut_off               = v;
  const float outer_cut_off_array[1] = {cosf(v)};
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 12 * sizeof(float),
                          outer_cut_off_array, sizeof(outer_cut_off_array));

  this->camera.field_of_view = v * 1.5f;
  perspective_camera_update_projection_matrix(&this->camera);

  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->projection_ubo.buffer, 0,
                          this->camera.projection_matrix, sizeof(mat4));
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->projection_ubo.buffer, 16 * sizeof(float),
                          this->camera.projection_inv_matrix, sizeof(mat4));
}

static float spot_light_get_intensity(spot_light_t* this)
{
  return this->_intensity;
}

static void spot_light_set_intensity(spot_light_t* this, float v)
{
  this->_intensity               = v;
  const float intensity_array[1] = {v};
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->light_info_ubo.buffer, 13 * sizeof(float),
                          intensity_array, sizeof(intensity_array));
}

static void spot_light_create(spot_light_t* this, webgpu_renderer_t* renderer,
                              ispot_light_t* ispot_light)
{
  this->renderer = renderer;
  perspective_camera_init(&this->camera, deg_to_rad(56.0f), 1.0f, 0.1f, 120.0f);
  perspective_camera_update_view_matrix(&this->camera);
  perspective_camera_update_projection_matrix(&this->camera);

  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Depth texture */
  WGPUExtent3D texture_extent = {
    .width              = settings_get_quality_level().shadow_res,
    .height             = settings_get_quality_level().shadow_res,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "spot light depth texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->depth_texture.texture != NULL);

  /* Depth texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "spot light depth texture view",
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

  /* Light info UBO */
  this->light_info_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = (4 * sizeof(float) + // position
                             4 * sizeof(float) + // direction
                             3 * sizeof(float) + // color
                             1 * sizeof(float) + // cutOff
                             1 * sizeof(float) + // outerCutOff
                             1 * sizeof(float)   // intensity,
                             ),
                  });

  /* Projection UBO */
  this->projection_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = (16 * sizeof(float) + // matrix
                             16 * sizeof(float) + // inverse matrix
                             8 * sizeof(float) +  // screen size
                             1 * sizeof(float) +  // near
                             1 * sizeof(float)    // far
                             ),
                  });
  wgpu_queue_write_buffer(wgpu_context, this->projection_ubo.buffer,
                          0 * sizeof(float), this->camera.projection_matrix,
                          sizeof(mat4));
  wgpu_queue_write_buffer(wgpu_context, this->projection_ubo.buffer,
                          16 * sizeof(float),
                          this->camera.projection_inv_matrix, sizeof(mat4));
  const float shadow_res[2] = {
    (float)settings_get_quality_level().shadow_res, //
    (float)settings_get_quality_level().shadow_res  //
  };
  wgpu_queue_write_buffer(wgpu_context, this->projection_ubo.buffer,
                          32 * sizeof(float), shadow_res, sizeof(shadow_res));
  const float camera_near[1] = {this->camera.near};
  wgpu_queue_write_buffer(wgpu_context, this->projection_ubo.buffer,
                          40 * sizeof(float), camera_near, sizeof(camera_near));
  wgpu_queue_write_buffer(wgpu_context, this->projection_ubo.buffer,
                          41 * sizeof(float), camera_near, sizeof(camera_near));

  /* View UBO */
  this->view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = (16 * sizeof(float) + // matrix
                             16 * sizeof(float) + // inverse matrix
                             3 * sizeof(float) +  // camera position
                             1 * sizeof(float) +  // time
                             1 * sizeof(float)    // delta time
                             ),
                  });
  wgpu_queue_write_buffer(wgpu_context, this->view_ubo.buffer,
                          0 * sizeof(float), this->camera.view_matrix,
                          sizeof(mat4));
  wgpu_queue_write_buffer(wgpu_context, this->view_ubo.buffer,
                          16 * sizeof(float), this->camera.view_inv_matrix,
                          sizeof(mat4));
  wgpu_queue_write_buffer(wgpu_context, this->view_ubo.buffer,
                          32 * sizeof(float), this->camera.position,
                          sizeof(vec3));
  const float empty_array_1[1] = {0.0f};
  wgpu_queue_write_buffer(wgpu_context, this->view_ubo.buffer,
                          35 * sizeof(float), empty_array_1,
                          sizeof(empty_array_1));
  wgpu_queue_write_buffer(wgpu_context, this->view_ubo.buffer,
                          36 * sizeof(float), empty_array_1,
                          sizeof(empty_array_1));

  glm_vec3_copy(ispot_light->position, this->_position);
  glm_vec3_copy(ispot_light->direction, this->_direction);
  glm_vec3_copy(ispot_light->color, this->_color);
  this->_cut_off       = ispot_light->cut_off;
  this->_outer_cut_off = ispot_light->outer_cut_off;
  this->_intensity     = ispot_light->intensity;

  /* Render pass descriptor */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view           = this->depth_texture.view,
      .depthLoadOp    = WGPULoadOp_Clear,
      .depthStoreOp   = WGPUStoreOp_Store,
      .clearDepth     = 1.0f,
      .stencilLoadOp  = WGPULoadOp_Clear,
      .stencilStoreOp = WGPUStoreOp_Store,
      .clearStencil   = 0,
    };
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
    .occlusionQuerySet      = NULL,
  };

  /* Bind group layouts */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->light_info_ubo.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->projection_ubo.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->view_ubo.size,
        },
        .sampler = {0},
      }
    };
    this->bind_group_layouts.ubos = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "spot light ubos bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layouts.ubos != NULL);
  }
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Texture view
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout) {
            .sampleType    = WGPUTextureSampleType_Depth,
            .viewDimension = WGPUTextureViewDimension_2D,
            .multisampled  = false,
          },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.depth_texture = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = "spot light depth texture bind group layout",
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layouts.depth_texture != NULL);
  }

  /* Bind groups */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->light_info_ubo.buffer,
        .size    = this->light_info_ubo.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->projection_ubo.buffer,
        .size    = this->projection_ubo.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->view_ubo.buffer,
        .size    = this->view_ubo.size,
      },
    };
    this->bind_groups.ubos = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "spot light ubos bind group",
                              .layout     = this->bind_group_layouts.ubos,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_groups.ubos != NULL);
  }
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->depth_texture.view,
        },
      };
    this->bind_groups.depth_texture = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .layout = this->bind_group_layouts.depth_texture,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_groups.depth_texture != NULL);
  }
}

static void spot_light_destroy(spot_light_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->light_info_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->projection_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->view_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Texture, this->depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->depth_texture.view)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.ubos)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.depth_texture)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.ubos)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.depth_texture)
}

/* -------------------------------------------------------------------------- *
 * Box Outline
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/box-outline.ts
 * -------------------------------------------------------------------------- */

#define BOX_OUTLINE_RADIUS 2.5f
#define BOX_OUTLINE_SIDE_COUNT 13u

typedef struct {
  webgpu_renderer_t* renderer;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
    wgpu_buffer_t instance_buffer;
  } buffers;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
} box_outline_t;

static void box_outline_init(box_outline_t* this)
{
  /* Box outline render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->renderer->bind_groups_layouts.frame, // Group 0
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "box outline render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /*  Box outline render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_LineStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 4 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 8 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 12 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 16 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 4,
        .attributes     = &attributes[1],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "BoxOutlineVertexShader WGSL",
            .file  = "shaders/compute_metaballs/BoxOutlineVertexShader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "BoxOutlineFragmentShader WGSL",
          .file  = "shaders/GroundFragmentShader/BoxOutlineFragmentShader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline
      = wgpuDeviceCreateRenderPipeline(this->renderer->wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label = "box outline render pipeline",
                                         .layout       = this->pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void box_outline_init_defaults(box_outline_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void box_outline_create(box_outline_t* this, webgpu_renderer_t* renderer)
{
  box_outline_init_defaults(this);

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  const float vertices[2 * 3] = {
    -BOX_OUTLINE_RADIUS, 0.0f, 0.0f, //
    BOX_OUTLINE_RADIUS,  0.0f, 0.0f  //
  };

  const uint16_t indices[16] = {
    0,  1,  2,  3,  //
    4,  5,  6,  7,  //
    8,  9,  10, 11, //
    12, 13, 14, 15, //
  };

  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  float instance_matrices[BOX_OUTLINE_SIDE_COUNT * 16] = {0};

  mat4 instance_matrix = GLM_MAT4_IDENTITY_INIT;

  /* Top rig */
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 0});
  memcpy(&instance_matrices[0 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[1 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[2 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[3 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* bottom rig */
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 0});
  memcpy(&instance_matrices[4 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[5 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[6 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[7 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Sides */
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[9 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[10 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[11 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[12 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  this->buffers.instance_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline instance matrices buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_matrices),
                    .initial.data = instance_matrices,
                  });

  /* Init render pipeline */
  box_outline_init(this);
}

static void box_outline_destroy(box_outline_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_buffer.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
}

static void box_outline_render(box_outline_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 2, BOX_OUTLINE_SIDE_COUNT, 0, 0,
                                   0);
}

/* -------------------------------------------------------------------------- *
 * Ground
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/ground.ts
 * -------------------------------------------------------------------------- */

#define GROUND_WORLD_Y -7.5f
#define GROUND_WIDTH 100u
#define GROUND_HEIGHT 100u
#define GROUND_COUNT 100u
#define GROUND_SPACING 0

typedef struct {
  webgpu_renderer_t* renderer;
  spot_light_t* spot_light;

  struct {
    WGPUPipelineLayout render_pipeline;
    WGPUPipelineLayout render_shadow_pipeline;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline render_pipeline;
    WGPURenderPipeline render_shadow_pipeline;
  } render_pipelines;

  WGPUBindGroupLayout model_bind_group_layout;
  WGPUBindGroup model_bind_group;

  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t normal_buffer;
    wgpu_buffer_t instance_offsets_buffer;
    wgpu_buffer_t instance_material_buffer;
    wgpu_buffer_t uniform_buffer;
  } buffers;

  uint32_t instance_count;
  mat4 model_matrix;
} ground_t;

static void ground_init(ground_t* this)
{
  /* Ground render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_groups_layouts.frame, // Group 0
      this->model_bind_group_layout,             // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "ground render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_pipeline = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);
  }

  /* Ground render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 1 * sizeof(float),
        .format         = WGPUVertexFormat_Float32,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[4] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
      [2] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &attributes[2],
      },
      [3] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 2,
        .attributes     = &attributes[3],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "GroundVertexShader WGSL",
            .file  = "shaders/compute_metaballs/GroundVertexShader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "GroundFragmentShader WGSL",
          .file  = "shaders/GroundFragmentShader/GroundVertexShader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = "ground render pipeline",
        .layout       = this->pipeline_layouts.render_pipeline,
        .primitive    = primitive_state,
        .vertex       = vertex_state,
        .fragment     = &fragment_state,
        .depthStencil = &depth_stencil_state,
        .multisample  = multisample_state,
      });
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Ground shadow render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->spot_light->bind_group_layouts.ubos, // Group 0
      this->model_bind_group_layout,             // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "ground shadow rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_shadow_pipeline
      = wgpuDeviceCreatePipelineLayout(this->renderer->wgpu_context->device,
                                       &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);
  }

  /* Ground shadow render pipeline layout */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[2] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "GroundShadowVertexShader WGSL",
            .file  = "shaders/compute_metaballs/GroundShadowVertexShader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_shadow_pipeline
      = wgpuDeviceCreateRenderPipeline(
        this->renderer->wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label        = "ground render pipeline",
          .layout       = this->pipeline_layouts.render_shadow_pipeline,
          .primitive    = primitive_state,
          .vertex       = vertex_state,
          .fragment     = NULL,
          .depthStencil = &depth_stencil_state,
          .multisample  = multisample_state,
        });
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  }
}

static void ground_init_defaults(ground_t* this)
{
  memset(this, 0, sizeof(*this));
  glm_mat4_identity(this->model_matrix);
}

static void ground_create(ground_t* this, webgpu_renderer_t* renderer,
                          spot_light_t* spot_light)
{
  ground_init_defaults(this);

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Create cube */
  cube_geometry_t cube_geometry;
  vec3 cube_dimensions = GLM_VEC3_ONE_INIT;
  create_cube(&cube_geometry, cube_dimensions);

  /* Ground vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = cube_geometry.positions.data_size,
                    .initial.data = cube_geometry.positions.data,
                  });

  /* Ground normal buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground normal buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = cube_geometry.normals.data_size,
                    .initial.data = cube_geometry.normals.data,
                  });

  /* Ground instance buffers */
  float instance_offsets[GROUND_WIDTH * GROUND_HEIGHT * 3];
  float instance_metallic_rougness[GROUND_WIDTH * GROUND_HEIGHT * 2];

  const float spacing_x = GROUND_WIDTH / GROUND_COUNT + GROUND_SPACING;
  const float spacing_y = GROUND_HEIGHT / GROUND_COUNT + GROUND_SPACING;

  float x_pos = 0.0f, y_pos = 0.0f;
  for (uint32_t x = 0, i = 0; x < GROUND_COUNT; x++) {
    for (uint32_t y = 0; y < GROUND_COUNT; y++) {
      x_pos = x * spacing_x;
      y_pos = y * spacing_y;

      // xyz offset
      instance_offsets[i * 3 + 0] = x_pos - GROUND_WIDTH / 2.0f;
      instance_offsets[i * 3 + 1] = y_pos - GROUND_HEIGHT / 2.0f;
      instance_offsets[i * 3 + 2] = random_float() * 3 + 1;

      // metallic
      instance_metallic_rougness[i * 2 + 0] = 1.0f;

      // roughness
      instance_metallic_rougness[i * 2 + 1] = 0.5f;

      ++i;
    }
  }

  this->instance_count = ARRAY_SIZE(instance_offsets) / 3;

  this->buffers.instance_offsets_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground instance xyz buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_offsets),
                    .initial.data = instance_offsets,
                  });

  this->buffers.instance_material_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground instance material buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_metallic_rougness),
                    .initial.data = instance_metallic_rougness,
                  });

  /* Ground uniform buffer */
  glm_translate(this->model_matrix, (vec3){0, GROUND_WORLD_Y, 0});
  this->buffers.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(this->model_matrix),
                    .initial.data = this->model_matrix,
                  });

  /* Ground bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->buffers.uniform_buffer.size,
        },
        .sampler = {0},
        },
      };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "ground bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->model_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->model_bind_group_layout != NULL);
  }

  /* Ground bind group*/
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->buffers.uniform_buffer.buffer,
        .size    = this->buffers.uniform_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "ground bind group",
      .layout     = this->model_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->model_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->model_bind_group != NULL);
  }

  /* Init render pipeline */
  ground_init(this);
}

static void ground_destroy(ground_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layouts.render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        this->pipeline_layouts.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipelines.render_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline,
                        this->render_pipelines.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->model_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->model_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.normal_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_offsets_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_material_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.uniform_buffer.buffer)
}

static ground_t* ground_render_shadow(ground_t* this,
                                      WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_shadow_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(
    render_pass, this->render_pipelines.render_shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->spot_light->bind_groups.ubos, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_offsets_buffer.buffer, 1,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

static ground_t* ground_render(ground_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass,
                                   this->render_pipelines.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.normal_buffer.buffer, 1, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, this->buffers.instance_offsets_buffer.buffer, 1,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, this->buffers.instance_material_buffer.buffer, 1,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

/* --------------------------------------------------------------------------
 * Metaballs
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/meshesmetaballs.ts
 * --------------------------------------------------------------------------
 */

typedef struct {
  webgpu_renderer_t* renderer;
  ivolume_settings_t volume;
  spot_light_t* spot_light;

  metaballs_compute_t metaballs_compute;

  struct {
    WGPUPipelineLayout render_pipeline;
    WGPUPipelineLayout render_shadow_pipeline;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline render_pipeline;
    WGPURenderPipeline render_shadow_pipeline;
  } render_pipelines;

  wgpu_buffer_t ubo;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  vec3 color_rgb;
  vec3 color_target_rgb;
  float roughness;
  float roughness_target;
  float metallic;
  float metallic_target;
} metaballs_t;

static bool metaballs_is_ready(metaballs_t* this)
{
  return (metaballs_compute_is_ready(&this->metaballs_compute) && //
          (this->render_pipelines.render_pipeline != NULL) &&     //
          (this->render_pipelines.render_shadow_pipeline != NULL) //
  );
}

static bool metaballs_has_updated_at_least_once(metaballs_t* this)
{
  return this->metaballs_compute.has_calced_once;
}

static void metaballs_init(metaballs_t* this)
{
  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_groups_layouts.frame, // Group 0
      this->bind_group_layout,                   // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "metaball rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_pipeline = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attribute_1 = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexAttribute attribute_2 = {
      .shaderLocation = 1,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute_1,
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute_2,
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "ParticlesFragmentShader WGSL",
            .file  = "shaders/compute_metaballs/MetaballsVertexShader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "MetaballsFragmentShader WGSL",
          .file  = "shaders/compute_metaballs/MetaballsFragmentShader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = "metaball rendering pipeline",
        .layout       = this->pipeline_layouts.render_pipeline,
        .primitive    = primitive_state,
        .vertex       = vertex_state,
        .fragment     = &fragment_state,
        .depthStencil = &depth_stencil_state,
        .multisample  = multisample_state,
      });
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Metaballs shadow render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->spot_light->bind_group_layouts.ubos, // Group 0
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "metaballs shadow rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_shadow_pipeline
      = wgpuDeviceCreatePipelineLayout(this->renderer->wgpu_context->device,
                                       &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);
  }

  /* Metaballs shadow render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attribute = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexBufferLayout vertex_buffers[1] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute,
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "MetaballsShadowVertexShader WGSL",
            .file  = "shaders/compute_metaballs/MetaballsShadowVertexShader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = "metaballs shadow rendering pipeline",
        .layout       = this->pipeline_layouts.render_pipeline,
        .primitive    = primitive_state,
        .vertex       = vertex_state,
        .fragment     = NULL,
        .depthStencil = &depth_stencil_state,
        .multisample  = multisample_state,
      });
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  }
}

static void metaballs_init_defaults(metaballs_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_one(this->color_rgb);
  glm_vec3_one(this->color_target_rgb);
  this->roughness       = 0.3f;
  this->metallic_target = this->roughness;
  this->metallic        = 0.1;
  this->metallic_target = this->metallic;
}

static void metaballs_create(metaballs_t* this, webgpu_renderer_t* renderer,
                             ivolume_settings_t volume,
                             spot_light_t* spot_light)
{
  metaballs_init_defaults(this);

  this->renderer = renderer;
  memcpy(&this->volume, &volume, sizeof(ivolume_settings_t));
  this->spot_light = spot_light;
  metaballs_compute_create(&this->metaballs_compute, renderer, volume);

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Metaballs ubo */
  {
    const float metaballs_ubo_data[5] = {1.0f, 1.0f, 1.0f, 0.3f, 0.1f};
    this->ubo
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "metaballs ubo",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = sizeof(metaballs_ubo_data),
                                           .initial.data = metaballs_ubo_data,
                                         });
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->ubo.size,
        },
        .sampler = {0},
        },
      };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "metaballs bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group*/
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->ubo.buffer,
        .size    = this->ubo.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "metaballs bind group",
      .layout     = this->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
  }

  /* Init render pipeline */
  metaballs_init(this);
}

static void metaballs_destroy(metaballs_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layouts.render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        this->pipeline_layouts.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipelines.render_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline,
                        this->render_pipelines.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubo.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void metaballs_rearrange(metaballs_t* this)
{
  this->color_target_rgb[0] = random_float();
  this->color_target_rgb[1] = random_float();
  this->color_target_rgb[2] = random_float();

  this->metallic_target  = 0.08f + random_float() * 0.92f;
  this->roughness_target = 0.08f + random_float() * 0.92f;

  metaballs_compute_rearrange(&this->metaballs_compute);
}

static metaballs_t* metaballs_update_sim(metaballs_t* this,
                                         WGPUComputePassEncoder compute_pass,
                                         float time, float time_delta)
{
  const float color_speed = time_delta * 2.0f;
  this->color_rgb[0]
    += (this->color_target_rgb[0] - this->color_rgb[0]) * color_speed;
  this->color_rgb[1]
    += (this->color_target_rgb[1] - this->color_rgb[1]) * color_speed;
  this->color_rgb[2]
    += (this->color_target_rgb[2] - this->color_rgb[2]) * color_speed;

  const float material_speed = time_delta * 3;
  this->metallic += (this->metallic_target - this->metallic) * material_speed;
  this->roughness
    += (this->roughness_target - this->roughness) * material_speed;

  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  wgpu_queue_write_buffer(wgpu_context, this->ubo.buffer, 0 * sizeof(float),
                          this->color_rgb, sizeof(this->color_rgb));
  wgpu_queue_write_buffer(wgpu_context, this->ubo.buffer, 3 * sizeof(float),
                          &this->roughness, sizeof(this->roughness));
  wgpu_queue_write_buffer(wgpu_context, this->ubo.buffer, 4 * sizeof(float),
                          &this->metallic, sizeof(this->metallic));

  metaballs_compute_update_sim(&this->metaballs_compute, compute_pass, time,
                               time_delta);
  return this;
}

static metaballs_t* metaballs_render_shadow(metaballs_t* this,
                                            WGPURenderPassEncoder render_pass)
{
  if (!metaballs_is_ready(this)) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(
    render_pass, this->render_pipelines.render_shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->spot_light->bind_groups.ubos, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->metaballs_compute.vertex_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->metaballs_compute.index_buffer.buffer,
    WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, this->metaballs_compute.index_count, 0, 0, 0, 0);
  return this;
}

static metaballs_t* metaballs_render(metaballs_t* this,
                                     WGPURenderPassEncoder render_pass)
{
  if (!metaballs_is_ready(this)) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass,
                                   this->render_pipelines.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->metaballs_compute.vertex_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->metaballs_compute.normal_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->metaballs_compute.index_buffer.buffer,
    WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, this->metaballs_compute.index_count, 0, 0, 0, 0);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Particles
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/meshes/particles.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
} particles_t;

static void particles_init(particles_t* this)
{
  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_groups_layouts.frame, // Group 0
      this->bind_group_layout,                   // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "particles render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "ParticlesFragmentShader WGSL",
            .file  = "shaders/compute_metaballs/ParticlesVertexShader.wgsl",
            .entry = "main",
           },
          .buffer_count = 0,
          .buffers = NULL,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Fragment shader WGSL
            .label = "ParticlesFragmentShader WGSL",
            .file  = "shaders/compute_metaballs/ParticlesFragmentShader.wgsl",
            .entry = "main",
          },
          .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
          .targets      = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline
      = wgpuDeviceCreateRenderPipeline(this->renderer->wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label  = "particles render pipeline",
                                         .layout = this->pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void particles_create(particles_t* this, webgpu_renderer_t* renderer,
                             wgpu_buffer_t* lights_buffer)
{
  this->renderer               = renderer;
  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Particles bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
             .type           = WGPUBufferBindingType_ReadOnlyStorage,
             .minBindingSize = lights_buffer->size,
         },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "particles bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Particles bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = lights_buffer->buffer,
        .size    = lights_buffer->size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "particles bind group",
      .layout     = this->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
  }

  /* Init render pipeline */
  particles_init(this);
}

static void particles_destroy(particles_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void particles_render(particles_t* this,
                             WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, 6, settings_get_quality_level().point_lights_count, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Effect
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/effect.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  iscreen_effect_t* screen_effect;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
  } buffers;
} effect_t;

static void effect_init(effect_t* this, const char* fragment_shader,
                        WGPUBindGroupLayout* bind_group_layouts,
                        uint32_t bind_group_layout_count, const char* label)
{
  /* Render pipeline layout */
  {
    char pipeline_layout_lbl[STRMAX] = {0};
    snprintf(pipeline_layout_lbl, strlen(label) + 7 + 1, "%s layout", label);
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = pipeline_layout_lbl,
      .bindGroupLayoutCount = bind_group_layout_count,
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(true);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = this->screen_effect->presentation_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attribute = {
      .shaderLocation = 0,
      .offset         = 0 * sizeof(float),
      .format         = WGPUVertexFormat_Float32x2,
    };
    WGPUVertexBufferLayout vertex_buffers[1] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute,
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "EffectVertexShader WGSL",
            .file  = "shaders/compute_metaballs/EffectVertexShader.wgsl",
            .entry = "main",
           },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Fragment shader WGSL
            .label            = "FragmentShader WGSL",
            .wgsl_code.source = fragment_shader,
            .entry            = "main",
          },
          .target_count = 1,
          .targets      = &color_target_state,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline
      = wgpuDeviceCreateRenderPipeline(this->renderer->wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label        = label,
                                         .layout       = this->pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void effect_create(effect_t* this, webgpu_renderer_t* renderer,
                          iscreen_effect_t* screen_effect)
{
  this->renderer      = renderer;
  this->screen_effect = screen_effect;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Vertex data & indices */
  const float vertex_data[8] = {
    -1.0f, 1.0f,  //
    -1.0f, -1.0f, //
    1.0f,  -1.0f, //
    1.0f,  1.0f,  //
  };

  const uint16_t indices[6] = {
    3, 2, 1, //
    3, 1, 0, //
  };

  /* Effect vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "effect vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertex_data),
                    .initial.data = vertex_data,
                  });

  /* Effect index buffer */
  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "effect index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  /* Init render pipeline */
  effect_init(this, screen_effect->fragment_shader,
              screen_effect->bind_group_layouts.items,
              screen_effect->bind_group_layouts.item_count,
              screen_effect->label);
}

static void effect_destroy(effect_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer)
}

static void effect_pre_render(effect_t* this, WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  for (uint32_t i = 0; i < this->screen_effect->bind_groups.item_count; ++i) {
    wgpuRenderPassEncoderSetBindGroup(
      render_pass, i, this->screen_effect->bind_groups.items[i], 0, 0);
  }
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
}

/* -------------------------------------------------------------------------- *
 * Bloom Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/bloom-pass.ts
 * -------------------------------------------------------------------------- */

#define BLOOM_PASS_TILE_DIM 128
#define BLOOM_PASS_BATCH                                                       \
  {                                                                            \
    4, 4                                                                       \
  }
#define BLOOM_PASS_FILTER_SIZE 10
#define BLOOM_PASS_ITERATIONS 2u

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t* point_lights;
  spot_light_t* spot_light;
  WGPURenderPassDescriptor framebuffer_descriptor;

  WGPUTexture bloom_yexture;
  WGPUTexture input_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } blur_textures[2];

  WGPUComputePipeline blur_pipeline;

  WGPUBindGroupLayout blur_constants_bind_group_layout;
  WGPUBindGroup blur_compute_constants_bindGroup;

  WGPUBindGroupLayout blur_compute_bind_group_layout;
  WGPUBindGroup blur_compute_bind_group_0;
  WGPUBindGroup blur_compute_bind_group_1;
  WGPUBindGroup blur_compute_bind_group_2;

  WGPUSampler sampler;
  uint32_t block_dim;
} bloom_pass_t;

static bool bloom_pass_is_ready(bloom_pass_t* this)
{
  return (this->effect.render_pipeline != NULL)
         && (this->blur_pipeline != NULL);
}

static void bloom_pass_update_bloom(bloom_pass_t* this,
                                    WGPUComputePassEncoder compute_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  const webgpu_renderer_t* renderer = this->renderer;
  const uint32_t block_dim          = this->block_dim;
  const uint32_t batch[2]           = BLOOM_PASS_BATCH;
  const uint32_t src_width          = renderer->output_size.width;
  const uint32_t src_height         = renderer->output_size.height;

  wgpuComputePassEncoderSetPipeline(compute_pass, this->blur_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->blur_compute_constants_bindGroup, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(compute_pass, 1,
                                     this->blur_compute_bind_group_0, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_width / (float)block_dim), // workgroupCountX
    (uint32_t)ceil(src_height / (float)batch[1]), // workgroupCountY
    1                                             // workgroupCountZ
  );
  wgpuComputePassEncoderSetBindGroup(compute_pass, 1,
                                     this->blur_compute_bind_group_1, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_height / (float)block_dim), // workgroupCountX
    (uint32_t)ceil(src_width / (float)batch[1]),   // workgroupCountY
    1                                              // workgroupCountZ
  );
  for (uint32_t i = 0; i < BLOOM_PASS_ITERATIONS - 1; ++i) {
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_group_2, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_width / (float)block_dim), // workgroupCountX
      (uint32_t)ceil(src_height / (float)batch[1]), // workgroupCountY
      1                                             // workgroupCountZ
    );
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_group_1, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_height / (float)block_dim), // workgroupCountX
      (uint32_t)ceil(src_width / (float)batch[1]),   // workgroupCountY
      1                                              // workgroupCountZ
    );
  }
}

static void bloom_pass_render(bloom_pass_t* this,
                              WGPURenderPassEncoder render_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Copy Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/copy-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } copy_texture;
} copy_pass_t;

static bool copy_pass_is_ready(copy_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void copy_pass_init_defaults(copy_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void copy_pass_create(copy_pass_t* this, webgpu_renderer_t* renderer)
{
  copy_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Copy texture */
  WGPUExtent3D texture_extent = {
    .width              = renderer->output_size.width,
    .height             = renderer->output_size.height,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "copy pass texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->copy_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->copy_texture.texture != NULL);

  /* Copy texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "copy pass texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->copy_texture.view
    = wgpuTextureCreateView(this->copy_texture.texture, &texture_view_dec);
  ASSERT(this->copy_texture.view != NULL);

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Texture view
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "copy pass bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = this->copy_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "copy pass bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Frame buffer Color attachments */
  this->framebuffer.color_attachments[0] =
    (WGPURenderPassColorAttachment) {
      .view       = this->copy_texture.view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };
}

static void copy_pass_destroy(copy_pass_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->copy_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->copy_texture.view)
}

static void copy_pass_render(copy_pass_t* this,
                             WGPURenderPassEncoder render_pass)
{
  if (!copy_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Deffered Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/deferred-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t point_lights;
  spot_light_t spot_light;

  struct {
    WGPURenderPassColorAttachment color_attachments[2];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_normal;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_diffuse;

  vec3 spot_light_target;
  vec3 spot_light_color_target;
} deferred_pass_t;

static bool deferred_pass_is_ready(deferred_pass_t* this)
{
  return point_lights_is_ready(&this->point_lights)
         && (this->effect.render_pipeline != NULL);
}

static void deferred_pass_init_defaults(deferred_pass_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_zero(this->spot_light_target);
  glm_vec3_one(this->spot_light_color_target);
}

static void deferred_pass_create(deferred_pass_t* this,
                                 webgpu_renderer_t* renderer)
{
  deferred_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Point light */
  point_lights_create(&this->point_lights);

  /* Spot light */
  ispot_light_t ispot_light = {
    .position      = {0.0f, 80.0f, 1.0f},
    .direction     = {0.0f, 1.0f, 0.0f},
    .color         = GLM_VEC3_ONE_INIT,
    .cut_off       = deg_to_rad(1.0f),
    .outer_cut_off = deg_to_rad(4.0f),
    .intensity     = 40.0f,
  };
  spot_light_create(&this->spot_light, renderer, &ispot_light);

  /* G-Buffer normal texture */
  {
    /* G-Buffer normal texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->output_size.width,
      .height             = renderer->output_size.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "gbuffer normal texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_normal.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_normal.texture != NULL);

    /* G-Buffer normal texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "gbuffer normal texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_normal.view = wgpuTextureCreateView(
      this->g_buffer_texture_normal.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_normal.view != NULL);
  }

  /* G-Buffer diffuse texture */
  {
    /* G-Buffer diffuse texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->output_size.width,
      .height             = renderer->output_size.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "gbuffer diffuse texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_diffuse.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_diffuse.texture != NULL);

    /* G-Buffer diffuse texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "gbuffer diffuse texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_diffuse.view = wgpuTextureCreateView(
      this->g_buffer_texture_diffuse.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_diffuse.view != NULL);
  }

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[5] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = this->point_lights.lights_buffer.size,
       },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
            .type           = WGPUBufferBindingType_Uniform,
            .minBindingSize = this->point_lights.lights_config_uniform_buffer.size,
        },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [3] = (WGPUBindGroupLayoutEntry) {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [4] = (WGPUBindGroupLayoutEntry) {
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "gbuffer bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[5] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = this->point_lights.lights_buffer.buffer,
      .size    = this->point_lights.lights_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .buffer  = this->point_lights.lights_config_uniform_buffer.buffer,
      .size    = this->point_lights.lights_config_uniform_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = this->g_buffer_texture_normal.view,
    },
    [3] = (WGPUBindGroupEntry) {
      .binding     = 3,
      .textureView = this->g_buffer_texture_diffuse.view,
    },
    [4] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = renderer->textures.depth_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "gbuffer bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Frame buffer Color attachments */
  {
    this->framebuffer.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_normal.view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };
  }
  {
    this->framebuffer.color_attachments[1] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_diffuse.view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearColor = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };
  }

  /* Frame buffer depth stencil attachment */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = renderer->textures.depth_texture.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthClearValue = 1.0f,
      .depthStoreOp    = WGPUStoreOp_Store,
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 2,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
  };
}

static void deferred_pass_destroy(deferred_pass_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_normal.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_normal.view)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_diffuse.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_diffuse.view)
}

static void deferred_pass_rearrange(deferred_pass_t* this)
{
  this->spot_light_target[0]       = (random_float() * 2 - 1) * 3;
  this->spot_light_target[2]       = (random_float() * 2 - 1) * 3;
  this->spot_light_color_target[0] = random_float();
  this->spot_light_color_target[1] = random_float();
  this->spot_light_color_target[2] = random_float();
}

static void deferred_pass_update_lights_sim(deferred_pass_t* this,
                                            WGPUComputePassEncoder compute_pass,
                                            float _time, float time_delta)
{
  point_lights_update_sim(&this->point_lights, compute_pass);
  const float speed = time_delta * 2.0f;
  glm_vec3_copy(
    (vec3){
      this->spot_light._position[0]
        + (this->spot_light_target[0] - this->spot_light._position[0])
            * speed, // x
      this->spot_light._position[1]
        + (this->spot_light_target[1] - this->spot_light._position[1])
            * speed, // y
      this->spot_light._position[2]
        + (this->spot_light_target[2] - this->spot_light._position[2])
            * speed, // z
    },
    this->spot_light._position);

  glm_vec3_copy(
    (vec3){
      (this->spot_light_color_target[0] - this->spot_light._color[0]) * speed
        * 4, // r
      (this->spot_light_color_target[1] - this->spot_light._color[1]) * speed
        * 4, // g
      (this->spot_light_color_target[2] - this->spot_light._color[2]) * speed
        * 4, // b
    },
    this->spot_light._color);
}

static void deferred_pass_render(deferred_pass_t* this,
                                 WGPURenderPassEncoder render_pass)
{
  if (!deferred_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Result Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/result-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } empty_texture;
} result_pass_t;

static bool result_pass_is_ready(result_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void result_pass_init_defaults(result_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void result_pass_create(result_pass_t* this, webgpu_renderer_t* renderer,
                               copy_pass_t* copy_pass, bloom_pass_t* bloom_pass)
{
  result_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "result pass bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* G-Buffer normal texture */
  WGPUExtent3D texture_extent = {
    .width              = 1,
    .height             = 1,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "empty texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_BGRA8Unorm,
    .usage         = WGPUTextureUsage_TextureBinding,
  };
  this->empty_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->empty_texture.texture != NULL);

  /* G-Buffer normal texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "empty texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->empty_texture.view
    = wgpuTextureCreateView(this->empty_texture.texture, &texture_view_dec);
  ASSERT(this->empty_texture.view != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = copy_pass->copy_texture.view,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = bloom_pass ? bloom_pass->blur_textures[1].view
                                  : this->empty_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "result pass bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);
}

static void result_pass_destroy(result_pass_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->empty_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->empty_texture.view)
}

static void result_pass_render(result_pass_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!result_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}
