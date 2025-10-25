#include "meshes.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

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

#include <stdbool.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cameras
 *
 * This example provides example camera implementations.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/cameras
 * https://github.com/pr0g/c-polymorphism
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* cube_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Math functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Calculates the square root of the sum of squares of its arguments.
 * @param a argument 1
 * @param b argument 2
 * @param c argument 3
 * @return the square root of the sum of squares of its arguments
 */
static float math_hypot3(float a, float b, float c)
{
  return sqrt(a * a + b * b + c * c);
}

/**
 * @brief Calculates the length of a vec3.
 * @param v A vector to calculate length of.
 * @returns The length of the vec3.
 */
static float glm_vec3_length(vec3 v)
{
  return math_hypot3(v[0], v[1], v[2]);
}

/**
 * @brief Adds two vectors, scaling the 2nd; assumes a and b have the same
 * dimension.
 * @param a - Operand vector.
 * @param b - Operand vector.
 * @param scale - Amount to scale b
 * @param dst - vector to hold result.
 * @returns A vector that is the sum of a + b * scale.
 */
static vec3* glm_vec3_add_scaled(vec3 a, vec3 b, float scale, vec3* dst)
{
  (*dst)[0] = a[0] + b[0] * scale;
  (*dst)[1] = a[1] + b[1] * scale;
  (*dst)[2] = a[2] + b[2] * scale;
  return dst;
}

/**
 * @brief Multiplies a vector by a scalar.
 * @param v - The vector.
 * @param k - The scalar.
 * @param dst - vector to hold result.
 * @returns The scaled vector.
 */
static vec3* glm_vec3_mul_scalar(vec3 v, float k, vec3* dst)
{
  (*dst)[0] = v[0] * k;
  (*dst)[1] = v[1] * k;
  (*dst)[2] = v[2] * k;
  return dst;
}

/**
 * @brief Transform vec4 by upper 3x3 matrix inside 4x4 matrix.
 * @param v - The direction.
 * @param m - The matrix.
 * @param dst - Vec3 to store result.
 * @returns The transformed vector.
 */
static vec3* glm_vec3_transform_mat4_upper3x3(vec3 v, mat4 m, vec3* dst)
{
  const float v0 = v[0];
  const float v1 = v[1];
  const float v2 = v[2];
  (*dst)[0]      = v0 * m[0][0] + v1 * m[1][0] + v2 * m[2][0];
  (*dst)[1]      = v0 * m[0][1] + v1 * m[1][1] + v2 * m[2][1];
  (*dst)[2]      = v0 * m[0][2] + v1 * m[1][2] + v2 * m[2][2];
  return dst;
}

/**
 * @brief Creates a 4-by-4 matrix which rotates around the given axis by the
 * given angle.
 * @param axis - The axis about which to rotate.
 * @param angle_in_radians - The angle by which to rotate (in radians).
 * @param dst - matrix to hold result.
 * @returns A matrix which rotates angle radians around the axis.
 */
static mat4* glm_mat4_axis_rotation(vec3 axis, float angle_in_radians,
                                    mat4* dst)
{
  float x       = axis[0];
  float y       = axis[1];
  float z       = axis[2];
  const float n = sqrt(x * x + y * y + z * z);
  x /= n;
  y /= n;
  z /= n;
  const float xx               = x * x;
  const float yy               = y * y;
  const float zz               = z * z;
  const float c                = cos(angle_in_radians);
  const float s                = sin(angle_in_radians);
  const float one_minus_cosine = 1.0f - c;
  (*dst)[0][0]                 = xx + (1.0f - xx) * c;
  (*dst)[0][1]                 = x * y * one_minus_cosine + z * s;
  (*dst)[0][2]                 = x * z * one_minus_cosine - y * s;
  (*dst)[0][3]                 = 0.0f;
  (*dst)[1][0]                 = x * y * one_minus_cosine - z * s;
  (*dst)[1][1]                 = yy + (1.0f - yy) * c;
  (*dst)[1][2]                 = y * z * one_minus_cosine + x * s;
  (*dst)[1][3]                 = 0.0f;
  (*dst)[2][0]                 = x * z * one_minus_cosine + y * s;
  (*dst)[2][1]                 = y * z * one_minus_cosine - x * s;
  (*dst)[2][2]                 = zz + (1.0f - zz) * c;
  (*dst)[2][3]                 = 0.0f;
  (*dst)[3][0]                 = 0.0f;
  (*dst)[3][1]                 = 0.0f;
  (*dst)[3][2]                 = 0.0f;
  (*dst)[3][3]                 = 1.0f;
  return dst;
}

/**
 * @brief Rotates the given 4-by-4 matrix around the x-axis by the given angle.
 * @param m - The matrix.
 * @param angle_in_radians - The angle by which to rotate (in radians).
 * @param dst - matrix to hold result.
 * @returns The rotated matrix.
 */
static mat4* glm_mat4_rotate_x(mat4 m, float angle_in_radians, mat4* dst)
{
  const float m10 = m[1][0];
  const float m11 = m[1][1];
  const float m12 = m[1][2];
  const float m13 = m[1][3];
  const float m20 = m[2][0];
  const float m21 = m[2][1];
  const float m22 = m[2][2];
  const float m23 = m[2][3];
  const float c   = cos(angle_in_radians);
  const float s   = sin(angle_in_radians);
  (*dst)[1][0]    = c * m10 + s * m20;
  (*dst)[1][1]    = c * m11 + s * m21;
  (*dst)[1][2]    = c * m12 + s * m22;
  (*dst)[1][3]    = c * m13 + s * m23;
  (*dst)[2][0]    = c * m20 - s * m10;
  (*dst)[2][1]    = c * m21 - s * m11;
  (*dst)[2][2]    = c * m22 - s * m12;
  (*dst)[2][3]    = c * m23 - s * m13;
  /* if (&m != dst)*/
  {
    (*dst)[0][0] = m[0][0];
    (*dst)[0][1] = m[0][1];
    (*dst)[0][2] = m[0][2];
    (*dst)[0][3] = m[0][3];
    (*dst)[3][0] = m[3][0];
    (*dst)[3][1] = m[3][1];
    (*dst)[3][2] = m[3][2];
    (*dst)[3][3] = m[3][3];
  }
  return dst;
}

/**
 * @brief Creates a 4-by-4 matrix which rotates around the y-axis by the given
 * angle.
 * @param angle_in_radians - The angle by which to rotate (in radians).
 * @param dst - matrix to hold result.
 * @returns The rotation matrix.
 */
static mat4* glm_mat4_rotation_y(float angle_in_radians, mat4* dst)
{
  glm_mat4_zero(*dst);
  const float c = cos(angle_in_radians);
  const float s = sin(angle_in_radians);
  (*dst)[0][0]  = c;
  (*dst)[0][2]  = -s;
  (*dst)[1][1]  = 1.0f;
  (*dst)[2][0]  = s;
  (*dst)[2][2]  = c;
  (*dst)[3][3]  = 1.0f;
  return dst;
}

/**
 * @brief Determines the sign of 2 boolean values.
 */
static int32_t sign(bool positive, bool negative)
{
  return (positive ? 1 : 0) - (negative ? 1 : 0);
}

/**
 * @brief Returns `x` clamped between [`min` .. `max`].
 */
static float clamp(float x, float min, float max)
{
  return MIN(MAX(x, min), max);
}

/**
 * @brief Returns `x` float-modulo `div`.
 */
static float mod(float x, float div)
{
  return x - floorf(fabs(x) / div) * div * glm_signf(x);
}

/**
 * @brief Returns `vec` rotated `angle` radians around `axis`.
 */
static vec3* rotate(vec3 vec, vec3 axis, float angle, vec3* dst)
{
  mat4 rotation = GLM_MAT4_ZERO_INIT;
  return glm_vec3_transform_mat4_upper3x3(
    vec, *glm_mat4_axis_rotation(axis, angle, &rotation), dst);
}

/**
 * @brief Returns the linear interpolation between 'a' and 'b' using 's'.
 */
static vec3* lerp(vec3 a, vec3 b, float s, vec3* dst)
{
  vec3 sub = GLM_VEC3_ZERO_INIT;
  glm_vec3_sub(b, a, sub);
  return glm_vec3_add_scaled(a, sub, s, dst);
}

/* -------------------------------------------------------------------------- *
 * The input event handling
 * -------------------------------------------------------------------------- */

typedef struct input_handler_t {
  /* Digital input (e.g keyboard state) */
  struct {
    bool forward;
    bool backward;
    bool left;
    bool right;
    bool up;
    bool down;
  } digital;
  /* Analog input (e.g mouse, touchscreen) */
  struct {
    vec2 prev_position;
    vec2 current_position;
    vec2 drag_distance;
    bool touching;
    float zoom;
  } analog;
} input_handler_t;

static void input_handler_init_defaults(input_handler_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void input_handler_init(input_handler_t* this)
{
  input_handler_init_defaults(this);
}

static void update_mouse_state(input_handler_t* this,
                               struct wgpu_context_t* wgpu_context)
{
#if 0
  vec2 mouse_position = {
    context->mouse_position[0],
    wgpu_context->surface.height - context->mouse_position[1],
  };

  /* Mouse move */
  if (!this->analog.touching && context->mouse_buttons.left) {
    glm_vec2_copy(mouse_position, this->analog.prev_position);
    this->analog.touching = true;
  }
  else if (this->analog.touching && context->mouse_buttons.left) {
    glm_vec2_sub(mouse_position, this->analog.prev_position,
                 this->analog.drag_distance);
    glm_vec2_add(this->analog.current_position, this->analog.drag_distance,
                 this->analog.current_position);
    glm_vec2_copy(mouse_position, this->analog.prev_position);
  }
  else if (this->analog.touching && !context->mouse_buttons.left) {
    this->analog.touching = false;
  }
#endif
}

static void reset_mouse_state(input_handler_t* this)
{
  memset(&this->digital, 0, sizeof(this->digital));
}

/* -------------------------------------------------------------------------- *
 * The common functionality between camera implementations
 * -------------------------------------------------------------------------- */

struct camera_base_t;

typedef struct camera_base_vtbl_t {
  mat4* (*get_matrix)(struct camera_base_t*);
  void (*set_matrix)(struct camera_base_t*, mat4);
  mat4* (*update)(struct camera_base_t*, float, input_handler_t*);
} camera_base_vtbl_t;

typedef struct camera_base_t {
  camera_base_vtbl_t _vtbl;
  /* The camera matrix */
  mat4 _matrix;
  /* The calculated view matrix */
  mat4 _view;
} camera_base_t;

static void camera_base_init_defaults(camera_base_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_mat4_identity(this->_matrix);
}

static void camera_base_init(camera_base_t* this)
{
  camera_base_init_defaults(this);
}

static mat4* camera_base__get_matrix(camera_base_t* this)
{
  return &this->_matrix;
}

static void camera_base__set_matrix(camera_base_t* this, mat4 mat)
{
  glm_mat4_copy(mat, this->_matrix);
}

/* Returns the camera matrix */
static mat4* camera_base_get_matrix(camera_base_t* this)
{
  return this->_vtbl.get_matrix(this);
}

/* Assigns `mat` to the camera matrix */
static void camera_base_set_matrix(camera_base_t* this, mat4 mat)
{
  this->_vtbl.set_matrix(this, mat);
}

static mat4* camera_base_update(struct camera_base_t* this, float delta_time,
                                input_handler_t* input_handler)
{
  return this->_vtbl.update(this, delta_time, input_handler);
}

/* Returns the camera view matrix */
static mat4* camera_base_get_view(camera_base_t* this)
{
  return &this->_view;
}

/* Assigns `mat` to the camera view */
static void camera_base_set_view(camera_base_t* this, mat4 mat)
{
  glm_mat4_copy(mat, this->_view);
}

/* Returns column vector 0 of the camera matrix */
static vec4* camera_base_get_right(camera_base_t* this)
{
  return &this->_matrix[0];
}

/* Assigns `vec` to the first 3 elements of column vector 0 of the camera matrix
 */
static void camera_base_set_right(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[0]);
}

/* Returns column vector 1 of the camera matrix */
static vec4* camera_base_get_up(camera_base_t* this)
{
  return &this->_matrix[1];
}

/* Assigns `vec` to the first 3 elements of column vector 1 of the camera matrix
 */
static void camera_base_set_up(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[1]);
}

/* Returns column vector 2 of the camera matrix */
static vec4* camera_base_get_back(camera_base_t* this)
{
  return &this->_matrix[2];
}

/* Assigns `vec` to the first 3 elements of column vector 2 of the camera matrix
 */
static void camera_base_set_back(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[2]);
}

/* Returns column vector 3 of the camera matrix  */
static vec4* camera_base_get_position(camera_base_t* this)
{
  return &this->_matrix[3];
}

/* Assigns `vec` to the first 3 elements of column vector 3 of the camera matrix
 */
static void camera_base_set_position(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[3]);
}

/* -------------------------------------------------------------------------- *
 * WASDCamera is a camera implementation that behaves similar to
 * first-person-shooter PC games.
 * -------------------------------------------------------------------------- */

typedef struct wasd_camera_t {
  /* The camera bass class */
  camera_base_t super;
  /* The camera absolute pitch angle */
  float pitch;
  /* The camera absolute yaw angle */
  float yaw;
  /* The movement veloicty */
  vec3 _velocity;
  /* Speed multiplier for camera movement */
  float movement_speed;
  /* Speed multiplier for camera rotation */
  float rotation_speed;
  /* Movement velocity drag coeffient [0 .. 1] */
  /* 0: Continues forever                      */
  /* 1: Instantly stops moving                 */
  float friction_coefficient;
} wasd_camera_t;

static void wasd_camera_recalculate_angles(wasd_camera_t* this, vec3 dir);
static mat4* wasd_camera_get_matrix(camera_base_t* this);
static void wasd_camera_set_matrix(camera_base_t* this, mat4 mat);
static mat4* wasd_camera_update(camera_base_t* this, float delta_time,
                                input_handler_t* input);

static void wasd_camera_init_defaults(wasd_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->pitch = 0.0f;
  this->yaw   = 0.0f;

  glm_vec3_zero(this->_velocity);

  this->movement_speed       = 10.0f;
  this->rotation_speed       = 1.0f;
  this->friction_coefficient = 0.99f;
}

static void wasd_camera_init_virtual_method_table(wasd_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->get_matrix = wasd_camera_get_matrix;
  vtbl->set_matrix = wasd_camera_set_matrix;
  vtbl->update     = wasd_camera_update;
}

/* Construtor */
static void wasd_camera_init(wasd_camera_t* this,
                             /* The initial position of the camera */
                             vec3* iposition,
                             /* The initial target of the camera */
                             vec3* itarget)
{
  wasd_camera_init_defaults(this);

  camera_base_init(&this->super);
  wasd_camera_init_virtual_method_table(this);

  if ((iposition != NULL) || (itarget != NULL)) {
    vec3 position, target, forward;
    glm_vec3_copy((iposition == NULL) ? (vec3){0.0f, 0.0f, -5.0f} : *iposition,
                  position);
    glm_vec3_copy((itarget == NULL) ? (vec3){0.0f, 0.0f, 0.0f} : *itarget,
                  target);
    glm_vec3_sub(target, position, forward);
    glm_vec3_normalize(forward);
    wasd_camera_recalculate_angles(this, forward);
    camera_base_set_position(&this->super, position);
  }
}

/* Returns velocity vector */
static vec3* wasd_camera_get_velocity(wasd_camera_t* this)
{
  return &this->_velocity;
}

/* Assigns `vec` to the velocity vector */
static void wasd_camera_set_velocity(wasd_camera_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_velocity);
}

/* Returns the camera matrix */
static mat4* wasd_camera_get_matrix(camera_base_t* this)
{
  wasd_camera_t* _this = (wasd_camera_t*)this;
  return camera_base__get_matrix(&_this->super);
}

/* Assigns `mat` to the camera matrix, and recalcuates the camera angles */
static void wasd_camera_set_matrix(camera_base_t* this, mat4 mat)
{
  wasd_camera_t* _this = (wasd_camera_t*)this;
  camera_base__set_matrix(&_this->super, mat);
  wasd_camera_recalculate_angles(_this, *camera_base_get_back(&_this->super));
}

static mat4* wasd_camera_update(camera_base_t* this, float delta_time,
                                input_handler_t* input)
{
  wasd_camera_t* _this = (wasd_camera_t*)this;

  /* Apply the delta rotation to the pitch and yaw angles */
  _this->yaw
    -= input->analog.current_position[0] * delta_time * _this->rotation_speed;
  _this->pitch
    -= input->analog.current_position[1] * delta_time * _this->rotation_speed;

  /* Wrap yaw between [0° .. 360°], just to prevent large accumulation. */
  _this->yaw = mod(_this->yaw, PI2);
  /* Clamp pitch between [-90° .. +90°] to prevent somersaults. */
  _this->pitch = clamp(_this->pitch, -PI_2, PI_2);

  /* Save the current position, as we're about to rebuild the camera matrix. */
  vec3 position = GLM_VEC3_ZERO_INIT;
  glm_vec3_copy(*camera_base_get_position(this), position);

  /* Reconstruct the camera's rotation, and store into the camera matrix. */
  mat4 matrix_rot_y = GLM_MAT4_ZERO_INIT;
  glm_mat4_rotation_y(_this->yaw, &matrix_rot_y);
  glm_mat4_rotate_x(matrix_rot_y, _this->pitch, &this->_matrix);

  // Calculate the new target velocity
  const int32_t delta_right = sign(input->digital.right, input->digital.left);
  const int32_t delta_up    = sign(input->digital.up, input->digital.down);
  vec3 target_velocity      = GLM_VEC3_ZERO_INIT;
  const int32_t delta_back
    = sign(input->digital.backward, input->digital.forward);
  glm_vec3_add_scaled(target_velocity, *camera_base_get_right(this),
                      delta_right, &target_velocity);
  glm_vec3_add_scaled(target_velocity, *camera_base_get_up(this), delta_up,
                      &target_velocity);
  glm_vec3_add_scaled(target_velocity, *camera_base_get_back(this), delta_back,
                      &target_velocity);
  glm_vec3_normalize(target_velocity);
  glm_vec3_mul_scalar(target_velocity, _this->movement_speed, &target_velocity);

  /* Mix new target velocity */
  vec3 velocity = GLM_VEC3_ZERO_INIT;
  lerp(target_velocity, *wasd_camera_get_velocity(_this),
       pow(1.0f - _this->friction_coefficient, delta_time), &velocity);
  wasd_camera_set_velocity(_this, velocity);

  /* Integrate velocity to calculate new position */
  glm_vec3_add_scaled(position, *wasd_camera_get_velocity(_this), delta_time,
                      &position);
  camera_base_set_position(this, position);

  /* Invert the camera matrix to build the view matrix */
  mat4 view = GLM_MAT4_ZERO_INIT;
  glm_mat4_inv(*wasd_camera_get_matrix(this), view);
  camera_base_set_view(this, view);

  return camera_base_get_view(this);
}

/* Recalculates the yaw and pitch values from a directional vector */
static void wasd_camera_recalculate_angles(wasd_camera_t* this, vec3 dir)
{
  this->yaw   = atan2(dir[0], dir[2]);
  this->pitch = -asin(dir[1]);
}

/* -------------------------------------------------------------------------- *
 * ArcballCamera implements a basic orbiting camera around the world origin
 * -------------------------------------------------------------------------- */

typedef struct arcball_camera_t {
  /* The camera bass class */
  camera_base_t super;
  /* The camera distance from the target */
  float distance;
  /* The current angular velocity  */
  float angular_velocity;
  /* The current rotation axis */
  vec3 _axis;
  /* Speed multiplier for camera rotation */
  float rotation_speed;
  /* Speed multiplier for camera zoom */
  float zoom_speed;
  /* Movement velocity drag coeffient [0 .. 1] */
  /* 0: Spins forever                          */
  /* 1: Instantly stops spinning               */
  float friction_coefficient;
} arcball_camera_t;

static mat4* arcball_camera_update(camera_base_t* this, float delta_time,
                                   input_handler_t* input);
static mat4* arcball_camera_get_matrix(camera_base_t* this);
static void arcball_camera_set_matrix(camera_base_t* this, mat4 mat);
static void arcball_camera_recalcuate_right(arcball_camera_t* this);
static void arcball_camera_recalcuate_up(arcball_camera_t* this);

static void arcball_camera_init_defaults(arcball_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->distance         = 0.0f;
  this->angular_velocity = 0.0f;

  glm_vec3_zero(this->_axis);

  this->rotation_speed       = 1.0f;
  this->zoom_speed           = 0.1f;
  this->friction_coefficient = 0.999f;
}

static void arcball_camera_init_virtual_method_table(arcball_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->get_matrix = arcball_camera_get_matrix;
  vtbl->set_matrix = arcball_camera_set_matrix;
  vtbl->update     = arcball_camera_update;
}

/* Construtor */
static void arcball_camera_init(arcball_camera_t* this,
                                /* The initial position of the camera */
                                vec3* iposition)
{
  arcball_camera_init_defaults(this);

  camera_base_init(&this->super);
  arcball_camera_init_virtual_method_table(this);

  if (iposition != NULL) {
    camera_base_set_position(&this->super, *iposition);
    this->distance = glm_vec3_length(*camera_base_get_position(&this->super));
    glm_vec3_normalize_to(*camera_base_get_position(&this->super),
                          *camera_base_get_back(&this->super));
    arcball_camera_recalcuate_right(this);
    arcball_camera_recalcuate_up(this);
  }
}

/* Returns the rotation axis */
static vec3* arcball_camera_get_axis(arcball_camera_t* this)
{
  return &this->_axis;
}

/* Assigns `vec` to the rotation axis */
static void arcball_camera_set_axis(arcball_camera_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_axis);
}

/* Returns the camera matrix */
static mat4* arcball_camera_get_matrix(camera_base_t* this)
{
  arcball_camera_t* _this = (arcball_camera_t*)this;
  return camera_base__get_matrix(&_this->super);
}

/* Assigns `mat` to the camera matrix, and recalcuates the distance */
static void arcball_camera_set_matrix(camera_base_t* this, mat4 mat)
{
  arcball_camera_t* _this = (arcball_camera_t*)this;
  camera_base__set_matrix(&_this->super, mat);
  _this->distance = glm_vec3_length(*camera_base_get_position(&_this->super));
}

static mat4* arcball_camera_update(camera_base_t* this, float delta_time,
                                   input_handler_t* input)
{
  arcball_camera_t* _this = (arcball_camera_t*)this;

  const float epsilon = 0.0000001f;

  if (input->analog.touching) {
    /* Currently being dragged. */
    _this->angular_velocity = 0.0f;
  }
  else {
    /* Dampen any existing angular velocity */
    _this->angular_velocity
      *= pow(1.0f - _this->friction_coefficient, delta_time);
  }

  /* Calculate the movement vector */
  vec3 movement = GLM_VEC3_ZERO_INIT;
  glm_vec3_add_scaled(movement, *camera_base_get_right(this),
                      input->analog.current_position[0], &movement);
  glm_vec3_add_scaled(movement, *camera_base_get_up(this),
                      -input->analog.current_position[1], &movement);

  /* Cross the movement vector with the view direction to calculate the rotation
   * axis x magnitude */
  vec3 cross_product = GLM_VEC3_ZERO_INIT;
  glm_vec3_cross(movement, *camera_base_get_back(this), cross_product);

  /* Calculate the magnitude of the drag */
  const float magnitude = glm_vec3_length(cross_product);

  if (magnitude > epsilon) {
    /* Normalize the crossProduct to get the rotation axis */
    vec3 tmp = GLM_VEC3_ZERO_INIT;
    glm_vec3_scale(cross_product, 1.0f / magnitude, tmp);
    arcball_camera_set_axis(_this, tmp);

    /* Remember the current angular velocity. This is used when the touch is
     * released for a fling. */
    _this->angular_velocity = magnitude * _this->rotation_speed;
  }

  /* The rotation around this.axis to apply to the camera matrix this update */
  const float rotation_angle = _this->angular_velocity * delta_time;
  if (rotation_angle > epsilon) {
    // Rotate the matrix around axis
    // Note: The rotation is not done as a matrix-matrix multiply as the
    // repeated multiplications will quickly introduce substantial error into
    // the matrix.
    vec3 rotated_vec = GLM_VEC3_ZERO_INIT;
    rotate(*camera_base_get_back(this), *arcball_camera_get_axis(_this),
           rotation_angle, &rotated_vec);
    glm_vec3_normalize(rotated_vec);
    camera_base_set_back(this, rotated_vec);
    arcball_camera_recalcuate_right(_this);
    arcball_camera_recalcuate_up(_this);
  }

  /* Recalculate `this.position` from `this.back` considering zoom */
  if (input->analog.zoom != 0.0f) {
    _this->distance *= 1 + input->analog.zoom * _this->zoom_speed;
  }
  vec3 position = GLM_VEC3_ZERO_INIT;
  glm_vec3_scale(*camera_base_get_back(this), _this->distance, position);
  camera_base_set_position(this, position);

  /* Invert the camera matrix to build the view matrix */
  mat4 view = GLM_MAT4_ZERO_INIT;
  glm_mat4_inv(*arcball_camera_get_matrix(this), view);

  camera_base_set_view(this, view);

  return camera_base_get_view(this);
}

/* Assigns `this.right` with the cross product of `this.up` and `this.back` */
static void arcball_camera_recalcuate_right(arcball_camera_t* this)
{
  vec3 cross = GLM_VEC3_ZERO_INIT;
  glm_vec3_cross(*camera_base_get_up(&this->super),
                 *camera_base_get_back(&this->super), cross);
  glm_vec3_normalize(cross);
  camera_base_set_right(&this->super, cross);
}

/* Assigns `this.up` with the cross product of `this.back` and `this.right` */
static void arcball_camera_recalcuate_up(arcball_camera_t* this)
{
  vec3 cross = GLM_VEC3_ZERO_INIT;
  glm_vec3_cross(*camera_base_get_back(&this->super),
                 *camera_base_get_right(&this->super), cross);
  glm_vec3_normalize(cross);
  camera_base_set_up(&this->super, cross);
}

/* --------------------------------------------------------------------------
 * Cameras example.
 * -------------------------------------------------------------------------- */

/* Camera parameters */
typedef enum camera_type_t {
  CameraType_Arcball,
  Renderer_WASD,
} camera_type_t;

/* State struct */
static struct {
  cube_mesh_t cube_mesh;
  struct {
    WGPUBindGroup uniform_buffer_bind_group;
    WGPUBindGroupLayout bind_group_layout;
    struct {
      mat4 model_view_projection;
    } view_mtx;
  } cube;
  wgpu_buffer_t vertices;
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    mat4 projection;
    mat4 view;
  } view_matrices;
  struct {
    wgpu_texture_t cube;
    wgpu_texture_t depth;
    WGPUSampler sampler;
  } textures;
  uint8_t file_buffer[512 * 512 * 4];
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    vec3 initial_camera_position;
    camera_type_t camera_type;
  } example_parms;
  arcball_camera_t arcball_camera;
  wasd_camera_t wasd_camera;
  camera_base_t* cameras[2];
  const char* camera_type_names[2];
  input_handler_t input_handler;
  float last_frame_ms;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.5, 0.5, 0.5, 1.0},
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
  .example_parms = {
    .initial_camera_position = {3.0f, 2.0f, 5.0f},
    .camera_type             = CameraType_Arcball,
  },
  .cameras = {
    [0] = (camera_base_t*)&state.arcball_camera,
    [1] = (camera_base_t*)&state.wasd_camera,
  },
  .camera_type_names = {
    "arcball",
    "WASD",
  }
};

static void init_cameras(void)
{
  arcball_camera_init(&state.arcball_camera,
                      &state.example_parms.initial_camera_position);
  wasd_camera_init(&state.wasd_camera,
                   &state.example_parms.initial_camera_position, NULL);
}

/* Initialize the cube geometry */
static void init_cube_mesh(void)
{
  cube_mesh_init(&state.cube_mesh);
}

/* Create a vertex buffer from the cube data. */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  state.vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(state.cube_mesh.vertex_array),
                    .initial.data = state.cube_mesh.vertex_array,
                  });
}

/**
 * @brief The fetch-callback is called by sokol_fetch.h when the data is loaded,
 * or when an error has occurred.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    state.textures.cube.desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D) {
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 4,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    state.textures.cube.desc.is_dirty = true;
  }
}

static void fetch_texture(void)
{
  /* Start loading the image file */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/Di-3d.png",
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Create a depth/stencil texture for the color rendering pipeline */
  {
    WGPUExtent3D texture_extent = {
      .width              = wgpu_context->width,
      .height             = wgpu_context->height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = STRVIEW("Depth - Texture"),
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    state.textures.depth.handle
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(state.textures.depth.handle != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = STRVIEW("Depth - Texture view"),
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    state.textures.depth.view
      = wgpuTextureCreateView(state.textures.depth.handle, &texture_view_dec);
    ASSERT(state.textures.depth.view != NULL);
  }

  /* Cube texture */
  {
    state.textures.cube = wgpu_create_color_bars_texture(wgpu_context, 16, 16);
    fetch_texture();
  }

  /* Create a sampler with linear filtering for smooth interpolation. */
  {
    state.textures.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .label         = STRVIEW("Texture - Sampler"),
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(state.textures.sampler != NULL);
  }
}

static void init_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->width / (float)wgpu_context->height;

  // Projection matrix
  glm_mat4_identity(state.view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  state.view_matrices.projection);
}

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Setup the view matrices for the camera */
  init_view_matrices(wgpu_context);

  /* Set the current time */
  state.last_frame_ms = stm_ms(stm_now());

  /* Uniform buffer */
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Camera - Uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(mat4), // 4x4 matrix
                  });
  ASSERT(state.uniform_buffer_vs.buffer != NULL);
}

static mat4* get_model_view_projection_matrix(float delta_time)
{
  camera_base_t* camera = state.cameras[state.example_parms.camera_type];
  glm_mat4_copy(*camera_base_update(camera, delta_time, &state.input_handler),
                state.view_matrices.view);
  glm_mat4_mul(state.view_matrices.projection, state.view_matrices.view,
               state.cube.view_mtx.model_view_projection);
  return &state.cube.view_mtx.model_view_projection;
}

static void update_model_view_projection_matrix(wgpu_context_t* wgpu_context)
{
  /* Get the model-view-projection matrix */
  const float now        = stm_ms(stm_now());
  const float delta_time = (now - state.last_frame_ms) / 1000.0f;
  state.last_frame_ms    = now;

  mat4* model_view_projection = get_model_view_projection_matrix(delta_time);

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       model_view_projection, state.uniform_buffer_vs.size);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Transform */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), // 4x4 matrix
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Texture view */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  state.cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Cube - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Render - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.cube.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.uniform_buffer_bind_group)

  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      /* Transform */
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      /* Sampler */
      .binding = 1,
      .sampler = state.textures.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      /* Texture view */
      .binding     = 2,
      .textureView = state.textures.cube.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Cube uniform buffer - Bind group"),
    .layout     = state.cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.cube.uniform_buffer_bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, cube_shader_wgsl);

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
  WGPU_VERTEX_BUFFER_LAYOUT(textured_cube, state.cube_mesh.vertex_size,
                            /* Attribute location 0: Position */
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               state.cube_mesh.position_offset),
                            /* Attribute location 1: UV */
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2,
                                               state.cube_mesh.uv_offset))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Textured cubes - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vertex_main"),
      .bufferCount = 1,
      .buffers     = &textured_cube_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("fragment_main"),
      .module      = shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW
    },
    .depthStencil = &depth_stencil_state,
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });
    input_handler_init(&state.input_handler);
    init_cameras();
    init_cube_mesh();
    init_vertex_buffer(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_texture(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
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

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.textures.cube.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.cube);
    FREE_TEXTURE_PIXELS(state.textures.cube);
    /* Upddate the bindgroup */
    init_bind_group(wgpu_context);
  }

  /* Update camera */
  update_mouse_state(&state.input_handler, wgpu_context);
  update_model_view_projection_matrix(wgpu_context);
  reset_mouse_state(&state.input_handler);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0,
                                    state.cube.uniform_buffer_bind_group, 0, 0);
  wgpuRenderPassEncoderDraw(rpass_enc, state.cube_mesh.vertex_count, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  if (input_event->type == INPUT_EVENT_TYPE_KEY_DOWN) {
    switch (input_event->key_code) {
      case KEY_W:
        state.input_handler.digital.forward = 1;
        break;
      case KEY_S:
        state.input_handler.digital.backward = 1;
        break;
      case KEY_A:
        state.input_handler.digital.left = 1;
        break;
      case KEY_D:
        state.input_handler.digital.right = 1;
        break;
      case KEY_SPACE:
        state.input_handler.digital.up = 1;
        break;
      case KEY_C:
        state.input_handler.digital.down = 1;
        break;
      default:
        break;
    }
  }
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  WGPU_RELEASE_RESOURCE(BindGroup, state.cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  wgpu_destroy_texture(&state.textures.cube);
  wgpu_destroy_texture(&state.textures.depth);
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Caneras",
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

static const char* cube_shader_wgsl = CODE(
  struct Uniforms {
   modelViewProjectionMatrix : mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) fragUV : vec2f,
  }

  @vertex
  fn vertex_main(
  @location(0) position : vec4f,
    @location(1) uv : vec2f
  ) -> VertexOutput {
    return VertexOutput(uniforms.modelViewProjectionMatrix * position, uv);
  }

  @fragment
  fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
    return textureSample(myTexture, mySampler, fragUV);
  }
);
// clang-format on
