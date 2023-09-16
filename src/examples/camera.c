#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "meshes.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cameras
 *
 * This example provides example camera implementations.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/cameras
 * https://github.com/pr0g/c-polymorphism
 * -------------------------------------------------------------------------- *

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
 *
 * @param {ReadonlyVec3} a vector to calculate length of
 * @returns {Number} length of a
 */
static float glm_vec3_length(vec3 v)
{
  return math_hypot3(v[0], v[1], v[2]);
}

/* -------------------------------------------------------------------------- *
 * The input event handling
 * -------------------------------------------------------------------------- */

typedef struct input_t {
  void* padding;
} input_t;

/* -------------------------------------------------------------------------- *
 * The common functionality between camera implementations
 * -------------------------------------------------------------------------- */

struct camera_base_t;

typedef struct camera_base_vtbl_t {
  mat4* (*update)(struct camera_base_t*, float, input_t*);
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

/* Returns the camera matrix */
static mat4* camera_base_get_matrix(camera_base_t* this)
{
  return &this->_matrix;
}

/* Assigns `mat` to the camera matrix */
static void camera_base_set_matrix(camera_base_t* this, mat4 mat)
{
  glm_mat4_copy(mat, this->_matrix);
}

/* Returns the camera view matrix */
static mat4* camera_base_view(camera_base_t* this)
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
  /* 0: Instantly stops moving                 */
  /* 1: Continues forever                      */
  float friction_coefficient;
} wasd_camera_t;

static void wasd_camera_recalculate_angles(wasd_camera_t* this, vec3 dir);
static mat4* wasd_camera_update(camera_base_t* this, float delta_time,
                                input_t* input);

static void wasd_camera_init_defaults(wasd_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->pitch = 0.0f;
  this->yaw   = 0.0f;

  glm_vec3_zero(this->_velocity);

  this->movement_speed       = 10.0f;
  this->rotation_speed       = 1.0f;
  this->friction_coefficient = 0.01f;
}

static void wasd_camera_init_virtual_method_table(wasd_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->update = wasd_camera_update;
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
static vec3* wasd_camera_set_velocity(wasd_camera_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_velocity);
}

/* Returns the camera matrix */
static mat4* wasd_camera_get_matrix(wasd_camera_t* this)
{
  return camera_base_get_matrix(&this->super);
}

/* Assigns `mat` to the camera matrix, and recalcuates the camera angles */
static void wasd_camera_set_matrix(wasd_camera_t* this, mat4 mat)
{
  camera_base_set_matrix(&this->super, mat);
  wasd_camera_recalculate_angles(this, *camera_base_get_back(&this->super));
}

static mat4* update(camera_base_t* this, float delta_time, input_t* input)
{
  return NULL;
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
  /* 0: Instantly stops spinning               */
  /* 1: Spins forever                          */
  float friction_coefficient;
} arcball_camera_t;

static mat4* arcball_camera_update(camera_base_t* this, float delta_time,
                                   input_t* input);
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
  this->friction_coefficient = 0.0001f;
}

static void arcball_camera_init_virtual_method_table(arcball_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->update = arcball_camera_update;
}

/* Construtor */
static void arcball_camera_init(arcball_camera_t* this,
                                /* The initial position of the camera */
                                vec3* iposition)
{
  arcball_camera_init_virtual_method_table(this);

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

/* --------------------------------------------------------------------------
 * Cameras example.
 * -------------------------------------------------------------------------- */

// GUI
typedef enum camera_type_t {
  CameraType_Arcball,
  Renderer_WASD,
} camera_type_t;

static struct {
  vec3 initial_camera_position;
  camera_type_t camera_type;
} example_parms = {
  .initial_camera_position = {3.0f, 2.0f, 5.0f},
  .camera_type             = CameraType_Arcball,
};

/* The camera types */
static struct {
  arcball_camera_t arcball;
  wasd_camera_t wasd;
} cameras = {0};

static const char* camera_type_names[2] = {"arcball", "WASD"};

// Other variables
static const char* example_title = "Cameras";
static bool prepared             = false;

static void initialize_cameras(void)
{
  arcball_camera_init(&cameras.arcball, &example_parms.initial_camera_position);
  wasd_camera_init(&cameras.wasd, &example_parms.initial_camera_position, NULL);
}
