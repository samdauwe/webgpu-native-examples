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
 * -------------------------------------------------------------------------- *

/* -------------------------------------------------------------------------- *
* WGSL Shaders
* -------------------------------------------------------------------------- */

static const char* cube_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * The common functionality between camera implementations
 * -------------------------------------------------------------------------- */

typedef struct camera_base_t {
  /* The camera matrix */
  mat4 _matrix;
  /* The calculated view matrix */
  mat4 _view;
  /* Aliases to column vectors of the matrix */
  vec4 _right;
  vec4 _up;
  vec4 _back;
  vec4 _position;
} camera_base_t;

static void camera_base_init_defaults(camera_base_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_mat4_identity(this->_matrix);

  glm_vec4_copy((vec4){this->_matrix[0][0], this->_matrix[0][1],
                       this->_matrix[0][2], this->_matrix[0][3]},
                this->_right);
  glm_vec4_copy((vec4){this->_matrix[1][0], this->_matrix[1][1],
                       this->_matrix[1][2], this->_matrix[1][3]},
                this->_up);
  glm_vec4_copy((vec4){this->_matrix[2][0], this->_matrix[2][1],
                       this->_matrix[2][2], this->_matrix[2][3]},
                this->_back);
  glm_vec4_copy((vec4){this->_matrix[3][0], this->_matrix[3][1],
                       this->_matrix[3][2], this->_matrix[3][3]},
                this->_position);
}

static void camera_base__init(camera_base_t* this)
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
  return &this->_right;
}

/* Assigns `vec` to the first 3 elements of column vector 0 of the camera matrix
 */
static void camera_base_set_right(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_right);
}

/* Returns column vector 1 of the camera matrix */
static vec4* camera_base_get_up(camera_base_t* this)
{
  return &this->_up;
}

/* Assigns `vec` to the first 3 elements of column vector 1 of the camera matrix
 */
static void camera_base_set_up(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_up);
}

/* Returns column vector 2 of the camera matrix */
static vec4* camera_base_get_back(camera_base_t* this)
{
  return &this->_back;
}

/* Assigns `vec` to the first 3 elements of column vector 2 of the camera matrix
 */
static void camera_base_set_back(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_back);
}

/* Returns column vector 3 of the camera matrix  */
static vec4* camera_base_get_position(camera_base_t* this)
{
  return &this->_position;
}

/* Assigns `vec` to the first 3 elements of column vector 3 of the camera matrix
 */
static void camera_base_set_position(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_position);
}
