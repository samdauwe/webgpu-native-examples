#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Raytracer
 *
 * WebGPU demo featuring realtime path tracing via WebGPU compute shaders.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float value;
  float damping;
} damped_action_t;

static void damped_action_add_force(damped_action_t* this, float force)
{
  this->value += force;
}

static void damped_action_stop(damped_action_t* this)
{
  this->value = 0.0f;
}

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

typedef enum {
  CAMERA_ACTION_STATE_ROTATE,
  CAMERA_ACTION_STATE_PAN,
} camera_action_state_t;

typedef struct {
  vec3 UP;
  vec3 position;
  float aspect_ratio;
  vec3 target;
  mat4 view_projection_matrix;
  mat4 view_matrix;
  mat4 projection_matrix;
  camera_action_state_t _state;
  struct {
    float x;
    float y;
  } _rotate_delta;
  struct {
    float x;
    float y;
  } _rotate_start;
  struct {
    float x;
    float y;
  } _rotate_end;
  struct {
    float x;
    float y;
  } _pan_delta;
  struct {
    float x;
    float y;
  } _pan_start;
  struct {
    float x;
    float y;
  } _pan_end;
  struct {
    float radius;
    float theta;
    float phi;
  } _spherical;
  damped_action_t _target_x_damped_action;
  damped_action_t _target_y_damped_action;
  damped_action_t _target_z_damped_action;
  damped_action_t _target_theta_damped_action;
  damped_action_t _target_phi_damped_action;
  damped_action_t _target_radius_damped_action;
} _camera_t;

static void _camera_init_defaults(_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->UP);

  glm_vec3_copy((vec3){0.0f, 0.0f, -1.0f}, this->target);
  glm_mat4_identity(this->view_projection_matrix);
  glm_mat4_identity(this->view_matrix);
  glm_mat4_identity(this->projection_matrix);

  this->_state = CAMERA_ACTION_STATE_ROTATE;

  damped_action_init(&this->_target_x_damped_action);
  damped_action_init(&this->_target_y_damped_action);
  damped_action_init(&this->_target_z_damped_action);
  damped_action_init(&this->_target_theta_damped_action);
  damped_action_init(&this->_target_phi_damped_action);
  damped_action_init(&this->_target_radius_damped_action);
}

static void _camera_create(_camera_t* this, vec3 position, float vfov,
                           float aspectRatio)
{
  _camera_init_defaults(this);

  const float dx     = position[0];
  const float dy     = position[1];
  const float dz     = position[2];
  const float radius = sqrt(dx * dx + dy * dy + dz * dz);
  const float theta  = atan2(dx, dz); // equator angle around y-up axis
  const float phi    = acos(CLAMP(dy / radius, -1.0f, 1.0f)); // polar angle
  this->_spherical.radius = radius;
  this->_spherical.theta  = theta;
  this->_spherical.phi    = phi;
}

static void _camera_handle_input_events(_camera_t* this,
                                        wgpu_example_context_t* context)
{
  if (context->mouse_buttons.left) {
    this->_state          = CAMERA_ACTION_STATE_ROTATE;
    this->_rotate_start.x = context->mouse_position[0];
    this->_rotate_start.y = context->mouse_position[1];
  }
  else {
    this->_state       = CAMERA_ACTION_STATE_PAN;
    this->_pan_start.x = context->mouse_position[0];
    this->_pan_start.y = context->mouse_position[1];
  }
}

static void _camera_update_damped_action(_camera_t* this)
{
  this->target[0] += damped_action_update(&this->_target_x_damped_action);
  this->target[1] += damped_action_update(&this->_target_y_damped_action);
  this->target[2] += damped_action_update(&this->_target_z_damped_action);

  this->_spherical.theta
    += damped_action_update(&this->_target_theta_damped_action);
  this->_spherical.phi
    += damped_action_update(&this->_target_phi_damped_action);
  this->_spherical.radius
    += damped_action_update(&this->_target_radius_damped_action);
}

static void _camera_update_camera(_camera_t* this)
{
  const float s_radius       = this->_spherical.radius;
  const float s_theta        = this->_spherical.theta;
  const float s_phi          = this->_spherical.phi;
  const float sin_phi_radius = sin(s_phi) * s_radius;

  this->position[0] = sin_phi_radius * sin(s_theta) + this->target[0];
  this->position[1] = cos(s_phi) * s_radius + this->target[1];
  this->position[2] = sin_phi_radius * cos(s_theta) + this->target[2];

  glm_lookat(this->position,   /* eye    */
             this->target,     /* center */
             this->UP,         /* up     */
             this->view_matrix /* dest   */
  );
  glm_perspective(45.0f,                  /* fovy   */
                  this->aspect_ratio,     /* aspect */
                  0.1f,                   /* nearZ  */
                  100.0f,                 /* farZ   */
                  this->projection_matrix /* dest   */
  );
  glm_mat4_mul(this->projection_matrix, this->view_matrix,
               this->view_projection_matrix);
}

static void _camera_tick(_camera_t* this)
{
  _camera_update_damped_action(this);
  _camera_update_camera(this);
}

/* -------------------------------------------------------------------------- *
 * Material
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Material.ts
 * -------------------------------------------------------------------------- */

typedef enum {
  MATERIAL_TYPE_EMISSIVE_MATERIAL,
  MATERIAL_TYPE_REFLECTIVE_MATERIAL,
  MATERIAL_TYPE_DIELECTRIC_MATERIAL,
  MATERIAL_TYPE_LAMBERTIAN_MATERIAL,
} material_type_t;

typedef struct {
  material_type_t EMISSIVE_MATERIAL;
  material_type_t REFLECTIVE_MATERIAL;
  material_type_t DIELECTRIC_MATERIAL;
  material_type_t LAMBERTIAN_MATERIAL;

  vec4 albedo;
  material_type_t mtl_type;
  float reflection_ratio;
  float reflection_gloss;
  float refraction_index;
} material_t;

static void material_init_defaults(material_t* this)
{
  this->EMISSIVE_MATERIAL   = MATERIAL_TYPE_EMISSIVE_MATERIAL;
  this->REFLECTIVE_MATERIAL = MATERIAL_TYPE_REFLECTIVE_MATERIAL;
  this->DIELECTRIC_MATERIAL = MATERIAL_TYPE_DIELECTRIC_MATERIAL;
  this->LAMBERTIAN_MATERIAL = MATERIAL_TYPE_LAMBERTIAN_MATERIAL;
}

static void material_create(material_t* this, vec4 albedo,
                            material_type_t mtl_type, float reflection_ratio,
                            float reflection_gloss, float refraction_index)
{
  material_init_defaults(this);

  glm_vec4_copy(albedo, this->albedo);
  this->mtl_type         = mtl_type;
  this->reflection_ratio = reflection_ratio;
  this->reflection_gloss = reflection_gloss;
  this->refraction_index = refraction_index;
}
