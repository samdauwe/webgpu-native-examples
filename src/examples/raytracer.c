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
 * Minimal DOM event structures
 * -------------------------------------------------------------------------- */

typedef struct {
  int button; /* 0 = left, 1 = middle, 2 = right */
  int page_x;
  int page_y;
} mouse_event_t;

typedef struct {
  float delta_y;
} wheel_event_t;

typedef struct {
  bool default_prevented;
} pointer_event_t;

/* Event callback function types */
typedef void (*mouse_down_callback_t)(mouse_event_t* event, void* user_data);
typedef void (*mouse_move_callback_t)(mouse_event_t* event, void* user_data);
typedef void (*mouse_up_callback_t)(mouse_event_t* event, void* user_data);
typedef void (*wheel_callback_t)(wheel_event_t* event, void* user_data);
typedef void (*context_menu_callback_t)(pointer_event_t* event,
                                        void* user_data);

/* DOM element structure */
typedef struct {
  mouse_down_callback_t on_mouse_down;
  mouse_move_callback_t on_mouse_move;
  mouse_up_callback_t on_mouse_up;
  wheel_callback_t on_wheel;
  context_menu_callback_t on_context_menu;
  void* user_data;
} dom_element_t;

/* DOM element functions */
static void dom_element_create(dom_element_t* this)
{
  if (this) {
    memset(this, 0, sizeof(dom_element_t));
  }
}

static void dom_element_destroy(dom_element_t* element)
{
  if (element) {
    free(element);
  }
}

static void dom_element_add_event_listener(dom_element_t* element,
                                           const char* event_type,
                                           void* callback, void* user_data)
{
  if (!element) {
    return;
  }

  element->user_data = user_data;

  if (strcmp(event_type, "mousedown") == 0) {
    element->on_mouse_down = (mouse_down_callback_t)callback;
  }
  else if (strcmp(event_type, "mousemove") == 0) {
    element->on_mouse_move = (mouse_move_callback_t)callback;
  }
  else if (strcmp(event_type, "mouseup") == 0) {
    element->on_mouse_up = (mouse_up_callback_t)callback;
  }
  else if (strcmp(event_type, "wheel") == 0) {
    element->on_wheel = (wheel_callback_t)callback;
  }
  else if (strcmp(event_type, "contextmenu") == 0) {
    element->on_context_menu = (context_menu_callback_t)callback;
  }
}

static void dom_element_remove_event_listener(dom_element_t* element,
                                              const char* event_type)
{
  if (!element) {
    return;
  }

  if (strcmp(event_type, "mousemove") == 0) {
    element->on_mouse_move = NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Camera.ts
 * -------------------------------------------------------------------------- */

static float clamp_f(float x, float min, float max)
{
  if (x < min) {
    return min;
  }
  if (x > max) {
    return max;
  }
  return x;
}

/* Damped action structure */
typedef struct {
  float value;
  float damping;
} damped_action_t;

static void damped_action_stop(damped_action_t* this);

static void damped_action_init(damped_action_t* this)
{
  this->value   = 0.0f;
  this->damping = 0.5f;
}

static void damped_action_add_force(damped_action_t* this, float force)
{
  this->value += force;
}

static float damped_action_update(damped_action_t* this)
{
  bool is_active = (this->value * this->value) > 0.000001f;
  if (is_active) {
    this->value *= this->damping;
  }
  else {
    damped_action_stop(this);
  }
  return this->value;
}

static void damped_action_stop(damped_action_t* this)
{
  this->value = 0.0f;
}

/* Point structure */
typedef struct {
  float x;
  float y;
} point_t;

/* Spherical coordinates structure */
typedef struct {
  float radius;
  float theta;
  float phi;
} spherical_t;

/* Camera state */
typedef enum {
  CAMERA_STATE_ROTATE = 0,
  CAMERA_STATE_PAN    = 1,
} camera_state_t;

/* Camera UP vector constant */
static vec3 CAMERA_UP = {0.0f, 1.0f, 0.0f};

/* Camera structure */
typedef struct {
  /* Public members */
  vec3 target;
  mat4 view_projection_matrix;
  mat4 view_matrix;
  mat4 projection_matrix;
  vec3 position;
  float vfov;
  float aspect_ratio;

  /* Private members */
  dom_element_t* dom_element;
  camera_state_t state;

  point_t rotate_delta;
  point_t rotate_start;
  point_t rotate_end;
  point_t pan_start;
  point_t pan_delta;
  point_t pan_end;
  spherical_t spherical;

  damped_action_t target_x_damped_action;
  damped_action_t target_y_damped_action;
  damped_action_t target_z_damped_action;
  damped_action_t target_theta_damped_action;
  damped_action_t target_phi_damped_action;
  damped_action_t target_radius_damped_action;

  /* Window dimensions for mouse calculations */
  int window_width;
  int window_height;
} rt_camera_t;

/* Camera function prototypes */
static void rt_camera_update_damped_action(rt_camera_t* this);
static void rt_camera_update_camera(rt_camera_t* this);

/* Event handlers */
static void rt_camera_on_mouse_down(mouse_event_t* event, void* user_data);
static void rt_camera_on_mouse_move(mouse_event_t* event, void* user_data);
static void rt_camera_on_mouse_up(mouse_event_t* event, void* user_data);
static void rt_camera_on_mouse_wheel(wheel_event_t* event, void* user_data);
static void rt_camera_on_context_menu(pointer_event_t* event, void* user_data);

static void rt_camera_init(rt_camera_t* this, dom_element_t* dom_element,
                           vec3 position, float vfov, float aspect_ratio)
{
  /* Initialize public members */
  glm_vec3_copy((vec3){0.0f, 0.0f, -1.0f}, this->target);
  glm_mat4_identity(this->view_projection_matrix);
  glm_mat4_identity(this->view_matrix);
  glm_mat4_identity(this->projection_matrix);
  glm_vec3_copy(position, this->position);
  this->vfov         = vfov;
  this->aspect_ratio = aspect_ratio;

  /* Initialize private members */
  this->dom_element   = dom_element;
  this->state         = CAMERA_STATE_ROTATE;
  this->window_width  = 800; /* Default values */
  this->window_height = 600;

  /* Initialize points */
  memset(&this->rotate_delta, 0, sizeof(point_t));
  memset(&this->rotate_start, 0, sizeof(point_t));
  memset(&this->rotate_end, 0, sizeof(point_t));
  memset(&this->pan_start, 0, sizeof(point_t));
  memset(&this->pan_delta, 0, sizeof(point_t));
  memset(&this->pan_end, 0, sizeof(point_t));

  /* Initialize spherical coordinates */
  float dx     = position[0];
  float dy     = position[1];
  float dz     = position[2];
  float radius = sqrtf(dx * dx + dy * dy + dz * dz);
  float theta  = atan2f(dx, dz);
  float phi    = acosf(clamp_f(dy / radius, -1.0f, 1.0f));

  this->spherical.radius = radius;
  this->spherical.theta  = theta;
  this->spherical.phi    = phi;

  /* Initialize damped actions */
  damped_action_init(&this->target_x_damped_action);
  damped_action_init(&this->target_y_damped_action);
  damped_action_init(&this->target_z_damped_action);
  damped_action_init(&this->target_theta_damped_action);
  damped_action_init(&this->target_phi_damped_action);
  damped_action_init(&this->target_radius_damped_action);

  /* Set up event listeners */
  if (dom_element) {
    dom_element_add_event_listener(dom_element, "mousedown",
                                   (void*)rt_camera_on_mouse_down, this);
    dom_element_add_event_listener(dom_element, "mouseup",
                                   (void*)rt_camera_on_mouse_up, this);
    dom_element_add_event_listener(dom_element, "wheel",
                                   (void*)rt_camera_on_mouse_wheel, this);
    dom_element_add_event_listener(dom_element, "contextmenu",
                                   (void*)rt_camera_on_context_menu, this);
  }
}

static void rt_camera_set_window_size(rt_camera_t* this, int width, int height)
{
  if (this) {
    this->window_width  = width;
    this->window_height = height;
  }
}

static void rt_camera_tick(rt_camera_t* this)
{
  rt_camera_update_damped_action(this);
  rt_camera_update_camera(this);
}

static void rt_camera_update_damped_action(rt_camera_t* this)
{
  this->target[0] += damped_action_update(&this->target_x_damped_action);
  this->target[1] += damped_action_update(&this->target_y_damped_action);
  this->target[2] += damped_action_update(&this->target_z_damped_action);

  this->spherical.theta
    += damped_action_update(&this->target_theta_damped_action);
  this->spherical.phi += damped_action_update(&this->target_phi_damped_action);
  this->spherical.radius
    += damped_action_update(&this->target_radius_damped_action);
}

static void rt_camera_update_camera(rt_camera_t* this)
{
  spherical_t* s       = &this->spherical;
  float sin_phi_radius = sinf(s->phi) * s->radius;

  this->position[0] = sin_phi_radius * sinf(s->theta) + this->target[0];
  this->position[1] = cosf(s->phi) * s->radius + this->target[1];
  this->position[2] = sin_phi_radius * cosf(s->theta) + this->target[2];

  glm_lookat(this->position, this->target, CAMERA_UP, this->view_matrix);
  glm_perspective(glm_rad(45.0f), this->aspect_ratio, 0.1f, 100.0f,
                  this->projection_matrix);
  glm_mat4_mul(this->projection_matrix, this->view_matrix,
               this->view_projection_matrix);
}

/* Event handlers */
static void rt_camera_on_mouse_down(mouse_event_t* event, void* user_data)
{
  rt_camera_t* camera = (rt_camera_t*)user_data;
  if (!camera || !event)
    return;

  if (event->button == 0) { /* Left button */
    camera->state          = CAMERA_STATE_ROTATE;
    camera->rotate_start.x = (float)event->page_x;
    camera->rotate_start.y = (float)event->page_y;
  }
  else { /* Other buttons */
    camera->state       = CAMERA_STATE_PAN;
    camera->pan_start.x = (float)event->page_x;
    camera->pan_start.y = (float)event->page_y;
  }

  if (camera->dom_element) {
    dom_element_add_event_listener(camera->dom_element, "mousemove",
                                   (void*)rt_camera_on_mouse_move, camera);
  }
}

static void rt_camera_on_mouse_move(mouse_event_t* event, void* user_data)
{
  rt_camera_t* camera = (rt_camera_t*)user_data;
  if (!camera || !event) {
    return;
  }

  if (camera->state == CAMERA_STATE_ROTATE) {
    camera->rotate_end.x = (float)event->page_x;
    camera->rotate_end.y = (float)event->page_y;

    camera->rotate_delta.x = camera->rotate_end.x - camera->rotate_start.x;
    camera->rotate_delta.y = camera->rotate_end.y - camera->rotate_start.y;

    damped_action_add_force(&camera->target_theta_damped_action,
                            -camera->rotate_delta.x
                              / (float)camera->window_width);
    damped_action_add_force(&camera->target_phi_damped_action,
                            -camera->rotate_delta.y
                              / (float)camera->window_height);

    camera->rotate_start.x = camera->rotate_end.x;
    camera->rotate_start.y = camera->rotate_end.y;
  }
  else { /* PAN state */
    camera->pan_end.x   = (float)event->page_x;
    camera->pan_end.y   = (float)event->page_y;
    camera->pan_delta.x = -0.5f * (camera->pan_end.x - camera->pan_start.x);
    camera->pan_delta.y = 0.5f * (camera->pan_end.y - camera->pan_start.y);
    camera->pan_start.x = camera->pan_end.x;
    camera->pan_start.y = camera->pan_end.y;

    vec3 x_dir, y_dir, z_dir;
    glm_vec3_sub(camera->target, camera->position, z_dir);
    glm_vec3_normalize(z_dir);

    vec3 up = {0.0f, 1.0f, 0.0f};
    glm_vec3_cross(z_dir, up, x_dir);
    glm_vec3_cross(x_dir, z_dir, y_dir);

    float scale = fmaxf(camera->spherical.radius / 2000.0f, 0.001f);

    damped_action_add_force(
      &camera->target_x_damped_action,
      (x_dir[0] * camera->pan_delta.x + y_dir[0] * camera->pan_delta.y)
        * scale);
    damped_action_add_force(
      &camera->target_y_damped_action,
      (x_dir[1] * camera->pan_delta.x + y_dir[1] * camera->pan_delta.y)
        * scale);
    damped_action_add_force(
      &camera->target_z_damped_action,
      (x_dir[2] * camera->pan_delta.x + y_dir[2] * camera->pan_delta.y)
        * scale);
  }
}

static void rt_camera_on_mouse_up(mouse_event_t* event, void* user_data)
{
  UNUSED_VAR(event);

  rt_camera_t* camera = (rt_camera_t*)user_data;
  if (!camera) {
    return;
  }

  if (camera->dom_element) {
    dom_element_remove_event_listener(camera->dom_element, "mousemove");
  }
}

static void rt_camera_on_mouse_wheel(wheel_event_t* event, void* user_data)
{
  rt_camera_t* camera = (rt_camera_t*)user_data;
  if (!camera || !event) {
    return;
  }

  float force = 0.1f;
  if (event->delta_y > 0.0f) {
    damped_action_add_force(&camera->target_radius_damped_action, force);
  }
  else {
    damped_action_add_force(&camera->target_radius_damped_action, -force);
  }
}

static void rt_camera_on_context_menu(pointer_event_t* event, void* user_data)
{
  UNUSED_VAR(user_data);

  if (event) {
    event->default_prevented = true; /* Equivalent to e.preventDefault() */
  }
}

/* -------------------------------------------------------------------------- *
 * Material
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Material.ts
 * -------------------------------------------------------------------------- */

/* Material type enumeration */
typedef enum material_type_enun {
  MATERIAL_TYPE_EMISSIVE_MATERIAL   = 0,
  MATERIAL_TYPE_REFLECTIVE_MATERIAL = 1,
  MATERIAL_TYPE_DIELECTRIC_MATERIAL = 2,
  MATERIAL_TYPE_LAMBERTIAN_MATERIAL = 3,
} material_type_enun;

/* Material structure */
typedef struct {
  vec4 albedo;
  material_type_enun mtl_type;
  float reflection_ratio;
  float reflection_gloss;
  float refraction_index;
} material_t;

static void material_init(material_t* this, vec4 albedo,
                          material_type_enun mtl_type, float reflection_ratio,
                          float reflection_gloss, float refraction_index)
{
  glm_vec4_copy(albedo, this->albedo);
  this->mtl_type         = mtl_type;
  this->reflection_ratio = reflection_ratio;
  this->reflection_gloss = reflection_gloss;
  this->refraction_index = refraction_index;
}

static void material_init_default(material_t* this, vec4 albedo)
{
  material_init(this, albedo, MATERIAL_TYPE_LAMBERTIAN_MATERIAL, 0.0f, 1.0f,
                1.0f);
}
