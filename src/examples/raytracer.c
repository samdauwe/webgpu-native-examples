#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* camera_wgsl;
static const char* color_wgsl;
static const char* common_wgsl;
static const char* interval_wgsl;
static const char* material_wgsl;
static const char* ray_wgsl;
static const char* utils_wgsl;
static const char* vec_wgsl;
static const char* vertex_wgsl;

/* -------------------------------------------------------------------------- *
 * Bounding Volume Hierarchy.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/BV.ts
 * -------------------------------------------------------------------------- */

/* Axis enumeration */
typedef enum {
  AXIS_X = 0,
  AXIS_Y = 1,
  AXIS_Z = 2,
} axis_t;

/* Constants */
#define BV_MIN_DELTA 0.01f
#define BV_MAX_FACES_PER_LEAF 2

/* Face structure (equivalent to TypeScript Face interface) */
typedef struct {
  vec3 p0, p1, p2; /* vertex positions */
  vec3 n0, n1, n2; /* vertex normals */
  vec3 fn;         /* face normal */
  int fi;          /* face index */
  int mi;          /* material index */
} face_t;

/* Bounding Volume structure */
typedef struct {
  vec4 min;  /* minimum bounds */
  vec4 max;  /* maximum bounds */
  int lt;    /* left child BV index */
  int rt;    /* right child BV index */
  int fi[2]; /* face indices (max 2 faces per leaf) */
} bv_t;

/* Dynamic array structure for BVs */
typedef struct {
  bv_t* data;
  size_t size;
  size_t capacity;
} bv_array_t;

/* Dynamic array functions for BVs */
bv_array_t* bv_array_create(size_t initial_capacity)
{
  bv_array_t* array = (bv_array_t*)malloc(sizeof(bv_array_t));
  if (!array) {
    return NULL;
  }

  array->data = (bv_t*)malloc(sizeof(bv_t) * initial_capacity);
  if (!array->data) {
    free(array);
    return NULL;
  }

  array->size     = 0;
  array->capacity = initial_capacity;
  return array;
}

void bv_array_destroy(bv_array_t* this)
{
  if (this) {
    free(this->data);
    free(this);
  }
}

void bv_array_resize(bv_array_t* this, size_t new_capacity)
{
  if (!this) {
    return;
  }

  bv_t* new_data = (bv_t*)realloc(this->data, sizeof(bv_t) * new_capacity);
  if (new_data) {
    this->data     = new_data;
    this->capacity = new_capacity;
  }
}

void bv_array_push(bv_array_t* this, bv_t bv)
{
  if (!this) {
    return;
  }

  if (this->size >= this->capacity) {
    bv_array_resize(this, this->capacity * 2);
  }

  this->data[this->size] = bv;
  this->size++;
}

/* Dynamic array structure for faces */
typedef struct {
  face_t* data;
  size_t size;
  size_t capacity;
} face_array_t;

float face_centroid_axis(const face_t* this, axis_t axis)
{
  return (this->p0[axis] + this->p1[axis] + this->p2[axis]) / 3.0f;
}

/* Dynamic array functions for faces */
face_array_t* face_array_create(size_t initial_capacity)
{
  face_array_t* array = (face_array_t*)malloc(sizeof(face_array_t));
  if (!array) {
    return NULL;
  }

  array->data = (face_t*)malloc(sizeof(face_t) * initial_capacity);
  if (!array->data) {
    free(array);
    return NULL;
  }

  array->size     = 0;
  array->capacity = initial_capacity;
  return array;
}

void face_array_destroy(face_array_t* this)
{
  if (this) {
    free(this->data);
    free(this);
  }
}

void face_array_resize(face_array_t* this, size_t new_capacity)
{
  if (!this) {
    return;
  }

  face_t* new_data
    = (face_t*)realloc(this->data, sizeof(face_t) * new_capacity);
  if (new_data) {
    this->data     = new_data;
    this->capacity = new_capacity;
  }
}

void face_array_push(face_array_t* this, face_t face)
{
  if (!this)
    return;

  if (this->size >= this->capacity) {
    face_array_resize(this, this->capacity * 2);
  }

  this->data[this->size] = face;
  this->size++;
}

static void bv_split_across(bv_t* this, axis_t axis, face_t* faces,
                            size_t face_count, bv_array_t* aabbs);

static void bv_init(bv_t* this, vec4 min, vec4 max)
{
  glm_vec4_copy(min, this->min);
  glm_vec4_copy(max, this->max);
  this->lt    = -1;
  this->rt    = -1;
  this->fi[0] = -1;
  this->fi[1] = -1;
}

/* BV creation and initialization */
static bv_t bv_create(vec4 min, vec4 max)
{
  bv_t bv;
  bv_init(&bv, min, max);
  return bv;
}

/* Face comparison function for qsort_r */
static axis_t
  g_sort_axis; /* Global variable to pass axis to comparison function */

static int bv_face_compare(const void* a, const void* b, void* axis_ptr)
{
  const face_t* face_a = (const face_t*)a;
  const face_t* face_b = (const face_t*)b;
  axis_t axis          = *(axis_t*)axis_ptr;

  float centroid_a = face_centroid_axis(face_a, axis);
  float centroid_b = face_centroid_axis(face_b, axis);

  if (centroid_a < centroid_b) {
    return -1;
  }
  if (centroid_a > centroid_b) {
    return 1;
  }
  return 0;
}

/* Alternative comparison function for systems without qsort_r */
static int bv_face_compare_global(const void* a, const void* b)
{
  const face_t* face_a = (const face_t*)a;
  const face_t* face_b = (const face_t*)b;

  float centroid_a = face_centroid_axis(face_a, g_sort_axis);
  float centroid_b = face_centroid_axis(face_b, g_sort_axis);

  if (centroid_a < centroid_b) {
    return -1;
  }
  if (centroid_a > centroid_b) {
    return 1;
  }
  return 0;
}

static void bv_calculate_bounds(face_t* faces, size_t face_count, vec4 min_out,
                                vec4 max_out)
{
  if (!faces || face_count == 0) {
    return;
  }

  /* Initialize with extreme values */
  min_out[0] = min_out[1] = min_out[2] = FLT_MAX;
  max_out[0] = max_out[1] = max_out[2] = -FLT_MAX;
  min_out[3] = max_out[3] = 1.0f;

  for (size_t i = 0; i < face_count; i++) {
    const face_t* face = &faces[i];

    /* Check p0 */
    min_out[0] = fminf(min_out[0], face->p0[0]);
    min_out[1] = fminf(min_out[1], face->p0[1]);
    min_out[2] = fminf(min_out[2], face->p0[2]);
    max_out[0] = fmaxf(max_out[0], face->p0[0]);
    max_out[1] = fmaxf(max_out[1], face->p0[1]);
    max_out[2] = fmaxf(max_out[2], face->p0[2]);

    /* Check p1 */
    min_out[0] = fminf(min_out[0], face->p1[0]);
    min_out[1] = fminf(min_out[1], face->p1[1]);
    min_out[2] = fminf(min_out[2], face->p1[2]);
    max_out[0] = fmaxf(max_out[0], face->p1[0]);
    max_out[1] = fmaxf(max_out[1], face->p1[1]);
    max_out[2] = fmaxf(max_out[2], face->p1[2]);

    /* Check p2 */
    min_out[0] = fminf(min_out[0], face->p2[0]);
    min_out[1] = fminf(min_out[1], face->p2[1]);
    min_out[2] = fminf(min_out[2], face->p2[2]);
    max_out[0] = fmaxf(max_out[0], face->p2[0]);
    max_out[1] = fmaxf(max_out[1], face->p2[1]);
    max_out[2] = fmaxf(max_out[2], face->p2[2]);
  }
}

static void bv_ensure_min_delta(vec4 min, vec4 max)
{
  if (max[0] - min[0] < BV_MIN_DELTA) {
    max[0] += BV_MIN_DELTA;
  }
  if (max[1] - min[1] < BV_MIN_DELTA) {
    max[1] += BV_MIN_DELTA;
  }
  if (max[2] - min[2] < BV_MIN_DELTA) {
    max[2] += BV_MIN_DELTA;
  }
}

static void bv_subdivide(bv_t* this, face_t* faces, size_t face_count,
                         bv_array_t* aabbs)
{
  if (!this || !faces || !aabbs) {
    return;
  }

  if (face_count <= BV_MAX_FACES_PER_LEAF) {
    /* Leaf node - store face indices */
    for (size_t i = 0; i < face_count && i < BV_MAX_FACES_PER_LEAF; i++) {
      this->fi[i] = faces[i].fi;
    }
  }
  else {
    /* Internal node - split along longest axis */
    float dx = fabsf(this->max[0] - this->min[0]);
    float dy = fabsf(this->max[1] - this->min[1]);
    float dz = fabsf(this->max[2] - this->min[2]);

    float largest_delta = fmaxf(dx, fmaxf(dy, dz));
    axis_t split_axis;

    if (largest_delta == dx) {
      split_axis = AXIS_X;
    }
    else if (largest_delta == dy) {
      split_axis = AXIS_Y;
    }
    else {
      split_axis = AXIS_Z;
    }

    bv_split_across(this, split_axis, faces, face_count, aabbs);
  }
}

static void bv_split_across(bv_t* this, axis_t axis, face_t* faces,
                            size_t face_count, bv_array_t* aabbs)
{
  if (!this || !faces || !aabbs || face_count == 0) {
    return;
  }

  /* Create a copy of faces array for sorting */
  face_t* sorted_faces = (face_t*)malloc(sizeof(face_t) * face_count);
  if (!sorted_faces) {
    return;
  }

  memcpy(sorted_faces, faces, sizeof(face_t) * face_count);

  /* Sort faces by centroid along the specified axis */
  g_sort_axis = axis;
  qsort(sorted_faces, face_count, sizeof(face_t), bv_face_compare_global);

  /* Split into left and right halves */
  size_t h             = face_count / 2;
  size_t lt_face_count = h;
  size_t rt_face_count = face_count - h;

  face_t* lt_faces = sorted_faces;
  face_t* rt_faces = sorted_faces + h;

  /* Create left child BV */
  if (lt_face_count > 0) {
    vec4 lt_min, lt_max;
    bv_calculate_bounds(lt_faces, lt_face_count, lt_min, lt_max);
    bv_ensure_min_delta(lt_min, lt_max);

    this->lt   = (int)aabbs->size;
    bv_t lt_bv = bv_create(lt_min, lt_max);
    bv_array_push(aabbs, lt_bv);
  }

  /* Create right child BV */
  if (rt_face_count > 0) {
    vec4 rt_min, rt_max;
    bv_calculate_bounds(rt_faces, rt_face_count, rt_min, rt_max);
    bv_ensure_min_delta(rt_min, rt_max);

    this->rt   = (int)aabbs->size;
    bv_t rt_bv = bv_create(rt_min, rt_max);
    bv_array_push(aabbs, rt_bv);
  }

  /* Recursively subdivide children */
  if (this->lt >= 0) {
    bv_t* lt_bv = &aabbs->data[this->lt];
    bv_subdivide(lt_bv, lt_faces, lt_face_count, aabbs);
  }

  if (this->rt >= 0) {
    bv_t* rt_bv = &aabbs->data[this->rt];
    bv_subdivide(rt_bv, rt_faces, rt_face_count, aabbs);
  }

  free(sorted_faces);
}

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
  const float dx     = position[0];
  const float dy     = position[1];
  const float dz     = position[2];
  const float radius = sqrtf(dx * dx + dy * dy + dz * dz);
  const float theta  = atan2f(dx, dz);
  const float phi    = acosf(clamp_f(dy / radius, -1.0f, 1.0f));

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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* camera_wgsl = CODE(
  struct Camera {
    viewportSize: vec2u,
    imageWidth: f32,
    imageHeight: f32,
    pixel00Loc: vec3<f32>,
    pixelDeltaU: vec3<f32>,
    pixelDeltaV: vec3<f32>,

    aspectRatio: f32,
    center: vec3<f32>,
    vfov: f32,

    lookFrom: vec3f,
    lookAt: vec3f,
    vup: vec3f,

    defocusAngle: f32,
    focusDist: f32,

    defocusDiscU: vec3f,
    defocusDiscV: vec3f
  }

  fn initCamera(camera: ptr<function, Camera>) {
    (*camera).imageHeight = (*camera).imageWidth / (*camera).aspectRatio;
    (*camera).imageHeight = select((*camera).imageHeight, 1, (*camera).imageHeight < 1);

    (*camera).center = (*camera).lookFrom;

    let theta = radians((*camera).vfov);
    let h = tan(theta * 0.5);
    let viewportHeight = 2.0 * h * (*camera).focusDist;
    let viewportWidth = viewportHeight * ((*camera).imageWidth / (*camera).imageHeight);

    let w = normalize((*camera).lookFrom - (*camera).lookAt);
    let u = normalize(cross((*camera).vup, w));
    let v = cross(w, u);

    let viewportU = viewportWidth * u;
    let viewportV = viewportHeight * -v;

    (*camera).pixelDeltaU = viewportU / (*camera).imageWidth;
    (*camera).pixelDeltaV = viewportV / (*camera).imageHeight;

    let viewportUpperLeft = (*camera).center - ((*camera).focusDist * w) - viewportU / 2 - viewportV / 2;
    (*camera).pixel00Loc = viewportUpperLeft + 0.5 * ((*camera).pixelDeltaU + (*camera).pixelDeltaV);

    let defocusRadius = (*camera).focusDist * tan(radians((*camera).defocusAngle * 0.5));
    (*camera).defocusDiscU = u * defocusRadius;
    (*camera).defocusDiscV = v * defocusRadius;
  }
);

static const char* color_wgsl = CODE(
  // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
  @must_use
  fn aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate(x * (a * x + b)) / (x * (c * x + d) + e);
  }

  // Filmic Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators/
  @must_use
  fn filmic(x: vec3f) -> vec3f {
    let X = max(vec3f(0.0), x - 0.004);
    let result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
    return pow(result, vec3(2.2));
  }

  // Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines"
  @must_use
  fn lottes(x: vec3f) -> vec3f {
    let a = vec3f(1.6);
    let d = vec3f(0.977);
    let hdrMax = vec3f(8.0);
    let midIn = vec3f(0.18);
    let midOut = vec3f(0.267);

    let b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    let c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
  }

  @must_use
  fn reinhard(x: vec3f) -> vec3f {
    return x / (1.0 + x);
  }
);

static const char* common_wgsl = CODE(
  struct CommonUniforms {
    // Random seed for the workgroup
    seed : vec3u,
    frameCounter: u32,
    maxBounces: u32,
    flatShading: u32,
    debugNormals: u32
  }

  struct HitRecord {
    p: vec3f,
    normal: vec3f,
    t: f32,
    frontFace: bool,
    materialIdx: u32,
    meshIdx: i32
  };

  struct Face {
    p0: vec3f,
    p1: vec3f,
    p2: vec3f,

    n0: vec3f,
    n1: vec3f,
    n2: vec3f,

    faceNormal: vec3f,
    materialIdx: u32
  }

  struct AABB {
    min: vec3f,
    max: vec3f,
    leftChildIdx: i32,
    rightChildIdx: i32,
    faceIdx0: i32,
    faceIdx1: i32
  }

  struct Mesh {
    aabbOffset: i32,
    faceOffset: i32
  }
);

static const char* interval_wgsl = CODE(
  struct Interval {
    min: f32,
    max: f32,
  };

  @must_use
  fn intervalContains(interval: Interval, x: f32) -> bool {
    return interval.min <= x && x <= interval.max;
  }

  @must_use
  fn intervalSurrounds(interval: Interval, x: f32) -> bool {
    return interval.min < x && x < interval.max;
  }

  @must_use
  fn intervalClamp(interval: Interval, x: f32) -> f32 {
    var out = x;
    if (x < interval.min) {
      out = interval.min;
    }
    if (x > interval.max) {
      out = interval.max;
    }
    return out;
  }

  const emptyInterval = Interval(f32max, f32min);
  const universeInterval = Interval(f32min, f32max);
  const positiveUniverseInterval = Interval(EPSILON, f32max);
);

static const char* material_wgsl = CODE(
  struct Material {
    materialType: u32,
    reflectionRatio: f32,
    reflectionGloss: f32,
    refractionIndex: f32,
    albedo: vec3f,
  };

  @must_use
  fn scatterLambertian(
    material: ptr<function, Material>,
    ray: ptr<function, Ray>,
    scattered: ptr<function, Ray>,
    hitRec: ptr<function, HitRecord>,
    attenuation: ptr<function, vec3f>,
    rngState: ptr<function, u32>
  ) -> bool {
    var scatterDirection = (*hitRec).normal + randomUnitVec3(rngState);
    if (nearZero(scatterDirection)) {
      scatterDirection = (*hitRec).normal;
    }
    (*scattered) = Ray((*hitRec).p, scatterDirection);
    (*attenuation) = (*material).albedo;
    return true;
  }

  @must_use
  fn scatterMetal(
    material: ptr<function, Material>,
    ray: ptr<function, Ray>,
    scattered: ptr<function, Ray>,
    hitRec: ptr<function, HitRecord>,
    attenuation: ptr<function, vec3f>,
    rngState: ptr<function, u32>
  ) -> bool {
    let reflected = reflect(normalize((*ray).direction), (*hitRec).normal);
    (*scattered) = Ray((*hitRec).p, reflected + (*material).reflectionGloss * randomUnitVec3(rngState));
    (*attenuation) = (*material).albedo;
    return (dot((*scattered).direction, (*hitRec).normal) >= 0);
  }

  @must_use
  fn scatterDielectric(
    material: ptr<function, Material>,
    ray: ptr<function, Ray>,
    scattered: ptr<function, Ray>,
    hitRec: ptr<function, HitRecord>,
    attenuation: ptr<function, vec3f>,
    rngState: ptr<function, u32>
  ) -> bool {
    *attenuation = vec3f(1);
    let refractRatio = select((*material).refractionIndex, 1.0 / (*material).refractionIndex, (*hitRec).frontFace);
    let unitDirection = normalize((*ray).direction);
    let cosTheta = dot(-unitDirection, (*hitRec).normal);
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    let cannotRefract = refractRatio * sinTheta > 1.0;
    let direction = select(
      refract(unitDirection, (*hitRec).normal, refractRatio),
      reflect(unitDirection, (*hitRec).normal),
      cannotRefract || reflectance(cosTheta, refractRatio) > rngNextFloat(rngState)
    );
    (*scattered) = Ray((*hitRec).p, direction);
    return true;
  }

  @must_use
  fn reflectance(cosine: f32, refractionIndex: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    var r0 = (1.0 - refractionIndex) / (1.0 + refractionIndex);
    r0 *= r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
  }
);

static const char* ray_wgsl = CODE(
  struct Ray {
    origin: vec3f,
    direction: vec3f,
  };

  @must_use
  fn rayAt(ray: ptr<function, Ray>, t: f32) -> vec3f {
    return (*ray).origin + (*ray).direction * t;
  }

  @must_use
  fn rayIntersectFace(
    ray: ptr<function, Ray>,
    face: ptr<function, Face>,
    rec: ptr<function, HitRecord>,
    interval: Interval
  ) -> bool {
    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm

    // let fnDotRayDir = dot((*face).faceNormal, (*ray).direction);
    // if (abs(fnDotRayDir) < EPSILON) {
    //   return false; // ray direction almost parallel
    // }

    let e1 = (*face).p1 - (*face).p0;
    let e2 = (*face).p2 - (*face).p0;

    let h = cross((*ray).direction, e2);
    let det = dot(e1, h);

    if det > -0.00001 && det < 0.00001 {
      return false;
    }

    let invDet = 1.0f / det;
    let s = (*ray).origin - (*face).p0;
    let u = invDet * dot(s, h);

    if u < 0.0f || u > 1.0f {
      return false;
    }

    let q = cross(s, e1);
    let v = invDet * dot((*ray).direction, q);

    if v < 0.0f || u + v > 1.0f {
      return false;
    }

    let t = invDet * dot(e2, q);

    if t > interval.min && t < interval.max {
      // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html

      let p = (*face).p0 + u * e1 + v * e2;
      // *hit = TriangleHit(offsetRay(p, n), b, t);
      (*rec).t = t;
      (*rec).p = p;
      (*rec).materialIdx = (*face).materialIdx;
      if (commonUniforms.flatShading == 1u) {
        (*rec).normal = (*face).faceNormal;
      } else {
        let b = vec3f(1f - u - v, u, v);
        let n = b[0] * (*face).n0 + b[1] * (*face).n1 + b[2] * (*face).n2;
        (*rec).normal = n;
      }
      return true;
    } else {
      return false;
    }
  }


  @must_use
  fn rayIntersectBV(ray: ptr<function, Ray>, aabb: ptr<function, AABB>) -> bool {
    let t0 = ((*aabb).min - (*ray).origin) / (*ray).direction;
    let t1 = ((*aabb).max - (*ray).origin) / (*ray).direction;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let maxMinT = max(tmin.x, max(tmin.y, tmin.z));
    let minMaxT = min(tmax.x, min(tmax.y, tmax.z));
    return maxMinT < minMaxT;
  }

  @must_use
  fn rayIntersectBVH(
    ray: ptr<function, Ray>,
    hitRec: ptr<function, HitRecord>,
    interval: Interval
  ) -> bool {

    var current: HitRecord;
    var didIntersect = false;
    var stack: array<i32, BV_MAX_STACK_DEPTH>;

    (*hitRec).t = f32max;

    var top: i32;

    for (var objIdx = 0u; objIdx < OBJECTS_COUNT_IN_SCENE; objIdx++) {
      top = 0;
      stack[0] = 0;

      while (top > -1) {
        var bvIdx = stack[top];
        top--;
        var aabb = AABBs[u32(bvIdx) + objIdx * MAX_BVs_COUNT_PER_MESH];

        if (rayIntersectBV(ray, &aabb)) {
          if (aabb.leftChildIdx != -1) {
            top++;
            stack[top] = aabb.leftChildIdx;
          }
          if (aabb.rightChildIdx != -1) {
            top++;
            stack[top] = aabb.rightChildIdx;
          }

          if (aabb.faceIdx0 != -1) {
            var face = faces[u32(aabb.faceIdx0) + objIdx * MAX_FACES_COUNT_PER_MESH];
            if (
              rayIntersectFace(ray, &face, &current, positiveUniverseInterval) &&
              current.t < (*hitRec).t
            ) {
              *hitRec = current;
              didIntersect = true;
            }
          }

          if (aabb.faceIdx1 != -1) {
            var face = faces[u32(aabb.faceIdx1) + objIdx * MAX_FACES_COUNT_PER_MESH];
            if (
              rayIntersectFace(ray, &face, &current, positiveUniverseInterval) &&
              current.t < (*hitRec).t
            ) {
              *hitRec = current;
              didIntersect = true;
            }
          }
        }
      }
    }
    return didIntersect;
  }

  @must_use
  fn getCameraRay(camera: ptr<function, Camera>, i: f32, j: f32, rngState: ptr<function, u32>) -> Ray {
    let pixelCenter = (*camera).pixel00Loc + (i * (*camera).pixelDeltaU) + (j * (*camera).pixelDeltaV);
    let pixelSample = pixelCenter + pixelSampleSquare(camera, rngState);
    let rayOrigin = select(defocusDiskSample(camera, rngState), (*camera).center, (*camera).defocusAngle <= 0);
    let rayDirection = pixelSample - rayOrigin;
    return Ray(rayOrigin, rayDirection);
  }

  @must_use
  fn defocusDiskSample(camera: ptr<function, Camera>, rngState: ptr<function, u32>) -> vec3f {
    let p = randomVec3InUnitDisc(rngState);
    return (*camera).center + (p.x * (*camera).defocusDiscU) + (p.y * (*camera).defocusDiscV);
  }

  @must_use
  fn pixelSampleSquare(camera: ptr<function, Camera>, rngState: ptr<function, u32>) -> vec3<f32> {
    let px = -0.5 + rngNextFloat(rngState);
    let py = -0.5 + rngNextFloat(rngState);
    return (px * (*camera).pixelDeltaU) + (py * (*camera).pixelDeltaV);
  }
);

static const char* utils_wgsl = CODE(
  const f32min = 0x1p-126f;
  const f32max = 0x1.fffffep+127;

  const pi = 3.141592653589793;

  @must_use
  fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    rngNextInt(state);
    return f32(*state) / f32(0xffffffffu);
  }

  fn rngNextInt(state: ptr<function, u32>) {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let oldState = *state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    *state = (word >> 22u) ^ word;
  }

  @must_use
  fn randInRange(min: f32, max: f32, state: ptr<function, u32>) -> f32 {
    return min + rngNextFloat(state) * (max - min);
  }
);

static const char* vec_wgsl = CODE(
  @must_use
  fn randomVec3(rngState: ptr<function, u32>) -> vec3f {
    return vec3f(rngNextFloat(rngState), rngNextFloat(rngState), rngNextFloat(rngState));
  }

  @must_use
  fn randomVec3InRange(min: f32, max: f32, rngState: ptr<function, u32>) -> vec3f {
    return vec3f(
      randInRange(min, max, rngState),
      randInRange(min, max, rngState),
      randInRange(min, max, rngState)
    );
  }

  fn randomVec3InUnitDisc(state: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rngNextFloat(state));
    let alpha = 2f * pi * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
  }

  fn randomVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    let r = pow(rngNextFloat(state), 0.33333f);
    let theta = pi * rngNextFloat(state);
    let phi = 2f * pi * rngNextFloat(state);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);

    return vec3(x, y, z);
  }

  @must_use
  fn randomUnitVec3(rngState: ptr<function, u32>) -> vec3f {
    return normalize(randomVec3InUnitSphere(rngState));
  }

  @must_use
  fn randomUnitVec3OnHemisphere(normal: vec3f, rngState: ptr<function, u32>) -> vec3f {
    let onUnitSphere = randomUnitVec3(rngState);
    return select(-onUnitSphere, onUnitSphere, dot(onUnitSphere, normal) > 0.0);
  }

  @must_use
  fn nearZero(v: vec3f) -> bool {
    let epsilon = vec3f(1e-8);
    return any(abs(v) < epsilon);
  }
);

static const char* vertex_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,
  }
);
// clang-format on
