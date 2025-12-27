#include "webgpu/wgpu_common.h"

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

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

/* Get the complete shader code for different shaders */
static const char* get_raytracer_shader(void);
static const char* get_present_shader(void);
static const char* get_debug_bvh_shader(void);

/* Cleanup shader strings */
static void free_shader_strings(void);

/* -------------------------------------------------------------------------- *
 * Bounding Volume Hierarchy.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/BV.ts
 * -------------------------------------------------------------------------- */

/* Forward declaration */
typedef struct face_t face_t;

/* Axis enumeration for splitting */
typedef enum {
  AXIS_X = 0,
  AXIS_Y = 1,
  AXIS_Z = 2,
} axis_t;

/* Bounding Volume structure */
typedef struct bv_t {
  vec4 min;      /* Minimum corner of AABB */
  vec4 max;      /* Maximum corner of AABB */
  int32_t lt;    /* Left child BV index */
  int32_t rt;    /* Right child BV index */
  int32_t fi[2]; /* Face indices */
} bv_t;

/* Face structure for BV subdivision */
struct face_t {
  vec3 p0, p1, p2; /* Triangle vertices */
  vec3 n0, n1, n2; /* Vertex normals */
  vec3 fn;         /* Face normal */
  int32_t fi;      /* Face index */
  int32_t mi;      /* Material index */
};

/* BV helper structure for dynamic array */
typedef struct {
  bv_t* data;
  uint32_t count;
  uint32_t capacity;
} bv_array_t;

/* BV constants */
#define BV_MIN_DELTA 0.01f

/* Comparison function for sorting faces along an axis */
static int compare_faces_x(const void* a, const void* b)
{
  const face_t* face_a = (const face_t*)a;
  const face_t* face_b = (const face_t*)b;
  float a_avg          = (face_a->p0[0] + face_a->p1[0] + face_a->p2[0]) / 3.0f;
  float b_avg          = (face_b->p0[0] + face_b->p1[0] + face_b->p2[0]) / 3.0f;
  if (a_avg < b_avg) {
    return -1;
  }
  if (a_avg > b_avg) {
    return 1;
  }
  return 0;
}

static int compare_faces_y(const void* a, const void* b)
{
  const face_t* face_a = (const face_t*)a;
  const face_t* face_b = (const face_t*)b;
  float a_avg          = (face_a->p0[1] + face_a->p1[1] + face_a->p2[1]) / 3.0f;
  float b_avg          = (face_b->p0[1] + face_b->p1[1] + face_b->p2[1]) / 3.0f;
  if (a_avg < b_avg) {
    return -1;
  }
  if (a_avg > b_avg) {
    return 1;
  }
  return 0;
}

static int compare_faces_z(const void* a, const void* b)
{
  const face_t* face_a = (const face_t*)a;
  const face_t* face_b = (const face_t*)b;
  float a_avg          = (face_a->p0[2] + face_a->p1[2] + face_a->p2[2]) / 3.0f;
  float b_avg          = (face_b->p0[2] + face_b->p1[2] + face_b->p2[2]) / 3.0f;
  if (a_avg < b_avg) {
    return -1;
  }
  if (a_avg > b_avg) {
    return 1;
  }
  return 0;
}

/* Split BV across the specified axis */
static void bv_split_across(bv_t* bv, axis_t axis, face_t* faces,
                            uint32_t face_count, bv_array_t* aabbs);

/* Initialize a bounding volume */
static void bv_init(bv_t* bv, vec4 min, vec4 max)
{
  glm_vec4_copy(min, bv->min);
  glm_vec4_copy(max, bv->max);
  bv->lt    = -1;
  bv->rt    = -1;
  bv->fi[0] = -1;
  bv->fi[1] = -1;
}

/* Initialize BV array */
static void bv_array_init(bv_array_t* array, uint32_t initial_capacity)
{
  array->data     = (bv_t*)malloc(initial_capacity * sizeof(bv_t));
  array->count    = 0;
  array->capacity = initial_capacity;
}

/* Add BV to array */
static void bv_array_push(bv_array_t* array, const bv_t* bv)
{
  if (array->count >= array->capacity) {
    array->capacity *= 2;
    array->data = (bv_t*)realloc(array->data, array->capacity * sizeof(bv_t));
  }
  array->data[array->count++] = *bv;
}

/* Free BV array */
static void bv_array_free(bv_array_t* array)
{
  free(array->data);
  array->data     = NULL;
  array->count    = 0;
  array->capacity = 0;
}

/* Subdivide BV recursively */
static void bv_subdivide(bv_t* bv, face_t* faces, uint32_t face_count,
                         bv_array_t* aabbs)
{
  if (face_count <= 2) {
    /* Base case: assign faces to this BV */
    for (uint32_t i = 0; i < face_count && i < 2; ++i) {
      bv->fi[i] = faces[i].fi;
    }
  }
  else {
    /* Find the largest axis delta */
    float dx            = fabsf(bv->max[0] - bv->min[0]);
    float dy            = fabsf(bv->max[1] - bv->min[1]);
    float dz            = fabsf(bv->max[2] - bv->min[2]);
    float largest_delta = fmaxf(dx, fmaxf(dy, dz));

    /* Split across the largest axis */
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

    bv_split_across(bv, split_axis, faces, face_count, aabbs);
  }
}

/* Split BV across the specified axis */
static void bv_split_across(bv_t* bv, axis_t axis, face_t* faces,
                            uint32_t face_count, bv_array_t* aabbs)
{
  /* Compute parent index within the dynamic array to avoid using the
     `bv` pointer after `bv_array_push()` may cause `realloc()` and move
     the backing storage (which would make `bv` dangling). */
  int32_t parent_index = -1;
  if (aabbs != NULL && aabbs->data != NULL) {
    ptrdiff_t d = bv - aabbs->data;
    if (d >= 0) {
      parent_index = (int32_t)d;
    }
  }

  /* Sort faces along the specified axis */
  switch (axis) {
    case AXIS_X:
      qsort(faces, face_count, sizeof(face_t), compare_faces_x);
      break;
    case AXIS_Y:
      qsort(faces, face_count, sizeof(face_t), compare_faces_y);
      break;
    case AXIS_Z:
      qsort(faces, face_count, sizeof(face_t), compare_faces_z);
      break;
  }

  /* Split faces into two halves */
  uint32_t half          = face_count / 2;
  face_t* lt_faces       = faces;
  uint32_t lt_face_count = half;
  face_t* rt_faces       = faces + half;
  uint32_t rt_face_count = face_count - half;

  /* Create left child BV */
  if (lt_face_count > 0) {
    vec4 lt_min = {FLT_MAX, FLT_MAX, FLT_MAX, 1.0f};
    vec4 lt_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f};

    for (uint32_t i = 0; i < lt_face_count; ++i) {
      face_t* face = &lt_faces[i];
      lt_min[0]
        = fminf(lt_min[0], fminf(face->p0[0], fminf(face->p1[0], face->p2[0])));
      lt_min[1]
        = fminf(lt_min[1], fminf(face->p0[1], fminf(face->p1[1], face->p2[1])));
      lt_min[2]
        = fminf(lt_min[2], fminf(face->p0[2], fminf(face->p1[2], face->p2[2])));
      lt_max[0]
        = fmaxf(lt_max[0], fmaxf(face->p0[0], fmaxf(face->p1[0], face->p2[0])));
      lt_max[1]
        = fmaxf(lt_max[1], fmaxf(face->p0[1], fmaxf(face->p1[1], face->p2[1])));
      lt_max[2]
        = fmaxf(lt_max[2], fmaxf(face->p0[2], fmaxf(face->p1[2], face->p2[2])));
    }

    /* Ensure minimum delta */
    if (lt_max[0] - lt_min[0] < BV_MIN_DELTA) {
      lt_max[0] += BV_MIN_DELTA;
    }
    if (lt_max[1] - lt_min[1] < BV_MIN_DELTA) {
      lt_max[1] += BV_MIN_DELTA;
    }
    if (lt_max[2] - lt_min[2] < BV_MIN_DELTA) {
      lt_max[2] += BV_MIN_DELTA;
    }

    bv_t lt_bv;
    bv_init(&lt_bv, lt_min, lt_max);
    if (parent_index >= 0) {
      aabbs->data[parent_index].lt = (int32_t)aabbs->count;
    }
    else {
      bv->lt = (int32_t)aabbs->count;
    }
    bv_array_push(aabbs, &lt_bv);
  }

  /* Create right child BV */
  if (rt_face_count > 0) {
    vec4 rt_min = {FLT_MAX, FLT_MAX, FLT_MAX, 1.0f};
    vec4 rt_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f};

    for (uint32_t i = 0; i < rt_face_count; ++i) {
      face_t* face = &rt_faces[i];
      rt_min[0]
        = fminf(rt_min[0], fminf(face->p0[0], fminf(face->p1[0], face->p2[0])));
      rt_min[1]
        = fminf(rt_min[1], fminf(face->p0[1], fminf(face->p1[1], face->p2[1])));
      rt_min[2]
        = fminf(rt_min[2], fminf(face->p0[2], fminf(face->p1[2], face->p2[2])));
      rt_max[0]
        = fmaxf(rt_max[0], fmaxf(face->p0[0], fmaxf(face->p1[0], face->p2[0])));
      rt_max[1]
        = fmaxf(rt_max[1], fmaxf(face->p0[1], fmaxf(face->p1[1], face->p2[1])));
      rt_max[2]
        = fmaxf(rt_max[2], fmaxf(face->p0[2], fmaxf(face->p1[2], face->p2[2])));
    }

    /* Ensure minimum delta */
    if (rt_max[0] - rt_min[0] < BV_MIN_DELTA) {
      rt_max[0] += BV_MIN_DELTA;
    }
    if (rt_max[1] - rt_min[1] < BV_MIN_DELTA) {
      rt_max[1] += BV_MIN_DELTA;
    }
    if (rt_max[2] - rt_min[2] < BV_MIN_DELTA) {
      rt_max[2] += BV_MIN_DELTA;
    }

    bv_t rt_bv;
    bv_init(&rt_bv, rt_min, rt_max);
    if (parent_index >= 0) {
      aabbs->data[parent_index].rt = (int32_t)aabbs->count;
    }
    else {
      bv->rt = (int32_t)aabbs->count;
    }
    bv_array_push(aabbs, &rt_bv);
  }

  /* Recursively subdivide children */
  if (parent_index >= 0) {
    int32_t lit = aabbs->data[parent_index].lt;
    int32_t rit = aabbs->data[parent_index].rt;
    if (lit != -1) {
      bv_subdivide(&aabbs->data[lit], lt_faces, lt_face_count, aabbs);
    }
    if (rit != -1) {
      bv_subdivide(&aabbs->data[rit], rt_faces, rt_face_count, aabbs);
    }
  }
  else {
    if (bv->lt != -1) {
      bv_subdivide(&aabbs->data[bv->lt], lt_faces, lt_face_count, aabbs);
    }
    if (bv->rt != -1) {
      bv_subdivide(&aabbs->data[bv->rt], rt_faces, rt_face_count, aabbs);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Camera.ts
 * -------------------------------------------------------------------------- */

/* Camera up vector */
static vec3 CAMERA_UP = {0.0f, 1.0f, 0.0f};

/* Camera state enumeration */
typedef enum {
  CAMERA_STATE_ROTATE,
  CAMERA_STATE_PAN,
} camera_state_t;

/* Damped action for smooth camera movement */
typedef struct {
  float value;
  float damping;
} damped_action_t;

/* Spherical coordinates for camera rotation */
typedef struct {
  float radius;
  float theta;
  float phi;
} spherical_t;

/* Raytracer camera structure */
typedef struct {
  /* Camera vectors and matrices */
  vec3 position;
  vec3 target;
  mat4 view_projection_matrix;
  mat4 view_matrix;
  mat4 projection_matrix;

  /* Viewport dimensions */
  uint32_t viewport_width;
  uint32_t viewport_height;

  /* Camera parameters */
  float vfov;         /* Vertical field of view in degrees */
  float aspect_ratio; /* Aspect ratio */
  float fov;          /* Field of view for calculations */

  /* Spherical coordinates (relative to target) */
  spherical_t spherical;

  /* Camera state */
  camera_state_t state;

  /* Mouse/touch tracking */
  struct {
    float x;
    float y;
  } rotate_delta, rotate_start, rotate_end;
  struct {
    float x;
    float y;
  } pan_start, pan_delta, pan_end;

  /* Touch dolly tracking */
  float touch_start_distance;
  float touch_end_distance;

  /* Constraints */
  float min_distance;
  float max_distance;
  float min_polar_angle;
  float max_polar_angle;

  /* Damped actions for smooth movement */
  damped_action_t target_x_damped;
  damped_action_t target_y_damped;
  damped_action_t target_z_damped;
  damped_action_t target_theta_damped;
  damped_action_t target_phi_damped;
  damped_action_t target_radius_damped;

  /* Sensitivity settings */
  float rotate_speed;
  float pan_speed;
  float zoom_speed;
} raytracer_camera_t;

/* Forward declarations of private functions */
static void damped_action_stop(damped_action_t* action);
static void update_spherical_from_position(raytracer_camera_t* camera);
static void update_damped_actions(raytracer_camera_t* camera);
static void update_camera_matrices(raytracer_camera_t* camera);
static void pan_camera(raytracer_camera_t* camera, float dx, float dy);

/* Damped action functions */
static void damped_action_init(damped_action_t* action)
{
  action->value   = 0.0f;
  action->damping = 0.5f;
}

static void damped_action_add_force(damped_action_t* action, float force)
{
  action->value += force;
}

static float damped_action_update(damped_action_t* action)
{
  bool is_active = (action->value * action->value) > 0.000001f;
  if (is_active) {
    action->value *= action->damping;
  }
  else {
    damped_action_stop(action);
  }
  return action->value;
}

static void damped_action_stop(damped_action_t* action)
{
  action->value = 0.0f;
}

/* Initialize camera */
static void raytracer_camera_init(raytracer_camera_t* camera, vec3 position,
                                  float vfov, float aspect_ratio,
                                  uint32_t viewport_width,
                                  uint32_t viewport_height)
{
  memset(camera, 0, sizeof(raytracer_camera_t));

  /* Set position and target */
  glm_vec3_copy(position, camera->position);
  glm_vec3_zero(camera->target);
  camera->target[2] = -1.0f;

  /* Set viewport dimensions */
  camera->viewport_width  = viewport_width;
  camera->viewport_height = viewport_height;

  /* Set camera parameters */
  camera->vfov         = vfov;
  camera->aspect_ratio = aspect_ratio;
  camera->fov          = 45.0f;

  /* Set constraints */
  camera->min_distance    = 0.1f;
  camera->max_distance    = 100.0f;
  camera->min_polar_angle = 0.0f;
  camera->max_polar_angle = GLM_PI;

  /* Set sensitivity */
  camera->rotate_speed = 1.0f;
  camera->pan_speed    = 0.5f;
  camera->zoom_speed   = 2.0f;

  /* Initialize damped actions */
  damped_action_init(&camera->target_x_damped);
  damped_action_init(&camera->target_y_damped);
  damped_action_init(&camera->target_z_damped);
  damped_action_init(&camera->target_theta_damped);
  damped_action_init(&camera->target_phi_damped);
  damped_action_init(&camera->target_radius_damped);

  /* Initialize spherical coordinates based on position */
  update_spherical_from_position(camera);

  /* Initialize matrices */
  glm_mat4_identity(camera->view_matrix);
  glm_mat4_identity(camera->projection_matrix);
  glm_mat4_identity(camera->view_projection_matrix);

  /* Update camera matrices */
  update_camera_matrices(camera);
}

/* Destroy camera */
static void raytracer_camera_destroy(raytracer_camera_t* camera)
{
  /* Nothing to cleanup for now */
  (void)camera;
}

/* Update spherical coordinates from position */
static void update_spherical_from_position(raytracer_camera_t* camera)
{
  vec3 offset;
  glm_vec3_sub(camera->position, camera->target, offset);

  float radius = glm_vec3_norm(offset);
  float theta  = atan2f(offset[0], offset[2]);
  float phi    = acosf(CLAMP(offset[1] / radius, -1.0f, 1.0f));

  camera->spherical.radius = radius;
  camera->spherical.theta  = theta;
  camera->spherical.phi    = phi;
}

/* Update damped actions */
static void update_damped_actions(raytracer_camera_t* camera)
{
  /* Update target position (for panning) */
  camera->target[0] += damped_action_update(&camera->target_x_damped);
  camera->target[1] += damped_action_update(&camera->target_y_damped);
  camera->target[2] += damped_action_update(&camera->target_z_damped);

  /* Update spherical coordinates (for rotation around target) */
  camera->spherical.theta += damped_action_update(&camera->target_theta_damped);
  camera->spherical.phi += damped_action_update(&camera->target_phi_damped);
  camera->spherical.radius
    += damped_action_update(&camera->target_radius_damped);

  /* Apply constraints */
  camera->spherical.radius = CLAMP(camera->spherical.radius,
                                   camera->min_distance, camera->max_distance);
  camera->spherical.phi = CLAMP(camera->spherical.phi, camera->min_polar_angle,
                                camera->max_polar_angle);
}

/* Update camera matrices */
static void update_camera_matrices(raytracer_camera_t* camera)
{
  /* Convert spherical coordinates to Cartesian position relative to target */
  spherical_t* s       = &camera->spherical;
  float sin_phi_radius = sinf(s->phi) * s->radius;

  camera->position[0] = sin_phi_radius * sinf(s->theta) + camera->target[0];
  camera->position[1] = cosf(s->phi) * s->radius + camera->target[1];
  camera->position[2] = sin_phi_radius * cosf(s->theta) + camera->target[2];

  /* Update view matrix */
  glm_lookat(camera->position, camera->target, CAMERA_UP, camera->view_matrix);

  /* Update projection matrix */
  glm_perspective(camera->fov, camera->aspect_ratio, 0.1f, 100.0f,
                  camera->projection_matrix);

  /* Update view-projection matrix */
  glm_mat4_mul(camera->projection_matrix, camera->view_matrix,
               camera->view_projection_matrix);
}

/* Pan camera */
static void pan_camera(raytracer_camera_t* camera, float dx, float dy)
{
  /* Calculate pan distance based on camera distance and field of view */
  float target_distance = glm_vec3_distance(camera->position, camera->target);
  float fov_radians     = (camera->fov * GLM_PI) / 180.0f;
  float scale           = (2.0f * target_distance * tanf(fov_radians * 0.5f))
                / (float)camera->viewport_height;

  /* Extract camera's right and up vectors from view matrix */
  vec3 right_vector = {camera->view_matrix[0][0], camera->view_matrix[1][0],
                       camera->view_matrix[2][0]};

  vec3 up_vector = {camera->view_matrix[0][1], camera->view_matrix[1][1],
                    camera->view_matrix[2][1]};

  /* Calculate pan offset */
  vec3 pan_offset;
  glm_vec3_scale(right_vector, -dx * scale, right_vector);
  glm_vec3_scale(up_vector, dy * scale, up_vector);
  glm_vec3_add(right_vector, up_vector, pan_offset);

  /* Apply pan to target using damped actions for smooth movement */
  damped_action_add_force(&camera->target_x_damped, pan_offset[0]);
  damped_action_add_force(&camera->target_y_damped, pan_offset[1]);
  damped_action_add_force(&camera->target_z_damped, pan_offset[2]);
}

/* Camera tick - update camera state */
static void raytracer_camera_tick(raytracer_camera_t* camera)
{
  update_damped_actions(camera);
  update_camera_matrices(camera);
}

/* Set camera target */
static void raytracer_camera_set_target(raytracer_camera_t* camera, float x,
                                        float y, float z)
{
  camera->target[0] = x;
  camera->target[1] = y;
  camera->target[2] = z;
}

/* Set camera position */
static void raytracer_camera_set_position(raytracer_camera_t* camera, float x,
                                          float y, float z)
{
  camera->position[0] = x;
  camera->position[1] = y;
  camera->position[2] = z;
  update_spherical_from_position(camera);
}

/* Set camera distance */
static void raytracer_camera_set_distance(raytracer_camera_t* camera,
                                          float distance)
{
  camera->spherical.radius
    = CLAMP(distance, camera->min_distance, camera->max_distance);
}

/* Get camera distance */
static float raytracer_camera_get_distance(const raytracer_camera_t* camera)
{
  return camera->spherical.radius;
}

/* Set camera constraints */
static void raytracer_camera_set_constraints(raytracer_camera_t* camera,
                                             float min_distance,
                                             float max_distance,
                                             float min_polar_angle,
                                             float max_polar_angle)
{
  camera->min_distance    = min_distance;
  camera->max_distance    = max_distance;
  camera->min_polar_angle = min_polar_angle;
  camera->max_polar_angle = max_polar_angle;
}

/* Mouse down event handler */
static void raytracer_camera_on_mouse_down(raytracer_camera_t* camera,
                                           int button, float x, float y)
{
  if (button == 0) {
    /* Left click - rotate */
    camera->state          = CAMERA_STATE_ROTATE;
    camera->rotate_start.x = x;
    camera->rotate_start.y = y;
  }
  else if (button == 1) {
    /* Right click - pan */
    camera->state       = CAMERA_STATE_PAN;
    camera->pan_start.x = x;
    camera->pan_start.y = y;
  }
}

/* Mouse up event handler */
static void raytracer_camera_on_mouse_up(raytracer_camera_t* camera)
{
  /* Reset deltas */
  (void)camera;
}

/* Mouse move event handler */
static void raytracer_camera_on_mouse_move(raytracer_camera_t* camera, float x,
                                           float y)
{
  if (camera->state == CAMERA_STATE_ROTATE) {
    camera->rotate_end.x = x;
    camera->rotate_end.y = y;

    camera->rotate_delta.x
      = (camera->rotate_end.x - camera->rotate_start.x) * camera->rotate_speed;
    camera->rotate_delta.y
      = (camera->rotate_end.y - camera->rotate_start.y) * camera->rotate_speed;

    /* Apply rotation around target using damped actions */
    damped_action_add_force(&camera->target_theta_damped,
                            -camera->rotate_delta.x
                              / (float)camera->viewport_width);
    damped_action_add_force(&camera->target_phi_damped,
                            -camera->rotate_delta.y
                              / (float)camera->viewport_height);

    camera->rotate_start.x = camera->rotate_end.x;
    camera->rotate_start.y = camera->rotate_end.y;
  }
  else if (camera->state == CAMERA_STATE_PAN) {
    camera->pan_end.x = x;
    camera->pan_end.y = y;

    camera->pan_delta.x
      = (camera->pan_end.x - camera->pan_start.x) * camera->pan_speed;
    camera->pan_delta.y
      = (camera->pan_end.y - camera->pan_start.y) * camera->pan_speed;

    pan_camera(camera, camera->pan_delta.x, camera->pan_delta.y);

    camera->pan_start.x = camera->pan_end.x;
    camera->pan_start.y = camera->pan_end.y;
  }
}

/* Mouse wheel event handler */
static void raytracer_camera_on_mouse_wheel(raytracer_camera_t* camera,
                                            float delta)
{
  float force
    = (delta > 0.0f) ? camera->zoom_speed * 0.1f : -camera->zoom_speed * 0.1f;
  damped_action_add_force(&camera->target_radius_damped, force);
}

/* -------------------------------------------------------------------------- *
 * Material
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Material.ts
 * -------------------------------------------------------------------------- */

/* Material types */
typedef enum {
  MATERIAL_TYPE_EMISSIVE   = 0,
  MATERIAL_TYPE_REFLECTIVE = 1,
  MATERIAL_TYPE_DIELECTRIC = 2,
  MATERIAL_TYPE_LAMBERTIAN = 3,
} material_type_t;

/* Material structure */
typedef struct {
  float albedo[4];          /* RGBA color */
  material_type_t mtl_type; /* Material type */
  float reflection_ratio;   /* Reflection ratio for reflective materials */
  float reflection_gloss;   /* Reflection glossiness */
  float refraction_index;   /* Index of refraction for dielectric materials */
} material_t;

/* Initialize a material with default values */
void material_init(material_t* material, float r, float g, float b, float a,
                   material_type_t type)
{
  material->albedo[0]        = r;
  material->albedo[1]        = g;
  material->albedo[2]        = b;
  material->albedo[3]        = a;
  material->mtl_type         = type;
  material->reflection_ratio = 0.0f;
  material->reflection_gloss = 1.0f;
  material->refraction_index = 1.0f;
}

/* Set material albedo color */
void material_set_albedo(material_t* material, float r, float g, float b,
                         float a)
{
  material->albedo[0] = r;
  material->albedo[1] = g;
  material->albedo[2] = b;
  material->albedo[3] = a;
}

/* Set material type */
void material_set_type(material_t* material, material_type_t type)
{
  material->mtl_type = type;
}

/* Set reflection ratio */
void material_set_reflection_ratio(material_t* material, float ratio)
{
  material->reflection_ratio = ratio;
}

/* Set reflection glossiness */
void material_set_reflection_gloss(material_t* material, float gloss)
{
  material->reflection_gloss = gloss;
}

/* Set refraction index */
void material_set_refraction_index(material_t* material, float index)
{
  material->refraction_index = index;
}

/* -------------------------------------------------------------------------- *
 * Scene
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-raytracer/blob/main/src/Scene.ts
 * -------------------------------------------------------------------------- */

/* Scene constants - based on analysis of raytraced-scene.obj */
#define SCENE_MAX_VERTICES 12000
#define SCENE_MAX_NORMALS 12000
#define SCENE_MAX_FACES 23000
#define SCENE_MAX_MATERIALS 10
#define SCENE_MAX_MODELS 10

/* Model structure */
typedef struct {
  char name[64];
  face_t* faces;
  uint32_t face_count;
  bv_t* aabbs;
  uint32_t aabb_count;
} model_t;

/* Scene structure */
typedef struct {
  WGPUDevice device;

  /* Scene data */
  model_t* models;
  uint32_t model_count;
  material_t* materials;
  uint32_t material_count;

  /* GPU buffers */
  WGPUBuffer faces_buffer;
  WGPUBuffer aabbs_buffer;
  WGPUBuffer materials_buffer;

  /* Scene statistics */
  uint32_t max_num_bvs_per_mesh;
  uint32_t max_num_faces_per_mesh;
  uint32_t suzanne_material_idx;
} scene_t;

/* Helper structures for parsing */
typedef struct {
  float x, y, z;
} vertex_3f_t;

typedef struct {
  int vertex_index;
  int texture_coords_index;
  int vertex_normal_index;
} face_vertex_t;

typedef struct {
  char material[64];
  face_vertex_t vertices[3];
} obj_face_t;

typedef struct {
  char name[64];
  vertex_3f_t* vertices;
  uint32_t vertex_count;
  vertex_3f_t* vertex_normals;
  uint32_t vertex_normal_count;
  obj_face_t* faces;
  uint32_t face_count;
} obj_model_t;

typedef struct {
  char name[64];
  float kd_red;
  float kd_green;
  float kd_blue;
} mtl_material_t;

/* Forward declarations */
static int parse_obj_file(const char* path, obj_model_t** models,
                          uint32_t* model_count);
static int parse_mtl_file(const char* path, mtl_material_t** materials,
                          uint32_t* material_count);
static void convert_obj_to_scene(obj_model_t* obj_models, uint32_t obj_count,
                                 mtl_material_t* mtl_materials,
                                 uint32_t mtl_count, scene_t* scene);
static void create_gpu_buffers(scene_t* scene);
static void free_obj_models(obj_model_t* models, uint32_t count);
static void free_mtl_materials(mtl_material_t* materials, uint32_t count);

/* Initialize scene */
void scene_init(scene_t* scene, WGPUDevice device)
{
  memset(scene, 0, sizeof(scene_t));
  scene->device = device;
}

/* Load models from OBJ and MTL files */
int scene_load_models(scene_t* scene, const char* obj_path,
                      const char* mtl_path)
{
  obj_model_t* obj_models       = NULL;
  uint32_t obj_model_count      = 0;
  mtl_material_t* mtl_materials = NULL;
  uint32_t mtl_material_count   = 0;

  /* Parse OBJ file */
  if (parse_obj_file(obj_path, &obj_models, &obj_model_count) != 0) {
    return -1;
  }

  /* Parse MTL file */
  if (parse_mtl_file(mtl_path, &mtl_materials, &mtl_material_count) != 0) {
    free_obj_models(obj_models, obj_model_count);
    return -1;
  }

  /* Convert to scene format */
  convert_obj_to_scene(obj_models, obj_model_count, mtl_materials,
                       mtl_material_count, scene);

  /* Create GPU buffers */
  create_gpu_buffers(scene);

  /* Cleanup temporary data */
  free_obj_models(obj_models, obj_model_count);
  free_mtl_materials(mtl_materials, mtl_material_count);

  return 0;
}

/* Parse OBJ file - simplified for raytraced-scene.obj */
static int parse_obj_file(const char* path, obj_model_t** models,
                          uint32_t* model_count)
{
  FILE* file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "Failed to open OBJ file: %s\n", path);
    return -1;
  }

  /* Allocate temporary storage */
  vertex_3f_t* vertices
    = (vertex_3f_t*)malloc(SCENE_MAX_VERTICES * sizeof(vertex_3f_t));
  vertex_3f_t* normals
    = (vertex_3f_t*)malloc(SCENE_MAX_NORMALS * sizeof(vertex_3f_t));

  uint32_t vertex_count = 0;
  uint32_t normal_count = 0;

  /* Allocate models array */
  *models      = (obj_model_t*)calloc(SCENE_MAX_MODELS, sizeof(obj_model_t));
  *model_count = 0;

  obj_model_t* current_model = NULL;
  char current_material[64]  = {0};
  char line[256];

  while (fgets(line, sizeof(line), file)) {
    if (line[0] == 'o' && line[1] == ' ') {
      /* New object */
      if (*model_count >= SCENE_MAX_MODELS) {
        break;
      }
      current_model = &(*models)[*model_count];
      (*model_count)++;
      sscanf(line, "o %63s", current_model->name);
      current_model->vertices       = vertices;
      current_model->vertex_normals = normals;
      current_model->faces
        = (obj_face_t*)malloc(SCENE_MAX_FACES * sizeof(obj_face_t));
      current_model->face_count = 0;
    }
    else if (line[0] == 'v' && line[1] == ' ') {
      /* Vertex position */
      if (vertex_count < SCENE_MAX_VERTICES) {
        sscanf(line, "v %f %f %f", &vertices[vertex_count].x,
               &vertices[vertex_count].y, &vertices[vertex_count].z);
        vertex_count++;
      }
    }
    else if (line[0] == 'v' && line[1] == 'n') {
      /* Vertex normal */
      if (normal_count < SCENE_MAX_NORMALS) {
        sscanf(line, "vn %f %f %f", &normals[normal_count].x,
               &normals[normal_count].y, &normals[normal_count].z);
        normal_count++;
      }
    }
    else if (strncmp(line, "usemtl", 6) == 0) {
      /* Material */
      sscanf(line, "usemtl %63s", current_material);
    }
    else if (line[0] == 'f' && line[1] == ' ' && current_model) {
      /* Face */
      if (current_model->face_count < SCENE_MAX_FACES) {
        obj_face_t* face = &current_model->faces[current_model->face_count];
        strncpy(face->material, current_material, sizeof(face->material) - 1);

        /* Robust face parser: support "v", "v//vn" and "v/vt/vn" formats */
        {
          /* Make a mutable copy (strtok modifies the string) */
          char buf[256];
          strncpy(buf, line + 2, sizeof(buf) - 1); /* skip leading "f " */
          buf[sizeof(buf) - 1] = '\0';

          int v_idx[3] = {0, 0, 0};
          int n_idx[3] = {0, 0, 0};

          char* tok  = strtok(buf, " \t\r\n");
          int vi     = 0;
          int vcount = 0;
          while (tok && vcount < 3) {
            if (strstr(tok, "//") != NULL) {
              /* v//vn */
              sscanf(tok, "%d//%d", &vi, &n_idx[vcount]);
              v_idx[vcount] = vi;
            }
            else if (strchr(tok, '/') != NULL) {
              /* v/vt/vn or v/vt */
              int a = 0, b = 0, c = 0;
              int matched = sscanf(tok, "%d/%d/%d", &a, &b, &c);
              if (matched == 3) {
                v_idx[vcount] = a;
                n_idx[vcount] = c;
              }
              else {
                /* maybe v/vt */
                matched = sscanf(tok, "%d/%d", &a, &b);
                if (matched >= 1) {
                  v_idx[vcount] = a;
                }
              }
            }
            else {
              /* just vertex index */
              sscanf(tok, "%d", &vi);
              v_idx[vcount] = vi;
            }

            vcount++;
            tok = strtok(NULL, " \t\r\n");
          }

          /* Assign parsed indices to face structure (OBJ is 1-indexed) */
          for (int k = 0; k < 3; ++k) {
            face->vertices[k].vertex_index        = v_idx[k];
            face->vertices[k].vertex_normal_index = n_idx[k];
          }
        }

        current_model->face_count++;
      }
    }
  }

  /* Set vertex and normal counts for all models */
  for (uint32_t i = 0; i < *model_count; ++i) {
    (*models)[i].vertex_count        = vertex_count;
    (*models)[i].vertex_normal_count = normal_count;
  }

  fclose(file);
  return 0;
}

/* Parse MTL file */
static int parse_mtl_file(const char* path, mtl_material_t** materials,
                          uint32_t* material_count)
{
  FILE* file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "Failed to open MTL file: %s\n", path);
    return -1;
  }

  *materials
    = (mtl_material_t*)calloc(SCENE_MAX_MATERIALS, sizeof(mtl_material_t));
  *material_count = 0;

  mtl_material_t* current_material = NULL;
  char line[256];

  while (fgets(line, sizeof(line), file)) {
    if (strncmp(line, "newmtl", 6) == 0) {
      if (*material_count >= SCENE_MAX_MATERIALS) {
        break;
      }
      current_material = &(*materials)[*material_count];
      (*material_count)++;
      sscanf(line, "newmtl %63s", current_material->name);
    }
    else if (strncmp(line, "Kd", 2) == 0 && current_material) {
      sscanf(line, "Kd %f %f %f", &current_material->kd_red,
             &current_material->kd_green, &current_material->kd_blue);
    }
  }

  fclose(file);
  return 0;
}

/* Convert OBJ models to scene format */
static void convert_obj_to_scene(obj_model_t* obj_models, uint32_t obj_count,
                                 mtl_material_t* mtl_materials,
                                 uint32_t mtl_count, scene_t* scene)
{
  /* Allocate scene data */
  scene->models         = (model_t*)calloc(obj_count, sizeof(model_t));
  scene->model_count    = obj_count;
  scene->materials      = (material_t*)calloc(mtl_count, sizeof(material_t));
  scene->material_count = mtl_count;

  /* Convert materials */
  for (uint32_t i = 0; i < mtl_count; ++i) {
    mtl_material_t* mtl = &mtl_materials[i];
    material_t* mat     = &scene->materials[i];

    material_init(mat, mtl->kd_red, mtl->kd_green, mtl->kd_blue, 1.0f,
                  MATERIAL_TYPE_LAMBERTIAN);

    /* Apply material-specific properties based on name */
    if (strcmp(mtl->name, "Light") == 0) {
      material_set_type(mat, MATERIAL_TYPE_EMISSIVE);
      material_set_albedo(mat, 6.0f, 6.0f, 6.0f, 1.0f);
    }
    else if (strcmp(mtl->name, "Floor") == 0) {
      material_set_type(mat, MATERIAL_TYPE_REFLECTIVE);
      material_set_reflection_ratio(mat, 0.2f);
      material_set_reflection_gloss(mat, 0.4f);
    }
    else if (strcmp(mtl->name, "Teapot") == 0) {
      material_set_type(mat, MATERIAL_TYPE_REFLECTIVE);
      material_set_reflection_ratio(mat, 1.0f);
      material_set_reflection_gloss(mat, 0.4f);
      material_set_refraction_index(mat, 1.523f);
    }
    else if (strcmp(mtl->name, "Suzanne") == 0) {
      material_set_type(mat, MATERIAL_TYPE_REFLECTIVE);
      material_set_reflection_ratio(mat, 0.1f);
      material_set_reflection_gloss(mat, 1.0f);
      material_set_refraction_index(mat, 0.1f);
      scene->suzanne_material_idx = i;
    }
    else if (strcmp(mtl->name, "Dodecahedron") == 0) {
      material_set_type(mat, MATERIAL_TYPE_DIELECTRIC);
      material_set_refraction_index(mat, 1.52f);
    }
  }

  /* Convert models */
  vec3 p1p0_diff, p2p0_diff, fn;

  for (uint32_t m = 0; m < obj_count; ++m) {
    obj_model_t* obj = &obj_models[m];
    model_t* model   = &scene->models[m];

    strncpy(model->name, obj->name, sizeof(model->name) - 1);
    model->faces      = (face_t*)malloc(obj->face_count * sizeof(face_t));
    model->face_count = obj->face_count;

    /* Convert faces */
    for (uint32_t f = 0; f < obj->face_count; ++f) {
      obj_face_t* obj_face = &obj->faces[f];
      face_t* face         = &model->faces[f];

      /* Get vertex indices (OBJ is 1-indexed) */
      int i0 = obj_face->vertices[0].vertex_index - 1;
      int i1 = obj_face->vertices[1].vertex_index - 1;
      int i2 = obj_face->vertices[2].vertex_index - 1;

      /* Set positions */
      face->p0[0] = obj->vertices[i0].x;
      face->p0[1] = obj->vertices[i0].y;
      face->p0[2] = obj->vertices[i0].z;
      face->p1[0] = obj->vertices[i1].x;
      face->p1[1] = obj->vertices[i1].y;
      face->p1[2] = obj->vertices[i1].z;
      face->p2[0] = obj->vertices[i2].x;
      face->p2[1] = obj->vertices[i2].y;
      face->p2[2] = obj->vertices[i2].z;

      /* Get normal indices */
      int j0 = obj_face->vertices[0].vertex_normal_index - 1;
      int j1 = obj_face->vertices[1].vertex_normal_index - 1;
      int j2 = obj_face->vertices[2].vertex_normal_index - 1;

      /* Set normals */
      face->n0[0] = obj->vertex_normals[j0].x;
      face->n0[1] = obj->vertex_normals[j0].y;
      face->n0[2] = obj->vertex_normals[j0].z;
      face->n1[0] = obj->vertex_normals[j1].x;
      face->n1[1] = obj->vertex_normals[j1].y;
      face->n1[2] = obj->vertex_normals[j1].z;
      face->n2[0] = obj->vertex_normals[j2].x;
      face->n2[1] = obj->vertex_normals[j2].y;
      face->n2[2] = obj->vertex_normals[j2].z;

      /* Calculate face normal */
      glm_vec3_sub(face->p1, face->p0, p1p0_diff);
      glm_vec3_sub(face->p2, face->p0, p2p0_diff);
      glm_vec3_cross(p1p0_diff, p2p0_diff, fn);
      glm_vec3_normalize(fn);
      glm_vec3_copy(fn, face->fn);

      /* Set face index */
      face->fi = (int32_t)f;

      /* Find material index */
      face->mi = -1;
      for (uint32_t mi = 0; mi < mtl_count; ++mi) {
        if (strcmp(mtl_materials[mi].name, obj_face->material) == 0) {
          face->mi = (int32_t)mi;
          break;
        }
      }
    }

    /* Build BVH for this model */
    bv_array_t aabbs;
    bv_array_init(&aabbs, 1024);

    /* Calculate root AABB */
    vec4 min = {FLT_MAX, FLT_MAX, FLT_MAX, 1.0f};
    vec4 max = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f};

    for (uint32_t f = 0; f < model->face_count; ++f) {
      face_t* face = &model->faces[f];
      min[0]
        = fminf(min[0], fminf(face->p0[0], fminf(face->p1[0], face->p2[0])));
      min[1]
        = fminf(min[1], fminf(face->p0[1], fminf(face->p1[1], face->p2[1])));
      min[2]
        = fminf(min[2], fminf(face->p0[2], fminf(face->p1[2], face->p2[2])));
      max[0]
        = fmaxf(max[0], fmaxf(face->p0[0], fmaxf(face->p1[0], face->p2[0])));
      max[1]
        = fmaxf(max[1], fmaxf(face->p0[1], fmaxf(face->p1[1], face->p2[1])));
      max[2]
        = fmaxf(max[2], fmaxf(face->p0[2], fmaxf(face->p1[2], face->p2[2])));
    }

    /* Ensure minimum delta */
    if (max[0] - min[0] < BV_MIN_DELTA)
      max[0] += BV_MIN_DELTA;
    if (max[1] - min[1] < BV_MIN_DELTA)
      max[1] += BV_MIN_DELTA;
    if (max[2] - min[2] < BV_MIN_DELTA)
      max[2] += BV_MIN_DELTA;

    /* Create root BV */
    bv_t root_bv;
    bv_init(&root_bv, min, max);
    bv_array_push(&aabbs, &root_bv);

    /* Subdivide */
    bv_subdivide(&aabbs.data[0], model->faces, model->face_count, &aabbs);

    /* Copy AABBs to model */
    model->aabb_count = aabbs.count;
    model->aabbs      = (bv_t*)malloc(aabbs.count * sizeof(bv_t));
    memcpy(model->aabbs, aabbs.data, aabbs.count * sizeof(bv_t));

    /* Update scene statistics */
    if (model->face_count > scene->max_num_faces_per_mesh) {
      scene->max_num_faces_per_mesh = model->face_count;
    }
    if (model->aabb_count > scene->max_num_bvs_per_mesh) {
      scene->max_num_bvs_per_mesh = model->aabb_count;
    }

    bv_array_free(&aabbs);
  }
}

/* Create GPU buffers */
static void create_gpu_buffers(scene_t* scene)
{
  const uint32_t num_floats_per_face     = 28;
  const uint32_t num_floats_per_bv       = 12;
  const uint32_t num_floats_per_material = 8;

  /* Create faces buffer */
  {
    uint64_t buffer_size = (uint64_t)num_floats_per_face * sizeof(float)
                           * scene->max_num_faces_per_mesh * scene->model_count;

    scene->faces_buffer = wgpuDeviceCreateBuffer(
      scene->device, &(WGPUBufferDescriptor){
                       .label            = STRVIEW("Raytracer Faces Buffer"),
                       .usage            = WGPUBufferUsage_Storage,
                       .size             = buffer_size,
                       .mappedAtCreation = true,
                     });

    void* mapped
      = wgpuBufferGetMappedRange(scene->faces_buffer, 0, buffer_size);
    float* face_data        = (float*)mapped;
    uint32_t* face_data_u32 = (uint32_t*)mapped;

    for (uint32_t m = 0; m < scene->model_count; ++m) {
      model_t* model = &scene->models[m];
      uint64_t idx
        = (uint64_t)m * num_floats_per_face * scene->max_num_faces_per_mesh;

      for (uint32_t f = 0; f < model->face_count; ++f) {
        face_t* face = &model->faces[f];

        face_data[idx + 0]      = face->p0[0];
        face_data[idx + 1]      = face->p0[1];
        face_data[idx + 2]      = face->p0[2];
        face_data[idx + 4]      = face->p1[0];
        face_data[idx + 5]      = face->p1[1];
        face_data[idx + 6]      = face->p1[2];
        face_data[idx + 8]      = face->p2[0];
        face_data[idx + 9]      = face->p2[1];
        face_data[idx + 10]     = face->p2[2];
        face_data[idx + 12]     = face->n0[0];
        face_data[idx + 13]     = face->n0[1];
        face_data[idx + 14]     = face->n0[2];
        face_data[idx + 16]     = face->n1[0];
        face_data[idx + 17]     = face->n1[1];
        face_data[idx + 18]     = face->n1[2];
        face_data[idx + 20]     = face->n2[0];
        face_data[idx + 21]     = face->n2[1];
        face_data[idx + 22]     = face->n2[2];
        face_data[idx + 24]     = face->fn[0];
        face_data[idx + 25]     = face->fn[1];
        face_data[idx + 26]     = face->fn[2];
        face_data_u32[idx + 27] = (uint32_t)face->mi;

        idx += num_floats_per_face;
      }
    }

    wgpuBufferUnmap(scene->faces_buffer);
  }

  /* Create AABBs buffer */
  {
    uint64_t buffer_size = (uint64_t)num_floats_per_bv * sizeof(float)
                           * scene->max_num_bvs_per_mesh * scene->model_count;

    scene->aabbs_buffer = wgpuDeviceCreateBuffer(
      scene->device, &(WGPUBufferDescriptor){
                       .label            = STRVIEW("Raytracer AABBs Buffer"),
                       .usage            = WGPUBufferUsage_Storage,
                       .size             = buffer_size,
                       .mappedAtCreation = true,
                     });

    void* mapped
      = wgpuBufferGetMappedRange(scene->aabbs_buffer, 0, buffer_size);
    float* aabb_data       = (float*)mapped;
    int32_t* aabb_data_i32 = (int32_t*)mapped;

    for (uint32_t m = 0; m < scene->model_count; ++m) {
      model_t* model = &scene->models[m];
      uint64_t idx
        = (uint64_t)num_floats_per_bv * scene->max_num_bvs_per_mesh * m;

      for (uint32_t a = 0; a < model->aabb_count; ++a) {
        bv_t* aabb = &model->aabbs[a];

        aabb_data[idx + 0]      = aabb->min[0];
        aabb_data[idx + 1]      = aabb->min[1];
        aabb_data[idx + 2]      = aabb->min[2];
        aabb_data[idx + 3]      = 1.0f;
        aabb_data[idx + 4]      = aabb->max[0];
        aabb_data[idx + 5]      = aabb->max[1];
        aabb_data[idx + 6]      = aabb->max[2];
        aabb_data_i32[idx + 7]  = aabb->lt;
        aabb_data_i32[idx + 8]  = aabb->rt;
        aabb_data_i32[idx + 9]  = aabb->fi[0];
        aabb_data_i32[idx + 10] = aabb->fi[1];
        aabb_data_i32[idx + 11] = 0;

        idx += num_floats_per_bv;
      }
    }

    wgpuBufferUnmap(scene->aabbs_buffer);
  }

  /* Create materials buffer */
  {
    uint64_t buffer_size = (uint64_t)num_floats_per_material * sizeof(float)
                           * scene->material_count;

    scene->materials_buffer = wgpuDeviceCreateBuffer(
      scene->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("Raytracer Materials Buffer"),
        .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size             = buffer_size,
        .mappedAtCreation = true,
      });

    void* mapped
      = wgpuBufferGetMappedRange(scene->materials_buffer, 0, buffer_size);
    float* material_data        = (float*)mapped;
    uint32_t* material_data_u32 = (uint32_t*)mapped;

    for (uint32_t i = 0; i < scene->material_count; ++i) {
      material_t* mat = &scene->materials[i];
      uint64_t idx    = (uint64_t)i * num_floats_per_material;

      material_data_u32[idx + 0] = (uint32_t)mat->mtl_type;
      material_data[idx + 1]     = mat->reflection_ratio;
      material_data[idx + 2]     = mat->reflection_gloss;
      material_data[idx + 3]     = mat->refraction_index;
      material_data[idx + 4]     = mat->albedo[0];
      material_data[idx + 5]     = mat->albedo[1];
      material_data[idx + 6]     = mat->albedo[2];
    }

    wgpuBufferUnmap(scene->materials_buffer);
  }
}

/* Set Suzanne material to glass or reflective */
void scene_set_suzanne_glass(scene_t* scene, bool is_glass)
{
  const uint32_t num_floats_per_material = 8;
  material_t* mtl = &scene->materials[scene->suzanne_material_idx];

  float buffer_data[8];
  uint32_t* buffer_data_u32 = (uint32_t*)buffer_data;

  buffer_data_u32[0]
    = is_glass ? MATERIAL_TYPE_DIELECTRIC : MATERIAL_TYPE_REFLECTIVE;
  buffer_data[1] = mtl->reflection_ratio;
  buffer_data[2] = mtl->reflection_gloss;
  buffer_data[3] = mtl->refraction_index;
  buffer_data[4] = is_glass ? 1.0f : mtl->albedo[0];
  buffer_data[5] = is_glass ? 1.0f : mtl->albedo[1];
  buffer_data[6] = is_glass ? 1.0f : mtl->albedo[2];

  wgpuQueueWriteBuffer(wgpuDeviceGetQueue(scene->device),
                       scene->materials_buffer,
                       (uint64_t)scene->suzanne_material_idx
                         * num_floats_per_material * sizeof(float),
                       buffer_data, num_floats_per_material * sizeof(float));
}

/* Cleanup scene */
void scene_destroy(scene_t* scene)
{
  /* Release GPU buffers */
  if (scene->faces_buffer) {
    wgpuBufferRelease(scene->faces_buffer);
  }
  if (scene->aabbs_buffer) {
    wgpuBufferRelease(scene->aabbs_buffer);
  }
  if (scene->materials_buffer) {
    wgpuBufferRelease(scene->materials_buffer);
  }

  /* Free models */
  if (scene->models) {
    for (uint32_t i = 0; i < scene->model_count; ++i) {
      free(scene->models[i].faces);
      free(scene->models[i].aabbs);
    }
    free(scene->models);
  }

  /* Free materials */
  if (scene->materials) {
    free(scene->materials);
  }

  memset(scene, 0, sizeof(scene_t));
}

/* Free OBJ models */
static void free_obj_models(obj_model_t* models, uint32_t count)
{
  if (models) {
    /* Free shared vertices and normals (only once) */
    if (count > 0 && models[0].vertices) {
      free(models[0].vertices);
    }
    if (count > 0 && models[0].vertex_normals) {
      free(models[0].vertex_normals);
    }

    /* Free faces for each model */
    for (uint32_t i = 0; i < count; ++i) {
      free(models[i].faces);
    }
    free(models);
  }
}

/* Free MTL materials */
static void free_mtl_materials(mtl_material_t* materials, uint32_t count)
{
  (void)count;
  if (materials) {
    free(materials);
  }
}

/* -------------------------------------------------------------------------- *
 * GPU Raytracer example
 * -------------------------------------------------------------------------- */

/* Compute workgroup sizes */
#define COMPUTE_WORKGROUP_SIZE_X 16
#define COMPUTE_WORKGROUP_SIZE_Y 16
#define MAX_BOUNCES_INTERACTING 1

/* Pad a size up to a multiple of 16 bytes (required for uniform buffers) */
#define PAD_TO_16(x) ((((x) + 15) / 16) * 16)

/* Uniforms structures matching WGSL */
typedef struct {
  uint32_t seed[3];
  uint32_t frame_counter;
  uint32_t max_bounces;
  uint32_t flat_shading;
  uint32_t debug_normals;
} common_uniforms_t;

typedef struct {
  uint32_t viewport_size[2];
  float image_width;
  float image_height;
  float pixel00_loc[3];
  float _padding0;
  float pixel_delta_u[3];
  float _padding1;
  float pixel_delta_v[3];
  float aspect_ratio;
  float center[3];
  float vfov;
  float look_from[3];
  float _padding2;
  float look_at[3];
  float _padding3;
  float vup[3];
  float defocus_angle;
  float focus_dist;
  float _padding4;
  float _padding5;
  float _padding6;
  float defocus_disc_u[3];
  float _padding7;
  float defocus_disc_v[3];
  float _padding8;
} camera_uniforms_t;

/* State struct */
static struct {
  /* Scene and camera */
  scene_t scene;
  raytracer_camera_t camera;

  /* Buffers */
  WGPUBuffer raytraced_storage_buffer;
  WGPUBuffer rng_state_buffer;
  WGPUBuffer common_uniforms_buffer;
  WGPUBuffer camera_uniform_buffer;
  WGPUBuffer camera_view_proj_matrix_buffer;

  /* Pipelines */
  WGPUComputePipeline compute_pipeline;
  WGPURenderPipeline blit_to_screen_pipeline;
  WGPURenderPipeline debug_bvh_pipeline;

  /* Bind groups */
  WGPUBindGroup compute_bind_group_0;
  WGPUBindGroup compute_bind_group_1;
  WGPUBindGroup blit_to_screen_bind_group;
  WGPUBindGroup debug_bvh_bind_group;

  /* Bind group layouts */
  WGPUBindGroupLayout compute_bind_group_0_layout;
  WGPUBindGroupLayout compute_bind_group_1_layout;
  WGPUBindGroupLayout blit_to_screen_bind_group_layout;
  WGPUBindGroupLayout debug_bvh_bind_group_layout;

  /* Pipeline layouts */
  WGPUPipelineLayout compute_pipeline_layout;
  WGPUPipelineLayout blit_to_screen_pipeline_layout;
  WGPUPipelineLayout debug_bvh_pipeline_layout;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Settings */
  uint32_t max_samples;
  uint32_t max_bounces;
  bool debug_bvh;
  bool debug_normals;
  bool use_phong_shading;
  bool crystal_suzanne;

  /* Frame state */
  uint32_t frame_counter;
  float time_expired_ms;
  float old_time_ms;
  float shader_seed[3];

  /* Mouse state */
  bool mouse_left_down;
  bool mouse_right_down;

  bool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .max_samples        = 5000,
  .max_bounces        = 4,
  .debug_bvh          = false,
  .debug_normals      = false,
  .use_phong_shading  = true,
  .crystal_suzanne    = false,
  .frame_counter      = 0,
  .time_expired_ms    = 0.0f,
  .old_time_ms        = 0.0f,
  .mouse_left_down    = false,
  .mouse_right_down   = false,
};

/* Reset render state */
static void reset_render(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  state.frame_counter   = 0;
  state.old_time_ms     = (float)stm_sec(stm_now()) * 1000.0f;
  state.time_expired_ms = 0.0f;
}

/* Create buffers */
static void create_buffers(wgpu_context_t* wgpu_context)
{
  /* Raytraced storage buffer */
  {
    uint64_t buffer_size
      = sizeof(float) * 4 * wgpu_context->width * wgpu_context->height;
    state.raytraced_storage_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device, &(WGPUBufferDescriptor){
                              .label = STRVIEW("Raytraced Image Buffer"),
                              .usage = WGPUBufferUsage_Storage,
                              .size  = buffer_size,
                            });
  }

  /* RNG state buffer */
  {
    uint64_t buffer_size
      = sizeof(uint32_t) * wgpu_context->width * wgpu_context->height;
    state.rng_state_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device, &(WGPUBufferDescriptor){
                              .label            = STRVIEW("RNG State Buffer"),
                              .usage            = WGPUBufferUsage_Storage,
                              .size             = buffer_size,
                              .mappedAtCreation = true,
                            });

    /* Initialize RNG state */
    uint32_t* rng_state = (uint32_t*)wgpuBufferGetMappedRange(
      state.rng_state_buffer, 0, buffer_size);
    for (int32_t i = 0; i < wgpu_context->width * wgpu_context->height; ++i) {
      rng_state[i] = i;
    }
    wgpuBufferUnmap(state.rng_state_buffer);
  }

  /* Common uniforms buffer */
  {
    state.common_uniforms_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Common Uniforms Buffer"),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = PAD_TO_16(sizeof(common_uniforms_t)),
      });
  }

  /* Camera uniforms buffer */
  {
    state.camera_uniform_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Camera Uniforms Buffer"),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = sizeof(camera_uniforms_t),
      });
  }

  /* Camera view-projection matrix buffer */
  {
    state.camera_view_proj_matrix_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label = STRVIEW("Camera ViewProjection Matrix"),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = 16 * sizeof(float),
      });
  }
}

/* Create compute pipeline */
static void create_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Create bind group layouts */
  {
    /* Bind group 0 layout */
    WGPUBindGroupLayoutEntry bg0_entries[4] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_Storage,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_Storage,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
      [3] = {
        .binding    = 3,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };

    state.compute_bind_group_0_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Compute Bind Group 0 Layout"),
                              .entryCount = 4,
                              .entries    = bg0_entries,
                            });

    /* Bind group 1 layout */
    WGPUBindGroupLayoutEntry bg1_entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
    };

    state.compute_bind_group_1_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Compute Bind Group 1 Layout"),
                              .entryCount = 3,
                              .entries    = bg1_entries,
                            });
  }

  /* Create pipeline layout */
  {
    WGPUBindGroupLayout layouts[2] = {
      state.compute_bind_group_0_layout,
      state.compute_bind_group_1_layout,
    };

    state.compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("Compute Pipeline Layout"),
                              .bindGroupLayoutCount = 2,
                              .bindGroupLayouts     = layouts,
                            });
  }

  /* Create shader module */
  WGPUShaderModule compute_shader_module
    = wgpu_create_shader_module(wgpu_context->device, get_raytracer_shader());

  /* Create compute pipeline */
  {
    WGPUConstantEntry constants[5] = {
      [0] = {
        .key   = STRVIEW("WORKGROUP_SIZE_X"),
        .value = COMPUTE_WORKGROUP_SIZE_X,
      },
      [1] = {
        .key   = STRVIEW("WORKGROUP_SIZE_Y"),
        .value = COMPUTE_WORKGROUP_SIZE_Y,
      },
      [2] = {
        .key   = STRVIEW("OBJECTS_COUNT_IN_SCENE"),
        .value = (double)state.scene.model_count,
      },
      [3] = {
        .key   = STRVIEW("MAX_BVs_COUNT_PER_MESH"),
        .value = (double)state.scene.max_num_bvs_per_mesh,
      },
      [4] = {
        .key   = STRVIEW("MAX_FACES_COUNT_PER_MESH"),
        .value = (double)state.scene.max_num_faces_per_mesh,
      },
    };

    state.compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label  = STRVIEW("Raytracer Compute Pipeline"),
        .layout = state.compute_pipeline_layout,
        .compute = {
          .module      = compute_shader_module,
          .entryPoint  = STRVIEW("main"),
          .constantCount = 5,
          .constants     = constants,
        },
      });
  }

  /* Release shader module */
  wgpuShaderModuleRelease(compute_shader_module);
}

/* Create render pipelines */
static void create_render_pipelines(wgpu_context_t* wgpu_context)
{
  /* Blit to screen pipeline */
  {
    /* Create bind group layout */
    WGPUBindGroupLayoutEntry bg_entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type = WGPUBufferBindingType_Storage,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };

    state.blit_to_screen_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Blit to Screen Bind Group Layout"),
        .entryCount = 3,
        .entries    = bg_entries,
      });

    /* Create pipeline layout */
    state.blit_to_screen_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Blit to Screen Pipeline Layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.blit_to_screen_bind_group_layout,
      });

    /* Create shader module */
    WGPUShaderModule blit_shader_module
      = wgpu_create_shader_module(wgpu_context->device, get_present_shader());

    /* Create render pipeline */
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.blit_to_screen_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Blit to Screen Pipeline"),
        .layout = state.blit_to_screen_pipeline_layout,
        .vertex = {
          .module     = blit_shader_module,
          .entryPoint = STRVIEW("vertexMain"),
        },
        .fragment = &(WGPUFragmentState){
          .module      = blit_shader_module,
          .entryPoint  = STRVIEW("fragmentMain"),
          .targetCount = 1,
          .targets     = &color_target,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
        },
        .multisample = {
          .count = 1,
          .mask  = 0xffffffff
        },
      });

    wgpuShaderModuleRelease(blit_shader_module);
  }

  /* Debug BVH pipeline */
  {
    /* Create bind group layout */
    WGPUBindGroupLayoutEntry bg_entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {
          .type = WGPUBufferBindingType_ReadOnlyStorage,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {
          .type = WGPUBufferBindingType_Uniform,
        },
      },
    };

    state.debug_bvh_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Debug BVH Bind Group Layout"),
                              .entryCount = 2,
                              .entries    = bg_entries,
                            });

    /* Create pipeline layout */
    state.debug_bvh_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Debug BVH Pipeline Layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.debug_bvh_bind_group_layout,
      });

    /* Create shader module */
    WGPUShaderModule debug_shader_module
      = wgpu_create_shader_module(wgpu_context->device, get_debug_bvh_shader());

    /* Create render pipeline */
    WGPUBlendState blend_state = {
      .color = {
        .srcFactor = WGPUBlendFactor_One,
        .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        .operation = WGPUBlendOperation_Add,
      },
      .alpha = {
        .srcFactor = WGPUBlendFactor_One,
        .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        .operation = WGPUBlendOperation_Add,
      },
    };

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.debug_bvh_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Debug BVH Pipeline"),
        .layout = state.debug_bvh_pipeline_layout,
        .vertex = {
          .module     = debug_shader_module,
          .entryPoint = STRVIEW("vertexMain"),
        },
        .fragment = &(WGPUFragmentState){
          .module      = debug_shader_module,
          .entryPoint  = STRVIEW("fragmentMain"),
          .targetCount = 1,
          .targets     = &color_target,
        },
        .primitive = {
          .topology = WGPUPrimitiveTopology_LineList,
        },
        .multisample = {
          .count = 1,
          .mask  = 0xffffffff
        },
      });

    wgpuShaderModuleRelease(debug_shader_module);
  }
}

/* Create bind groups */
static void create_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Compute bind group 0 */
  {
    WGPUBindGroupEntry entries[4] = {
      [0] = {
        .binding = 0,
        .buffer  = state.raytraced_storage_buffer,
        .size    = wgpuBufferGetSize(state.raytraced_storage_buffer),
      },
      [1] = {
        .binding = 1,
        .buffer  = state.rng_state_buffer,
        .size    = wgpuBufferGetSize(state.rng_state_buffer),
      },
      [2] = {
        .binding = 2,
        .buffer  = state.common_uniforms_buffer,
        .size    = PAD_TO_16(sizeof(common_uniforms_t)),
      },
      [3] = {
        .binding = 3,
        .buffer  = state.camera_uniform_buffer,
        .size    = sizeof(camera_uniforms_t),
      },
    };

    state.compute_bind_group_0 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Compute Bind Group 0"),
                              .layout     = state.compute_bind_group_0_layout,
                              .entryCount = 4,
                              .entries    = entries,
                            });
  }

  /* Compute bind group 1 */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.scene.faces_buffer,
        .size    = wgpuBufferGetSize(state.scene.faces_buffer),
      },
      [1] = {
        .binding = 1,
        .buffer  = state.scene.aabbs_buffer,
        .size    = wgpuBufferGetSize(state.scene.aabbs_buffer),
      },
      [2] = {
        .binding = 2,
        .buffer  = state.scene.materials_buffer,
        .size    = wgpuBufferGetSize(state.scene.materials_buffer),
      },
    };

    state.compute_bind_group_1 = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Compute Bind Group 1"),
                              .layout     = state.compute_bind_group_1_layout,
                              .entryCount = 3,
                              .entries    = entries,
                            });
  }

  /* Blit to screen bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.raytraced_storage_buffer,
        .size    = wgpuBufferGetSize(state.raytraced_storage_buffer),
      },
      [1] = {
        .binding = 1,
        .buffer  = state.camera_uniform_buffer,
        .size    = sizeof(camera_uniforms_t),
      },
      [2] = {
        .binding = 2,
        .buffer  = state.common_uniforms_buffer,
        .size    = PAD_TO_16(sizeof(common_uniforms_t)),
      },
    };

    state.blit_to_screen_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Blit to Screen Bind Group"),
                              .layout = state.blit_to_screen_bind_group_layout,
                              .entryCount = 3,
                              .entries    = entries,
                            });
  }

  /* Debug BVH bind group */
  {
    WGPUBindGroupEntry entries[2] = {
      [0] = {
        .binding = 0,
        .buffer  = state.scene.aabbs_buffer,
        .size    = wgpuBufferGetSize(state.scene.aabbs_buffer),
      },
      [1] = {
        .binding = 1,
        .buffer  = state.camera_view_proj_matrix_buffer,
        .size    = 16 * sizeof(float),
      },
    };

    state.debug_bvh_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Debug BVH Bind Group"),
                              .layout     = state.debug_bvh_bind_group_layout,
                              .entryCount = 2,
                              .entries    = entries,
                            });
  }
}

/* Update uniforms */
static void update_uniforms(wgpu_context_t* wgpu_context)
{
  /* Update common uniforms */
  {
    common_uniforms_t common_uniforms = {
      .seed          = {(uint32_t)(state.shader_seed[0] * 0xffffff),
                        (uint32_t)(state.shader_seed[1] * 0xffffff),
                        (uint32_t)(state.shader_seed[2] * 0xffffff)},
      .frame_counter = state.frame_counter,
      .max_bounces   = state.max_bounces,
      .flat_shading  = state.use_phong_shading ? 0 : 1,
      .debug_normals = state.debug_normals ? 1u : 0u,
    };

    wgpuQueueWriteBuffer(wgpu_context->queue, state.common_uniforms_buffer, 0,
                         &common_uniforms, sizeof(common_uniforms_t));
  }

  /* Update camera uniforms */
  {
    camera_uniforms_t camera_uniforms = {
      .viewport_size = {wgpu_context->width, wgpu_context->height},
      .image_width   = (float)wgpu_context->width,
      .aspect_ratio  = state.camera.aspect_ratio,
      .vfov          = state.camera.vfov,
      .defocus_angle = 0.0f,
      .focus_dist    = 3.4f,
    };

    /* Copy camera position and target */
    memcpy(camera_uniforms.look_from, state.camera.position, sizeof(float) * 3);
    memcpy(camera_uniforms.look_at, state.camera.target, sizeof(float) * 3);
    camera_uniforms.vup[0] = 0.0f;
    camera_uniforms.vup[1] = 1.0f;
    camera_uniforms.vup[2] = 0.0f;

    wgpuQueueWriteBuffer(wgpu_context->queue, state.camera_uniform_buffer, 0,
                         &camera_uniforms, sizeof(camera_uniforms_t));
  }

  /* Update camera view-projection matrix */
  {
    wgpuQueueWriteBuffer(
      wgpu_context->queue, state.camera_view_proj_matrix_buffer, 0,
      state.camera.view_projection_matrix, 16 * sizeof(float));
  }
}

/* Initialize */
static int init(struct wgpu_context_t* wgpu_context)
{
  /* Initialize sokol time */
  stm_setup();

  /* Initialize scene */
  scene_init(&state.scene, wgpu_context->device);
  if (scene_load_models(
        &state.scene,
        "/home/sdauwe/GitHub/webgpu-raytracer/public/raytraced-scene.obj",
        "/home/sdauwe/GitHub/webgpu-raytracer/public/raytraced-scene.mtl")
      != 0) {
    fprintf(stderr, "Failed to load scene models\n");
    return EXIT_FAILURE;
  }

  /* Initialize camera */
  vec3 camera_pos = {0.0f, 0.0f, 3.5f};
  raytracer_camera_init(&state.camera, camera_pos, 60.0f,
                        (float)wgpu_context->width
                          / (float)wgpu_context->height,
                        wgpu_context->width, wgpu_context->height);

  /* Create buffers */
  create_buffers(wgpu_context);

  /* Create pipelines */
  create_compute_pipeline(wgpu_context);
  create_render_pipelines(wgpu_context);

  /* Create bind groups */
  create_bind_groups(wgpu_context);

  /* Initialize frame state */
  state.shader_seed[0] = (float)rand() / (float)RAND_MAX;
  state.shader_seed[1] = (float)rand() / (float)RAND_MAX;
  state.shader_seed[2] = (float)rand() / (float)RAND_MAX;
  state.old_time_ms    = (float)stm_sec(stm_now()) * 1000.0f;

  state.initialized = true;

  return EXIT_SUCCESS;
}

/* Input event callback */
static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    state.camera.viewport_width  = input_event->window_width;
    state.camera.viewport_height = input_event->window_height;
    state.camera.aspect_ratio
      = (float)input_event->window_width / (float)input_event->window_height;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_DOWN) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.mouse_left_down = true;
      state.max_bounces     = MAX_BOUNCES_INTERACTING;
      raytracer_camera_on_mouse_down(&state.camera, 0,
                                     (float)input_event->mouse_x,
                                     (float)input_event->mouse_y);
    }
    else if (input_event->mouse_button == BUTTON_RIGHT) {
      state.mouse_right_down = true;
      state.max_bounces      = MAX_BOUNCES_INTERACTING;
      raytracer_camera_on_mouse_down(&state.camera, 1,
                                     (float)input_event->mouse_x,
                                     (float)input_event->mouse_y);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    if (input_event->mouse_button == BUTTON_LEFT) {
      state.mouse_left_down = false;
      state.max_bounces     = 4;
      raytracer_camera_on_mouse_up(&state.camera);
    }
    else if (input_event->mouse_button == BUTTON_RIGHT) {
      state.mouse_right_down = false;
      state.max_bounces      = 4;
      raytracer_camera_on_mouse_up(&state.camera);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (state.mouse_left_down || state.mouse_right_down) {
      raytracer_camera_on_mouse_move(&state.camera, (float)input_event->mouse_x,
                                     (float)input_event->mouse_y);
      reset_render(wgpu_context);
    }
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_SCROLL) {
    raytracer_camera_on_mouse_wheel(&state.camera,
                                    (float)input_event->scroll_y);
    reset_render(wgpu_context);
  }
}

/* Frame rendering */
static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Check if max samples reached */
  if (state.frame_counter >= state.max_samples) {
    return EXIT_SUCCESS;
  }

  /* Update time */
  float now_ms      = (float)stm_sec(stm_now()) * 1000.0f;
  float diff        = now_ms - state.old_time_ms;
  state.old_time_ms = now_ms;
  state.time_expired_ms += diff;

  /* Update camera */
  raytracer_camera_tick(&state.camera);

  /* Update random seeds */
  state.shader_seed[0] = (float)rand() / (float)RAND_MAX;
  state.shader_seed[1] = (float)rand() / (float)RAND_MAX;
  state.shader_seed[2] = (float)rand() / (float)RAND_MAX;

  /* Update uniforms */
  update_uniforms(wgpu_context);

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Compute pass */
  {
    WGPUComputePassEncoder compute_pass
      = wgpuCommandEncoderBeginComputePass(cmd_encoder, NULL);
    wgpuComputePassEncoderSetPipeline(compute_pass, state.compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0,
                                       state.compute_bind_group_0, 0, NULL);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 1,
                                       state.compute_bind_group_1, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (wgpu_context->width + COMPUTE_WORKGROUP_SIZE_X - 1)
        / COMPUTE_WORKGROUP_SIZE_X,
      (wgpu_context->height + COMPUTE_WORKGROUP_SIZE_Y - 1)
        / COMPUTE_WORKGROUP_SIZE_Y,
      1);
    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
  }

  /* Render pass */
  {
    state.color_attachment.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
      cmd_encoder, &state.render_pass_descriptor);

    /* Blit raytraced image to screen */
    wgpuRenderPassEncoderSetPipeline(render_pass,
                                     state.blit_to_screen_pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                      state.blit_to_screen_bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

    /* Debug BVH if enabled */
    if (state.debug_bvh) {
      uint32_t total_aabbs
        = state.scene.max_num_bvs_per_mesh * state.scene.model_count;
      wgpuRenderPassEncoderSetPipeline(render_pass, state.debug_bvh_pipeline);
      wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                        state.debug_bvh_bind_group, 0, NULL);
      wgpuRenderPassEncoderDraw(render_pass, 2, total_aabbs * 12, 0, 0);
    }

    wgpuRenderPassEncoderEnd(render_pass);
    wgpuRenderPassEncoderRelease(render_pass);
  }

  /* Submit commands */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_encoder);

  /* Increment frame counter */
  state.frame_counter++;

  return EXIT_SUCCESS;
}

/* Cleanup */
static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Release buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.raytraced_storage_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.rng_state_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.common_uniforms_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.camera_view_proj_matrix_buffer)

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blit_to_screen_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.debug_bvh_pipeline)

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_group_0)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_group_1)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blit_to_screen_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.debug_bvh_bind_group)

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute_bind_group_0_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute_bind_group_1_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.blit_to_screen_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.debug_bvh_bind_group_layout)

  /* Release pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.blit_to_screen_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.debug_bvh_pipeline_layout)

  /* Cleanup scene */
  scene_destroy(&state.scene);

  /* Cleanup camera */
  raytracer_camera_destroy(&state.camera);

  /* Free shader strings */
  free_shader_strings();
}

/* Main entry point */
int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "GPU Raytracer",
    .width          = 1280,
    .height         = 720,
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Chunks
 * -------------------------------------------------------------------------- */

// clang-format off

/* Camera shader chunk */
static const char* camera_shader_chunk = CODE(
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

/* Color shader chunk */
static const char* color_shader_chunk = CODE(
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

/* Common shader chunk */
static const char* common_shader_chunk = CODE(
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

/* Interval shader chunk */
static const char* interval_shader_chunk = CODE(
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

/* Material shader chunk */
static const char* material_shader_chunk = CODE(
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

static const char* ray_shader_chunk = CODE(
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

/* Utility shader chunk */
static const char* utils_shader_chunk = CODE(
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

/* Vec shader chunk */
static const char* vec_shader_chunk = CODE(
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

static const char* vertex_shader_chunk = CODE(
  struct VertexOutput {
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,
  }
);

/* Global storage for concatenated shaders */
static char* raytracer_shader_code = NULL;
static char* present_shader_code   = NULL;
static char* debug_bvh_shader_code = NULL;

/* Helper function to concatenate strings */
static char* concat_strings(const char** strings, int count)
{
  size_t total_len = 0;
  for (int i = 0; i < count; ++i) {
    total_len += strlen(strings[i]);
  }

  char* result = (char*)malloc(total_len + 1);
  if (result == NULL) {
    return NULL;
  }

  size_t offset = 0;
  for (int i = 0; i < count; ++i) {
    size_t len = strlen(strings[i]);
    if (len > 0) {
      memcpy(result + offset, strings[i], len);
      offset += len;
    }
  }
  result[offset] = '\0';

  return result;
}

/* Get the complete debug BVH shader */
static const char* get_debug_bvh_shader(void)
{
  if (debug_bvh_shader_code != NULL) {
    return debug_bvh_shader_code;
  }

  const char* debug_bvh_main = CODE(
    @group(0) @binding(0) var<storage, read> AABBs: array<AABB>;
    @group(0) @binding(1) var<uniform> viewProjectionMatrix: mat4x4f;

    const EDGES_PER_CUBE = 12u;

    @vertex
    fn vertexMain(
      @builtin(instance_index) instanceIndex: u32,
      @builtin(vertex_index) vertexIndex: u32
    ) -> @builtin(position) vec4f {
      let lineInstanceIdx = instanceIndex % EDGES_PER_CUBE;
      let aabbInstanceIdx = instanceIndex / EDGES_PER_CUBE;
      let a = AABBs[aabbInstanceIdx];
      var pos: vec3f;
      let fVertexIndex = f32(vertexIndex);

      //        a7 _______________ a6
      //         / |             /|
      //        /  |            / |
      //    a4 /   |       a5  /  |
      //      /____|__________/   |
      //      |    |__________|___|
      //      |   / a3        |   / a2
      //      |  /            |  /
      //      | /             | /
      //      |/______________|/
      //      a0              a1

      let dx = a.max.x - a.min.x;
      let dy = a.max.y - a.min.y;
      let dz = a.max.z - a.min.z;

      let a0 = a.min;
      let a1 = vec3f(a.min.x + dx, a.min.y,      a.min.z     );
      let a2 = vec3f(a.min.x + dx, a.min.y,      a.min.z + dz);
      let a3 = vec3f(a.min.x,      a.min.y,      a.min.z + dz);
      let a4 = vec3f(a.min.x,      a.min.y + dy, a.min.z     );
      let a5 = vec3f(a.min.x + dx, a.min.y + dy, a.min.z     );
      let a6 = a.max;
      let a7 = vec3f(a.min.x,      a.min.y + dy, a.min.z + dz);

      if (lineInstanceIdx == 0) {
        pos = mix(a0, a1, fVertexIndex);
      } else if (lineInstanceIdx == 1) {
        pos = mix(a1, a2, fVertexIndex);
      } else if (lineInstanceIdx == 2) {
        pos = mix(a2, a3, fVertexIndex);
      } else if (lineInstanceIdx == 3) {
        pos = mix(a0, a3, fVertexIndex);
      } else if (lineInstanceIdx == 4) {
        pos = mix(a0, a4, fVertexIndex);
      } else if (lineInstanceIdx == 5) {
        pos = mix(a1, a5, fVertexIndex);
      } else if (lineInstanceIdx == 6) {
        pos = mix(a2, a6, fVertexIndex);
      } else if (lineInstanceIdx == 7) {
        pos = mix(a3, a7, fVertexIndex);
      } else if (lineInstanceIdx == 8) {
        pos = mix(a4, a5, fVertexIndex);
      } else if (lineInstanceIdx == 9) {
        pos = mix(a5, a6, fVertexIndex);
      } else if (lineInstanceIdx == 10) {
        pos = mix(a6, a7, fVertexIndex);
      } else if (lineInstanceIdx == 11) {
        pos = mix(a7, a4, fVertexIndex);
      }
      return viewProjectionMatrix * vec4(pos, 1);
    }

    @fragment
    fn fragmentMain() -> @location(0) vec4f {
      return vec4f(0.01);
    }
  );

  const char* parts[] = {common_shader_chunk, vertex_shader_chunk,
                         debug_bvh_main};

  debug_bvh_shader_code = concat_strings(parts, 3);
  return debug_bvh_shader_code;
}

/* Get the complete present shader */
static const char* get_present_shader(void)
{
  if (present_shader_code != NULL) {
    return present_shader_code;
  }

  const char* present_main = CODE(
    @group(0) @binding(0) var<storage, read_write> raytraceImageBuffer: array<vec3f>;
    @group(0) @binding(1) var<uniform> cameraUniforms: Camera;
    @group(0) @binding(2) var<uniform> commonUniforms: CommonUniforms;

    // xy pos + uv
    const FULLSCREEN_QUAD = array<vec4<f32>, 6>(
      vec4(-1, 1, 0, 0),
      vec4(-1, -1, 0, 1),
      vec4(1, -1, 1, 1),
      vec4(-1, 1, 0, 0),
      vec4(1, -1, 1, 1),
      vec4(1, 1, 1, 0)
    );

    @vertex
    fn vertexMain(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
      var output: VertexOutput;
      output.Position = vec4<f32>(FULLSCREEN_QUAD[VertexIndex].xy, 0.0, 1.0);
      output.uv = FULLSCREEN_QUAD[VertexIndex].zw;
      return output;
    }

    @fragment
    fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4f {
      let x = u32(uv.x * f32(cameraUniforms.viewportSize.x));
      let y = u32(uv.y * f32(cameraUniforms.viewportSize.y));
      let idx = x + y * cameraUniforms.viewportSize.x;
      let color = lottes(raytraceImageBuffer[idx] / f32(commonUniforms.frameCounter + 1));
      return vec4f(color, 1.0);
    }
  );

  const char* parts[] = {camera_shader_chunk, color_shader_chunk,
                         vertex_shader_chunk, common_shader_chunk, present_main};

  present_shader_code = concat_strings(parts, 5);
  return present_shader_code;
}

/* Get the complete raytracer shader */
static const char* get_raytracer_shader(void)
{
  if (raytracer_shader_code != NULL) {
    return raytracer_shader_code;
  }

  const char* raytracer_main = CODE(
    const BV_MAX_STACK_DEPTH = 16;
    const EPSILON = 0.001;

    @group(0) @binding(0) var<storage, read_write> raytraceImageBuffer: array<vec3f>;
    @group(0) @binding(1) var<storage, read_write> rngStateBuffer: array<u32>;
    @group(0) @binding(2) var<uniform> commonUniforms: CommonUniforms;
    @group(0) @binding(3) var<uniform> cameraUniforms: Camera;

    @group(1) @binding(0) var<storage, read> faces: array<Face>;
    @group(1) @binding(1) var<storage, read> AABBs: array<AABB>;
    @group(1) @binding(2) var<storage, read> materials: array<Material>;

    override WORKGROUP_SIZE_X: u32;
    override WORKGROUP_SIZE_Y: u32;
    override OBJECTS_COUNT_IN_SCENE: u32;
    override MAX_BVs_COUNT_PER_MESH: u32;
    override MAX_FACES_COUNT_PER_MESH: u32;

    @compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
    fn main(@builtin(global_invocation_id) globalInvocationId : vec3<u32>,) {
      if (any(globalInvocationId.xy > cameraUniforms.viewportSize)) {
        return;
      }

      let pos = globalInvocationId.xy;
      let x = f32(pos.x);
      let y = f32(pos.y);
      let idx = pos.x + pos.y * cameraUniforms.viewportSize.x;

      var rngState = rngStateBuffer[idx];

      var camera = cameraUniforms;
      initCamera(&camera);

      var hitRec: HitRecord;

      var r = getCameraRay(&camera, x, y, &rngState);

      var color = vec3f(0);

      var mtlStack: array<Material, 16>;
      var bLoop = true;
      var i = 0u;

      while(bLoop && rayIntersectBVH(&r, &hitRec, positiveUniverseInterval)) {
        var scattered: Ray;
        var material = materials[hitRec.materialIdx];
        var albedo = material.albedo;

        mtlStack[i] = material;

        if (commonUniforms.debugNormals == 1u) {
          color = (hitRec.normal + 1) * 0.5;
          break;
        }

        switch material.materialType {
          case 0: {
            color = material.albedo;
            bLoop = false;
            break;
          }
          case 1: {
            if (i < commonUniforms.maxBounces) {
              var scatters = scatterMetal(&material, &r, &scattered, &hitRec, &albedo, &rngState);
              if (scatters) {
                i++;
                r = scattered;
              } else {
                color = vec3f(0);
                bLoop = false;
                i = 0u;
              }
            } else {
              color = material.albedo;
              bLoop = false;
            }
            break;
          }
          case 2: {
            if (i < commonUniforms.maxBounces) {
              var scatters = scatterDielectric(&material, &r, &scattered, &hitRec, &albedo, &rngState);
              r = scattered;
              i++;
            } else {
              color = mtlStack[i].albedo;
              bLoop = false;
            }
            break;
          }
          case 3: {
            var scatters = scatterLambertian(&material, &r, &scattered, &hitRec, &albedo, &rngState);
            if (i < commonUniforms.maxBounces) {
              i++;
              r = scattered;
            } else {
              bLoop = false;
            }
            break;
          }
          default: {
            // ...
          }
        }
      }


      while (i > 0) {
        i--;
        color *= mtlStack[i].albedo;
      }

      var pixel = raytraceImageBuffer[idx];

      if (commonUniforms.frameCounter == 0) {
        pixel = vec3f(0);
      }

      pixel += color;
      raytraceImageBuffer[idx] = pixel;

      rngStateBuffer[idx] = rngState;
    }
  );

  const char* parts[] = {utils_shader_chunk,   common_shader_chunk,
                         ray_shader_chunk,     vec_shader_chunk,
                         interval_shader_chunk, camera_shader_chunk,
                         color_shader_chunk,   material_shader_chunk,
                         raytracer_main};

  raytracer_shader_code = concat_strings(parts, 9);
  return raytracer_shader_code;
}

// clang-format on

/* Cleanup shader strings */
static void free_shader_strings(void)
{
  if (raytracer_shader_code) {
    free(raytracer_shader_code);
    raytracer_shader_code = NULL;
  }
  if (present_shader_code) {
    free(present_shader_code);
    present_shader_code = NULL;
  }
  if (debug_bvh_shader_code) {
    free(debug_bvh_shader_code);
    debug_bvh_shader_code = NULL;
  }
}
