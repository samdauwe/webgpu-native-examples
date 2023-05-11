#include "camera.h"

#include <string.h>

/* camera creating/releasing */

camera_t* camera_create(void)
{
  camera_t* camera = (camera_t*)malloc(sizeof(camera_t));
  memset(camera, 0, sizeof(camera_t));

  camera->rotation_speed = 1.0f;
  camera->movement_speed = 1.0f;

  camera->updated = false;
  camera->flip_y  = false;

  camera->keys.left  = false;
  camera->keys.right = false;
  camera->keys.up    = false;
  camera->keys.down  = false;

  return camera;
}

void camera_release(camera_t* camera)
{
  free(camera);
}

/* camera updating */

void camera_update(camera_t* camera, float delta_time)
{
  camera->updated = false;
  if (camera->type == CameraType_FirstPerson) {
    if (camera_moving(camera)) {
      vec3 cam_front = GLM_VEC3_ZERO_INIT;
      cam_front[0]   = -cos(glm_rad(camera->rotation[0]))
                     * sin(glm_rad(camera->rotation[1]));
      cam_front[1] = sin(glm_rad(camera->rotation[0]));
      cam_front[2]
        = cos(glm_rad(camera->rotation[0])) * cos(glm_rad(camera->rotation[1]));
      glm_normalize(cam_front);

      const float move_speed = delta_time * camera->movement_speed;

      if (camera->keys.up) {
        glm_vec3_scale(cam_front, move_speed, cam_front);
        glm_vec3_add(camera->position, cam_front, camera->position);
      }
      if (camera->keys.down) {
        glm_vec3_scale(cam_front, move_speed, cam_front);
        glm_vec3_sub(camera->position, cam_front, camera->position);
      }
      if (camera->keys.left) {
        glm_cross(cam_front, (vec3){0.0f, 1.0f, 0.0f}, cam_front);
        glm_normalize(cam_front);
        glm_vec3_scale(cam_front, move_speed, cam_front);
        glm_vec3_sub(camera->position, cam_front, camera->position);
      }
      if (camera->keys.right) {
        glm_cross(cam_front, (vec3){0.0f, 1.0f, 0.0f}, cam_front);
        glm_normalize(cam_front);
        glm_vec3_scale(cam_front, move_speed, cam_front);
        glm_vec3_add(camera->position, cam_front, camera->position);
      }

      camera_update_view_matrix(camera);
    }
  }
}

/**
 * @brief Update camera passing separate axis data (gamepad)
 * @returns true if view or position has been changed
 */
bool camera_update_pad(camera_t* camera, vec2 axis_left, vec2 axis_right,
                       float delta_time)
{
  bool ret_val = false;

  if (camera->type == CameraType_FirstPerson) {
    // Use the common console thumbstick layout
    // Left = view, right = move

    const float dead_zone = 0.0015f;
    const float range     = 1.0f - dead_zone;

    vec3 cam_front = GLM_VEC3_ZERO_INIT;
    cam_front[0]
      = -cos(glm_rad(camera->rotation[0])) * sin(glm_rad(camera->rotation[1]));
    cam_front[1] = sin(glm_rad(camera->rotation[0]));
    cam_front[2]
      = cos(glm_rad(camera->rotation[0])) * cos(glm_rad(camera->rotation[1]));
    glm_normalize(cam_front);

    const float move_speed = delta_time * camera->movement_speed * 2.0f;
    const float rot_speed  = delta_time * camera->movement_speed * 50.0f;

    // Move
    if (fabsf(axis_left[1]) > dead_zone) {
      vec3 cam_front_left_y = GLM_VEC3_ZERO_INIT;
      glm_vec3_copy(cam_front, cam_front_left_y);
      const float pos = (fabsf(axis_left[1]) - dead_zone) / range;
      glm_vec3_scale(cam_front_left_y,
                     pos * ((axis_left[1] < 0.0f) ? -1.0f : 1.0f) * move_speed,
                     cam_front_left_y);
      glm_vec3_sub(camera->position, cam_front_left_y, camera->position);
      ret_val = true;
    }
    if (fabsf(axis_left[0]) > dead_zone) {
      vec3 cam_front_left_x = GLM_VEC3_ZERO_INIT;
      glm_vec3_copy(cam_front, cam_front_left_x);
      const float pos = (fabsf(axis_left[0]) - dead_zone) / range;
      glm_cross(cam_front_left_x, (vec3){0.0f, 1.0f, 0.0f}, cam_front_left_x);
      glm_normalize(cam_front_left_x);
      glm_vec3_scale(cam_front_left_x,
                     pos * ((axis_left[0] < 0.0f) ? -1.0f : 1.0f) * move_speed,
                     cam_front_left_x);
      glm_vec3_add(camera->position, cam_front_left_x, camera->position);
      ret_val = true;
    }

    // Rotate
    if (fabsf(axis_right[0]) > dead_zone) {
      const float pos = (fabsf(axis_right[0]) - dead_zone) / range;
      camera->rotation[1]
        += pos * ((axis_right[0] < 0.0f) ? -1.0f : 1.0f) * rot_speed;
      ret_val = true;
    }
    if (fabsf(axis_right[1]) > dead_zone) {
      const float pos = (fabsf(axis_right[1]) - dead_zone) / range;
      camera->rotation[0]
        -= pos * ((axis_right[1] < 0.0f) ? -1.0f : 1.0f) * rot_speed;
      ret_val = true;
    }
  }

  if (ret_val) {
    camera_update_view_matrix(camera);
  }

  return ret_val;
}

void camera_update_view_matrix(camera_t* camera)
{
  mat4 rot_mat   = GLM_MAT4_IDENTITY_INIT;
  mat4 trans_mat = GLM_MAT4_IDENTITY_INIT;

  glm_rotate(rot_mat,
             glm_rad(camera->rotation[0] * (camera->flip_y ? -1.0f : 1.0f)),
             (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate(rot_mat, glm_rad(camera->rotation[1]), (vec3){0.0f, 1.0f, 0.0f});
  glm_rotate(rot_mat, glm_rad(camera->rotation[2]), (vec3){0.0f, 0.0f, 1.0f});

  vec3 translation;
  glm_vec3_copy(camera->position, translation);
  if (camera->flip_y) {
    translation[1] *= -1.0f;
  }
  glm_translate(trans_mat, translation);

  if (camera->type == CameraType_FirstPerson) {
    glm_mat4_mul(rot_mat, trans_mat, camera->matrices.view);
  }
  else {
    glm_mat4_mul(trans_mat, rot_mat, camera->matrices.view);
  }

  glm_vec4_mul(
    (vec4){camera->position[0], camera->position[1], camera->position[2], 0.0f},
    (vec4){-1.0f, 1.0f, -1.0f, 1.0f}, camera->view_pos);

  camera->updated = true;
}

void camera_set_position(camera_t* camera, vec3 position)
{
  glm_vec3_copy((vec3){position[0], -position[1], position[2]},
                camera->position);
  camera_update_view_matrix(camera);
}

void camera_set_rotation(camera_t* camera, vec3 rotation)
{
  glm_vec3_copy((vec3){rotation[0], rotation[1], rotation[2]},
                camera->rotation);
  camera_update_view_matrix(camera);
}

void camera_rotate(camera_t* camera, vec3 delta)
{
  glm_vec3_add(camera->rotation, (vec3){delta[0], -delta[1], delta[2]},
               camera->rotation);
  camera_update_view_matrix(camera);
}

void camera_set_translation(camera_t* camera, vec3 translation)
{
  glm_vec3_copy(translation, camera->position);
  camera_update_view_matrix(camera);
}

void camera_translate(camera_t* camera, vec3 delta)
{
  glm_vec3_add(camera->position, (vec3){-delta[0], delta[1], -delta[2]},
               camera->position);
  camera_update_view_matrix(camera);
}

void camera_set_rotation_speed(camera_t* camera, float rotation_speed)
{
  camera->rotation_speed = rotation_speed;
}

void camera_set_movement_speed(camera_t* camera, float movement_speed)
{
  camera->movement_speed = movement_speed;
}

void camera_set_perspective(camera_t* camera, float fov, float aspect,
                            float znear, float zfar)
{
  camera->fov   = fov;
  camera->znear = znear;
  camera->zfar  = zfar;
  glm_perspective(glm_rad(fov), aspect, znear, zfar,
                  camera->matrices.perspective);
  if (camera->flip_y) {
    camera->matrices.perspective[1][1] *= -1.0f;
  }
}

void camera_update_aspect_ratio(camera_t* camera, float aspect)
{
  glm_perspective(glm_rad(camera->fov), aspect, camera->znear, camera->zfar,
                  camera->matrices.perspective);
  if (camera->flip_y) {
    camera->matrices.perspective[1][1] *= -1.0f;
  }
}

/* property retrieving */

bool camera_moving(camera_t* camera)
{
  return camera->keys.left || camera->keys.right || camera->keys.up
         || camera->keys.down;
}

float camera_get_near_clip(camera_t* camera)
{
  return camera->znear;
}

float camera_get_far_clip(camera_t* camera)
{
  return camera->zfar;
}

/* projection helpers */

/**
 * @brief Converts a projection matrix from WebGPU-style Z range [0, 1] to
 * OpenGL-style Z range [-1, 1]
 */
static void projection_matrix_wgpu_to_opengl(mat4* proj_mtx)
{
  static mat4 WGPU_TO_OPENGL_MATRIX = {
    {1.0f, 0.0f, 0.0f, 0.0f}, //
    {0.0f, 1.0f, 0.0f, 0.0f}, //
    {0.0f, 0.0f, 2.0f, 0.0f}, //
    {0.0f, 0.0f, -1.0f, 1.0f} //
  };
  mat4 tmp = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_mulN((mat4*[]){&WGPU_TO_OPENGL_MATRIX, proj_mtx}, 2, tmp);
  glm_mat4_copy(tmp, *proj_mtx);
}

/**
 * @brief Converts a projection matrix from OpenGL-style Z range [-1, 1] to
 * WebGPU-style Z range [0, 1]
 */
static void projection_matrix_opengl_to_wgpu(mat4* proj_mtx)
{
  static mat4 OPENGL_TO_WGPU_MATRIX = {
    {1.0f, 0.0f, 0.0f, 0.0f}, //
    {0.0f, 1.0f, 0.0f, 0.0f}, //
    {0.0f, 0.0f, 0.5f, 0.0f}, //
    {0.0f, 0.0f, 0.5f, 1.0f}  //
  };
  mat4 tmp = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_mulN((mat4*[]){&OPENGL_TO_WGPU_MATRIX, proj_mtx}, 2, tmp);
  glm_mat4_copy(tmp, *proj_mtx);
}

void projection_matrix_convert_clip_space_near_z(mat4* proj_mtx,
                                                 clip_space_near_z_enum dest,
                                                 clip_space_near_z_enum src)
{
  if (dest == src) {
    return;
  }

  if (dest == ClipSpaceNearZ_NegativeOne) {
    projection_matrix_wgpu_to_opengl(proj_mtx);
  }
  else if (dest == ClipSpaceNearZ_Zero) {
    projection_matrix_opengl_to_wgpu(proj_mtx);
  }
}

void perspective_matrix_reversed_z(float fovy, float aspect, float near,
                                   float far, mat4 dest)
{
  glm_mat4_zero(dest);
  float f     = 1.0f / tanf(fovy * 0.5f);
  float range = far / (near - far);
  memcpy(dest,
         &(mat4){
           {f / aspect, 0.0f, 0.0f, 0.0f},     // first COLUMN
           {0.0f, f, 0.0f, 0.0f},              // second COLUMN
           {0.0f, 0.0f, -range - 1.0f, -1.0f}, // third COLUMN
           {0.0f, 0.0f, -near * range, 0.0f}   // fourth COLUMN
         },
         sizeof(mat4));
}

void perspective_matrix_reversed_z_infinite_far(float fovy, float aspect,
                                                float near, mat4 dest)
{
  glm_mat4_zero(dest);
  float f = 1.0f / tanf(fovy * 0.5f);
  memcpy(dest,
         &(mat4){
           {f / aspect, 0.0f, 0.0f, 0.0f}, // first COLUMN
           {0.0f, f, 0.0f, 0.0f},          // second COLUMN
           {0.0f, 0.0f, 0.0f, -1.0f},      // third COLUMN
           {0.0f, 0.0f, near, 0.0f}        // fourth COLUMN
         },
         sizeof(mat4));
}

mat4* perspective_zo(mat4* out, float fovy, float aspect, float near,
                     const float* far)
{
  const float f = 1.0f / tan(fovy / 2.0f);
  (*out)[0][0]  = f / aspect;
  (*out)[0][1]  = 0.0f;
  (*out)[0][2]  = 0.0f;
  (*out)[0][3]  = 0.0f;
  (*out)[1][0]  = 0.0f;
  (*out)[1][1]  = f;
  (*out)[1][2]  = 0.0f;
  (*out)[1][3]  = 0.0f;
  (*out)[2][0]  = 0.0f;
  (*out)[2][1]  = 0.0f;
  (*out)[2][3]  = -1.0f;
  (*out)[3][0]  = 0.0f;
  (*out)[3][1]  = 0.0f;
  (*out)[3][3]  = 0.0f;
  if (far != NULL && *far != INFINITY) {
    const float nf = 1.0f / (near - *far);
    (*out)[2][2]   = *far * nf;
    (*out)[3][2]   = *far * near * nf;
  }
  else {
    (*out)[2][2] = -1.0f;
    (*out)[3][2] = -near;
  }
  return out;
}
