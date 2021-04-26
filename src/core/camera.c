#include "camera.h"

#include <string.h>

/* camera creating/releasing */

camera_t* camera_create()
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
      vec3 camFront = {0};
      camFront[0]   = -cos(glm_rad(camera->rotation[0]))
                    * sin(glm_rad(camera->rotation[1]));
      camFront[1] = sin(glm_rad(camera->rotation[0]));
      camFront[2]
        = cos(glm_rad(camera->rotation[0])) * cos(glm_rad(camera->rotation[1]));
      glm_normalize(camFront);

      const float moveSpeed = delta_time * camera->movement_speed;

      if (camera->keys.up) {
        glm_vec3_scale(camFront, moveSpeed, camFront);
        glm_vec3_add(camera->position, camFront, camera->position);
      }
      if (camera->keys.down) {
        glm_vec3_scale(camFront, moveSpeed, camFront);
        glm_vec3_sub(camera->position, camFront, camera->position);
      }
      if (camera->keys.left) {
        glm_cross(camFront, (vec3){0.0f, 1.0f, 0.0f}, camFront);
        glm_normalize(camFront);
        glm_vec3_scale(camFront, moveSpeed, camFront);
        glm_vec3_sub(camera->position, camFront, camera->position);
      }
      if (camera->keys.right) {
        glm_cross(camFront, (vec3){0.0f, 1.0f, 0.0f}, camFront);
        glm_normalize(camFront);
        glm_vec3_scale(camFront, moveSpeed, camFront);
        glm_vec3_add(camera->position, camFront, camera->position);
      }

      camera_update_view_matrix(camera);
    }
  }
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
    (vec4){camera->position[0], camera->position[1], camera->position[2], 0.f},
    (vec4){-1.0f, 1.0f, -1.0f, 1.0f}, camera->view_pos);

  camera->updated = true;
}

void camera_set_position(camera_t* camera, vec3 position)
{
  glm_vec3_copy(position, camera->position);
  camera_update_view_matrix(camera);
}

void camera_set_rotation(camera_t* camera, vec3 rotation)
{
  glm_vec3_copy(rotation, camera->rotation);
  camera_update_view_matrix(camera);
}

void camera_rotate(camera_t* camera, vec3 delta)
{
  glm_vec3_add(camera->rotation, delta, camera->rotation);
  camera_update_view_matrix(camera);
}

void camera_set_translation(camera_t* camera, vec3 translation)
{
  memcpy(camera->position, translation, sizeof(vec3));
  camera_update_view_matrix(camera);
}

void camera_translate(camera_t* camera, vec3 delta)
{
  glm_vec3_add(camera->position, delta, camera->position);
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
