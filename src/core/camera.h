#ifndef CAMERA_H
#define CAMERA_H

#include <cglm/cglm.h>

typedef enum camera_type_enum {
  CameraType_LookAt      = 0,
  CameraType_FirstPerson = 1
} camera_type_enum;

/**
 * @brief Basic camera class
 */
typedef struct camera_t {
  vec3 rotation;
  vec3 position;
  vec4 view_pos;
  enum camera_type_enum type;
  float fov;
  float znear;
  float zfar;
  float rotation_speed;
  float movement_speed;
  bool updated;
  bool flip_y;
  struct {
    mat4 perspective;
    mat4 view;
  } matrices;
  struct {
    bool left;
    bool right;
    bool up;
    bool down;
  } keys;
} camera_t;

/* camera creating/releasing */
camera_t* camera_create();
void camera_release(camera_t* camera);

/* camera updating */
void camera_update(camera_t* camera, float delta_time);
void camera_update_view_matrix(camera_t* camera);
void camera_set_position(camera_t* camera, vec3 position);
void camera_set_rotation(camera_t* camera, vec3 rotation);
void camera_rotate(camera_t* camera, vec3 delta);
void camera_set_translation(camera_t* camera, vec3 translation);
void camera_translate(camera_t* camera, vec3 delta);
void camera_set_rotation_speed(camera_t* camera, float rotation_speed);
void camera_set_movement_speed(camera_t* camera, float movement_speed);
void camera_set_perspective(camera_t* camera, float fov, float aspect,
                            float znear, float zfar);
void camera_update_aspect_ratio(camera_t* camera, float aspect);

/* property retrieving */
bool camera_moving(camera_t* camera);
float camera_get_near_clip(camera_t* camera);
float camera_get_far_clip(camera_t* camera);

#endif
