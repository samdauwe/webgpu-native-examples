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
typedef struct {
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

/* Camera creating/releasing */
camera_t* camera_create(void);
void camera_release(camera_t* camera);

/* Camera updating */
void camera_update(camera_t* camera, float delta_time);
bool camera_update_pad(camera_t* camera, vec2 axis_left, vec2 axis_right,
                       float delta_time);
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

/* Property retrieving */
bool camera_moving(camera_t* camera);
float camera_get_near_clip(camera_t* camera);
float camera_get_far_clip(camera_t* camera);

/* Projection helpers */

typedef enum {
  ClipSpaceNearZ_NegativeOne = 0x00000000, // OpenGL
  ClipSpaceNearZ_Zero        = 0x00000001  // WebGPU
} clip_space_near_z_enum;

/**
 * @brief Convert a projection matrix {@param proj_mtx} between differing clip
 * spaces.
 *
 * There are two kinds of clip-space conventions in active use in graphics APIs,
 * differing in the range of the Z axis: OpenGL (and thus GL ES and WebGL) use a
 * Z range of [-1, 1] which matches the X and Y axis ranges. Direct3D, Vulkan,
 * Metal, and WebGPU all use a Z range of [0, 1], which differs from the X and Y
 * axis ranges, but makes sense from the perspective of a camera: a camera can
 * see to the left and right of it, above and below it, but only in front and
 * not behind it.
 *
 * The [0, 1] convention for Z range also has better characteristics for
 * "reversed depth". Since floating point numbers have higher precision around 0
 * than around 1. We then get to choose where to put the extra precise bits:
 * close to the near plane, or close to the far plane.
 *
 * With OpenGL's [-1, 1] convention, both -1 and 1 have similar amounts of
 * precision, so we don't get to make the same choice, and our higher precision
 * around 0 is stuck in the middle of the scene, which doesn't particularly
 * help.
 *
 * @ref
 * https://github.com/magcius/noclip.website/blob/master/src/gfx/helpers/ProjectionHelpers.ts
 *
 * This function does nothing if {@param dst} and {@param src} are the same.
 */
void projection_matrix_convert_clip_space_near_z(mat4* proj_mtx,
                                                 clip_space_near_z_enum dest,
                                                 clip_space_near_z_enum src);

/**
 * @brief Creates a perspective matrix [near = 1, far = 0].
 * @ref http://dev.theomader.com/depth-precision/
 */
void perspective_matrix_reversed_z(float fovy, float aspect, float near,
                                   float far, mat4 dest);

/**
 * @brief Creates a perspective matrix [near = 1, infinite = 0] without far
 * plane.
 * @ref http://dev.theomader.com/depth-precision/
 */
void perspective_matrix_reversed_z_infinite_far(float fovy, float aspect,
                                                float near, mat4 dest);

/**
 * @brief Generates a perspective projection matrix suitable for WebGPU with the
 * given bounds. The near/far clip planes correspond to a normalized device
 * coordinate Z range of [0, 1], which matches WebGPU/Vulkan/DirectX/Metal's
 * clip volume. Passing null/undefined/no value for far will generate infinite
 * projection matrix.
 *
 * @param out mat4 frustum matrix will be written into
 * @param fovy Vertical field of view in radians
 * @param aspect Aspect ratio. typically viewport width/height
 * @param near Near bound of the frustum
 * @param far Far bound of the frustum, can be null or Infinity
 * @returns out perspective projection matrix
 * @see
 * https://github.com/toji/gl-matrix/commit/e906eb7bb02822a81b1d197c6b5b33563c0403c0
 */
mat4* perspective_zo(mat4* out, float fovy, float aspect, float near,
                     const float* far);

#endif
