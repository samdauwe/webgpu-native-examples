#include "example_base.h"
#include "examples.h"

#include <limits.h>
#include <string.h>
#include <unistd.h>

#include <cJSON.h>
#include <sc_array.h>
#include <sc_queue.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Aquarium
 *
 * Aquarium is a native implementation of WebGL Aquarium.
 *
 * Ref:
 * https://github.com/webatintel/aquarium
 * https://webglsamples.org/aquarium/aquarium.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Aquarium Assert
 * -------------------------------------------------------------------------- */

#ifndef NDEBUG
#define AQUARIUM_ASSERT(expression)                                            \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
      abort();                                                                 \
    }                                                                          \
  }
#else
#define AQUARIUM_ASSERT(expression) NULL;
#endif

#ifndef NDEBUG
#define SWALLOW_ERROR(expression)                                              \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
    }                                                                          \
  }
#else
#define SWALLOW_ERROR(expression) expression
#endif

/* -------------------------------------------------------------------------- *
 * Matrix: Do matrix calculations including multiply, addition, substraction,
 * transpose, inverse, translation, etc.
 * -------------------------------------------------------------------------- */

static long long MATRIX_RANDOM_RANGE_ = 4294967296;

static void matrix_mul_matrix_matrix4(float* dst, const float* a,
                                      const float* b)
{
  float a00 = a[0];
  float a01 = a[1];
  float a02 = a[2];
  float a03 = a[3];
  float a10 = a[4 + 0];
  float a11 = a[4 + 1];
  float a12 = a[4 + 2];
  float a13 = a[4 + 3];
  float a20 = a[8 + 0];
  float a21 = a[8 + 1];
  float a22 = a[8 + 2];
  float a23 = a[8 + 3];
  float a30 = a[12 + 0];
  float a31 = a[12 + 1];
  float a32 = a[12 + 2];
  float a33 = a[12 + 3];
  float b00 = b[0];
  float b01 = b[1];
  float b02 = b[2];
  float b03 = b[3];
  float b10 = b[4 + 0];
  float b11 = b[4 + 1];
  float b12 = b[4 + 2];
  float b13 = b[4 + 3];
  float b20 = b[8 + 0];
  float b21 = b[8 + 1];
  float b22 = b[8 + 2];
  float b23 = b[8 + 3];
  float b30 = b[12 + 0];
  float b31 = b[12 + 1];
  float b32 = b[12 + 2];
  float b33 = b[12 + 3];
  dst[0]    = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
  dst[1]    = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
  dst[2]    = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
  dst[3]    = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
  dst[4]    = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
  dst[5]    = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
  dst[6]    = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
  dst[7]    = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
  dst[8]    = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
  dst[9]    = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
  dst[10]   = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
  dst[11]   = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;
  dst[12]   = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
  dst[13]   = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
  dst[14]   = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
  dst[15]   = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
}

static void matrix_inverse4(float* dst, const float* m)
{
  float m00    = m[0 * 4 + 0];
  float m01    = m[0 * 4 + 1];
  float m02    = m[0 * 4 + 2];
  float m03    = m[0 * 4 + 3];
  float m10    = m[1 * 4 + 0];
  float m11    = m[1 * 4 + 1];
  float m12    = m[1 * 4 + 2];
  float m13    = m[1 * 4 + 3];
  float m20    = m[2 * 4 + 0];
  float m21    = m[2 * 4 + 1];
  float m22    = m[2 * 4 + 2];
  float m23    = m[2 * 4 + 3];
  float m30    = m[3 * 4 + 0];
  float m31    = m[3 * 4 + 1];
  float m32    = m[3 * 4 + 2];
  float m33    = m[3 * 4 + 3];
  float tmp_0  = m22 * m33;
  float tmp_1  = m32 * m23;
  float tmp_2  = m12 * m33;
  float tmp_3  = m32 * m13;
  float tmp_4  = m12 * m23;
  float tmp_5  = m22 * m13;
  float tmp_6  = m02 * m33;
  float tmp_7  = m32 * m03;
  float tmp_8  = m02 * m23;
  float tmp_9  = m22 * m03;
  float tmp_10 = m02 * m13;
  float tmp_11 = m12 * m03;
  float tmp_12 = m20 * m31;
  float tmp_13 = m30 * m21;
  float tmp_14 = m10 * m31;
  float tmp_15 = m30 * m11;
  float tmp_16 = m10 * m21;
  float tmp_17 = m20 * m11;
  float tmp_18 = m00 * m31;
  float tmp_19 = m30 * m01;
  float tmp_20 = m00 * m21;
  float tmp_21 = m20 * m01;
  float tmp_22 = m00 * m11;
  float tmp_23 = m10 * m01;

  float t0 = (tmp_0 * m11 + tmp_3 * m21 + tmp_4 * m31)
             - (tmp_1 * m11 + tmp_2 * m21 + tmp_5 * m31);
  float t1 = (tmp_1 * m01 + tmp_6 * m21 + tmp_9 * m31)
             - (tmp_0 * m01 + tmp_7 * m21 + tmp_8 * m31);
  float t2 = (tmp_2 * m01 + tmp_7 * m11 + tmp_10 * m31)
             - (tmp_3 * m01 + tmp_6 * m11 + tmp_11 * m31);
  float t3 = (tmp_5 * m01 + tmp_8 * m11 + tmp_11 * m21)
             - (tmp_4 * m01 + tmp_9 * m11 + tmp_10 * m21);

  float d = 1.0f / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);

  dst[0] = d * t0;
  dst[1] = d * t1;
  dst[2] = d * t2;
  dst[3] = d * t3;
  dst[4] = d
           * ((tmp_1 * m10 + tmp_2 * m20 + tmp_5 * m30)
              - (tmp_0 * m10 + tmp_3 * m20 + tmp_4 * m30));
  dst[5] = d
           * ((tmp_0 * m00 + tmp_7 * m20 + tmp_8 * m30)
              - (tmp_1 * m00 + tmp_6 * m20 + tmp_9 * m30));
  dst[6] = d
           * ((tmp_3 * m00 + tmp_6 * m10 + tmp_11 * m30)
              - (tmp_2 * m00 + tmp_7 * m10 + tmp_10 * m30));
  dst[7] = d
           * ((tmp_4 * m00 + tmp_9 * m10 + tmp_10 * m20)
              - (tmp_5 * m00 + tmp_8 * m10 + tmp_11 * m20));
  dst[8] = d
           * ((tmp_12 * m13 + tmp_15 * m23 + tmp_16 * m33)
              - (tmp_13 * m13 + tmp_14 * m23 + tmp_17 * m33));
  dst[9] = d
           * ((tmp_13 * m03 + tmp_18 * m23 + tmp_21 * m33)
              - (tmp_12 * m03 + tmp_19 * m23 + tmp_20 * m33));
  dst[10] = d
            * ((tmp_14 * m03 + tmp_19 * m13 + tmp_22 * m33)
               - (tmp_15 * m03 + tmp_18 * m13 + tmp_23 * m33));
  dst[11] = d
            * ((tmp_17 * m03 + tmp_20 * m13 + tmp_23 * m23)
               - (tmp_16 * m03 + tmp_21 * m13 + tmp_22 * m23));
  dst[12] = d
            * ((tmp_14 * m22 + tmp_17 * m32 + tmp_13 * m12)
               - (tmp_16 * m32 + tmp_12 * m12 + tmp_15 * m22));
  dst[13] = d
            * ((tmp_20 * m32 + tmp_12 * m02 + tmp_19 * m22)
               - (tmp_18 * m22 + tmp_21 * m32 + tmp_13 * m02));
  dst[14] = d
            * ((tmp_18 * m12 + tmp_23 * m32 + tmp_15 * m02)
               - (tmp_22 * m32 + tmp_14 * m02 + tmp_19 * m12));
  dst[15] = d
            * ((tmp_22 * m22 + tmp_16 * m02 + tmp_21 * m12)
               - (tmp_20 * m12 + tmp_23 * m22 + tmp_17 * m02));
}

static void matrix_transpose4(float* dst, const float* m)
{
  float m00 = m[0 * 4 + 0];
  float m01 = m[0 * 4 + 1];
  float m02 = m[0 * 4 + 2];
  float m03 = m[0 * 4 + 3];
  float m10 = m[1 * 4 + 0];
  float m11 = m[1 * 4 + 1];
  float m12 = m[1 * 4 + 2];
  float m13 = m[1 * 4 + 3];
  float m20 = m[2 * 4 + 0];
  float m21 = m[2 * 4 + 1];
  float m22 = m[2 * 4 + 2];
  float m23 = m[2 * 4 + 3];
  float m30 = m[3 * 4 + 0];
  float m31 = m[3 * 4 + 1];
  float m32 = m[3 * 4 + 2];
  float m33 = m[3 * 4 + 3];

  dst[0]  = m00;
  dst[1]  = m10;
  dst[2]  = m20;
  dst[3]  = m30;
  dst[4]  = m01;
  dst[5]  = m11;
  dst[6]  = m21;
  dst[7]  = m31;
  dst[8]  = m02;
  dst[9]  = m12;
  dst[10] = m22;
  dst[11] = m32;
  dst[12] = m03;
  dst[13] = m13;
  dst[14] = m23;
  dst[15] = m33;
}

static void matrix_frustum(float* dst, float left, float right, float bottom,
                           float top, float near_, float far_)
{
  const float dx = right - left;
  const float dy = top - bottom;
  const float dz = near_ - far_;

  dst[0]  = 2 * near_ / dx;
  dst[1]  = 0;
  dst[2]  = 0;
  dst[3]  = 0;
  dst[4]  = 0;
  dst[5]  = 2 * near_ / dy;
  dst[6]  = 0;
  dst[7]  = 0;
  dst[8]  = (left + right) / dx;
  dst[9]  = (top + bottom) / dy;
  dst[10] = far_ / dz;
  dst[11] = -1;
  dst[12] = 0;
  dst[13] = 0;
  dst[14] = near_ * far_ / dz;
  dst[15] = 0;
}

static void matrix_get_axis(float* dst, const float* m, int axis)
{
  const int off = axis * 4;
  dst[0]        = m[off + 0];
  dst[1]        = m[off + 1];
  dst[2]        = m[off + 2];
}

static void matrix_mul_scalar_vector(float k, float* v, size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    v[i] = v[i] * k;
  }
}

static void matrix_add_vector(float* dst, const float* a, const float* b,
                              size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    dst[i] = a[i] + b[i];
  }
}

static void matrix_normalize(float* dst, const float* a, size_t length)
{
  float n = 0.0f;

  for (size_t i = 0; i < length; ++i) {
    n += a[i] * a[i];
  }
  n = sqrt(n);
  if (n > 0.00001f) {
    for (size_t i = 0; i < length; ++i) {
      dst[i] = a[i] / n;
    }
  }
  else {
    for (size_t i = 0; i < length; ++i) {
      dst[i] = 0;
    }
  }
}

static void matrix_sub_vector(float* dst, const float* a, const float* b,
                              size_t length)
{
  for (size_t i = 0; i < length; ++i) {
    dst[i] = a[i] - b[i];
  }
}

static void matrix_cross(float* dst, const float* a, const float* b)
{
  dst[0] = a[1] * b[2] - a[2] * b[1];
  dst[1] = a[2] * b[0] - a[0] * b[2];
  dst[2] = a[0] * b[1] - a[1] * b[0];
}

static float matrix_dot(float* a, float* b)
{
  return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

static void matrix_camera_look_at(float* dst, const float* eye,
                                  const float* target, const float* up)
{
  float t0[3];
  float t1[3];
  float t2[3];
  matrix_sub_vector(t0, eye, target, 3);
  matrix_normalize(t0, t0, 3);
  matrix_cross(t1, up, t0);
  matrix_normalize(t1, t1, 3);
  matrix_cross(t2, t0, t1);

  dst[0]  = t1[0];
  dst[1]  = t1[1];
  dst[2]  = t1[2];
  dst[3]  = 0;
  dst[4]  = t2[0];
  dst[5]  = t2[1];
  dst[6]  = t2[2];
  dst[7]  = 0;
  dst[8]  = t0[0];
  dst[9]  = t0[1];
  dst[10] = t0[2];
  dst[11] = 0;
  dst[12] = eye[0];
  dst[13] = eye[1];
  dst[14] = eye[2];
  dst[15] = 1;
}

static long long matrix_random_seed_ = 0;

void matrix_reset_pseudoRandom()
{
  matrix_random_seed_ = 0;
}

static double matrix_pseudo_random()
{
  matrix_random_seed_
    = (134775813 * matrix_random_seed_ + 1) % MATRIX_RANDOM_RANGE_;
  return ((double)matrix_random_seed_) / ((double)MATRIX_RANDOM_RANGE_);
}

void matrix_translation(float* dst, const float* v)
{
  dst[0]  = 1;
  dst[1]  = 0;
  dst[2]  = 0;
  dst[3]  = 0;
  dst[4]  = 0;
  dst[5]  = 1;
  dst[6]  = 0;
  dst[7]  = 0;
  dst[8]  = 0;
  dst[9]  = 0;
  dst[10] = 1;
  dst[11] = 0;
  dst[12] = v[0];
  dst[13] = v[1];
  dst[14] = v[2];
  dst[15] = 1;
}

void matrix_translate(float* m, const float* v)
{
  float v0  = v[0];
  float v1  = v[1];
  float v2  = v[2];
  float m00 = m[0];
  float m01 = m[1];
  float m02 = m[2];
  float m03 = m[3];
  float m10 = m[1 * 4 + 0];
  float m11 = m[1 * 4 + 1];
  float m12 = m[1 * 4 + 2];
  float m13 = m[1 * 4 + 3];
  float m20 = m[2 * 4 + 0];
  float m21 = m[2 * 4 + 1];
  float m22 = m[2 * 4 + 2];
  float m23 = m[2 * 4 + 3];
  float m30 = m[3 * 4 + 0];
  float m31 = m[3 * 4 + 1];
  float m32 = m[3 * 4 + 2];
  float m33 = m[3 * 4 + 3];

  m[12] = m00 * v0 + m10 * v1 + m20 * v2 + m30;
  m[13] = m01 * v0 + m11 * v1 + m21 * v2 + m31;
  m[14] = m02 * v0 + m12 * v1 + m22 * v2 + m32;
  m[15] = m03 * v0 + m13 * v1 + m23 * v2 + m33;
}

float deg_to_rad(float degrees)
{
  return degrees * PI / 180.0f;
}

/* -------------------------------------------------------------------------- *
 * Aquarium - Global enums
 * -------------------------------------------------------------------------- */

typedef enum {
  /* begin of background */
  MODELRUINCOLUMN,
  MODELARCH,
  MODELROCKA,
  MODELROCKB,
  MODELROCKC,
  MODELSUNKNSHIPBOXES,
  MODELSUNKNSHIPDECK,
  MODELSUNKNSHIPHULL,
  MODELFLOORBASE_BAKED,
  MODELSUNKNSUB,
  MODELCORAL,
  MODELSTONE,
  MODELCORALSTONEA,
  MODELCORALSTONEB,
  MODELGLOBEBASE,
  MODELTREASURECHEST,
  MODELENVIRONMENTBOX,
  MODELSUPPORTBEAMS,
  MODELSKYBOX,
  MODELGLOBEINNER,
  MODELSEAWEEDA,
  MODELSEAWEEDB,

  /* begin of fish */
  MODELSMALLFISHA,
  MODELMEDIUMFISHA,
  MODELMEDIUMFISHB,
  MODELBIGFISHA,
  MODELBIGFISHB,
  MODELSMALLFISHAINSTANCEDDRAWS,
  MODELMEDIUMFISHAINSTANCEDDRAWS,
  MODELMEDIUMFISHBINSTANCEDDRAWS,
  MODELBIGFISHAINSTANCEDDRAWS,
  MODELBIGFISHBINSTANCEDDRAWS,
  MODELMAX,
} model_name_t;

typedef enum {
  FISH,
  FISHINSTANCEDDRAW,
  INNER,
  SEAWEED,
  GENERIC,
  OUTSIDE,
  GROUPMAX,
} model_group_t;

typedef enum {
  FISHENUM_BIG,
  FISHENUM_MEDIUM,
  FISHENUM_SMALL,
  FISHENUM_MAX,
} fish_num_t;

typedef enum {
  /* Enable alpha blending */
  ENABLEALPHABLENDING,
  /* Go through instanced draw */
  ENABLEINSTANCEDDRAWS,
  // The toggle is only supported on Dawn backend
  // By default, the app will enable dynamic buffer offset
  // The toggle is to disable dbo feature
  ENABLEDYNAMICBUFFEROFFSET,
  /* Turn off render pass on dawn_d3d12 */
  DISABLED3D12RENDERPASS,
  /* Turn off dawn validation */
  DISABLEDAWNVALIDATION,
  /* Disable control panel */
  DISABLECONTROLPANEL,
  /* Select integrated gpu if available */
  INTEGRATEDGPU,
  /* Select discrete gpu if available */
  DISCRETEGPU,
  /* Draw per instance or model */
  DRAWPERMODEL,
  /* Support Full Screen mode */
  ENABLEFULLSCREENMODE,
  /* Print logs such as avg fps */
  PRINTLOG,
  /* Use async buffer mapping to upload data */
  BUFFERMAPPINGASYNC,
  /* Simulate fish come and go for Dawn backend */
  SIMULATINGFISHCOMEANDGO,
  /* Turn off vsync, donot limit fps to 60 */
  TURNOFFVSYNC,
  TOGGLEMAX,
} toggle_t;

/* -------------------------------------------------------------------------- *
 * Aquarium - Global classes
 * -------------------------------------------------------------------------- */

typedef struct {
  const char* name_str;
  model_name_t name;
  model_group_t type;
  struct {
    const char* vertex;
    const char* fragment;
  } shader;
  bool fog;
} g_scene_info_t;

typedef struct {
  const char* name;
  model_name_t model_name;
  fish_num_t type;
  float speed;
  float speed_range;
  float radius;
  float radius_range;
  float tail_speed;
  float height_offset;
  float height_range;

  float fish_length;
  float fish_wave_length;
  float fish_bend_amount;

  bool lasers;
  float laser_rot;
  vec3 laser_off;
  vec3 laser_scale;
} fish_t;

typedef struct {
  float tail_offset_mult;
  float end_of_dome;
  float tank_radius;
  float tank_height;
  float stand_height;
  float shark_speed;
  float shark_clock_offset;
  float shark_xclock;
  float shark_yclock;
  float shark_zclock;
  int numBubble_sets;
  float laser_eta;
  float laser_len_fudge;
  int num_light_rays;
  int light_ray_y;
  int light_ray_duration_min;
  int light_ray_duration_range;
  int light_ray_speed;
  int light_ray_spread;
  int light_ray_pos_range;
  float light_ray_rot_range;
  float light_ray_rot_lerp;
  float light_ray_offset;
  float bubble_timer;
  int bubble_index;

  int num_fish_small;
  int num_fish_medium;
  int num_fish_big;
  int num_fish_left_small;
  int num_fish_left_big;
  float sand_shininess;
  float sand_specular_factor;
  float generic_shininess;
  float generic_specular_factor;
  float outside_shininess;
  float outside_specular_factor;
  float seaweed_shininess;
  float seaweed_specular_factor;
  float inner_shininess;
  float inner_specular_factor;
  float fish_shininess;
  float fish_specular_factor;

  float speed;
  float target_height;
  float target_radius;
  float eye_height;
  float eye_speed;
  float filed_of_view;
  float ambient_red;
  float ambient_green;
  float ambient_blue;
  float fog_power;
  float fog_mult;
  float fog_offset;
  float fog_red;
  float fog_green;
  float fog_blue;
  float fish_height_range;
  float fish_height;
  float fish_speed;
  float fish_offset;
  float fish_xclock;
  float fish_yclock;
  float fish_zclock;
  float fish_tail_speed;
  float refraction_fudge;
  float eta;
  float tank_colorfudge;
  float fov_fudge;
  vec3 net_offset;
  float net_offset_mult;
  float eye_radius;
  float field_of_view;
} g_settings_t;

typedef struct {
  float projection[16];
  float view[16];
  float world_inverse[16];
  float view_projection_inverse[16];
  float sky_view[16];
  float sky_view_projection[16];
  float sky_view_projection_inverse[16];
  float eye_position[3];
  float target[3];
  float up[3];
  float v3t0[3];
  float v3t1[3];
  float m4t0[16];
  float m4t1[16];
  float m4t2[16];
  float m4t3[16];
  float color_mult[4];
  float start;
  float then;
  float mclock;
  float eye_clock;
  const char* alpha;
} global_t;

typedef struct {
  float light_world_pos[3];
  float padding;
  float view_projection[16];
  float view_inverse[16];
} light_world_position_uniform_t;

typedef struct {
  float world[16];
  float world_unverse_transpose[16];
  float world_view_projection[16];
} world_uniforms_t;

typedef struct {
  vec4 light_color;
  vec4 specular;
  vec4 ambient;
} light_uniforms_t;

typedef struct {
  float fog_power;
  float fog_mult;
  float fog_offset;
  float padding;
  vec4 fog_color;
} fog_uniforms_t;

typedef struct {
  vec3 world_position;
  float scale;
  vec3 next_position;
  float time;
  float padding[56]; // Padding to align with 256 byte offset.
} fish_per_t;

/* -------------------------------------------------------------------------- *
 * Aquarium - Global (constant) variables
 * -------------------------------------------------------------------------- */

enum {
  /* Begin of background */
  MODELRUINCOLUMN_COUNT      = 2,
  MODELARCH_COUNT            = 2,
  MODELROCKA_COUNT           = 5,
  MODELROCKB_COUNT           = 4,
  MODELROCKC_COUNT           = 4,
  MODELSUNKNSHIPBOXES_COUNT  = 3,
  MODELSUNKNSHIPDECK_COUNT   = 3,
  MODELSUNKNSHIPHULL_COUNT   = 3,
  MODELFLOORBASE_BAKED_COUNT = 1,
  MODELSUNKNSUB_COUNT        = 2,
  MODELCORAL_COUNT           = 16,
  MODELSTONE_COUNT           = 11,
  MODELCORALSTONEA_COUNT     = 4,
  MODELCORALSTONEB_COUNT     = 3,
  MODELGLOBEBASE_COUNT       = 1,
  MODELTREASURECHEST_COUNT   = 6,
  MODELENVIRONMENTBOX_COUNT  = 1,
  MODELSUPPORTBEAMS_COUNT    = 1,
  MODELSKYBOX_COUNT          = 1,
  MODELGLOBEINNER_COUNT      = 1,
  MODELSEAWEEDA_COUNT        = 11,
  MODELSEAWEEDB_COUNT        = 11,
};

static struct {
  bool enable_alpha_blending;        /* Enable alpha blending */
  bool enable_instanced_draw;        /* Go through instanced draw */
  bool enable_dynamic_buffer_offset; /* Enable dynamic buffer offset */
  bool draw_per_model;               /*  Draw per instance or model */
  bool print_log;                    /* Print logs such as avg fps */
  bool buffer_mapping_async;      /* Use async buffer mapping to upload data */
  bool simulate_fish_come_and_go; /* Simulate fish come and go */
  bool turn_off_vsync;            /* Turn off vsync, donot limit fps to 60 */
  uint32_t msaa_sample_count;     /* MSAA sample count */
} aquarium_settings;

static const g_scene_info_t g_scene_info[MODELMAX] = {
  {
    .name_str        = "SmallFishA",
    .name            = MODELSMALLFISHA,
    .type            = FISH,
    .shader.vertex   = "fishVertexShader",
    .shader.fragment = "fishReflectionFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "MediumFishA",
    .name            = MODELMEDIUMFISHA,
    .type            = FISH,
    .shader.vertex   = "fishVertexShader",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "MediumFishB",
    .name            = MODELMEDIUMFISHB,
    .type            = FISH,
    .shader.vertex   = "fishVertexShader",
    .shader.fragment = "fishReflectionFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "BigFishA",
    .name            = MODELBIGFISHA,
    .type            = FISH,
    .shader.vertex   = "fishVertexShader",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "BigFishB",
    .name            = MODELBIGFISHB,
    .type            = FISH,
    .shader.vertex   = "fishVertexShader",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "SmallFishA",
    .name            = MODELSMALLFISHAINSTANCEDDRAWS,
    .type            = FISHINSTANCEDDRAW,
    .shader.vertex   = "fishVertexShaderInstancedDraws",
    .shader.fragment = "fishReflectionFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "MediumFishA",
    .name            = MODELMEDIUMFISHAINSTANCEDDRAWS,
    .type            = FISHINSTANCEDDRAW,
    .shader.vertex   = "fishVertexShaderInstancedDraws",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "MediumFishB",
    .name            = MODELMEDIUMFISHBINSTANCEDDRAWS,
    .type            = FISHINSTANCEDDRAW,
    .shader.vertex   = "fishVertexShaderInstancedDraws",
    .shader.fragment = "fishReflectionFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "BigFishA",
    .name            = MODELBIGFISHAINSTANCEDDRAWS,
    .type            = FISHINSTANCEDDRAW,
    .shader.vertex   = "fishVertexShaderInstancedDraws",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "BigFishB",
    .name            = MODELBIGFISHBINSTANCEDDRAWS,
    .type            = FISHINSTANCEDDRAW,
    .shader.vertex   = "fishVertexShaderInstancedDraws",
    .shader.fragment = "fishNormalMapFragmentShader",
    .fog             = true,
  },
  {
    .name_str        = "Arch",
    .name            = MODELARCH,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "Coral",
    .name            = MODELCORAL,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "CoralStoneA",
    .name            = MODELCORALSTONEA,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "CoralStoneB",
    .name            = MODELCORALSTONEB,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "EnvironmentBox",
    .name            = MODELENVIRONMENTBOX,
    .type            = OUTSIDE,
    .shader.vertex   = "diffuseVertexShader",
    .shader.fragment = "diffuseFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "FloorBase_Baked",
    .name            = MODELFLOORBASE_BAKED,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "GlobeBase",
    .name            = MODELGLOBEBASE,
    .type            = GENERIC,
    .shader.vertex   = "diffuseVertexShader",
    .shader.fragment = "diffuseFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "GlobeInner",
    .name            = MODELGLOBEINNER,
    .type            = INNER,
    .shader.vertex   = "innerRefractionMapVertexShader",
    .shader.fragment = "innerRefractionMapFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "RockA",
    .name            = MODELROCKA,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "RockB",
    .name            = MODELROCKB,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "RockC",
    .name            = MODELROCKC,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "RuinColumn",
    .name            = MODELRUINCOLUMN,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "Stone",
    .name            = MODELSTONE,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "SunknShipBoxes",
    .name            = MODELSUNKNSHIPBOXES,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "SunknShipDeck",
    .name            = MODELSUNKNSHIPDECK,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "SunknShipHull",
    .name            = MODELSUNKNSHIPHULL,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "SunknSub",
    .name            = MODELSUNKNSUB,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
  {
    .name_str        = "SeaweedA",
    .name            = MODELSEAWEEDA,
    .type            = SEAWEED,
    .shader.vertex   = "seaweedVertexShader",
    .shader.fragment = "seaweedFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "SeaweedB",
    .name            = MODELSEAWEEDB,
    .type            = SEAWEED,
    .shader.vertex   = "seaweedVertexShader",
    .shader.fragment = "seaweedFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "Skybox",
    .name            = MODELSKYBOX,
    .type            = OUTSIDE,
    .shader.vertex   = "diffuseVertexShader",
    .shader.fragment = "diffuseFragmentShader",
    .fog             = false,
  },
  {
    .name_str        = "SupportBeams",
    .name            = MODELSUPPORTBEAMS,
    .type            = OUTSIDE,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = false,
  },
  {
    .name_str        = "TreasureChest",
    .name            = MODELTREASURECHEST,
    .type            = GENERIC,
    .shader.vertex   = "",
    .shader.fragment = "",
    .fog             = true,
  },
};

static const fish_t fish_table[5] = {
  {
    .name             = "SmallFishA",
    .model_name       = MODELSMALLFISHA,
    .type             = FISHENUM_SMALL,
    .speed            = 1.0f,
    .speed_range      = 1.5f,
    .radius           = 30.0f,
    .radius_range     = 25.0f,
    .tail_speed       = 10.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = 1.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "MediumFishA",
    .model_name       = MODELMEDIUMFISHA,
    .type             = FISHENUM_MEDIUM,
    .speed            = 1.0f,
    .speed_range      = 2.0f,
    .radius           = 10.0f,
    .radius_range     = 20.0f,
    .tail_speed       = 1.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -2.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "MediumFishB",
    .model_name       = MODELMEDIUMFISHB,
    .type             = FISHENUM_MEDIUM,
    .speed            = 0.5f,
    .speed_range      = 4.0f,
    .radius           = 10.0f,
    .radius_range     = 20.0f,
    .tail_speed       = 3.0f,
    .height_offset    = -8.0f,
    .height_range     = 5.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -2.0f,
    .fish_bend_amount = 2.0f,
  },
  {
    .name             = "BigFishA",
    .model_name       = MODELBIGFISHA,
    .type             = FISHENUM_BIG,
    .speed            = 0.5f,
    .speed_range      = 0.5f,
    .radius           = 50.0f,
    .radius_range     = 3.0f,
    .tail_speed       = 1.5f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -1.0f,
    .fish_bend_amount = 0.5f,
    .lasers           = true,
    .laser_rot        = 0.04f,
    .laser_off        = {0.0f, 0.1f, 9.0f},
    .laser_scale      = {0.3f, 0.3f, 1000.0f},
  },
  {
    .name             = "BigFishB",
    .model_name       = MODELBIGFISHA,
    .type             = FISHENUM_BIG,
    .speed            = 0.5f,
    .speed_range      = 0.5f,
    .radius           = 45.0f,
    .radius_range     = 3.0f,
    .tail_speed       = 1.0f,
    .height_offset    = 0.0f,
    .height_range     = 16.0f,
    .fish_length      = 10.0f,
    .fish_wave_length = -0.7f,
    .fish_bend_amount = 0.3f,
    .lasers           = true,
    .laser_rot        = 0.04f,
    .laser_off        = {0.0f, -0.3f, 9.0f},
    .laser_scale      = {0.3f, 0.3f, 1000.0f},
  },
};

static const int g_num_light_rays = 5;

static g_settings_t g_settings = {
  .tail_offset_mult         = 1.0f,
  .end_of_dome              = PI / 8.f,
  .tank_radius              = 74.0f,
  .tank_height              = 36.0f,
  .stand_height             = 25.0f,
  .shark_speed              = 0.3f,
  .shark_clock_offset       = 17.0f,
  .shark_xclock             = 1.0f,
  .shark_yclock             = 0.17f,
  .shark_zclock             = 1.0f,
  .numBubble_sets           = 10,
  .laser_eta                = 1.2f,
  .laser_len_fudge          = 1.0f,
  .num_light_rays           = g_num_light_rays,
  .light_ray_y              = 50,
  .light_ray_duration_min   = 1,
  .light_ray_duration_range = 1,
  .light_ray_speed          = 4,
  .light_ray_spread         = 7,
  .light_ray_pos_range      = 20,
  .light_ray_rot_range      = 1.0f,
  .light_ray_rot_lerp       = 0.2f,
  .light_ray_offset         = PI2 / (float)g_num_light_rays,
  .bubble_timer             = 0.0f,
  .bubble_index             = 0,

  .num_fish_small          = 100,
  .num_fish_medium         = 1000,
  .num_fish_big            = 10000,
  .num_fish_left_small     = 80,
  .num_fish_left_big       = 160,
  .sand_shininess          = 5.0f,
  .sand_specular_factor    = 0.3f,
  .generic_shininess       = 50.0f,
  .generic_specular_factor = 1.0f,
  .outside_shininess       = 50.0f,
  .outside_specular_factor = 0.0f,
  .seaweed_shininess       = 50.0f,
  .seaweed_specular_factor = 1.0f,
  .inner_shininess         = 50.0f,
  .inner_specular_factor   = 1.0f,
  .fish_shininess          = 5.0f,
  .fish_specular_factor    = 0.3f,

  .speed             = 1.0f,
  .target_height     = 63.3f,
  .target_radius     = 91.6f,
  .eye_height        = 7.5f,
  .eye_speed         = 0.0258f,
  .filed_of_view     = 82.699f,
  .ambient_red       = 0.218f,
  .ambient_green     = 0.502f,
  .ambient_blue      = 0.706f,
  .fog_power         = 16.5f,
  .fog_mult          = 1.5f,
  .fog_offset        = 0.738f,
  .fog_red           = 0.338f,
  .fog_green         = 0.81f,
  .fog_blue          = 1.0f,
  .fish_height_range = 1.0f,
  .fish_height       = 25.0f,
  .fish_speed        = 0.124f,
  .fish_offset       = 0.52f,
  .fish_xclock       = 1.0f,
  .fish_yclock       = 0.556f,
  .fish_zclock       = 1.0f,
  .fish_tail_speed   = 1.0f,
  .refraction_fudge  = 3.0f,
  .eta               = 1.0f,
  .tank_colorfudge   = 0.796f,
  .fov_fudge         = 1.0f,
  .net_offset        = {0.0f, 0.0f, 0.0f},
  .net_offset_mult   = 1.21f,
  .eye_radius        = 13.2f,
  .field_of_view     = 82.699f,
};

/* -------------------------------------------------------------------------- *
 * FPSTimer - Defines fps timer, uses millseconds time unit.
 * -------------------------------------------------------------------------- */

#define NUM_HISTORY_DATA 100
#define NUM_FRAMES_TO_AVERAGE 128
#define FPS_VALID_THRESHOLD 5

sc_array_def(float, float);

typedef struct {
  float total_time;
  struct sc_array_float time_table;
  int32_t time_table_cursor;
  struct sc_array_float history_fps;
  struct sc_array_float history_frame_time;
  struct sc_array_float log_fps;
  float average_fps;
} fps_timer_t;

static void fps_timer_init_defaults(fps_timer_t* this)
{
  memset(this, 0, sizeof(*this));

  this->total_time = NUM_FRAMES_TO_AVERAGE * 1000.0f;

  for (uint32_t i = 0; i < NUM_FRAMES_TO_AVERAGE; ++i) {
    sc_array_add(&this->time_table, 1000.0f);
  }

  this->time_table_cursor = 0;

  for (uint32_t i = 0; i < NUM_HISTORY_DATA; ++i) {
    sc_array_add(&this->history_fps, 1.0f);
    sc_array_add(&this->history_frame_time, 100.0f);
  }

  this->average_fps = 0.0f;
}

static void fps_timer_create(fps_timer_t* this)
{
  fps_timer_init_defaults(this);
}

static void fps_timer_update(fps_timer_t* this, float elapsed_time,
                             float rendering_time, float test_time)
{
  this->total_time
    += elapsed_time - this->time_table.elems[this->time_table_cursor];
  this->time_table.elems[this->time_table_cursor] = elapsed_time;

  ++this->time_table_cursor;
  if (this->time_table_cursor == NUM_FRAMES_TO_AVERAGE) {
    this->time_table_cursor = 0;
  }

  float frame_time  = this->total_time / NUM_FRAMES_TO_AVERAGE;
  this->average_fps = floor(1000.0f / frame_time + 0.5f);

  for (uint32_t i = 0; i < NUM_HISTORY_DATA - 1; ++i) {
    this->history_fps.elems[i]        = this->history_fps.elems[i + 1];
    this->history_frame_time.elems[i] = this->history_frame_time.elems[i + 1];
  }

  this->history_fps.elems[NUM_HISTORY_DATA - 1] = this->average_fps;
  this->history_frame_time.elems[NUM_HISTORY_DATA - 1]
    = 1000.0 / this->average_fps;

  if (test_time - rendering_time > 5000.0f
      && test_time - rendering_time < 25000.0f) {
    sc_array_add(&this->log_fps, this->average_fps);
  }
}

static float fps_timer_get_average_fps(fps_timer_t* this)
{
  return this->average_fps;
}

static int32_t fps_timer_variance(fps_timer_t* this)
{
  float avg = 0.0f;

  for (size_t i = 0; i < sc_array_size(&this->log_fps); ++i) {
    avg += this->log_fps.elems[i];
  }
  avg /= sc_array_size(&this->log_fps);

  float var = 0.0f;
  for (size_t i = 0; i < sc_array_size(&this->log_fps); ++i) {
    var += pow(this->log_fps.elems[i] - avg, 2);
  }
  var /= sc_array_size(&this->log_fps);

  if (var < FPS_VALID_THRESHOLD) {
    return (int32_t)ceil(avg);
  }

  return 0;
}

/* -------------------------------------------------------------------------- *
 * Behavior - Base class for behavior.
 * -------------------------------------------------------------------------- */

typedef enum {
  OPERATION_PLUS,
  OPERATION_MINUS,
} behavior_op_t;

typedef struct {
  int32_t frame;
  behavior_op_t op;
  int32_t count;
} behavior_t;

static void behavior_create(behavior_t* this, int32_t frame, char op,
                            int32_t count)
{
  this->frame = frame;
  this->op    = (op == '+') ? OPERATION_PLUS : OPERATION_MINUS;
  this->count = count;
}

static int32_t behavior_get_frame(behavior_t* this)
{
  return this->frame;
}

static behavior_op_t behavior_get_op(behavior_t* this)
{
  return this->op;
}

static int32_t behavior_get_count(behavior_t* this)
{
  return this->count;
}

static void behavior_set_frame(behavior_t* this, int32_t frame)
{
  this->frame = frame;
}

/* -------------------------------------------------------------------------- *
 * Program - Load programs from shaders.
 * -------------------------------------------------------------------------- */

typedef struct {
  void* context;
  char vertex_shader_wgsl_path[STRMAX];
  char fragment_shader_wgsl_path[STRMAX];
  wgpu_shader_t vs_module;
  wgpu_shader_t fs_module;
} program_t;

/* Forward declarations */
static wgpu_shader_t context_create_shader_module(void* context,
                                                  WGPUShaderStage shader_stage,
                                                  const char* shader_wgsl_path);

static void program_init_defaults(program_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void program_create(program_t* this, void* context,
                           const char* vertex_shader_wgsl_path,
                           const char* fragment_shader_wgsl_path)
{
  program_init_defaults(this);
  this->context = context;

  snprintf(this->vertex_shader_wgsl_path, strlen(vertex_shader_wgsl_path) + 1,
           "%s", vertex_shader_wgsl_path);
  snprintf(this->fragment_shader_wgsl_path,
           strlen(fragment_shader_wgsl_path) + 1, "%s",
           fragment_shader_wgsl_path);
}

static void program_destroy(program_t* this)
{
  WGPU_RELEASE_RESOURCE(ShaderModule, this->vs_module.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, this->fs_module.module);
}

static WGPUShaderModule program_get_vs_module(program_t* this)
{
  return this->vs_module.module;
}

static WGPUShaderModule program_get_fs_module(program_t* this)
{
  return this->fs_module.module;
}

static void program_compile_program(program_t* this)
{
  this->vs_module = context_create_shader_module(
    this->context, WGPUShaderStage_Vertex, this->vertex_shader_wgsl_path);
  this->fs_module = context_create_shader_module(
    this->context, WGPUShaderStage_Fragment, this->fragment_shader_wgsl_path);
}

/* -------------------------------------------------------------------------- *
 * Dawn Buffer - Defines the buffer wrapper of dawn, abstracting the vetex and
 * index buffer binding.
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUBuffer buffer;
  WGPUBufferUsage usage;
  int32_t total_components;
  uint32_t stride;
  void* offset;
  int32_t size;
  bool valid;
} buffer_dawn_t;

/* Forward declarations */
static WGPUBuffer context_create_buffer(void* context,
                                        WGPUBufferDescriptor const* descriptor);
static void context_set_buffer_data(void* context, WGPUBuffer buffer,
                                    uint32_t buffer_size, const void* data,
                                    uint32_t data_size);
static void context_update_buffer_data(void* context, WGPUBuffer buffer,
                                       size_t buffer_size, void* data,
                                       size_t data_size);

static size_t calc_constant_buffer_byte_size(size_t byte_size)
{
  return (byte_size + 255) & ~255;
}

static void buffer_dawn_create_f32(buffer_dawn_t* this, void* context,
                                   int32_t total_components,
                                   int32_t num_components, float* buffer,
                                   bool is_index)
{
  this->usage = is_index ? WGPUBufferUsage_Index : WGPUBufferUsage_Vertex;
  this->total_components = total_components;
  this->stride           = 0;
  this->offset           = NULL;

  this->size = num_components * sizeof(float);
  // Create buffer for vertex buffer. Because float is multiple of 4 bytes,
  // dummy padding isnt' needed.
  uint64_t buffer_size             = sizeof(float) * num_components;
  WGPUBufferDescriptor buffer_desc = {
    .usage            = this->usage | WGPUBufferUsage_CopyDst,
    .size             = buffer_size,
    .mappedAtCreation = false,
  };
  this->buffer = context_create_buffer(context, &buffer_desc);

  context_set_buffer_data(context, this->buffer, buffer_size, buffer,
                          buffer_size);
}

static void buffer_dawn_create_uint16(buffer_dawn_t* this, void* context,
                                      int32_t total_components,
                                      int32_t num_components, uint16_t* buffer,
                                      uint64_t buffer_count, bool is_index)
{
  this->usage = is_index ? WGPUBufferUsage_Index : WGPUBufferUsage_Vertex;
  this->total_components = total_components;
  this->stride           = 0;
  this->offset           = NULL;

  this->size = num_components * sizeof(uint16_t);
  // Create buffer for index buffer. Because unsigned short is multiple of 2
  // bytes, in order to align with 4 bytes of dawn metal, dummy padding need to
  // be added.
  if (total_components % 2 != 0) {
    ASSERT((uint64_t)num_components <= buffer_count);
    buffer[num_components] = 0;
    num_components++;
  }

  uint64_t buffer_size             = sizeof(uint16_t) * num_components;
  WGPUBufferDescriptor buffer_desc = {
    .usage            = this->usage | WGPUBufferUsage_CopyDst,
    .size             = buffer_size,
    .mappedAtCreation = false,
  };
  this->buffer = context_create_buffer(context, &buffer_desc);

  context_set_buffer_data(context, this->buffer, buffer_size, buffer,
                          buffer_size);
}

static void buffer_dawn_destroy(buffer_dawn_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer)
  memset(this, 0, sizeof(*this));
}

static WGPUBuffer buffer_dawn_get_buffer(buffer_dawn_t* this)
{
  return this->buffer;
}

static int32_t buffer_dawn_get_total_components(buffer_dawn_t* this)
{
  return this->total_components;
}

static uint32_t buffer_dawn_get_stride(buffer_dawn_t* this)
{
  return this->stride;
}

static void* buffer_dawn_get_offset(buffer_dawn_t* this)
{
  return this->offset;
}

static WGPUBufferUsage buffer_dawn_get_usage_bit(buffer_dawn_t* this)
{
  return this->usage;
}

static int32_t buffer_dawn_get_data_size(buffer_dawn_t* this)
{
  return this->size;
}

/* -------------------------------------------------------------------------- *
 * Buffer Manager - Implements buffer pool to manage buffer allocation and
 * recycle.
 * -------------------------------------------------------------------------- */

#define BUFFER_POOL_MAX_SIZE 409600000ull
#define BUFFER_MAX_COUNT 10ull
#define BUFFER_PER_ALLOCATE_SIZE (BUFFER_POOL_MAX_SIZE / BUFFER_MAX_COUNT)

typedef struct {
  size_t head;
  size_t tail;
  size_t size;

  void* buffer_manager;
  wgpu_context_t* wgpu_context;
  WGPUBuffer buf;
  void* mapped_data;
  void* pixels;
} ring_buffer_t;

sc_array_def(ring_buffer_t*, ring_buffer);
sc_queue_def(ring_buffer_t*, ring_buffer);

typedef struct {
  wgpu_context_t* wgpu_context;
  struct sc_queue_ring_buffer mapped_buffer_list;
  struct sc_array_ring_buffer enqueued_buffer_list;
  size_t buffer_pool_size;
  size_t used_size;
  size_t count;
  WGPUCommandEncoder encoder;
  bool sync;
} buffer_manager_t;

static size_t ring_buffer_get_size(ring_buffer_t* this)
{
  return this->size;
}

static size_t ring_buffer_get_available_size(ring_buffer_t* this)
{
  return this->size - this->tail;
}

bool ring_buffer_push(ring_buffer_t* this, WGPUCommandEncoder encoder,
                      WGPUBuffer dest_buffer, size_t src_offset,
                      size_t dest_offset, void* pixels, size_t size)
{
  memcpy(((unsigned char*)this->pixels) + src_offset, pixels, size);
  wgpuCommandEncoderCopyBufferToBuffer(encoder, this->buf, src_offset,
                                       dest_buffer, dest_offset, size);
  return true;
}

/* Reset current buffer and reuse the buffer. */
static bool ring_buffer_reset(ring_buffer_t* this, size_t size)
{
  if (size > this->size) {
    return false;
  }

  this->head = 0;
  this->tail = 0;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopyDst,
    .size             = this->size,
    .mappedAtCreation = true,
  };
  this->buf = context_create_buffer(this->wgpu_context, &buffer_desc);
  ASSERT(this->buf);
  this->pixels = wgpuBufferGetMappedRange(this->buf, 0, this->size);

  return true;
}

static void ring_buffer_create(ring_buffer_t* this,
                               buffer_manager_t* buffer_manager, size_t size)
{
  this->head = 0;
  this->tail = size;
  this->size = size;

  this->buffer_manager = buffer_manager;
  this->wgpu_context   = buffer_manager->wgpu_context;
  this->mapped_data    = NULL;
  this->pixels         = NULL;

  ring_buffer_reset(this, size);
}

static void ring_buffer_map_callback(WGPUBufferMapAsyncStatus status,
                                     void* user_data)
{
  if (status == WGPUBufferMapAsyncStatus_Success) {
    ring_buffer_t* ring_buffer = (ring_buffer_t*)user_data;
    ring_buffer->mapped_data   = (uint64_t*)wgpuBufferGetMappedRange(
        ring_buffer->buf, 0, ring_buffer->size);
    ASSERT(ring_buffer->mapped_data);
  }
}

static void ring_buffer_flush(ring_buffer_t* this)
{
  this->head = 0;
  this->tail = 0;

  wgpuBufferUnmap(this->buf);
}

static void ring_buffer_destroy(ring_buffer_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buf);
}

static void ring_buffer_re_map(ring_buffer_t* this)
{
  wgpuBufferMapAsync(this->buf, WGPUMapMode_Write, 0, 0,
                     ring_buffer_map_callback, this);
}

/* allocate size in a ring_buffer_t, return offset of the buffer */
static size_t ring_buffer_allocate(ring_buffer_t* this, size_t size)
{
  this->tail += size;
  ASSERT(this->tail < this->size);

  return this->tail - size;
}

static size_t buffer_manager_find(buffer_manager_t* this,
                                  ring_buffer_t* ring_buffer);

static void buffer_manager_init_defaults(buffer_manager_t* this)
{
  memset(this, 0, sizeof(*this));

  this->buffer_pool_size = BUFFER_POOL_MAX_SIZE;
  this->used_size        = 0;
  this->count            = 0;
}

static void buffer_manager_create(buffer_manager_t* this)
{
  buffer_manager_init_defaults(this);
  this->encoder
    = wgpuDeviceCreateCommandEncoder(this->wgpu_context->device, NULL);
}

static void buffer_manager_destroy_buffer_pool(buffer_manager_t* this)
{
  if (!this->sync) {
    return;
  }

  for (size_t i = 0; i < sc_array_size(&this->enqueued_buffer_list); i++) {
    ring_buffer_destroy(this->enqueued_buffer_list.elems[i]);
  }
  sc_array_clear(&this->enqueued_buffer_list);
}

static void buffer_manager_destroy(buffer_manager_t* this)
{
  buffer_manager_destroy_buffer_pool(this);
  WGPU_RELEASE_RESOURCE(CommandEncoder, this->encoder)
}

static size_t buffer_manager_get_size(buffer_manager_t* this)
{
  return this->buffer_pool_size;
}

static bool buffer_manager_reset_buffer(buffer_manager_t* this,
                                        ring_buffer_t* ring_buffer, size_t size)
{
  size_t index = buffer_manager_find(this, ring_buffer);

  if (index >= sc_array_size(&this->enqueued_buffer_list)) {
    return false;
  }

  size_t old_size = ring_buffer_get_size(ring_buffer);

  bool result = ring_buffer_reset(ring_buffer, size);
  // If the size is larger than the ring buffer size, reset fails and the ring
  // buffer retains.
  // If the size is equal or smaller than the ring buffer size, reset success
  // and the used size need to be updated.
  if (!result) {
    return false;
  }
  else {
    this->used_size = this->used_size - old_size + size;
  }

  return true;
}

static bool buffer_manager_destroy_buffer(buffer_manager_t* this,
                                          ring_buffer_t* ring_buffer)
{
  size_t index = buffer_manager_find(this, ring_buffer);

  if (index >= sc_array_size(&this->enqueued_buffer_list)) {
    return false;
  }

  this->used_size -= ring_buffer_get_size(ring_buffer);
  ring_buffer_destroy(ring_buffer);
  sc_array_del(&this->enqueued_buffer_list, index);

  return true;
}

static size_t buffer_manager_find(buffer_manager_t* this,
                                  ring_buffer_t* ring_buffer)
{
  size_t index = 0;
  for (index = 0; index < sc_array_size(&this->enqueued_buffer_list); index++) {
    if (this->enqueued_buffer_list.elems[index] == ring_buffer) {
      break;
    }
  }
  return index;
}

/* Flush copy commands in buffer pool */
static void buffer_manager_flush(buffer_manager_t* this)
{
  // The front buffer in MappedBufferList will be remap after submit, pop the
  // buffer from MappedBufferList.
  if (sc_array_size(&this->enqueued_buffer_list) == 0
      && (sc_array_last(&this->enqueued_buffer_list))
           == (sc_queue_peek_first(&this->mapped_buffer_list))) {
    sc_queue_del_first(&this->mapped_buffer_list);
  }

  ring_buffer_t* buffer = NULL;
  sc_array_foreach(&this->enqueued_buffer_list, buffer)
  {
    ring_buffer_flush(buffer);
  }

  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(this->encoder, NULL);
  wgpuQueueSubmit(this->wgpu_context->queue, 1, &copy);

  /* Async function */
  if (!this->sync) {
    sc_array_foreach(&this->enqueued_buffer_list, buffer)
    {
      ring_buffer_re_map(buffer);
    }
  }
  else {
    /* All buffers are used once in buffer sync mode. */
    for (size_t i = 0; i < sc_array_size(&this->enqueued_buffer_list); i++) {
      free(this->enqueued_buffer_list.elems[i]);
    }
    this->used_size = 0;
  }

  sc_array_clear(&this->enqueued_buffer_list);
  this->encoder
    = wgpuDeviceCreateCommandEncoder(this->wgpu_context->device, NULL);
}

/* Allocate new buffer from buffer pool. */
static ring_buffer_t* buffer_manager_allocate(buffer_manager_t* this,
                                              size_t size, size_t* offset)
{
  // If update data by sync method, create new buffer to upload every frame.
  // If updaye data by async method, get new buffer from pool if available. If
  // no available buffer and size is enough in the buffer pool, create a new
  // buffer. If size reach the limit of the buffer pool, force wait for the
  // buffer on mapping. Get the last one and check if the ring buffer is full.
  // If the buffer can hold extra size space, use the last one directly.
  // TODO(yizhou): Return nullptr if size reach the limit or no available
  // buffer, this means small bubbles in some of the ring buffers and we haven't
  // deal with the problem now.

  ring_buffer_t* ring_buffer = NULL;
  size_t cur_offset          = 0;
  if (!this->sync) {
    /* Upper limit */
    if (this->used_size + size > this->buffer_pool_size) {
      return NULL;
    }

    ring_buffer = malloc(sizeof(ring_buffer_t));
    ring_buffer_create(ring_buffer, this, size);
    sc_array_add(&this->enqueued_buffer_list, ring_buffer);
  }
  else { /* Buffer mapping async */
    while (!(sc_queue_size(&this->mapped_buffer_list) == 0)) {
      ring_buffer = sc_queue_peek_first(&this->mapped_buffer_list);
      if (ring_buffer_get_available_size(ring_buffer) < size) {
        sc_queue_del_first(&this->mapped_buffer_list);
        ring_buffer = NULL;
      }
      else {
        break;
      }
    }

    if (ring_buffer == NULL) {
      if (this->count < BUFFER_MAX_COUNT) {
        this->used_size += size;
        ring_buffer = malloc(sizeof(ring_buffer_t));
        ring_buffer_create(ring_buffer, this, BUFFER_PER_ALLOCATE_SIZE);
        sc_queue_add_last(&this->mapped_buffer_list, ring_buffer);
        this->count++;
      }
      else if (sc_queue_size(&this->mapped_buffer_list)
                 + sc_array_size(&this->enqueued_buffer_list)
               < this->count) {
        /* Force wait for the buffer remapping */
        while (sc_queue_size(&this->mapped_buffer_list) == 0) {
          printf("mContext->WaitABit();\n");
        }

        ring_buffer = sc_queue_peek_first(&this->mapped_buffer_list);
        if (ring_buffer_get_available_size(ring_buffer) < size) {
          sc_queue_del_first(&this->mapped_buffer_list);
          ring_buffer = NULL;
        }
      }
      else { /* Upper limit */
        return NULL;
      }
    }

    if (sc_array_size(&this->enqueued_buffer_list) == 0
        && (sc_array_last(&this->enqueued_buffer_list)) != ring_buffer) {
      sc_queue_add_last(&this->mapped_buffer_list, ring_buffer);
    }

    /* allocate size in the ring buffer */
    cur_offset = ring_buffer_allocate(ring_buffer, size);
    *offset    = cur_offset;
  }

  return ring_buffer;
}

/* -------------------------------------------------------------------------- *
 * Aquarium context - Defines outside model of Dawn
 * -------------------------------------------------------------------------- */

sc_array_def(WGPUCommandBuffer, command_buffer);

typedef struct {
  wgpu_context_t* wgpu_context;
  WGPUDevice device;
  uint32_t client_width;
  uint32_t client_height;
  uint32_t pre_total_instance;
  uint32_t cur_total_instance;
  uint32_t msaa_sample_count;
  struct sc_array_command_buffer command_buffers;
  struct {
    WGPUBindGroupLayout general;
    WGPUBindGroupLayout world;
    WGPUBindGroupLayout fish_per;
  } bind_group_layouts;
  struct {
    WGPUBindGroup general;
    WGPUBindGroup world;
    WGPUBindGroup fish_per;
  } bind_groups;
  WGPUBuffer fish_pers_buffer;
  WGPUBindGroup* bind_group_fish_pers;
  fish_per_t* fish_pers;
  WGPUTextureUsage swapchain_back_buffer_usage;
  bool is_swapchain_out_of_date;
  WGPUCommandEncoder command_encoder;
  WGPURenderPassEncoder render_pass;
  WGPURenderPassDescriptor render_pass_descriptor;
  struct {
    WGPUTextureView backbuffer;
    texture_t scene_render_target;
    texture_t scene_depth_stencil;
  } texture_views;
  WGPURenderPipeline pipeline;
  WGPUBindGroup bind_group;
  WGPUTextureFormat preferred_swap_chain_format;
  struct {
    WGPUBuffer light_world_position;
    WGPUBuffer light;
    WGPUBuffer fog;
  } uniform_buffers;
  bool enable_dynamic_buffer_offset;
  buffer_manager_t* buffer_manager;
} context_t;

sc_queue_def(behavior_t*, behavior);

// All state is in a single nested struct
typedef struct {
  wgpu_context_t* wgpu_context;
  context_t* context;
  light_world_position_uniform_t light_world_position_uniform;
  world_uniforms_t world_uniforms;
  light_uniforms_t light_uniforms;
  fog_uniforms_t fog_uniforms;
  global_t g;
  int32_t fish_count[5];
  struct {
    char key[STRMAX];
    model_name_t value;
  } model_enum_map[MODELMAX];
  int32_t cur_fish_count;
  int32_t pre_fish_count;
  int32_t test_time;
  struct sc_queue_behavior fish_behavior;
} aquarium_t;

/* Forward declarations context */
static void context_realloc_resource(context_t* this,
                                     uint32_t pre_total_instance,
                                     uint32_t cur_total_instance,
                                     bool enable_dynamic_buffer_offset);

/* Forward declarations aquarium */
static int32_t aquarium_get_cur_fish_count(aquarium_t* this);
static int32_t aquarium_get_pre_fish_count(aquarium_t* this);

static void context_update_world_uniforms(context_t* this,
                                          aquarium_t* aquarium);

static void context_init_defaults(context_t* this)
{
  memset(this, 0, sizeof(*this));

  this->preferred_swap_chain_format = WGPUTextureFormat_RGBA8Unorm;
  this->msaa_sample_count           = 1;
}

static void context_create(context_t* this)
{
  context_init_defaults(this);
}

static void context_detroy(context_t* this)
{
}

static void context_initialize(context_t* this)
{
}

static void context_set_window_size(context_t* this, uint32_t window_width,
                                    uint32_t window_height)
{
  if (window_width != 0) {
    this->client_width = window_width;
  }
  if (window_height != 0) {
    this->client_height = window_height;
  }
}

static uint32_t context_get_client_width(context_t* this)
{
  return this->client_width;
}

static uint32_t context_get_client_height(context_t* this)
{
  return this->client_height;
}

static void set_msaa_sample_count(context_t* this, uint32_t msaa_sample_count)
{
  this->msaa_sample_count = msaa_sample_count;
}

static WGPUSampler
context_create_sampler(context_t* this, WGPUSamplerDescriptor const* descriptor)
{
  return wgpuDeviceCreateSampler(this->device, descriptor);
}

static WGPUBuffer context_create_buffer_from_data(context_t* this,
                                                  const void* data,
                                                  uint32_t size,
                                                  uint32_t max_size,
                                                  WGPUBufferUsage usage)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = usage | WGPUBufferUsage_CopyDst,
    .size             = max_size,
    .mappedAtCreation = false,
  };
  WGPUBuffer buffer = context_create_buffer(wgpu_context, &buffer_desc);

  context_set_buffer_data(wgpu_context, buffer, max_size, data, size);
  ASSERT(buffer != NULL);
  return buffer;
}

static WGPUImageCopyBuffer
context_create_image_copy_buffer(context_t* this, WGPUBuffer buffer,
                                 uint32_t offset, uint32_t bytes_per_row,
                                 uint32_t rows_per_image)
{
  UNUSED_VAR(this);

  WGPUImageCopyBuffer image_copy_buffer_desc = {
    .layout.offset       = offset,
    .layout.bytesPerRow  = bytes_per_row,
    .layout.rowsPerImage = rows_per_image,
    .buffer              = buffer,
  };

  return image_copy_buffer_desc;
}

static WGPUImageCopyTexture
context_create_image_copy_texture(context_t* this, WGPUTexture texture,
                                  uint32_t level, WGPUOrigin3D origin)
{
  UNUSED_VAR(this);

  WGPUImageCopyTexture image_copy_texture_desc = {
    .texture  = texture,
    .mipLevel = level,
    .origin   = origin,
  };

  return image_copy_texture_desc;
}

static WGPUCommandBuffer context_copy_buffer_to_texture(
  context_t* this, WGPUImageCopyBuffer const* image_copy_buffer,
  WGPUImageCopyTexture const* image_copy_texture, WGPUExtent3D const* ext_3d)
{
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  wgpuCommandEncoderCopyBufferToTexture(encoder, image_copy_buffer,
                                        image_copy_texture, ext_3d);
  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return copy;
}

static WGPUCommandBuffer
context_copy_buffer_to_buffer(context_t* this, WGPUBuffer src_buffer,
                              uint64_t src_offset, WGPUBuffer dest_buffer,
                              uint64_t dest_offset, uint64_t size)
{
  WGPUCommandEncoder encoder
    = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  wgpuCommandEncoderCopyBufferToBuffer(encoder, src_buffer, src_offset,
                                       dest_buffer, dest_offset, size);
  WGPUCommandBuffer copy = wgpuCommandEncoderFinish(encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder)

  return copy;
}

static wgpu_shader_t context_create_shader_module(void* context,
                                                  WGPUShaderStage shader_stage,
                                                  const char* shader_wgsl_path)
{
  UNUSED_VAR(shader_stage);

  wgpu_shader_t shader = wgpu_shader_create(((context_t*)context)->wgpu_context,
                                            &(wgpu_shader_desc_t){
                                              // Shader WGSL
                                              .file  = shader_wgsl_path,
                                              .entry = "main",
                                            });
  ASSERT(shader.module);

  return shader;
}

static WGPUBindGroupLayout context_make_bind_group_layout(
  context_t* this, WGPUBindGroupLayoutEntry const* bind_group_layout_entries,
  uint32_t bind_group_layout_entry_count)
{
  WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    this->device, &(WGPUBindGroupLayoutDescriptor){
                    .entryCount = bind_group_layout_entry_count,
                    .entries    = bind_group_layout_entries,
                  });
  ASSERT(bind_group_layout != NULL);
  return bind_group_layout;
}

static WGPUPipelineLayout context_make_basic_pipeline_layout(
  context_t* this, WGPUBindGroupLayout const* bind_group_layouts,
  uint32_t bind_group_layout_count)
{
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    this->device, &(WGPUPipelineLayoutDescriptor){
                    .bindGroupLayoutCount = bind_group_layout_count,
                    .bindGroupLayouts     = bind_group_layouts,
                  });
  ASSERT(pipeline_layout != NULL);
  return pipeline_layout;
}

static WGPURenderPipeline context_create_render_pipeline(
  context_t* this, WGPUPipelineLayout pipeline_layout,
  WGPUShaderModule fs_module, WGPUVertexState const* vertex_state,
  bool enable_blend)
{
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  WGPUStencilFaceState stencil_face_state = {
    .compare     = WGPUCompareFunction_Always,
    .failOp      = WGPUStencilOperation_Keep,
    .depthFailOp = WGPUStencilOperation_Keep,
    .passOp      = WGPUStencilOperation_Keep,
  };

  WGPUDepthStencilState depth_stencil_state = {
    .format              = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled   = true,
    .depthCompare        = WGPUCompareFunction_Less,
    .stencilFront        = stencil_face_state,
    .stencilBack         = stencil_face_state,
    .stencilReadMask     = 0xffffffff,
    .stencilWriteMask    = 0xffffffff,
    .depthBias           = 0,
    .depthBiasSlopeScale = 0.0f,
    .depthBiasClamp      = 0.0f,
  };

  WGPUMultisampleState multisample_state = {
    .count                  = aquarium_settings.msaa_sample_count,
    .mask                   = 0xffffffff,
    .alphaToCoverageEnabled = false,
  };

  WGPUBlendComponent blend_component = {
    .operation = WGPUBlendOperation_Add,
  };
  if (enable_blend) {
    blend_component.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_component.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
  }
  else {
    blend_component.srcFactor = WGPUBlendFactor_One;
    blend_component.dstFactor = WGPUBlendFactor_Zero;
  }

  WGPUBlendState blend_state = {
    .color = blend_component,
    .alpha = blend_component,
  };

  WGPUColorTargetState color_target_state = {
    .format    = this->preferred_swap_chain_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUFragmentState fragment_state = {
    .module      = fs_module,
    .entryPoint  = "main",
    .targetCount = 1,
    .targets     = &color_target_state,
  };

  WGPURenderPipelineDescriptor pipeline_descriptor = {
    .layout       = pipeline_layout,
    .vertex       = *vertex_state,
    .primitive    = primitive_state,
    .depthStencil = &depth_stencil_state,
    .multisample  = multisample_state,
    .fragment     = &fragment_state,
  };

  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(this->device, &pipeline_descriptor);
  ASSERT(pipeline != NULL);
  return pipeline;
}

static texture_t context_create_multisampled_render_target_view(context_t* this)
{
  texture_t texture = {0};

  WGPUTextureDescriptor texture_desc = {
    .dimension               = WGPUTextureDimension_2D,
    .size.width              = this->client_width,
    .size.height             = this->client_height,
    .size.depthOrArrayLayers = 1,
    .sampleCount             = this->msaa_sample_count,
    .format                  = this->preferred_swap_chain_format,
    .mipLevelCount           = 1,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  texture.texture = wgpuDeviceCreateTexture(this->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);

  return texture;
}

static texture_t context_create_depth_stencil_view(context_t* this)
{
  texture_t texture = {0};

  WGPUTextureDescriptor texture_desc = {
    .dimension               = WGPUTextureDimension_2D,
    .size.width              = this->client_width,
    .size.height             = this->client_height,
    .size.depthOrArrayLayers = 1,
    .sampleCount             = this->msaa_sample_count,
    .format                  = WGPUTextureFormat_Depth24PlusStencil8,
    .mipLevelCount           = 1,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  texture.texture = wgpuDeviceCreateTexture(this->device, &texture_desc);
  ASSERT(texture.texture != NULL);

  // Create the texture view
  WGPUTextureViewDescriptor texture_view_dec = {
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  texture.view = wgpuTextureCreateView(texture.texture, &texture_view_dec);
  ASSERT(texture.view != NULL);

  return texture;
}

static WGPUBuffer context_create_buffer(void* this,
                                        WGPUBufferDescriptor const* descriptor)
{
  return wgpuDeviceCreateBuffer(((context_t*)this)->device, descriptor);
}

static void context_set_buffer_data(void* context, WGPUBuffer buffer,
                                    uint32_t buffer_size, const void* data,
                                    uint32_t data_size)
{
  context_t* this              = (context_t*)context;
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUBufferDescriptor buffer_desc = {
    .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
    .size             = buffer_size,
    .mappedAtCreation = true,
  };
  WGPUBuffer staging = context_create_buffer(wgpu_context, &buffer_desc);
  ASSERT(staging);
  void* mapping = wgpuBufferGetMappedRange(staging, 0, buffer_size);
  ASSERT(mapping);
  memcpy(mapping, data, data_size);
  wgpuBufferUnmap(staging);

  WGPUCommandBuffer command
    = context_copy_buffer_to_buffer(this, staging, 0, buffer, 0, buffer_size);
  WGPU_RELEASE_RESOURCE(Buffer, staging);
  sc_array_add(&((context_t*)this)->command_buffers, command);
}

static WGPUBindGroup
context_make_bind_group(context_t* this, WGPUBindGroupLayout layout,
                        WGPUBindGroupEntry const* bind_group_entries,
                        uint32_t bind_group_entry_count)
{
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    this->device, &(WGPUBindGroupDescriptor){
                    .layout     = layout,
                    .entryCount = bind_group_entry_count,
                    .entries    = bind_group_entries,
                  });
  ASSERT(bind_group != NULL);
  return bind_group;
}

static void context_init_general_resources(context_t* this,
                                           aquarium_t* aquarium)
{
  /* Initialize general uniform buffers */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.general = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  this->uniform_buffers.light = context_create_buffer_from_data(
    this, &aquarium->light_uniforms, sizeof(aquarium->light_uniforms),
    sizeof(aquarium->light_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.fog = context_create_buffer_from_data(
    this, &aquarium->fog_uniforms, sizeof(aquarium->fog_uniforms),
    sizeof(aquarium->fog_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  /* Initialize world uniform buffers */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.world = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  this->uniform_buffers.light_world_position = context_create_buffer_from_data(
    this, &aquarium->light_world_position_uniform,
    sizeof(aquarium->light_world_position_uniform),
    calc_constant_buffer_byte_size(
      sizeof(aquarium->light_world_position_uniform)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.light_world_position,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(
          sizeof(aquarium->light_world_position_uniform)),
      },
    };
    this->bind_groups.world
      = context_make_bind_group(this, this->bind_group_layouts.world,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  bool enable_dynamic_buffer_offset
    = aquarium_settings.enable_dynamic_buffer_offset;

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {0};
    if (enable_dynamic_buffer_offset) {
      bgl_entries[0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      };
    }
    else {
      bgl_entries[0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      };
    }
    this->bind_group_layouts.fish_per = context_make_bind_group_layout(
      this, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  context_realloc_resource(this, aquarium_get_pre_fish_count(aquarium),
                           aquarium_get_cur_fish_count(aquarium),
                           enable_dynamic_buffer_offset);
}

static void context_update_world_uniforms(context_t* this, aquarium_t* aquarium)
{
  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.light_world_position,
    calc_constant_buffer_byte_size(sizeof(light_world_position_uniform_t)),
    &aquarium->light_world_position_uniform,
    sizeof(light_world_position_uniform_t));
}

static buffer_dawn_t* context_create_buffer_f32(context_t* this,
                                                int32_t num_components,
                                                float* buf, size_t buf_count,
                                                bool is_index)
{
  buffer_dawn_t* buffer = malloc(sizeof(buffer_dawn_t));
  buffer_dawn_create_f32(buffer, this, (int)buf_count, num_components, buf,
                         is_index);

  return buffer;
}

static buffer_dawn_t*
context_create_buffer_uint16(context_t* this, int32_t num_components,
                             uint16_t* buf, size_t buf_count,
                             uint64_t buffer_count, bool is_index)
{
  buffer_dawn_t* buffer = malloc(sizeof(buffer_dawn_t));
  buffer_dawn_create_uint16(buffer, this, (int)buf_count, num_components, buf,
                            buffer_count, is_index);

  return buffer;
}

static void context_flush(context_t* this)
{
  /* Submit to the queue */
  wgpuQueueSubmit(this->wgpu_context->queue,
                  sc_array_size(&this->command_buffers),
                  this->command_buffers.elems);

  /* Release command buffer */
  for (size_t i = 0; i < sc_array_size(&this->command_buffers); ++i) {
    WGPU_RELEASE_RESOURCE(CommandBuffer, this->command_buffers.elems[i])
  }
  sc_array_clear(&this->command_buffers);
}

/* Submit commands of the frame */
static void context_do_flush(context_t* this)
{
  // End render pass
  wgpuRenderPassEncoderEnd(this->render_pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, this->render_pass)

  buffer_manager_flush(this->buffer_manager);

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(this->command_encoder, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, this->command_encoder)
  sc_array_add(&this->command_buffers, cmd);

  context_flush(this);

  wgpuSwapChainPresent(this->wgpu_context->swap_chain.instance);
}

static void context_pre_frame(context_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  if (this->is_swapchain_out_of_date) {
    this->client_width  = wgpu_context->surface.width;
    this->client_height = wgpu_context->surface.height;
    if (this->msaa_sample_count > 1) {
      this->texture_views.scene_render_target
        = context_create_multisampled_render_target_view(this);
    }
    this->texture_views.scene_depth_stencil
      = context_create_depth_stencil_view(this);
    // mSwapchain.Configure(mPreferredSwapChainFormat,
    // kSwapchainBackBufferUsage,
    //                     mClientWidth, mClientHeight);
    this->is_swapchain_out_of_date = false;
  }

  this->command_encoder = wgpuDeviceCreateCommandEncoder(this->device, NULL);
  this->texture_views.backbuffer
    = wgpuSwapChainGetCurrentTextureView(wgpu_context->swap_chain.instance);

  WGPURenderPassColorAttachment color_attachment = {0};
  if (this->msaa_sample_count) {
    // If MSAA is enabled, we render to a multisampled texture and then resolve
    // to the backbuffer
    color_attachment.view = this->texture_views.scene_render_target.view;
    color_attachment.resolveTarget = this->texture_views.backbuffer;
    color_attachment.loadOp        = WGPULoadOp_Clear;
    color_attachment.storeOp       = WGPUStoreOp_Store;
    color_attachment.clearColor    = (WGPUColor){0.f, 0.8f, 1.f, 0.f};
  }
  else {
    /* When MSAA is off, we render directly to the backbuffer */
    color_attachment.view       = this->texture_views.backbuffer;
    color_attachment.loadOp     = WGPULoadOp_Clear;
    color_attachment.storeOp    = WGPUStoreOp_Store;
    color_attachment.clearColor = (WGPUColor){0.f, 0.8f, 1.f, 0.f};
  }

  WGPURenderPassDepthStencilAttachment depth_stencil_attachment = {0};
  depth_stencil_attachment.view = this->texture_views.scene_depth_stencil.view;
  depth_stencil_attachment.depthLoadOp    = WGPULoadOp_Clear;
  depth_stencil_attachment.depthStoreOp   = WGPUStoreOp_Store;
  depth_stencil_attachment.clearDepth     = 1.f;
  depth_stencil_attachment.stencilLoadOp  = WGPULoadOp_Clear;
  depth_stencil_attachment.stencilStoreOp = WGPUStoreOp_Store;
  depth_stencil_attachment.clearStencil   = 0;

  this->render_pass_descriptor.colorAttachmentCount = 1;
  this->render_pass_descriptor.colorAttachments     = &color_attachment;
  this->render_pass_descriptor.depthStencilAttachment
    = &depth_stencil_attachment;

  this->render_pass = wgpuCommandEncoderBeginRenderPass(
    this->command_encoder, &this->render_pass_descriptor);
}

static void context_destroy_fish_resource(context_t* this)
{
}

static void context_realloc_resource(context_t* this,
                                     uint32_t pre_total_instance,
                                     uint32_t cur_total_instance,
                                     bool enable_dynamic_buffer_offset)
{
  this->pre_total_instance           = pre_total_instance;
  this->cur_total_instance           = cur_total_instance;
  this->enable_dynamic_buffer_offset = enable_dynamic_buffer_offset;

  if (cur_total_instance == 0) {
    return;
  }

  // If current fish number > pre fish number, allocate a new bigger buffer.
  // If current fish number <= prefish number, do not allocate a new one.
  if (pre_total_instance >= cur_total_instance) {
    return;
  }

  context_destroy_fish_resource(this);

  this->fish_pers = malloc(sizeof(fish_per_t) * cur_total_instance);

  if (enable_dynamic_buffer_offset) {
    this->bind_group_fish_pers = malloc(sizeof(WGPUBindGroup) * 1);
  }
  else {
    this->bind_group_fish_pers
      = malloc(sizeof(WGPUBindGroup) * cur_total_instance);
  }

  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
    .size
    = calc_constant_buffer_byte_size(sizeof(fish_per_t) * cur_total_instance),
    .mappedAtCreation = false,
  };
  this->fish_pers_buffer
    = context_create_buffer(this->wgpu_context, &buffer_desc);

  if (this->enable_dynamic_buffer_offset) {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->fish_pers_buffer,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(fish_per_t)),
      },
    };
    this->bind_group_fish_pers[0]
      = context_make_bind_group(this, this->bind_group_layouts.fish_per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }
  else {
    for (uint32_t i = 0; i < cur_total_instance; ++i) {
      WGPUBindGroupEntry bg_entries[1] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer  = this->fish_pers_buffer,
          .offset  = calc_constant_buffer_byte_size(sizeof(fish_per_t) * i),
          .size    = calc_constant_buffer_byte_size(sizeof(fish_per_t)),
        },
      };
      this->bind_group_fish_pers[i]
        = context_make_bind_group(this, this->bind_group_layouts.fish_per,
                                  bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
    }
  }
}

static void context_wait_a_bit(context_t* this)
{
  wgpuDeviceTick(this->device);

  usleep(100);
}

static WGPUCommandEncoder context_create_command_encoder(context_t* this)
{
  return wgpuDeviceCreateCommandEncoder(this->device, NULL);
}

static void context_update_buffer_data(void* context, WGPUBuffer buffer,
                                       size_t buffer_size, void* data,
                                       size_t data_size)
{
  context_t* this = (context_t*)context;
  size_t offset   = 0;
  ring_buffer_t* ring_buffer
    = buffer_manager_allocate(this->buffer_manager, buffer_size, &offset);

  if (ring_buffer == NULL) {
    log_error("Memory upper limit.");
    return;
  }

  ring_buffer_push(ring_buffer, this->buffer_manager->encoder, buffer, offset,
                   0, data, data_size);
}

static void context_update_all_fish_data(context_t* this)
{
  size_t size = calc_constant_buffer_byte_size(sizeof(fish_per_t)
                                               * this->cur_total_instance);
  context_update_buffer_data(this, this->fish_pers_buffer, size,
                             this->fish_pers,
                             sizeof(fish_per_t) * this->cur_total_instance);
}

static void context_destory_fish_resource(context_t* this)
{
  buffer_manager_destroy_buffer_pool(this->buffer_manager);
}

/* -------------------------------------------------------------------------- *
 * Aquarium - Main class functions
 * -------------------------------------------------------------------------- */

static float get_current_time_point()
{
  return 0.0f;
}

static void aquarium_init_defaults(aquarium_t* this)
{
  memset(this, 0, sizeof(*this));

  this->cur_fish_count = 500;
  this->pre_fish_count = 0;
  this->test_time      = INT_MAX;

  this->g.then      = get_current_time_point();
  this->g.mclock    = 0.0f;
  this->g.eye_clock = 0.0f;
  this->g.alpha     = "1";

  this->light_uniforms.light_color[0] = 1.0f;
  this->light_uniforms.light_color[1] = 1.0f;
  this->light_uniforms.light_color[2] = 1.0f;
  this->light_uniforms.light_color[3] = 1.0f;

  this->light_uniforms.specular[0] = 1.0f;
  this->light_uniforms.specular[1] = 1.0f;
  this->light_uniforms.specular[2] = 1.0f;
  this->light_uniforms.specular[3] = 1.0f;

  this->fog_uniforms.fog_color[0] = g_settings.fog_red;
  this->fog_uniforms.fog_color[1] = g_settings.fog_green;
  this->fog_uniforms.fog_color[2] = g_settings.fog_blue;
  this->fog_uniforms.fog_color[3] = 1.0f;

  this->fog_uniforms.fog_power  = g_settings.fog_power;
  this->fog_uniforms.fog_mult   = g_settings.fog_mult;
  this->fog_uniforms.fog_offset = g_settings.fog_offset;

  this->light_uniforms.ambient[0] = g_settings.ambient_red;
  this->light_uniforms.ambient[1] = g_settings.ambient_green;
  this->light_uniforms.ambient[2] = g_settings.ambient_blue;
  this->light_uniforms.ambient[3] = 0.0f;

  memset(this->fish_count, 0, sizeof(this->fish_count));
}

static void aquarium_create(aquarium_t* this)
{
  aquarium_init_defaults(this);
}

static void aquarium_calculate_fish_count(aquarium_t* this)
{
  /* Calculate fish count for each type of fish */
  int32_t num_left = this->cur_fish_count;
  for (int i = 0; i < FISHENUM_MAX; ++i) {
    for (uint32_t i = 0; i < (uint32_t)ARRAY_SIZE(fish_table); ++i) {
      const fish_t* fish_info = &fish_table[i];
      if (fish_info->type != i) {
        continue;
      }
      int32_t num_float = num_left;
      if (i == FISHENUM_BIG) {
        int32_t temp = this->cur_fish_count < g_settings.num_fish_small ? 1 : 2;
        num_float    = MIN(num_left, temp);
      }
      else if (i == FISHENUM_MEDIUM) {
        if (this->cur_fish_count < g_settings.num_fish_medium) {
          num_float = MIN(num_left, this->cur_fish_count / 10);
        }
        else if (this->cur_fish_count < g_settings.num_fish_big) {
          num_float = MIN(num_left, g_settings.num_fish_left_small);
        }
        else {
          num_float = MIN(num_left, g_settings.num_fish_left_big);
        }
      }
      num_left = num_left - num_float;
      this->fish_count[fish_info->model_name - MODELSMALLFISHA] = num_float;
    }
  }
}

static void aquarium_init(aquarium_t* this)
{
  aquarium_calculate_fish_count(this);

  /* Avoid resource allocation in the first render loop */
  this->pre_fish_count = this->cur_fish_count;
}

static int32_t aquarium_get_cur_fish_count(aquarium_t* this)
{
  return this->cur_fish_count;
}

static int32_t aquarium_get_pre_fish_count(aquarium_t* this)
{
  return this->pre_fish_count;
}

static void aquarium_reset_fps_time(aquarium_t* this)
{
  this->g.start = get_current_time_point();
  this->g.then  = this->g.start;
}

static void aquarium_load_models(aquarium_t* this)
{
}

static void aquarium_load_placement(aquarium_t* this)
{
}

static void aquarium_load_fish_scenario(aquarium_t* this)
{
}

static void aquarium_load_resource(aquarium_t* this)
{
  aquarium_load_models(this);
  aquarium_load_placement(this);
  if (aquarium_settings.simulate_fish_come_and_go) {
    aquarium_load_fish_scenario(this);
  }
}

static model_name_t
aquarium_map_model_name_str_to_model_name(aquarium_t* this,
                                          const char* model_name_str)
{
  model_name_t model_name = MODELMAX;
  for (uint32_t i = 0; i < MODELMAX; ++i) {
    if (strcmp(this->model_enum_map[i].key, model_name_str) == 0) {
      model_name = this->model_enum_map[i].value;
      break;
    }
  }
  return model_name;
}

static void aquarium_setup_model_enum_map(aquarium_t* this)
{
  for (uint32_t i = 0; i < MODELMAX; ++i) {
    snprintf(this->model_enum_map[i].key, strlen(g_scene_info[i].name_str) + 1,
             "%s", g_scene_info[i].name_str);
    this->model_enum_map[i].value = g_scene_info[i].name;
  }
}

static float
aquarium_get_elapsed_time(aquarium_t* this,
                          wgpu_example_context_t* wgpu_example_context)
{
  // Update our time
  float now          = wgpu_example_context->frame.timestamp_millis;
  float elapsed_time = now - this->g.then;
  this->g.then       = now;

  return elapsed_time;
}

static void aquarium_update_world_uniforms(aquarium_t* this)
{
  context_update_world_uniforms(this->context, this);
}

static void
aquarium_update_global_uniforms(aquarium_t* this,
                                wgpu_example_context_t* wgpu_example_context)
{
  global_t* g = &this->g;
  light_world_position_uniform_t* light_world_position_uniform
    = &this->light_world_position_uniform;

  float elapsed_time = aquarium_get_elapsed_time(this, wgpu_example_context);
  g->mclock += elapsed_time * g_settings.speed;
  g->eye_clock += elapsed_time * g_settings.eye_speed;

  g->eye_position[0] = sin(g->eye_clock) * g_settings.eye_radius;
  g->eye_position[1] = g_settings.eye_height;
  g->eye_position[2] = cos(g->eye_clock) * g_settings.eye_radius;
  g->target[0] = (float)(sin(g->eye_clock + PI)) * g_settings.target_radius;
  g->target[1] = g_settings.target_height;
  g->target[2] = (float)(cos(g->eye_clock + PI)) * g_settings.target_radius;

  float near_plane   = 1.0f;
  float far_plane    = 25000.0f;
  const float aspect = (float)context_get_client_width(this->context)
                       / (float)context_get_client_height(this->context);
  float top
    = tan(deg_to_rad(g_settings.field_of_view * g_settings.fov_fudge) * 0.5f)
      * near_plane;
  float bottom = -top;
  float left   = aspect * bottom;
  float right  = aspect * top;
  float width  = fabs(right - left);
  float height = fabs(top - bottom);
  float xOff   = width * g_settings.net_offset[0] * g_settings.net_offset_mult;
  float yOff   = height * g_settings.net_offset[1] * g_settings.net_offset_mult;

  // set frustum and camera look at
  matrix_frustum(g->projection, left + xOff, right + xOff, bottom + yOff,
                 top + yOff, near_plane, far_plane);
  matrix_camera_look_at(light_world_position_uniform->view_inverse,
                        g->eye_position, g->target, g->up);
  matrix_inverse4(g->view, light_world_position_uniform->view_inverse);
  matrix_mul_matrix_matrix4(light_world_position_uniform->view_projection,
                            g->view, g->projection);
  matrix_inverse4(g->view_projection_inverse,
                  light_world_position_uniform->view_projection);

  memcpy(g->sky_view, g->view, 16 * sizeof(float));
  g->sky_view[12] = 0.0f;
  g->sky_view[13] = 0.0f;
  g->sky_view[14] = 0.0f;
  matrix_mul_matrix_matrix4(g->sky_view_projection, g->sky_view, g->projection);
  matrix_inverse4(g->sky_view_projection_inverse, g->sky_view_projection);

  matrix_get_axis(g->v3t0, light_world_position_uniform->view_inverse, 0);
  matrix_get_axis(g->v3t1, light_world_position_uniform->view_inverse, 1);
  matrix_mul_scalar_vector(20.0f, g->v3t0, 3);
  matrix_mul_scalar_vector(30.0f, g->v3t1, 3);
  matrix_add_vector(light_world_position_uniform->light_world_pos,
                    g->eye_position, g->v3t0, 3);
  matrix_add_vector(light_world_position_uniform->light_world_pos,
                    light_world_position_uniform->light_world_pos, g->v3t1, 3);

  /* Update world uniforms  */
  context_update_world_uniforms(this->context, this);
}

static void aquarium_update_and_draw(aquarium_t* this);

static void aquarium_render(aquarium_t* this,
                            wgpu_example_context_t* wgpu_example_context)
{
  matrix_reset_pseudoRandom();

  context_pre_frame(this->context);

  /* Global Uniforms should update after command reallocation. */
  aquarium_update_global_uniforms(this, wgpu_example_context);

  if (aquarium_settings.simulate_fish_come_and_go) {
    if (!sc_queue_empty(&this->fish_behavior)) {
      behavior_t* behave = sc_queue_peek_first(&this->fish_behavior);
      int32_t frame      = behave->frame;
      if (frame == 0) {
        sc_queue_del_first(&this->fish_behavior);
        if (behave->op == OPERATION_PLUS) {
          this->cur_fish_count += behave->count;
        }
        else {
          this->cur_fish_count -= behave->count;
        }
      }
      else {
        behave->frame = --frame;
      }
    }
  }

  if (!aquarium_settings.enable_instanced_draw) {
    if (this->cur_fish_count != this->pre_fish_count) {
      aquarium_calculate_fish_count(this);
      bool enable_dynamic_buffer_offset
        = aquarium_settings.enable_dynamic_buffer_offset;
      context_realloc_resource(this->context, this->pre_fish_count,
                               this->cur_fish_count,
                               enable_dynamic_buffer_offset);
      this->pre_fish_count = this->cur_fish_count;
    }
  }

  aquarium_update_and_draw(this);
}

static void aquarium_update_and_draw(aquarium_t* this)
{
  bool draw_per_model = aquarium_settings.draw_per_model;
  int32_t fish_begin  = aquarium_settings.enable_instanced_draw ?
                          MODELSMALLFISHAINSTANCEDDRAWS :
                          MODELSMALLFISHA;
  int32_t fish_end    = aquarium_settings.enable_instanced_draw ?
                          MODELBIGFISHBINSTANCEDDRAWS :
                          MODELBIGFISHB;

  for (uint32_t i = MODELRUINCOLUMN; i <= MODELSEAWEEDB; ++i) {
  }
}

/* -------------------------------------------------------------------------- *
 * Model - Defines generic model.
 * -------------------------------------------------------------------------- */

typedef enum {
  BUFFERTYPE_POSITION,
  BUFFERTYPE_NORMAL,
  BUFFERTYPE_TEX_COORD,
  BUFFERTYPE_TANGENT,
  BUFFERTYPE_BI_NORMAL,
  BUFFERTYPE_INDICES,
  BUFFERTYPE_MAX,
} buffer_type_t;

typedef enum {
  TEXTURETYPE_DIFFUSE,
  TEXTURETYPE_NORMAL_MAP,
  TEXTURETYPE_REFLECTION_MAP,
  TEXTURETYPE_SKYBOX,
  TEXTURETYPE_MAX,
} texture_type_t;

typedef struct {
  float* data;
  size_t data_size;
} float_array_t;
sc_array_def(float_array_t, float_array);

typedef struct {
  model_group_t type;
  model_name_t name;
  program_t* program;
  bool blend;

  struct sc_array_float_array world_matrices;
  texture_t* texture_map[TEXTURETYPE_MAX];
  buffer_dawn_t* buffer_map[BUFFERTYPE_MAX];
} model_t;

static void model_init_defaults(model_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void model_create(model_t* this, model_group_t type, model_name_t name,
                         bool blend)
{
  this->type  = type;
  this->name  = name;
  this->blend = blend;
}

static void model_destroy(model_t* this)
{
  for (uint32_t i = 0; i < BUFFERTYPE_MAX; ++i) {
    buffer_dawn_t* buf = this->buffer_map[i];
    if (buf) {
      buffer_dawn_destroy(buf);
      buf = NULL;
    }
  }
}

static void model_set_program(model_t* this, program_t* prgm)
{
  this->program = prgm;
}

/* -------------------------------------------------------------------------- *
 * Fish model - Defined fish model
 *  - Updates fish specific uniforms.
 *  - Implement common functions of fish models.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t model;
  int32_t pre_instance;
  int32_t cur_instance;
  int32_t fish_per_offset;
  aquarium_t* aquarium;
} fish_model_t;

static void fish_model_init_defaults(fish_model_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void fish_model_create(fish_model_t* this, model_group_t type,
                              model_name_t name, bool blend,
                              aquarium_t* aquarium)
{
  fish_model_init_defaults(this);

  this->aquarium = aquarium;

  this->model = (model_t){
    .type  = type,
    .name  = name,
    .blend = blend,
  };
}

static void fish_model_prepare_for_draw(fish_model_t* this)
{
  this->fish_per_offset = 0;
  for (uint32_t i = 0; i < this->model.name - MODELSMALLFISHA; ++i) {
    const fish_t* fish_info = &fish_table[i];
    this->fish_per_offset
      += this->aquarium->fish_count[fish_info->model_name - MODELSMALLFISHA];
  }

  const fish_t* fish_info = &fish_table[this->model.name - MODELSMALLFISHA];
  this->cur_instance
    = this->aquarium->fish_count[fish_info->model_name - MODELSMALLFISHA];
}

/* -------------------------------------------------------------------------- *
 * Fish model - Defines the base fish model.
 * -------------------------------------------------------------------------- */

typedef struct {
  fish_model_t fish_model;
  struct {
    float fish_length;
    float fish_wave_length;
    float fish_bend_amount;
  } fish_vertex_uniforms;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t position;
    buffer_dawn_t normal;
    buffer_dawn_t tex_coord;
    buffer_dawn_t tangent;
    buffer_dawn_t bi_normal;
    buffer_dawn_t indices;
  } buffers;
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  WGPUBindGroupLayout bind_group_layout_model;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroup bind_group_model;
  WGPUBuffer fish_vertex_buffer;
  struct {
    WGPUBuffer light_factor;
  } uniform_buffers;
  struct {
    WGPUShaderModule vertex;
    WGPUShaderModule fragment;
  } shader_modules;
  wgpu_context_t* wgpu_context;
  context_t* context;
  bool enable_dynamic_buffer_offset;
} fish_model_draw_t;

static void fish_model_draw_init_defaults(fish_model_draw_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 5.0f;
  this->light_factor_uniforms.specular_factor = 0.3f;
}

static void fish_model_draw_create(fish_model_draw_t* this, context_t* context,
                                   aquarium_t* aquarium, model_group_t type,
                                   model_name_t name, bool blend)
{
  fish_model_draw_init_defaults(this);

  fish_model_create(&this->fish_model, type, name, blend, aquarium);

  this->context      = context;
  this->wgpu_context = context->wgpu_context;

  const fish_t* fish_info                = &fish_table[name - MODELSMALLFISHA];
  this->fish_vertex_uniforms.fish_length = fish_info->fish_length;
  this->fish_vertex_uniforms.fish_bend_amount = fish_info->fish_bend_amount;
  this->fish_vertex_uniforms.fish_wave_length = fish_info->fish_wave_length;

  this->fish_model.cur_instance
    = aquarium->fish_count[fish_info->model_name - MODELSMALLFISHA];
  this->fish_model.pre_instance = this->fish_model.cur_instance;
}

static void fish_model_draw_init(fish_model_draw_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.position.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.normal.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.tex_coord.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.tangent.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.bi_normal.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  this->vertex_state.module      = this->shader_modules.vertex;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[8] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [7] = (WGPUBindGroupLayoutEntry) {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    uint32_t bgl_entry_count = 0;
    if (this->textures.skybox && this->textures.reflection) {
      bgl_entry_count = 8;
    }
    else {
      bgl_entry_count = 5;
      bgl_entries[3]  = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      };
    }

    this->bind_group_layout_model = context_make_bind_group_layout(
      this->context, bgl_entries, bgl_entry_count);
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->context->bind_group_layouts.general,  // Group 0
    this->context->bind_group_layouts.world,    // Group 1
    this->bind_group_layout_model,              // Group 2
    this->context->bind_group_layouts.fish_per, // Group 3
  };
  this->pipeline_layout = context_make_basic_pipeline_layout(
    this->context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    this->context, this->pipeline_layout, this->shader_modules.fragment,
    &this->vertex_state, this->fish_model.model.blend);

  this->fish_vertex_buffer = context_create_buffer_from_data(
    this->context, &this->fish_vertex_uniforms,
    sizeof(this->fish_vertex_uniforms), sizeof(this->fish_vertex_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.light_factor = context_create_buffer_from_data(
    this->context, &this->light_factor_uniforms,
    sizeof(this->light_factor_uniforms), sizeof(this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  // Fish models includes small, medium and big. Some of them contains
  // reflection and skybox texture, but some doesn't.
  {
    WGPUBindGroupEntry bg_entries[8] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->fish_vertex_buffer,
        .offset  = 0,
        .size    = sizeof(this->fish_vertex_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(this->light_factor_uniforms)
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = this->textures.reflection->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .sampler = this->textures.skybox->sampler,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding     = 4,
        .textureView = this->textures.diffuse->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = this->textures.normal->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = this->textures.reflection->view,
      },
      [7] = (WGPUBindGroupEntry) {
        .binding     = 7,
        .textureView = this->textures.skybox->view,
      },
    };
    uint32_t bg_entry_count = 0;
    if (this->textures.skybox && this->textures.reflection) {
      bg_entry_count = 8;
    }
    else {
      bg_entry_count = 5;
      bg_entries[2]  = (WGPUBindGroupEntry){
         .binding = 2,
         .sampler = this->textures.diffuse->sampler,
      };
      bg_entries[3] = (WGPUBindGroupEntry){
        .binding     = 3,
        .textureView = this->textures.diffuse->view,
      };
      bg_entries[4] = (WGPUBindGroupEntry){
        .binding     = 4,
        .textureView = this->textures.normal->view,
      };
    }
    this->bind_group_model = context_make_bind_group(
      this->context, this->bind_group_layout_model, bg_entries, bg_entry_count);
  }

  context_set_buffer_data(wgpu_context, this->uniform_buffers.light_factor,
                          sizeof(this->light_factor_uniforms),
                          &this->light_factor_uniforms,
                          sizeof(this->light_factor_uniforms));
  context_set_buffer_data(
    wgpu_context, this->fish_vertex_buffer, sizeof(this->fish_vertex_uniforms),
    &this->fish_vertex_uniforms, sizeof(this->fish_vertex_uniforms));
}

static void fish_model_draw_destroy(fish_model_draw_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout_model)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group_model)
  WGPU_RELEASE_RESOURCE(Buffer, this->fish_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.light_factor)
}

static void fish_model_draw_draw(fish_model_draw_t* this)
{
  if (this->fish_model.cur_instance == 0) {
    return;
  }

  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPURenderPassEncoder render_pass = this->context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_group_model, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                       this->buffers.tangent.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                       this->buffers.bi_normal.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices.buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);

  if (this->enable_dynamic_buffer_offset) {
    for (int32_t i = 0; i < this->fish_model.cur_instance; ++i) {
      const uint32_t offset = 256u * (i + this->fish_model.fish_per_offset);
      wgpuRenderPassEncoderSetBindGroup(
        render_pass, 3, this->context->bind_group_fish_pers[0], 1, &offset);
      wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                       this->buffers.indices.total_components,
                                       1, 0, 0, 0);
    }
  }
  else {
    for (int32_t i = 0; i < this->fish_model.cur_instance; ++i) {
      const uint32_t offset = i + this->fish_model.fish_per_offset;
      wgpuRenderPassEncoderSetBindGroup(
        render_pass, 3, this->context->bind_group_fish_pers[offset], 0, NULL);
      wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                       this->buffers.indices.total_components,
                                       1, 0, 0, 0);
    }
  }
}

void fish_model_draw_update_fish_per_uniforms(fish_model_draw_t* this, float x,
                                              float y, float z, float next_x,
                                              float next_y, float next_z,
                                              float scale, float time,
                                              int index)
{
  index += this->fish_model.fish_per_offset;
  context_t* ctx = this->context;

  ctx->fish_pers[index].world_position[0] = x;
  ctx->fish_pers[index].world_position[1] = y;
  ctx->fish_pers[index].world_position[2] = z;
  ctx->fish_pers[index].next_position[0]  = next_x;
  ctx->fish_pers[index].next_position[1]  = next_y;
  ctx->fish_pers[index].next_position[2]  = next_z;
  ctx->fish_pers[index].scale             = scale;
  ctx->fish_pers[index].time              = time;
}

/* -------------------------------------------------------------------------- *
 * Fish model Instanced Draw - Defines instance fish model.
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 world_position;
  float scale;
  vec3 next_position;
  float time;
} fish_model_instanced_draw_fish_per;

typedef struct {
  fish_model_t fish_model;
  struct {
    float fish_length;
    float fish_wave_length;
    float fish_bend_amount;
  } fish_vertex_uniforms;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  fish_model_instanced_draw_fish_per* fish_pers;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t position;
    buffer_dawn_t normal;
    buffer_dawn_t tex_coord;
    buffer_dawn_t tangent;
    buffer_dawn_t bi_normal;
    buffer_dawn_t indices;
  } buffers;
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } bind_group_layouts;
  WGPUPipelineLayout pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } bind_groups;
  WGPUBuffer fish_vertex_buffer;
  struct {
    WGPUBuffer light_factor;
  } uniform_buffers;
  WGPUBuffer fish_pers_buffer;
  int32_t instance;
  struct {
    WGPUShaderModule vertex;
    WGPUShaderModule fragment;
  } shader_modules;
  wgpu_context_t* wgpu_context;
  context_t* aquarium_context;
} fish_model_instanced_draw_t;

static void
fish_model_instanced_draw_init_defaults(fish_model_instanced_draw_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 5.0f;
  this->light_factor_uniforms.specular_factor = 0.3f;

  this->instance = 0;
}

static void fish_model_instanced_draw_create(fish_model_instanced_draw_t* this,
                                             context_t* aquarium_context,
                                             aquarium_t* aquarium,
                                             model_group_t type,
                                             model_name_t name, bool blend)
{
  fish_model_instanced_draw_init_defaults(this);

  fish_model_create(&this->fish_model, type, name, blend, aquarium);

  this->aquarium_context = aquarium_context;
  this->wgpu_context     = aquarium_context->wgpu_context;

  const fish_t* fish_info = &fish_table[name - MODELSMALLFISHAINSTANCEDDRAWS];
  this->fish_vertex_uniforms.fish_length      = fish_info->fish_length;
  this->fish_vertex_uniforms.fish_bend_amount = fish_info->fish_bend_amount;
  this->fish_vertex_uniforms.fish_wave_length = fish_info->fish_wave_length;

  this->instance
    = aquarium->fish_count[fish_info->model_name - MODELSMALLFISHA];
  this->fish_pers
    = malloc(this->instance + sizeof(fish_model_instanced_draw_fish_per));
}

static void fish_model_instanced_draw_destroy(fish_model_instanced_draw_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, this->fish_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, this->fish_pers_buffer)
  free(this->fish_pers);
}

static void fish_model_instanced_draw_init(fish_model_instanced_draw_t* this)
{
  if (this->instance == 0) {
    return;
  }

  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
    .size  = sizeof(fish_model_instanced_draw_fish_per) * this->instance,
    .mappedAtCreation = false,
  };
  this->fish_pers_buffer = context_create_buffer(wgpu_context, &buffer_desc);

  WGPUVertexAttribute vertex_attributes[9] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, world_position),
      .shaderLocation = 4,
    },
    [5] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 5,
    },
    [6] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32,
      .offset = offsetof(fish_model_instanced_draw_fish_per, scale),
      .shaderLocation = 6,
    },
    [7] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, next_position),
      .shaderLocation = 7,
    },
    [8] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = offsetof(fish_model_instanced_draw_fish_per, time),
      .shaderLocation = 8,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[6] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.position.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.normal.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.tex_coord.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.tangent.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = this->buffers.bi_normal.size,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
    [5] = (WGPUVertexBufferLayout) {
      .arrayStride    = sizeof(fish_model_instanced_draw_fish_per),
      .stepMode       = WGPUVertexStepMode_Instance,
      .attributeCount = 4,
      .attributes     = &vertex_attributes[5],
    },
  };

  this->vertex_state.module      = this->shader_modules.vertex;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[8] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [7] = (WGPUBindGroupLayoutEntry) {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    uint32_t bgl_entry_count = 0;
    if (this->textures.skybox && this->textures.reflection) {
      bgl_entry_count = 8;
    }
    else {
      bgl_entry_count = 5;
      bgl_entries[3]  = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      };
    }

    this->bind_group_layouts.model = context_make_bind_group_layout(
      wgpu_context, bgl_entries, bgl_entry_count);
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.per = context_make_bind_group_layout(
      wgpu_context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->aquarium_context->bind_group_layouts.general, /* Group 0 */
    this->aquarium_context->bind_group_layouts.world,   /* Group 1 */
    this->bind_group_layouts.model,                     /* Group 2 */
    this->bind_group_layouts.per,                       /* Group 3 */
  };
  this->pipeline_layout = context_make_basic_pipeline_layout(
    wgpu_context, bind_group_layouts, (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    wgpu_context, this->pipeline_layout, this->shader_modules.fragment,
    &this->vertex_state, this->fish_model.model.blend);

  this->fish_vertex_buffer = context_create_buffer_from_data(
    wgpu_context, &this->fish_vertex_uniforms,
    sizeof(this->fish_vertex_uniforms), sizeof(this->fish_vertex_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.light_factor = context_create_buffer_from_data(
    wgpu_context, &this->light_factor_uniforms,
    sizeof(this->light_factor_uniforms), sizeof(this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  // Fish models includes small, medium and big. Some of them contains
  // reflection and skybox texture, but some doesn't.
  {
    WGPUBindGroupEntry bg_entries[8] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = this->fish_vertex_buffer,
        .offset = 0,
        .size = sizeof(this->fish_vertex_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer = this->uniform_buffers.light_factor,
        .offset = 0,
        .size = sizeof(this->light_factor_uniforms)
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = this->textures.reflection->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .sampler = this->textures.skybox->sampler,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .textureView = this->textures.diffuse->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding = 5,
        .textureView = this->textures.normal->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding = 6,
        .textureView = this->textures.reflection->view,
      },
      [7] = (WGPUBindGroupEntry) {
        .binding = 7,
        .textureView = this->textures.skybox->view,
      },
    };
    uint32_t bg_entry_count = 0;
    if (this->textures.skybox && this->textures.reflection) {
      bg_entry_count = 8;
    }
    else {
      bg_entry_count = 5;
      bg_entries[2]  = (WGPUBindGroupEntry){
         .binding = 2,
         .sampler = this->textures.diffuse->sampler,
      };
      bg_entries[3] = (WGPUBindGroupEntry){
        .binding     = 3,
        .textureView = this->textures.diffuse->view,
      };
      bg_entries[4] = (WGPUBindGroupEntry){
        .binding     = 4,
        .textureView = this->textures.normal->view,
      };
    }
    this->bind_groups.model = context_make_bind_group(
      wgpu_context, this->bind_group_layouts.model, bg_entries, bg_entry_count);
  }

  context_set_buffer_data(wgpu_context, this->uniform_buffers.light_factor,
                          sizeof(this->light_factor_uniforms),
                          &this->light_factor_uniforms,
                          sizeof(this->light_factor_uniforms));
  context_set_buffer_data(
    wgpu_context, this->fish_vertex_buffer, sizeof(this->fish_vertex_uniforms),
    &this->fish_vertex_uniforms, sizeof(this->fish_vertex_uniforms));
}

static void fish_model_instanced_draw_draw(fish_model_instanced_draw_t* this)
{
  if (this->instance == 0) {
    return;
  }

  wgpu_context_t* wgpu_context = this->wgpu_context;

  context_set_buffer_data(
    wgpu_context, this->fish_pers_buffer,
    sizeof(fish_model_instanced_draw_fish_per) * this->instance,
    this->fish_pers,
    sizeof(fish_model_instanced_draw_fish_per) * this->instance);

  WGPURenderPassEncoder render_pass = this->aquarium_context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(
    render_pass, 0, this->aquarium_context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(
    render_pass, 1, this->aquarium_context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_groups.model, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                       this->buffers.tangent.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                       this->buffers.bi_normal.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    wgpu_context->rpass_enc, 5, this->fish_pers_buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices.buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->buffers.indices.total_components,
                                   this->instance, 0, 0, 0);
  this->instance = 0;
}

void fish_model_instanced_draw_update_fish_per_uniforms(
  fish_model_instanced_draw_t* this, float x, float y, float z, float next_x,
  float next_y, float next_z, float scale, float time, int index)
{
  this->fish_pers[index].world_position[0] = x;
  this->fish_pers[index].world_position[1] = y;
  this->fish_pers[index].world_position[2] = z;
  this->fish_pers[index].next_position[0]  = next_x;
  this->fish_pers[index].next_position[1]  = next_y;
  this->fish_pers[index].next_position[2]  = next_z;
  this->fish_pers[index].scale             = scale;
  this->fish_pers[index].time              = time;
}

/* -------------------------------------------------------------------------- *
 * Generic model - Defines generic model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    world_uniforms_t world_uniforms[20];
  } world_uniform_per;
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } bind_group_layouts;
  WGPUPipelineLayout pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer world;
  } uniform_buffers;
  wgpu_context_t* wgpu_context;
  context_t* context;
  int32_t instance;
} generic_model_t;

static void generic_model_init_defaults(generic_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 1.0f;
}

static void generic_model_create(generic_model_t* this, context_t* context,
                                 model_group_t type, model_name_t name,
                                 bool blend)
{
  generic_model_init_defaults(this);

  this->context      = context;
  this->wgpu_context = context->wgpu_context;

  model_create(&this->model, type, name, blend);
}

static void generic_model_destroy(generic_model_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.world)
}

static void generic_model_initialize(generic_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  // Generic models use reflection, normal or diffuse shaders, of which
  // groupLayouts are diiferent in texture binding.  MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  WGPUVertexAttribute vertex_attributes[5] = {0};
  {
    vertex_attributes[0].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[0].offset         = 0;
    vertex_attributes[0].shaderLocation = 0;
    vertex_attributes[1].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[1].offset         = 0;
    vertex_attributes[1].shaderLocation = 1;
    vertex_attributes[2].format         = WGPUVertexFormat_Float32x2;
    vertex_attributes[2].offset         = 0;
    vertex_attributes[2].shaderLocation = 2;
    vertex_attributes[3].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[3].offset         = 0;
    vertex_attributes[3].shaderLocation = 3;
    vertex_attributes[4].format         = WGPUVertexFormat_Float32x3;
    vertex_attributes[4].offset         = 0;
    vertex_attributes[4].shaderLocation = 4;
  }

  // Generic models use reflection, normal or diffuse shaders, of which
  // groupLayouts are diiferent in texture binding.  MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {0};
  uint32_t vertex_buffer_layout_count             = 0;
  {
    vertex_buffer_layouts[0].arrayStride    = this->buffers.position.size,
    vertex_buffer_layouts[0].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[0].attributeCount = 1;
    vertex_buffer_layouts[0].attributes     = &vertex_attributes[0];
    vertex_buffer_layouts[1].arrayStride    = this->buffers.normal.size,
    vertex_buffer_layouts[1].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[1].attributeCount = 1;
    vertex_buffer_layouts[1].attributes     = &vertex_attributes[1];
    vertex_buffer_layouts[2].arrayStride    = this->buffers.tex_coord.size,
    vertex_buffer_layouts[2].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[2].attributeCount = 1;
    vertex_buffer_layouts[2].attributes     = &vertex_attributes[2];
    vertex_buffer_layouts[3].arrayStride    = this->buffers.tangent.size,
    vertex_buffer_layouts[3].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[3].attributeCount = 1;
    vertex_buffer_layouts[3].attributes     = &vertex_attributes[3];
    vertex_buffer_layouts[4].arrayStride    = this->buffers.normal.size,
    vertex_buffer_layouts[4].stepMode       = WGPUVertexStepMode_Vertex;
    vertex_buffer_layouts[4].attributeCount = 1;
    vertex_buffer_layouts[4].attributes     = &vertex_attributes[4];
    if (this->textures.normal && this->model.name != MODELGLOBEBASE) {
      vertex_buffer_layout_count = 5;
    }
    else {
      vertex_buffer_layout_count = 3;
    }
  }

  this->vertex_state.module      = this->shader_modules.vertex;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = vertex_buffer_layout_count;
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[7] = {0};
    uint32_t bgl_entry_count                = 0;
    bgl_entries[0].binding                  = 0;
    bgl_entries[0].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[0].buffer.type              = WGPUBufferBindingType_Uniform;
    bgl_entries[0].buffer.hasDynamicOffset  = false;
    bgl_entries[0].buffer.minBindingSize    = 0;
    bgl_entries[1].binding                  = 1;
    bgl_entries[1].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[1].sampler.type             = WGPUSamplerBindingType_Filtering;
    bgl_entries[2].binding                  = 2;
    bgl_entries[2].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[2].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[2].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[2].texture.multisampled     = false;
    bgl_entries[3].binding                  = 3;
    bgl_entries[3].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[3].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[3].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[3].texture.multisampled     = false;
    bgl_entries[4].binding                  = 4;
    bgl_entries[4].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[4].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[4].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[4].texture.multisampled     = false;
    bgl_entries[5].binding                  = 5;
    bgl_entries[5].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[5].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[5].texture.viewDimension    = WGPUTextureViewDimension_2D;
    bgl_entries[5].texture.multisampled     = false;
    bgl_entries[6].binding                  = 6;
    bgl_entries[6].visibility               = WGPUShaderStage_Fragment;
    bgl_entries[6].texture.sampleType       = WGPUTextureSampleType_Float;
    bgl_entries[6].texture.viewDimension    = WGPUTextureViewDimension_Cube;
    bgl_entries[6].texture.multisampled     = false;
    if (this->textures.skybox && this->textures.reflection
        && this->model.name != MODELGLOBEBASE) {
      bgl_entry_count = 7;
    }
    else if (this->textures.normal && this->model.name != MODELGLOBEBASE) {
      bgl_entry_count = 4;
    }
    else {
      bgl_entry_count = 3;
    }
    this->bind_group_layouts.model = context_make_bind_group_layout(
      this->context, bgl_entries, bgl_entry_count);
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.per = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->context->bind_group_layouts.general, // Group 0
    this->context->bind_group_layouts.world,   // Group 1
    this->bind_group_layouts.model,            // Group 2
    this->bind_group_layouts.per,              // Group 3
  };
  this->pipeline_layout = context_make_basic_pipeline_layout(
    this->context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    this->context, this->pipeline_layout, this->shader_modules.fragment,
    &this->vertex_state, this->model.blend);

  this->uniform_buffers.light_factor = context_create_buffer_from_data(
    this->context, &this->light_factor_uniforms,
    sizeof(this->light_factor_uniforms), sizeof(this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  this->uniform_buffers.world = context_create_buffer_from_data(
    this->context, &this->world_uniform_per, sizeof(this->world_uniform_per),
    calc_constant_buffer_byte_size(sizeof(this->world_uniform_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  // Generic models use reflection, normal or diffuse shaders, of which
  // grouplayouts are diiferent in texture binding. MODELGLOBEBASE use diffuse
  // shader though it contains normal and reflection textures.
  {
    WGPUBindGroupEntry bg_entries[7] = {0};
    uint32_t bg_entry_count          = 0;
    bg_entries[0].binding            = 0;
    bg_entries[0].buffer             = this->uniform_buffers.light_factor;
    bg_entries[0].offset             = 0;
    bg_entries[0].size               = sizeof(light_uniforms_t);
    if (this->textures.skybox && this->textures.reflection
        && this->model.name != MODELGLOBEBASE) {
      bg_entry_count            = 7;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = this->textures.reflection->sampler,
      bg_entries[2].binding     = 2;
      bg_entries[2].sampler     = this->textures.skybox->sampler,
      bg_entries[3].binding     = 3;
      bg_entries[3].textureView = this->textures.diffuse->view,
      bg_entries[4].binding     = 4;
      bg_entries[4].textureView = this->textures.normal->view,
      bg_entries[5].binding     = 5;
      bg_entries[5].textureView = this->textures.reflection->view,
      bg_entries[6].binding     = 6;
      bg_entries[6].textureView = this->textures.skybox->view;
    }
    else if (this->textures.normal && this->model.name != MODELGLOBEBASE) {
      bg_entry_count            = 4;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = this->textures.diffuse->sampler;
      bg_entries[2].binding     = 2;
      bg_entries[2].textureView = this->textures.diffuse->view;
      bg_entries[3].binding     = 3;
      bg_entries[3].textureView = this->textures.normal->view;
    }
    else {
      bg_entry_count            = 3;
      bg_entries[1].binding     = 1;
      bg_entries[1].sampler     = this->textures.diffuse->sampler;
      bg_entries[2].binding     = 2;
      bg_entries[2].textureView = this->textures.diffuse->view;
    }
    this->bind_groups.model
      = context_make_bind_group(this->context, this->bind_group_layouts.model,
                                bg_entries, bg_entry_count);
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.world,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
      },
    };
    this->bind_groups.per
      = context_make_bind_group(this->context, this->bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(wgpu_context, this->uniform_buffers.light_factor,
                          sizeof(this->light_factor_uniforms),
                          &this->light_factor_uniforms,
                          sizeof(this->light_factor_uniforms));
}

static void generic_model_prepare_for_draw(generic_model_t* this)
{
  context_update_buffer_data(this->wgpu_context, this->uniform_buffers.world,
                             sizeof(this->world_uniform_per),
                             &this->world_uniform_per,
                             sizeof(this->world_uniform_per));
}

static void generic_model_draw(generic_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPURenderPassEncoder render_pass = this->context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_groups.model, 0,
                                    0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, this->bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  // diffuseShader doesn't have to input tangent buffer or binormal buffer.
  if (this->buffers.tangent.valid && this->buffers.bi_normal.valid) {
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                         this->buffers.tangent.buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                         this->buffers.bi_normal.buffer, 0,
                                         WGPU_WHOLE_SIZE);
  }
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices.buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->buffers.indices.total_components, 1, 0,
                                   0, 0);
  this->instance = 0;
}

static void generic_model_update_per_instance_uniforms(
  generic_model_t* this, const world_uniforms_t* world_uniforms)
{
  memcpy(&this->world_uniform_per.world_uniforms[this->instance],
         world_uniforms, sizeof(world_uniforms_t));

  this->instance++;
}

/* -------------------------------------------------------------------------- *
 * Inner model - Defines inner model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t model;
  struct {
    float eta;
    float tank_color_fudge;
    float refraction_fudge;
    float padding;
  } inner_uniforms;
  world_uniforms_t world_uniform_per;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } bind_group_layouts;
  WGPUPipelineLayout pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } bind_groups;
  struct {
    WGPUBuffer inner;
    WGPUBuffer view;
  } uniform_buffers;
  wgpu_context_t* wgpu_context;
  context_t* context;
} inner_model_t;

static void inner_model_init_defaults(inner_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->inner_uniforms.eta              = 1.0f;
  this->inner_uniforms.tank_color_fudge = 0.796f;
  this->inner_uniforms.refraction_fudge = 3.0f;
}

static void inner_model_create(inner_model_t* this, context_t* context,
                               model_group_t type, model_name_t name,
                               bool blend)
{
  inner_model_init_defaults(this);

  this->context      = context;
  this->wgpu_context = context->wgpu_context;

  model_create(&this->model, type, name, blend);
}

static void inner_model_destroy(inner_model_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.inner)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.view)
}

static void inner_model_initialize(inner_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUShaderModule vs_module = program_get_vs_module(this->model.program);

  texture_t** texture_map   = this->model.texture_map;
  this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = this->model.buffer_map;
  this->buffers.position     = buffer_map[BUFFERTYPE_POSITION];
  this->buffers.normal       = buffer_map[BUFFERTYPE_NORMAL];
  this->buffers.tex_coord    = buffer_map[BUFFERTYPE_TEX_COORD];
  this->buffers.tangent      = buffer_map[BUFFERTYPE_TANGENT];
  this->buffers.bi_normal    = buffer_map[BUFFERTYPE_BI_NORMAL];
  this->buffers.indices      = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(this->buffers.normal),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  this->vertex_state.module      = vs_module;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[7] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type  =WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [3] = (WGPUBindGroupLayoutEntry) {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [4] = (WGPUBindGroupLayoutEntry) {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [5] = (WGPUBindGroupLayoutEntry) {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
      [6] = (WGPUBindGroupLayoutEntry) {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_Cube,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.model = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.per = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->context->bind_group_layouts.general, /* Group 0 */
    this->context->bind_group_layouts.world,   /* Group 1 */
    this->bind_group_layouts.model,            /* Group 2 */
    this->bind_group_layouts.per,              /* Group 3 */
  };
  this->pipeline_layout = context_make_basic_pipeline_layout(
    this->context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    this->context, this->pipeline_layout, this->model.program->fs_module.module,
    &this->vertex_state, this->model.blend);

  this->uniform_buffers.inner = context_create_buffer_from_data(
    this->context, &this->inner_uniforms, sizeof(this->inner_uniforms),
    sizeof(this->inner_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  this->uniform_buffers.view = context_create_buffer_from_data(
    this->context, &this->world_uniform_per, sizeof(world_uniforms_t),
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[7] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.inner,
        .offset  = 0,
        .size    = sizeof(this->inner_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = this->textures.reflection->sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = this->textures.skybox->sampler,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding     = 3,
        .textureView = this->textures.diffuse->view,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding     = 4,
        .textureView = this->textures.normal->view,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding     = 5,
        .textureView = this->textures.reflection->view,
      },
      [6] = (WGPUBindGroupEntry) {
        .binding     = 6,
        .textureView = this->textures.skybox->view,
      },
    };
    this->bind_groups.model
      = context_make_bind_group(this->context, this->bind_group_layouts.model,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.view,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
      },
    };
    this->bind_groups.per
      = context_make_bind_group(this->context, this->bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(wgpu_context, this->uniform_buffers.inner,
                          sizeof(this->inner_uniforms), &this->inner_uniforms,
                          sizeof(this->inner_uniforms));
}

static void inner_model_draw(inner_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPURenderPassEncoder render_pass = this->context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_groups.model, 0,
                                    0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, this->bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                       this->buffers.tangent->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                       this->buffers.bi_normal->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices->buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->buffers.indices->total_components, 1,
                                   0, 0, 0);
}

static void
inner_model_update_per_instance_uniforms(inner_model_t* this,
                                         const world_uniforms_t* world_uniforms)
{
  memcpy(&this->world_uniform_per, world_uniforms, sizeof(world_uniforms_t));

  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t)),
    &this->world_uniform_per, sizeof(world_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Outside model - Defines outside model.
 * -------------------------------------------------------------------------- */

typedef struct {
  model_t model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* tangent;
    buffer_dawn_t* bi_normal;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  world_uniforms_t world_uniform_per[20];
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } bind_group_layouts;
  WGPUPipelineLayout pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer view;
  } uniform_buffers;
  wgpu_context_t* wgpu_context;
  context_t* context;
} outside_model_t;

static void outside_model_init_defaults(outside_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 0.0f;
}

static void outside_model_create(outside_model_t* this, context_t* context,
                                 model_group_t type, model_name_t name,
                                 bool blend)
{
  outside_model_init_defaults(this);

  this->context      = context;
  this->wgpu_context = context->wgpu_context;

  model_create(&this->model, type, name, blend);
}

static void outside_model_destroy(outside_model_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.view)
}

static void outside_model_initialize(outside_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUShaderModule vs_module = program_get_vs_module(this->model.program);

  texture_t** texture_map   = this->model.texture_map;
  this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = this->model.buffer_map;
  this->buffers.position     = buffer_map[BUFFERTYPE_POSITION];
  this->buffers.normal       = buffer_map[BUFFERTYPE_NORMAL];
  this->buffers.tex_coord    = buffer_map[BUFFERTYPE_TEX_COORD];
  this->buffers.tangent      = buffer_map[BUFFERTYPE_TANGENT];
  this->buffers.bi_normal    = buffer_map[BUFFERTYPE_BI_NORMAL];
  this->buffers.indices      = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[5] = {
    [0] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = 0,
      .shaderLocation = 2,
    },
    [3] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 3,
    },
    [4] = (WGPUVertexAttribute) {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = 0,
      .shaderLocation = 4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[5] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.position),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.tex_coord),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[2],
    },
    [3] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.tangent),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[3],
    },
    [4] = (WGPUVertexBufferLayout) {
      .arrayStride    = buffer_dawn_get_data_size(this->buffers.normal),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &vertex_attributes[4],
    },
  };

  this->vertex_state.module      = vs_module;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.per = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  /* Outside models use diffuse shaders. */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.model = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->context->bind_group_layouts.general, /* Group 0 */
    this->context->bind_group_layouts.world,   /* Group 1 */
    this->bind_group_layouts.model,            /* Group 2 */
    this->bind_group_layouts.per,              /* Group 3 */
  };
  this->pipeline_layout = context_make_basic_pipeline_layout(
    this->context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    this->context, this->pipeline_layout, this->model.program->fs_module.module,
    &this->vertex_state, this->model.blend);

  this->uniform_buffers.light_factor = context_create_buffer_from_data(
    this->context, &this->light_factor_uniforms,
    sizeof(this->light_factor_uniforms), sizeof(this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  this->uniform_buffers.view = context_create_buffer_from_data(
    this->context, &this->world_uniform_per, sizeof(world_uniforms_t) * 20,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(this->light_factor_uniforms)
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = this->textures.diffuse->sampler,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding     = 2,
        .textureView = this->textures.diffuse->view,
      },
    };
    this->bind_groups.model
      = context_make_bind_group(this->context, this->bind_group_layouts.model,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.view,
        .offset  = 0,
        .size  = calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
      },
    };
    this->bind_groups.per
      = context_make_bind_group(this->context, this->bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(
    wgpu_context, this->uniform_buffers.light_factor, sizeof(light_uniforms_t),
    &this->light_factor_uniforms, sizeof(light_uniforms_t));
}

static void outside_model_draw(outside_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPURenderPassEncoder render_pass = this->context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_groups.model, 0,
                                    0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, this->bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  // diffuseShader doesn't have to input tangent buffer or binormal buffer.
  if (this->buffers.tangent->valid && this->buffers.bi_normal->valid) {
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 3,
                                         this->buffers.tangent->buffer, 0,
                                         WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 4,
                                         this->buffers.bi_normal->buffer, 0,
                                         WGPU_WHOLE_SIZE);
  }
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices->buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->buffers.indices->total_components, 1,
                                   0, 0, 0);
}

static void outside_model_update_per_instance_uniforms(
  outside_model_t* this, const world_uniforms_t* world_uniforms)
{
  memcpy(&this->world_uniform_per, world_uniforms, sizeof(world_uniforms_t));

  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(world_uniforms_t) * 20),
    &this->world_uniform_per, sizeof(world_uniforms_t));
}

/* -------------------------------------------------------------------------- *
 * Seaweed model - Defines seaweed model.
 * -------------------------------------------------------------------------- */

typedef struct {
  float time;
  vec3 padding;
} seaweed_t;

typedef struct {
  model_t model;
  struct {
    texture_t* diffuse;
    texture_t* normal;
    texture_t* reflection;
    texture_t* skybox;
  } textures;
  struct {
    buffer_dawn_t* position;
    buffer_dawn_t* normal;
    buffer_dawn_t* tex_coord;
    buffer_dawn_t* indices;
  } buffers;
  struct {
    float shininess;
    float specular_factor;
  } light_factor_uniforms;
  struct {
    seaweed_t seaweed[20];
  } seaweed_per;
  struct {
    world_uniforms_t world_uniforms[20];
  } world_uniform_per;
  WGPUVertexState vertex_state;
  WGPURenderPipeline pipeline;
  struct {
    WGPUBindGroupLayout model;
    WGPUBindGroupLayout per;
  } bind_group_layouts;
  WGPUPipelineLayout pipeline_layout;
  struct {
    WGPUBindGroup model;
    WGPUBindGroup per;
  } bind_groups;
  struct {
    WGPUBuffer light_factor;
    WGPUBuffer time;
    WGPUBuffer view;
  } uniform_buffers;
  aquarium_t* aquarium;
  wgpu_context_t* wgpu_context;
  context_t* context;
  int32_t instance;
} seaweed_model_t;

static void seaweed_model_init_defaults(seaweed_model_t* this)
{
  memset(this, 0, sizeof(*this));

  this->light_factor_uniforms.shininess       = 50.0f;
  this->light_factor_uniforms.specular_factor = 1.0f;

  this->instance = 0;
}

static void seaweed_model_create(seaweed_model_t* this, context_t* context,
                                 aquarium_t* aquarium, model_group_t type,
                                 model_name_t name, bool blend)
{
  seaweed_model_init_defaults(this);

  model_create(&this->model, type, name, blend);

  this->aquarium     = aquarium;
  this->context      = context;
  this->wgpu_context = context->wgpu_context;
}

static void seaweed_model_initialize(seaweed_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPUShaderModule vs_module = program_get_vs_module(this->model.program);

  texture_t** texture_map   = this->model.texture_map;
  this->textures.diffuse    = texture_map[TEXTURETYPE_DIFFUSE];
  this->textures.normal     = texture_map[TEXTURETYPE_NORMAL_MAP];
  this->textures.reflection = texture_map[TEXTURETYPE_REFLECTION_MAP];
  this->textures.skybox     = texture_map[TEXTURETYPE_SKYBOX];

  buffer_dawn_t** buffer_map = this->model.buffer_map;
  this->buffers.position     = buffer_map[BUFFERTYPE_POSITION];
  this->buffers.normal       = buffer_map[BUFFERTYPE_NORMAL];
  this->buffers.tex_coord    = buffer_map[BUFFERTYPE_TEX_COORD];
  this->buffers.indices      = buffer_map[BUFFERTYPE_INDICES];

  WGPUVertexAttribute vertex_attributes[3] = {
    [0] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = 0,
      .shaderLocation = 0,
    },
    [1] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x3,
      .offset = 0,
      .shaderLocation = 1,
    },
    [2] = (WGPUVertexAttribute) {
      .format = WGPUVertexFormat_Float32x2,
      .offset = 0,
      .shaderLocation = 2,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layouts[3] = {
    [0] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(this->buffers.position),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[0],
    },
    [1] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(this->buffers.normal),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[1],
    },
    [2] = (WGPUVertexBufferLayout) {
      .arrayStride = buffer_dawn_get_data_size(this->buffers.tex_coord),
      .stepMode = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes = &vertex_attributes[2],
    },
  };

  this->vertex_state.module      = vs_module;
  this->vertex_state.entryPoint  = "main";
  this->vertex_state.bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts);
  this->vertex_state.buffers     = vertex_buffer_layouts;

  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = (WGPUSamplerBindingLayout){
          .type=WGPUSamplerBindingType_Filtering,
        },
        .texture = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled = false,
        },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.model = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 0,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding = 1,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize = 0,
        },
        .sampler = {0},
      },
    };
    this->bind_group_layouts.per = context_make_bind_group_layout(
      this->context, bgl_entries, (uint32_t)ARRAY_SIZE(bgl_entries));
  }

  WGPUBindGroupLayout bind_group_layouts[4] = {
    this->context->bind_group_layouts.general, /* Group 0 */
    this->context->bind_group_layouts.world,   /* Group 1 */
    this->bind_group_layouts.model,            /* Group 2 */
    this->bind_group_layouts.per,              /* Group 3 */
  };

  this->pipeline_layout = context_make_basic_pipeline_layout(
    this->context, bind_group_layouts,
    (uint32_t)ARRAY_SIZE(bind_group_layouts));

  this->pipeline = context_create_render_pipeline(
    this->context, this->pipeline_layout, this->model.program->fs_module.module,
    &this->vertex_state, this->model.blend);

  this->uniform_buffers.light_factor = context_create_buffer_from_data(
    this->context, &this->light_factor_uniforms,
    sizeof(this->light_factor_uniforms), sizeof(this->light_factor_uniforms),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.time = context_create_buffer_from_data(
    this->context, &this->seaweed_per, sizeof(this->seaweed_per),
    calc_constant_buffer_byte_size(sizeof(this->seaweed_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);
  this->uniform_buffers.view = context_create_buffer_from_data(
    this->context, &this->world_uniform_per, sizeof(this->world_uniform_per),
    calc_constant_buffer_byte_size(sizeof(this->world_uniform_per)),
    WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform);

  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.light_factor,
        .offset  = 0,
        .size    = sizeof(this->light_factor_uniforms)
      },
      [1] = (WGPUBindGroupEntry){
         .binding = 1,
         .sampler = this->textures.diffuse->sampler,
      },
      [2] = (WGPUBindGroupEntry){
        .binding     = 2,
        .textureView = this->textures.diffuse->view,
      },
      };
    this->bind_groups.model
      = context_make_bind_group(this->context, this->bind_group_layouts.model,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->uniform_buffers.view,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(this->world_uniform_per)),
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->uniform_buffers.time,
        .offset  = 0,
        .size    = calc_constant_buffer_byte_size(sizeof(this->seaweed_per)),
      },
    };
    this->bind_groups.per
      = context_make_bind_group(this->context, this->bind_group_layouts.per,
                                bg_entries, (uint32_t)ARRAY_SIZE(bg_entries));
  }

  context_set_buffer_data(wgpu_context, this->uniform_buffers.light_factor,
                          sizeof(this->light_factor_uniforms),
                          &this->light_factor_uniforms,
                          sizeof(this->light_factor_uniforms));
}

static void seaweed_model_destroy(seaweed_model_t* this)
{
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.model)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.per)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.model)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.per)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.light_factor)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.time)
  WGPU_RELEASE_RESOURCE(Buffer, this->uniform_buffers.view)
}

static void seaweed_model_prepare_for_draw(seaweed_model_t* this)
{
  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.view,
    calc_constant_buffer_byte_size(sizeof(this->world_uniform_per)),
    &this->world_uniform_per, sizeof(this->world_uniform_per));
  context_update_buffer_data(
    this->wgpu_context, this->uniform_buffers.time,
    calc_constant_buffer_byte_size(sizeof(this->seaweed_per)),
    &this->seaweed_per, sizeof(this->seaweed_per));
}

static void seaweed_model_draw(seaweed_model_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  WGPURenderPassEncoder render_pass = this->context->render_pass;
  wgpuRenderPassEncoderSetPipeline(render_pass, this->pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->context->bind_groups.general, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->context->bind_groups.world, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 2, this->bind_groups.model, 0,
                                    0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 3, this->bind_groups.per, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       this->buffers.position->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 1,
                                       this->buffers.normal->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 2,
                                       this->buffers.tex_coord->buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    wgpu_context->rpass_enc, this->buffers.indices->buffer,
    WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                   this->buffers.indices->total_components, 1,
                                   0, 0, 0);
  this->instance = 0;
}

static void seaweed_model_update_per_instance_uniforms(
  seaweed_model_t* this, const world_uniforms_t* world_uniforms)
{
  memcpy(&this->world_uniform_per.world_uniforms[this->instance],
         world_uniforms, sizeof(world_uniforms_t));
  this->seaweed_per.seaweed[this->instance].time
    = this->aquarium->g.mclock + this->instance;

  this->instance++;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int load_placement(aquarium_t* aquarium, const char* const placement)
{
  const cJSON* objects           = NULL;
  const cJSON* object            = NULL;
  const cJSON* name              = NULL;
  const cJSON* world_matrix      = NULL;
  const cJSON* world_matrix_item = NULL;
  int status                     = 0;
  cJSON* placement_json          = cJSON_Parse(placement);
  if (placement_json == NULL) {
    const char* error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      fprintf(stderr, "Error before: %s\n", error_ptr);
    }
    status = 0;
    goto load_placement_end;
  }

  if (!cJSON_IsObject(placement_json)
      || !cJSON_HasObjectItem(placement_json, "objects")) {
    fprintf(stderr, "Invalid placements file\n");
    status = 0;
    goto load_placement_end;
  }

  objects = cJSON_GetObjectItemCaseSensitive(placement_json, "objects");
  if (!cJSON_IsArray(objects)) {
    fprintf(stderr, "Objects item is not an array\n");
    status = 0;
    goto load_placement_end;
  }

  uint32_t cnts[MODELMAX] = {0};

  cJSON_ArrayForEach(object, objects)
  {
    name = cJSON_GetObjectItemCaseSensitive(object, "name");
    if (cJSON_IsString(name) && (name->valuestring != NULL)) {
      printf("name \"%s\"\n", name->valuestring);
    }

    world_matrix = cJSON_GetObjectItemCaseSensitive(object, "worldMatrix");
    if (cJSON_IsArray(world_matrix) && cJSON_GetArraySize(world_matrix) == 16) {
      cJSON_ArrayForEach(world_matrix_item, world_matrix)
      {
        if (!cJSON_IsNumber(world_matrix_item)) {
          status = 0;
          goto load_placement_end;
        }
        // printf("\t\t\tvalue = %f\n", world_matrix_item->valuedouble);
      }
      model_name_t mn = aquarium_map_model_name_str_to_model_name(
        aquarium, name->valuestring);
      cnts[mn]++;
    }
  }

  for (uint32_t i = 0; i < MODELMAX; ++i) {
    printf("%d,", cnts[i]);
  }
load_placement_end:
  cJSON_Delete(placement_json);
  return status;
}

void example_aquarium(int argc, char* argv[])
{
#if 10
  aquarium_t aquarium;
  aquarium_setup_model_enum_map(&aquarium);

  const char* filename = "/home/sdauwe/GitHub/aquarium/assets/PropPlacement.js";

  if (!file_exists(filename)) {
    log_fatal("Could not load texture from %s", filename);
    return;
  }

  file_read_result_t file_read_result = {0};
  read_file(filename, &file_read_result, true);

  load_placement(&aquarium, (const char* const)file_read_result.data);

  free(file_read_result.data);
#elif 1
  aquarium_t aquarium;
  aquarium_setup_model_enum_map(&aquarium);
  model_name_t mn
    = aquarium_map_model_name_str_to_model_name(&aquarium, "BigFishB");
  printf("DONE;");

#endif
}
