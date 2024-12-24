#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Metaballs
 *
 * WebGPU demo featuring marching cubes and bloom post-processing via compute
 * shaders, physically based shading, deferred rendering, gamma correction and
 * shadow mapping.
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/constants.ts
 * -------------------------------------------------------------------------- */

#define MAX_METABALLS 256u

static const WGPUTextureFormat DEPTH_FORMAT = WGPUTextureFormat_Depth24Plus;

static const uint32_t SHADOW_MAP_SIZE = 128;

static const uint32_t METABALLS_COMPUTE_WORKGROUP_SIZE[3] = {4, 4, 4};

static const vec4 BACKGROUND_COLOR = {0.1f, 0.1f, 0.1f, 1.0f};

/* -------------------------------------------------------------------------- *
 * Protocol
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/protocol.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float x_min;
  float y_min;
  float z_min;
  float x_step;
  float y_step;
  float z_step;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  float iso_level;
} ivolume_settings_t;

typedef struct {
  float x;
  float y;
  float z;
  float vx;
  float vy;
  float vz;
  float speed;
} imetaball_pos_t;

typedef struct {
  const char* fragment_shader_file;
  struct {
    WGPUBindGroupLayout* items;
    uint32_t item_count;
  } bind_group_layouts;
  struct {
    WGPUBindGroup* items;
    uint32_t item_count;
  } bind_groups;
  WGPUTextureFormat presentation_format;
  const char* label;
} iscreen_effect_t;

typedef struct {
  vec3 position;
  vec3 direction;
  vec3 color;
  float cut_off;
  float outer_cut_off;
  float intensity;
} ispot_light_t;

typedef enum quality_settings_enum {
  QualitySettings_Low    = 0,
  QualitySettings_Medium = 1,
  QualitySettings_High   = 2,
} quality_settings_enum;

typedef struct {
  bool bloom_toggle;
  uint32_t shadow_res;
  uint32_t point_lights_count;
  float output_scale;
  bool update_metaballs;
} quality_option_t;

/* -------------------------------------------------------------------------- *
 * Shared Chunks
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/shaders/shared-chunks.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 matrix;         /* matrix         */
  mat4 inverse_matrix; /* inverse matrix */
  vec2 output_size;    /* screen size    */
  float z_near;        /* near           */
  float z_far;         /* far            */
} projection_uniforms_t;

typedef struct {
  mat4 matrix;         /* matrix          */
  mat4 inverse_matrix; /* inverse matrix  */
  vec3 position;       /* camera position */
  float time;          /* time            */
  float delta_time;    /* delta time      */
  vec3 padding; /* padding required for struct alignment: size % 8 == 0 */
} view_uniforms_t;

typedef struct {
  mat4 matrix; // matrix
} screen_projection_uniforms_t;

typedef struct {
  mat4 matrix; // matrix
} screen_view_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Settings
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/settings.ts
 * -------------------------------------------------------------------------- */

static const quality_option_t QUALITIES[3] = {
  [QualitySettings_Low] = {
    .bloom_toggle       = false,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 1.0f,
    .update_metaballs   = false,
  },
  [QualitySettings_Medium] = {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 32,
    .output_scale       = 1.0f,
    .update_metaballs   = true,
   },
  [QualitySettings_High] = {
    .bloom_toggle       = true,
    .shadow_res         = 512,
    .point_lights_count = 128,
    .output_scale       = 1.0f,
    .update_metaballs   = true,
  },
};

static quality_settings_enum _quality = QualitySettings_Low;

static quality_option_t settings_get_quality_level(void)
{
  return QUALITIES[_quality];
}

static quality_settings_enum settings_get_quality(void)
{
  return _quality;
}

static void settings_set_quality(quality_settings_enum v)
{
  _quality = v;
}

/* -------------------------------------------------------------------------- *
 * Orthographic Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/orthographic-camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 UP_VECTOR;
  float left;
  float right;
  float top;
  float bottom;
  float near;
  float far;
  float zoom;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 view_matrix;
} orthographic_camera_t;

static void orthographic_camera_set_position(orthographic_camera_t* this,
                                             vec3 target)
{
  glm_vec3_copy(target, this->position);
}

static void orthographic_camera_update_view_matrix(orthographic_camera_t* this)
{
  glm_lookat(this->position,         /* eye    */
             this->look_at_position, /* center */
             this->UP_VECTOR,        /* up     */
             this->view_matrix       /* dest   */
  );
}

static void
orthographic_camera_update_projection_matrix(orthographic_camera_t* this)
{
  glm_ortho(this->left,             /* left   */
            this->right,            /* right  */
            this->bottom,           /* bottom */
            this->top,              /* top    */
            this->near,             /* nearZ  */
            this->far,              /* farZ   */
            this->projection_matrix /* dest   */
  );
}

static void orthographic_camera_look_at(orthographic_camera_t* this,
                                        vec3 target)
{
  glm_vec3_copy(target, this->look_at_position);
  orthographic_camera_update_view_matrix(this);
}

static void orthographic_camera_init_defaults(orthographic_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->UP_VECTOR);

  this->left   = -1.0f;
  this->right  = 1.0f;
  this->top    = 1.0f;
  this->bottom = -1.0f;
  this->near   = 0.1f;
  this->far    = 2000.0f;
  this->zoom   = 1.0f;

  glm_vec3_zero(this->position);
  glm_vec3_zero(this->look_at_position);
  glm_mat4_identity(this->projection_matrix);
  glm_mat4_identity(this->view_matrix);
}

static void orthographic_camera_init(orthographic_camera_t* this, float left,
                                     float right, float top, float bottom,
                                     float near, float far)
{
  orthographic_camera_init_defaults(this);

  this->left   = left;
  this->right  = right;
  this->top    = top;
  this->bottom = bottom;

  this->near = near;
  this->far  = far;

  orthographic_camera_update_projection_matrix(this);
}

/* -------------------------------------------------------------------------- *
 * Perspective Camera
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/perspective-camera.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec3 UP_VECTOR;
  vec3 position;
  vec3 look_at_position;
  mat4 projection_matrix;
  mat4 projection_inv_matrix;
  mat4 view_matrix;
  mat4 view_inv_matrix;
  float zoom;
  float field_of_view;
  float aspect;
  float near;
  float far;
} perspective_camera_t;

static void perspective_camera_set_position(perspective_camera_t* this,
                                            vec3 target)
{
  glm_vec3_copy(target, this->position);
}

static void perspective_camera_update_view_matrix(perspective_camera_t* this)
{
  glm_lookat(this->position,         /* eye    */
             this->look_at_position, /* center */
             this->UP_VECTOR,        /* up     */
             this->view_matrix       /* dest   */
  );
  glm_mat4_inv(this->view_matrix, this->view_inv_matrix);
}

static void
perspective_camera_update_projection_matrix(perspective_camera_t* this)
{
  glm_perspective(this->field_of_view,    /* fovy   */
                  this->aspect,           /* aspect */
                  this->near,             /* nearZ  */
                  this->far,              /* farZ   */
                  this->projection_matrix /* dest   */
  );
  glm_mat4_inv(this->projection_matrix, this->projection_inv_matrix);
}

static void perspective_camera_look_at(perspective_camera_t* this, vec3 target)
{
  glm_vec3_copy(target, this->look_at_position);
  perspective_camera_update_view_matrix(this);
}

static void perspective_camera_init_defaults(perspective_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, this->UP_VECTOR);

  glm_vec3_zero(this->position);
  glm_vec3_zero(this->look_at_position);

  glm_mat4_identity(this->projection_matrix);
  glm_mat4_identity(this->projection_inv_matrix);
  glm_mat4_identity(this->view_matrix);
  glm_mat4_identity(this->view_inv_matrix);

  this->zoom = 1.0f;
}

static void perspective_camera_init(perspective_camera_t* this,
                                    float field_of_view, float aspect,
                                    float near, float far)
{
  perspective_camera_init_defaults(this);

  this->field_of_view = field_of_view;
  this->aspect        = aspect;
  this->near          = near;
  this->far           = far;

  perspective_camera_update_projection_matrix(this);
}

/* -------------------------------------------------------------------------- *
 * Camera Controller
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/camera/camera-controller.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  float value;
  float damping;
} damped_action_t;

static void damped_action_add_force(damped_action_t* this, float force)
{
  this->value += force;
}

/**
 * @brief Stops the damping.
 */
static void damped_action_stop(damped_action_t* this)
{
  this->value = 0.0f;
}

/**
 * @brief Updates the damping and calls.
 */
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
  CAMERA_ACTION_STATE_IDLE,
  CAMERA_ACTION_STATE_ROTATE,
  CAMERA_ACTION_STATE_PAN,
  CAMERA_ACTION_STATE_ZOOM,
} camera_action_state_t;

typedef struct {
  perspective_camera_t* camera;
  vec3 target;
  float min_distance;
  float max_distance;
  bool is_enabled;
  bool is_damping;
  float damping_factor;
  bool is_zoom;
  float zoom_speed;
  bool is_rotate;
  float rotate_speed;
  bool is_pan;
  float key_pan_speed;
  bool enable_keys;
  vec3 origin_target;
  vec3 origin_position;
  damped_action_t target_x_damped_action;
  damped_action_t target_y_damped_action;
  damped_action_t target_z_damped_action;
  damped_action_t target_theta_damped_action;
  damped_action_t target_phi_damped_action;
  damped_action_t target_radius_damped_action;
  bool _is_shift_down;
  struct {
    vec2 start;
    vec2 end;
    vec2 delta;
  } _rotate;
  struct {
    float radius;
    float theta;
    float phi;
  } _spherical;
  float _zoom_distance_end;
  float _zoom_distance;
  camera_action_state_t state;

  uint64_t loop_id;
  struct {
    vec2 start;
    vec2 end;
    vec2 delta;
  } _pan;
  struct {
    vec2 start;
    vec2 end;
    vec2 delta;
  } _zoom;
  bool _paused;
  bool _is_debug;
  vec3 camera_position_debug;
  float mouse_wheel_force;
} camera_controller_t;

static void camera_controller_init_defaults(camera_controller_t* this)
{
  memset(this, 0, sizeof(*this));

  this->min_distance = 0;
  this->max_distance = INFINITY;
  this->is_enabled   = true;

  damped_action_init(&this->target_x_damped_action);
  damped_action_init(&this->target_y_damped_action);
  damped_action_init(&this->target_z_damped_action);
  damped_action_init(&this->target_theta_damped_action);
  damped_action_init(&this->target_phi_damped_action);
  damped_action_init(&this->target_radius_damped_action);

  this->_rotate.start[0] = 9999.0f;
  this->_rotate.start[1] = 9999.0f;

  this->_rotate.end[0] = 9999.0f;
  this->_rotate.end[1] = 9999.0f;

  this->_rotate.delta[0] = 9999.0f;
  this->_rotate.delta[1] = 9999.0f;

  this->_zoom_distance_end = 0.0f;
  this->_zoom_distance     = 0.0f;
  this->loop_id            = 0;

  this->_paused   = false;
  this->_is_debug = false;

  this->mouse_wheel_force = 1.0f;
}

static void camera_controller_create(camera_controller_t* this,
                                     perspective_camera_t* camera,
                                     bool is_debug, float mouse_wheel_force)
{
  camera_controller_init_defaults(this);

  this->mouse_wheel_force = mouse_wheel_force;

  ASSERT(camera);
  this->camera = camera;

  // Set to true to enable damping (inertia)
  // If damping is enabled, you must call controls.update() in your animation
  // loop
  this->is_damping     = false;
  this->damping_factor = 0.25f;

  // This option actually enables dollying in and out; left as "zoom" for
  // backwards compatibility. Set to false to disable zooming
  this->is_zoom    = true;
  this->zoom_speed = 1.0f;

  // Set to false to disable rotating
  this->is_rotate    = true;
  this->rotate_speed = 1.0f;

  // Set to false to disable panning
  this->is_pan        = true;
  this->key_pan_speed = 7.0f; // pixels moved per arrow key push

  // Set to false to disable use of the keys
  this->enable_keys = true;

  // for reset
  this->origin_position[0] = camera->position[0];
  this->origin_position[1] = camera->position[0];
  this->origin_position[2] = camera->position[0];

  const float dx     = this->camera->position[0];
  const float dy     = this->camera->position[1];
  const float dz     = this->camera->position[2];
  const float radius = sqrt(dx * dx + dy * dy + dz * dz);
  const float theta
    = atan2(this->camera->position[0],
            this->camera->position[2]); // equator angle around y-up axis
  const float phi = acos(
    CLAMP(this->camera->position[1] / radius, -1.0f, 1.0f)); // polar angle
  this->_spherical.radius = radius;
  this->_spherical.theta  = theta;
  this->_spherical.phi    = phi;

  this->_is_debug = is_debug;
}

static void camera_controller_look_at(camera_controller_t* this, vec3 target)
{
  glm_vec3_copy(target, this->target);
}

static void camera_controller_pause(camera_controller_t* this)
{
  this->_paused = true;
}

static void camera_controller_start(camera_controller_t* this)
{
  this->_paused = false;
}

static void camera_controller_update_pan_handler(camera_controller_t* this)
{
  vec3 x_dir = GLM_VEC3_ZERO_INIT;
  vec3 y_dir = GLM_VEC3_ZERO_INIT;
  vec3 z_dir = GLM_VEC3_ZERO_INIT;
  glm_vec3_sub(this->target, this->camera->position, z_dir);
  glm_vec3_normalize(z_dir);

  glm_vec3_cross(z_dir, (vec3){0.0f, 1.0f, 0.0f}, x_dir);
  glm_vec3_cross(x_dir, z_dir, y_dir);

  const float scale = MAX(this->_spherical.radius / 2000.0f, 0.001f);

  damped_action_add_force(
    &this->target_x_damped_action,
    (x_dir[0] * this->_pan.delta[0] + y_dir[0] * this->_pan.delta[1]) * scale);
  damped_action_add_force(
    &this->target_y_damped_action,
    (x_dir[1] * this->_pan.delta[0] + y_dir[1] * this->_pan.delta[1]) * scale);
  damped_action_add_force(
    &this->target_z_damped_action,
    (x_dir[2] * this->_pan.delta[0] + y_dir[2] * this->_pan.delta[1]) * scale);
}

static void
camera_controller_update_rotate_handler(camera_controller_t* this,
                                        wgpu_example_context_t* context)
{
  damped_action_add_force(&this->target_theta_damped_action,
                          -this->_rotate.delta[0]
                            / (float)context->wgpu_context->surface.width);
  damped_action_add_force(&this->target_phi_damped_action,
                          -this->_rotate.delta[1]
                            / (float)context->wgpu_context->surface.height);
}

static void camera_controller_update_zoom_handler(camera_controller_t* this)
{
  const float force = this->mouse_wheel_force;
  if (this->_zoom.delta[1] > 0.0f) {
    damped_action_add_force(&this->target_radius_damped_action, force);
  }
  else if (this->_zoom.delta[1] < 0.0f) {
    damped_action_add_force(&this->target_radius_damped_action, -force);
  }
}

static void
camera_controller_handle_input_events(camera_controller_t* this,
                                      wgpu_example_context_t* context)
{
  if (!this->is_enabled) {
    return;
  }

  /* Camera rotation handling */
  if (context->mouse_buttons.left) {
    if (this->state != CAMERA_ACTION_STATE_ROTATE) {
      this->state = CAMERA_ACTION_STATE_ROTATE;
      /* Rotation start */
      glm_vec2_copy(context->mouse_position, this->_rotate.start);
    }
    else if (this->state == CAMERA_ACTION_STATE_ROTATE
             && context->mouse_buttons.left) {
      /* Rotation end */
      glm_vec2_copy(context->mouse_position, this->_rotate.end);
      /* Rotation delta */
      glm_vec2_sub(this->_rotate.end, this->_rotate.start, this->_rotate.delta);
      /* Update camera rotation */
      camera_controller_update_rotate_handler(this, context);
      /* Rotation start */
      glm_vec2_copy(context->mouse_position, this->_rotate.start);
    }
  }
  else if (this->state == CAMERA_ACTION_STATE_ROTATE
           && !context->mouse_buttons.left) {
    this->state = CAMERA_ACTION_STATE_IDLE;
  }

  /* Camera pan handling */
  if (context->mouse_buttons.middle) {
    if (this->state != CAMERA_ACTION_STATE_PAN) {
      this->state = CAMERA_ACTION_STATE_PAN;
      /* Pan start */
      glm_vec2_copy(context->mouse_position, this->_pan.start);
    }
    else if (this->state == CAMERA_ACTION_STATE_PAN
             && context->mouse_buttons.middle) {
      /* Pan end */
      glm_vec2_copy(context->mouse_position, this->_pan.end);
      /* Pan delta */
      this->_pan.delta[0] = -0.5f * (this->_pan.end[0] - this->_pan.start[0]);
      this->_pan.delta[1] = 0.5f * (this->_pan.end[1] - this->_pan.start[1]);
      /* Update camera panning */
      camera_controller_update_pan_handler(this);
      /* Pan start */
      glm_vec2_copy(context->mouse_position, this->_pan.start);
    }
  }
  else if (this->state == CAMERA_ACTION_STATE_PAN
           && !context->mouse_buttons.middle) {
    this->state = CAMERA_ACTION_STATE_IDLE;
  }

  /* Camera zoom handling */
  if (context->mouse_buttons.right) {
    if (this->state != CAMERA_ACTION_STATE_ZOOM) {
      this->state = CAMERA_ACTION_STATE_ZOOM;
      /* Zoom start */
      glm_vec2_copy(context->mouse_position, this->_zoom.start);
    }
    else if (this->state == CAMERA_ACTION_STATE_ZOOM
             && context->mouse_buttons.right) {
      /* Zoom end */
      glm_vec2_copy(context->mouse_position, this->_zoom.end);
      /* Zoom delta */
      glm_vec2_sub(this->_zoom.end, this->_zoom.start, this->_zoom.delta);
      /* Update camera zoom */
      camera_controller_update_zoom_handler(this);
      /* Zoom start */
      glm_vec2_copy(context->mouse_position, this->_zoom.start);
    }
  }
  else if (this->state == CAMERA_ACTION_STATE_ZOOM
           && !context->mouse_buttons.right) {
    this->state = CAMERA_ACTION_STATE_IDLE;
  }
}

static void camera_controller_update_damped_action(camera_controller_t* this)
{
  this->target[0] += damped_action_update(&this->target_x_damped_action);
  this->target[1] += damped_action_update(&this->target_y_damped_action);
  this->target[2] += damped_action_update(&this->target_z_damped_action);

  this->_spherical.theta
    += damped_action_update(&this->target_theta_damped_action);
  this->_spherical.phi += damped_action_update(&this->target_phi_damped_action);
  this->_spherical.radius
    += damped_action_update(&this->target_radius_damped_action);
}

static void camera_controller_update_camera(camera_controller_t* this)
{
  const float s_radius       = this->_spherical.radius;
  const float s_theta        = this->_spherical.theta;
  const float s_phi          = this->_spherical.phi;
  const float sin_phi_radius = sin(s_phi) * s_radius;

  this->camera->position[0] = sin_phi_radius * sin(s_theta) + this->target[0];
  this->camera->position[1] = cos(s_phi) * s_radius + this->target[1];
  this->camera->position[2] = sin_phi_radius * cos(s_theta) + this->target[2];

  this->camera->look_at_position[0] = this->target[0];
  this->camera->look_at_position[1] = this->target[1];
  this->camera->look_at_position[2] = this->target[2];

  perspective_camera_update_view_matrix(this->camera);
}

static void camera_controller_tick(camera_controller_t* this)
{
  if (!this->_paused) {
    camera_controller_update_damped_action(this);
    camera_controller_update_camera(this);

    if (this->_is_debug) {
      this->camera_position_debug[0]
        = round(this->camera->position[0] * 100) / 100.0f;
      this->camera_position_debug[1]
        = round(this->camera->position[1] * 100) / 100.0f;
      this->camera_position_debug[2]
        = round(this->camera->position[2] * 100) / 100.0f;
    }
  }
  this->loop_id++;
}

/* -------------------------------------------------------------------------- *
 * WebGPU Renderer
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/webgpu-renderer.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  wgpu_context_t* wgpu_context;
  uint32_t output_size[2];
  float device_pixel_ratio;
  struct {
    WGPUBindGroupLayout frame;
  } bind_group_layouts;
  struct {
    WGPUBindGroup frame;
  } bind_groups;
  struct {
    wgpu_buffer_t projection_ubo;
    wgpu_buffer_t view_ubo;
    wgpu_buffer_t screen_projection_ubo;
    wgpu_buffer_t screen_view_ubo;
  } ubos;
  struct {
    projection_uniforms_t projection_ubo;
    view_uniforms_t view_ubo;
    screen_projection_uniforms_t screen_projection_ubo;
    screen_view_uniforms_t screen_view_ubo;
  } ubos_data;
  struct {
    struct {
      WGPUTexture texture;
      WGPUTextureView view;
    } depth_texture;
  } textures;
  WGPUSampler default_sampler;
  WGPUTextureFormat presentation_format;
  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;
} webgpu_renderer_t;

static WGPUTextureFormat
webgpu_renderer_get_presentation_Format(webgpu_renderer_t* this)
{
  return this->wgpu_context ? this->wgpu_context->swap_chain.format :
                              WGPUTextureFormat_BGRA8Unorm;
}

static void webgpu_renderer_set_output_size(webgpu_renderer_t* this,
                                            uint32_t width, uint32_t height)
{
  this->output_size[0] = width;
  this->output_size[1] = height;
}

static void webgpu_renderer_get_output_size(webgpu_renderer_t* this,
                                            uint32_t* width, uint32_t* height)
{
  *width  = this->output_size[0];
  *height = this->output_size[1];
}

static void webgpu_renderer_init_defaults(webgpu_renderer_t* this)
{
  memset(this, 0, sizeof(*this));

  this->output_size[0]     = 512;
  this->output_size[1]     = 512;
  this->device_pixel_ratio = 1.0f;
}

static void webgpu_renderer_create(webgpu_renderer_t* this,
                                   wgpu_context_t* wgpu_context)
{
  webgpu_renderer_init_defaults(this);

  this->wgpu_context = wgpu_context;
}

static void webgpu_renderer_init(webgpu_renderer_t* this)
{
  wgpu_context_t* wgpu_context = this->wgpu_context;

  /* Set presentation format */
  this->presentation_format = webgpu_renderer_get_presentation_Format(this);

  /* default sampler */
  this->default_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "Default WebGPU renderer sampler",
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(this->default_sampler != NULL);

  /* Projection UBO */
  this->ubos.projection_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Projection UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(projection_uniforms_t),
                  });

  /* View UBO */
  this->ubos.view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "View UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(view_uniforms_t),
                  });

  /* Screen projection UBO*/
  this->ubos.screen_projection_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Screen projection UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(screen_projection_uniforms_t),
                  });

  /* Screen view UBO*/
  this->ubos.screen_view_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Screen view UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(screen_view_uniforms_t),
                  });

  /* Frame buffer Color attachment */
  this->framebuffer.color_attachments[0] =
    (WGPURenderPassColorAttachment) {
      .view          = NULL,
      .resolveTarget = NULL,
      .depthSlice    = ~0,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = BACKGROUND_COLOR[0],
        .g = BACKGROUND_COLOR[1],
        .b = BACKGROUND_COLOR[2],
        .a = BACKGROUND_COLOR[3],
      },
    };

  /* Depth texture */
  WGPUExtent3D texture_extent = {
    .width              = this->output_size[0],
    .height             = this->output_size[1],
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "WebGPU renderer depth texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = DEPTH_FORMAT,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->textures.depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->textures.depth_texture.texture != NULL);

  /* Depth texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "WebGPU renderer depth texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->textures.depth_texture.view = wgpuTextureCreateView(
    this->textures.depth_texture.texture, &texture_view_dec);
  ASSERT(this->textures.depth_texture.view != NULL);

  /* Frame buffer depth stencil attachment */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = this->textures.depth_texture.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthClearValue = 1.0f,
      .depthStoreOp    = WGPUStoreOp_Discard,
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };

  /* Frame bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = this->ubos.projection_ubo.size,
    },
    .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = this->ubos.view_ubo.size,
      },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };
  this->bind_group_layouts.frame = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "Frame bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layouts.frame != NULL);

  /* Frame bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->ubos.projection_ubo.buffer,
        .size    = this->ubos.projection_ubo.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->ubos.view_ubo.buffer,
        .size    = this->ubos.view_ubo.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .sampler = this->default_sampler,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "Frame bind group",
      .layout     = this->bind_group_layouts.frame,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_groups.frame
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_groups.frame != NULL);
  }
}

static void webgpu_renderer_destroy(webgpu_renderer_t* this)
{
  WGPU_RELEASE_RESOURCE(Texture, this->textures.depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->textures.depth_texture.view)
  WGPU_RELEASE_RESOURCE(Sampler, this->default_sampler)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.projection_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.view_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.screen_projection_ubo.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.screen_view_ubo.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.frame)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.frame)
}

static void webgpu_renderer_on_render(webgpu_renderer_t* this)
{
  this->framebuffer.color_attachments[0].view
    = this->wgpu_context->swap_chain.frame_buffer;
}

/* -------------------------------------------------------------------------- *
 * Marching Cubes
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/geometry/marching-cubes.ts
 * -------------------------------------------------------------------------- */

static const uint16_t MARCHING_CUBES_EDGE_TABLE[256] = {
  0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
  0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f,
  0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
  0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
  0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5,
  0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
  0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
  0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
  0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
  0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc,
  0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
  0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
  0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055,
  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
  0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
  0x066, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
  0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
  0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
  0x13a, 0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
  0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09,
  0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
  0x203, 0x109, 0x000,
};

// Each set of 16 values corresponds to one of the entries in the
// MarchingCubesEdgeTable above. The first value is the number of valid indices
// for this entry. The following 15 values are the triangle indices for this
// entry, with -1 representing "no index".
static const int8_t MARCHING_CUBES_TRI_TABLE[4096] = {
  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  8,
  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  1,  9,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  8,  3,  9,  8,  1,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 6,  9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,
  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, 3,  3,  11, 2,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  11, 2,  8,  11, 0,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 6,  1,  9,  0,  2,  3,  11, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1,
  -1, -1, 6,  3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, 9,  3,  9,  0,
  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, 6,  9,  8,  10, 10, 8,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  4,  7,  8,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 6,  4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 6,  0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, 6,  1,  2,
  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  3,  4,  7,  3,  0,
  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, 9,  9,  2,  10, 9,  0,  2,  8,  4,
  7,  -1, -1, -1, -1, -1, -1, 12, 2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,
  4,  -1, -1, -1, 6,  8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 9,  11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, 9,  9,
  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, 12, 4,  7,  11, 9,
  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, 9,  3,  10, 1,  3,  11, 10, 7,
  8,  4,  -1, -1, -1, -1, -1, -1, 12, 1,  11, 10, 1,  4,  11, 1,  0,  4,  7,
  11, 4,  -1, -1, -1, 12, 4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1,
  -1, -1, 9,  4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, 3,
  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  5,  4,
  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  5,  4,  1,  5,  0,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  5,  4,  8,  3,  5,  3,  1,  5,
  -1, -1, -1, -1, -1, -1, 6,  1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1,
  9,  5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, 12, 2,  10,
  5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, 6,  9,  5,  4,  2,  3,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  11, 2,  0,  8,  11, 4,  9,
  5,  -1, -1, -1, -1, -1, -1, 9,  0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1,
  -1, -1, -1, -1, 12, 2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1,
  -1, 9,  10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, 12, 4,
  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, 12, 5,  4,  0,  5,
  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, 9,  5,  4,  8,  5,  8,  10, 10,
  8,  11, -1, -1, -1, -1, -1, -1, 6,  9,  7,  8,  5,  7,  9,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 9,  9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1,
  -1, -1, 9,  0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, 6,
  1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  9,  7,  8,
  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, 12, 10, 1,  2,  9,  5,  0,
  5,  3,  0,  5,  7,  3,  -1, -1, -1, 12, 8,  0,  2,  8,  2,  5,  8,  5,  7,
  10, 5,  2,  -1, -1, -1, 9,  2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1,
  -1, -1, -1, 9,  7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1,
  12, 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, 12, 2,  3,
  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, 9,  11, 2,  1,  11, 1,
  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, 12, 9,  5,  8,  8,  5,  7,  10, 1,
  3,  10, 3,  11, -1, -1, -1, 15, 5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,
  10, 11, 10, 0,  15, 11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,
  0,  6,  11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10,
  6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  8,  3,  5,
  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  9,  0,  1,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  8,  3,  1,  9,  8,  5,  10, 6,  -1,
  -1, -1, -1, -1, -1, 6,  1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, 9,
  9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, 12, 5,  9,  8,
  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, 6,  2,  3,  11, 10, 6,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 0,  8,  11, 2,  0,  10, 6,  5,
  -1, -1, -1, -1, -1, -1, 9,  0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1,
  -1, -1, -1, 12, 5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1,
  9,  6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,
  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, 12, 3,  11, 6,  0,  3,
  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, 9,  6,  5,  9,  6,  9,  11, 11, 9,
  8,  -1, -1, -1, -1, -1, -1, 6,  5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 9,  4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1,
  -1, 9,  1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, 12, 10,
  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, 9,  6,  1,  2,  6,
  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  5,  5,  2,  6,  3,
  0,  4,  3,  4,  7,  -1, -1, -1, 12, 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,
  2,  6,  -1, -1, -1, 15, 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,
  6,  9,  9,  3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, 12,
  5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, 12, 0,  1,  9,
  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, 15, 9,  2,  1,  9,  11, 2,
  9,  4,  11, 7,  11, 4,  5,  10, 6,  12, 8,  4,  7,  3,  11, 5,  3,  5,  1,
  5,  11, 6,  -1, -1, -1, 15, 5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,
  0,  4,  11, 15, 0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,
  12, 6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, 6,  10, 4,
  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  10, 6,  4,  9,
  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, 9,  10, 0,  1,  10, 6,  0,  6,  4,
  0,  -1, -1, -1, -1, -1, -1, 12, 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,
  10, -1, -1, -1, 9,  1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1,
  -1, 12, 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, 6,  0,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  8,  3,  2,  8,
  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, 9,  10, 4,  9,  10, 6,  4,  11,
  2,  3,  -1, -1, -1, -1, -1, -1, 12, 0,  8,  2,  2,  8,  11, 4,  9,  10, 4,
  10, 6,  -1, -1, -1, 12, 3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1,
  -1, -1, 15, 6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  12,
  9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, 15, 8,  11, 1,
  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  9,  3,  11, 6,  3,  6,  0,
  0,  6,  4,  -1, -1, -1, -1, -1, -1, 6,  6,  4,  8,  11, 6,  8,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1,
  -1, -1, -1, 12, 0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1,
  12, 10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, 9,  10, 6,
  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  6,  1,  6,
  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,  6,  9,  2,  9,  1,  6,  7,
  9,  0,  9,  3,  7,  3,  9,  9,  7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1,
  -1, -1, -1, -1, 6,  7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 12, 2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, 15, 2,
  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  15, 1,  8,  0,  1,
  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, 12, 11, 2,  1,  11, 1,  7,  10,
  6,  1,  6,  7,  1,  -1, -1, -1, 15, 8,  9,  6,  8,  6,  7,  9,  1,  6,  11,
  6,  3,  1,  3,  6,  6,  0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 12, 7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, 3,
  7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  7,  6,  11,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  3,  0,  8,  11, 7,  6,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  0,  1,  9,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 9,  8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1,
  -1, -1, -1, 6,  10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
  9,  1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 9,  2,  9,
  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 12, 6,  11, 7,  2,  10,
  3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, 6,  7,  2,  3,  6,  2,  7,  -1, -1,
  -1, -1, -1, -1, -1, -1, -1, 9,  7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1,
  -1, -1, -1, -1, 9,  2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1,
  -1, 12, 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, 9,  10,
  7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, 12, 10, 7,  6,  1,
  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, 12, 0,  3,  7,  0,  7,  10, 0,
  10, 9,  6,  10, 7,  -1, -1, -1, 9,  7,  6,  10, 7,  10, 8,  8,  10, 9,  -1,
  -1, -1, -1, -1, -1, 6,  6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 9,  3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, 9,
  8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, 12, 9,  4,  6,
  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, 9,  6,  8,  4,  6,  11, 8,
  2,  10, 1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 3,  0,  11, 0,  6,  11,
  0,  4,  6,  -1, -1, -1, 12, 4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,
  -1, -1, -1, 15, 10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,
  9,  8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, 6,  0,  4,
  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 1,  9,  0,  2,  3,
  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, 9,  1,  9,  4,  1,  4,  2,  2,  4,
  6,  -1, -1, -1, -1, -1, -1, 12, 8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10,
  1,  -1, -1, -1, 9,  10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1,
  -1, 15, 4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  6,  10,
  9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  4,  9,  5,  7,
  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  3,  4,  9,  5,  11,
  7,  6,  -1, -1, -1, -1, -1, -1, 9,  5,  0,  1,  5,  4,  0,  7,  6,  11, -1,
  -1, -1, -1, -1, -1, 12, 11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1,
  -1, -1, 9,  9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, 12,
  6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, 12, 7,  6,  11,
  5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, 15, 3,  4,  8,  3,  5,  4,
  3,  2,  5,  10, 5,  2,  11, 7,  6,  9,  7,  2,  3,  7,  6,  2,  5,  4,  9,
  -1, -1, -1, -1, -1, -1, 12, 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,
  -1, -1, -1, 12, 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1,
  15, 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  12, 9,  5,
  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, 15, 1,  6,  10, 1,  7,
  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  15, 4,  0,  10, 4,  10, 5,  0,  3,
  10, 6,  10, 7,  3,  7,  10, 12, 7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,
  10, -1, -1, -1, 9,  6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1,
  -1, 12, 3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, 12, 0,
  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, 9,  6,  11, 3,  6,
  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, 12, 1,  2,  10, 9,  5,  11, 9,
  11, 8,  11, 5,  6,  -1, -1, -1, 15, 0,  11, 3,  0,  6,  11, 0,  9,  6,  5,
  6,  9,  1,  2,  10, 15, 11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,
  2,  5,  12, 6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, 12,
  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, 9,  9,  5,  6,
  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, 15, 1,  5,  8,  1,  8,  0,
  5,  6,  8,  3,  8,  2,  6,  2,  8,  6,  1,  5,  6,  2,  1,  6,  -1, -1, -1,
  -1, -1, -1, -1, -1, -1, 15, 1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,
  8,  9,  6,  12, 10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1,
  6,  0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  10, 5,
  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6,  11, 5,  10, 7,  5,
  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  11, 5,  10, 11, 7,  5,  8,  3,
  0,  -1, -1, -1, -1, -1, -1, 9,  5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1,
  -1, -1, -1, -1, 12, 10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1,
  -1, 9,  11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, 12, 0,
  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, 12, 9,  7,  5,  9,
  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, 15, 7,  5,  2,  7,  2,  11, 5,
  9,  2,  3,  2,  8,  9,  8,  2,  9,  2,  5,  10, 2,  3,  5,  3,  7,  5,  -1,
  -1, -1, -1, -1, -1, 12, 8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1,
  -1, -1, 12, 9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, 15,
  9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  6,  1,  3,  5,
  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  8,  7,  0,  7,  1,
  1,  7,  5,  -1, -1, -1, -1, -1, -1, 9,  9,  0,  3,  9,  3,  5,  5,  3,  7,
  -1, -1, -1, -1, -1, -1, 6,  9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1,
  -1, -1, -1, 9,  5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1,
  12, 5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, 12, 0,  1,
  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, 15, 10, 11, 4,  10, 4,
  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  12, 2,  5,  1,  2,  8,  5,  2,  11,
  8,  4,  5,  8,  -1, -1, -1, 15, 0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11,
  1,  5,  1,  11, 15, 0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,
  5,  6,  9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,
  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, 9,  5,  10, 2,  5,
  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, 15, 3,  10, 2,  3,  5,  10, 3,
  8,  5,  4,  5,  8,  0,  1,  9,  12, 5,  10, 2,  5,  2,  4,  1,  9,  2,  9,
  4,  2,  -1, -1, -1, 9,  8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1,
  -1, -1, 6,  0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,
  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, 3,  9,  4,  5,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  11, 7,  4,  9,  11,
  9,  10, 11, -1, -1, -1, -1, -1, -1, 12, 0,  8,  3,  4,  9,  7,  9,  11, 7,
  9,  10, 11, -1, -1, -1, 12, 1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11,
  -1, -1, -1, 15, 3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,
  12, 4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, 15, 9,  7,
  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  9,  11, 7,  4,  11, 4,
  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, 12, 11, 7,  4,  11, 4,  2,  8,  3,
  4,  3,  2,  4,  -1, -1, -1, 12, 2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,
  9,  -1, -1, -1, 15, 9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,
  7,  15, 3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, 6,  1,
  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  4,  9,  1,  4,
  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, 12, 4,  9,  1,  4,  1,  7,  0,
  8,  1,  8,  7,  1,  -1, -1, -1, 6,  4,  0,  3,  7,  4,  3,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, 3,  4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, 6,  9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
  3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, 9,  0,  1,  10,
  0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, 6,  3,  1,  10, 11, 3,  10,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  1,  2,  11, 1,  11, 9,  9,  11, 8,
  -1, -1, -1, -1, -1, -1, 12, 3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,
  -1, -1, -1, 6,  0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  3,  3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,  3,
  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, 6,  9,  10, 2,  0,  9,
  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 2,  3,  8,  2,  8,  10, 0,  1,
  8,  1,  10, 8,  -1, -1, -1, 3,  1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, 6,  1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 3,  0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,
  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

/* -------------------------------------------------------------------------- *
 * Metaballs Compute
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/compute/metaballs.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec4 position;
  float radius;
  float strength;
  float subtract;
  float padding;
} metaball_t;

typedef struct {
  uint32_t ball_count[4];
  metaball_t balls[MAX_METABALLS];
} metaball_list;

typedef struct {
  webgpu_renderer_t* renderer;

  ivolume_settings_t volume;
  imetaball_pos_t ball_positions[MAX_METABALLS];

  wgpu_buffer_t tables_buffer;
  wgpu_buffer_t metaball_buffer;
  wgpu_buffer_t volume_buffer;
  wgpu_buffer_t indirect_render_buffer;

  WGPUComputePipeline compute_metaballs_pipeline;
  WGPUComputePipeline compute_marching_cubes_pipeline;

  WGPUBindGroup compute_metaballs_bind_group;
  WGPUBindGroup compute_marching_cubes_bind_group;

  uint32_t indirect_render_array[9];
  metaball_list metaball_array;
  uint32_t* metaball_array_header;
  metaball_t* metaball_array_balls;

  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t normal_buffer;
  wgpu_buffer_t index_buffer;

  uint32_t index_count;

  float strength;
  float strength_target;
  float subtract;
  float subtract_target;

  bool has_calced_once;
} metaballs_compute_t;

static bool metaballs_compute_is_ready(metaballs_compute_t* this)
{
  return (this->compute_metaballs_pipeline != NULL)
         && (this->compute_marching_cubes_pipeline != NULL);
}

static void metaballs_compute_init(metaballs_compute_t* this)
{
  /* Compute metaballs pipeline */
  {
    /* Compute shader */
    wgpu_shader_t comp_shader = wgpu_shader_create(
      this->renderer->wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .label = "metaballs isosurface compute shader",
        .file  = "shaders/compute_metaballs/metaball_field_compute_shader.wgsl",
        .entry = "main",
      });

    /* Create pipeline */
    this->compute_metaballs_pipeline = wgpuDeviceCreateComputePipeline(
      this->renderer->wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .compute = comp_shader.programmable_stage_descriptor,
      });
    ASSERT(this->compute_metaballs_pipeline != NULL);

    /* Partial clean-up */
    wgpu_shader_release(&comp_shader);
  }

  /* Compute metaballs bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->metaball_buffer.buffer,
        .size    = this->metaball_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->volume_buffer.buffer,
        .size    = this->volume_buffer.size,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .label  = "compute metaballs bind group",
      .layout = wgpuComputePipelineGetBindGroupLayout(
        this->compute_metaballs_pipeline, 0),
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    this->compute_metaballs_bind_group = wgpuDeviceCreateBindGroup(
      this->renderer->wgpu_context->device, &bg_desc);
    ASSERT(this->compute_metaballs_bind_group != NULL);
  }

  /* Compute marching cubes pipeline */
  {
    /* Compute shader */
    wgpu_shader_t comp_shader = wgpu_shader_create(
      this->renderer->wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .label = "marching cubes computer shader",
        .file  = "shaders/compute_metaballs/marching_cubes_compute_shader.wgsl",
        .entry = "main",
      });

    /* Create pipeline */
    this->compute_marching_cubes_pipeline = wgpuDeviceCreateComputePipeline(
      this->renderer->wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "compute marching cubes pipeline",
        .compute = comp_shader.programmable_stage_descriptor,
      });
    ASSERT(this->compute_marching_cubes_pipeline != NULL);

    /* Partial clean-up */
    wgpu_shader_release(&comp_shader);
  }

  /* Compute marching cubes bind group */
  {
    WGPUBindGroupEntry bg_entries[6] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->tables_buffer.buffer,
        .size    = this->tables_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->volume_buffer.buffer,
        .size    = this->volume_buffer.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->vertex_buffer.buffer,
        .size    = this->vertex_buffer.size,
      },
      [3] = (WGPUBindGroupEntry) {
        .binding = 3,
        .buffer  = this->normal_buffer.buffer,
        .size    = this->normal_buffer.size,
      },
      [4] = (WGPUBindGroupEntry) {
        .binding = 4,
        .buffer  = this->index_buffer.buffer,
        .size    = this->index_buffer.size,
      },
      [5] = (WGPUBindGroupEntry) {
        .binding = 5,
        .buffer  = this->indirect_render_buffer.buffer,
        .size    = this->indirect_render_buffer.size,
      },
    };

    WGPUBindGroupDescriptor bg_desc = {
      .layout = wgpuComputePipelineGetBindGroupLayout(
        this->compute_marching_cubes_pipeline, 0),
      .label      = "compute marching cubes bind group",
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };

    this->compute_marching_cubes_bind_group = wgpuDeviceCreateBindGroup(
      this->renderer->wgpu_context->device, &bg_desc);
    ASSERT(this->compute_marching_cubes_bind_group != NULL);
  }
}

static void metaballs_compute_init_defaults(metaballs_compute_t* this)
{
  memset(this, 0, sizeof(*this));

  this->strength        = 1.0f;
  this->strength_target = this->strength;
  this->subtract        = 1.0f;
  this->subtract_target = this->subtract;

  this->has_calced_once = false;
}

static void metaballs_compute_create(metaballs_compute_t* this,
                                     webgpu_renderer_t* renderer,
                                     ivolume_settings_t* volume)
{
  metaballs_compute_init_defaults(this);
  this->renderer = renderer;

  memcpy(&this->volume, volume, sizeof(ivolume_settings_t));

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Metaballs table buffer */
  {
    size_t table_size = (ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE)
                         + ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE))
                        * sizeof(int32_t);
    WGPUBufferDescriptor buffer_desc = {
      .label            = "metaballs table buffer",
      .usage            = WGPUBufferUsage_Storage,
      .size             = table_size,
      .mappedAtCreation = true,
    };
    this->tables_buffer = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
      .count  = table_size / sizeof(int32_t),
    };
    ASSERT(this->tables_buffer.buffer);
    int32_t* tables_array = (int32_t*)wgpuBufferGetMappedRange(
      this->tables_buffer.buffer, 0, table_size);

    size_t j = 0;
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_EDGE_TABLE); ++i) {
      tables_array[j++] = (int32_t)MARCHING_CUBES_EDGE_TABLE[i];
    }
    for (size_t i = 0; i < ARRAY_SIZE(MARCHING_CUBES_TRI_TABLE); ++i) {
      tables_array[j++] = (int32_t)MARCHING_CUBES_TRI_TABLE[i];
    }

    wgpuBufferUnmap(this->tables_buffer.buffer);
  }

  /* Metaballs buffer */
  {
    this->metaball_array_header
      = (uint32_t*)(&this->metaball_array.ball_count[0]);
    this->metaball_array_balls = (metaball_t*)(&this->metaball_array.balls[0]);
    this->metaball_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "metaballs buffer",
                                           .usage = WGPUBufferUsage_Storage
                                                    | WGPUBufferUsage_CopyDst,
                                           .size = sizeof(this->metaball_array),
                                         });
  }

  /* Metaballs volume buffer */
  {
    const uint32_t volume_elements
      = volume->width * volume->height * volume->depth;
    const uint64_t volume_buffer_size = sizeof(float) * 12
                                        + sizeof(uint32_t) * 4
                                        + sizeof(float) * volume_elements;
    WGPUBufferDescriptor buffer_desc = {
      .label            = "metaballs volume buffer",
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = volume_buffer_size,
      .mappedAtCreation = true,
    };
    this->volume_buffer = (wgpu_buffer_t){
      .buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc),
      .usage  = buffer_desc.usage,
      .size   = buffer_desc.size,
      .count  = 12 + 4 + volume_elements,
    };
    ASSERT(this->volume_buffer.buffer);
    float* volume_mapped_array = (float*)wgpuBufferGetMappedRange(
      this->volume_buffer.buffer, 0, volume_buffer_size);
    float* volume_float32 = volume_mapped_array;
    uint32_t* volume_size = (uint32_t*)(&volume_mapped_array[12]);

    volume_float32[0] = volume->x_min;
    volume_float32[1] = volume->y_min;
    volume_float32[2] = volume->z_min;

    volume_float32[8]  = volume->x_step;
    volume_float32[9]  = volume->y_step;
    volume_float32[10] = volume->z_step;

    volume_size[0] = volume->width;
    volume_size[1] = volume->height;
    volume_size[2] = volume->depth;

    volume_float32[15] = volume->iso_level;
    wgpuBufferUnmap(this->volume_buffer.buffer);
  }

  const uint32_t marching_cube_cells
    = (volume->width - 1) * (volume->height - 1) * (volume->depth - 1);
  const size_t vertex_buffer_size
    = sizeof(float) * 3 * 12 * marching_cube_cells;
  const size_t index_buffer_size = sizeof(uint32_t) * 15 * marching_cube_cells;

  this->vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs vertex buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  this->normal_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs normal buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
                    .size  = vertex_buffer_size,
                  });

  this->index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "metaballs index buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
                    .size  = index_buffer_size,
                  });
  this->index_count = index_buffer_size / sizeof(uint32_t);

  this->indirect_render_array[0] = 500;
  this->indirect_render_buffer   = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "metaballs indirect draw buffer",
                      .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect
                             | WGPUBufferUsage_CopyDst,
                      .size = sizeof(this->indirect_render_array),
                  });

  for (uint32_t i = 0; i < MAX_METABALLS; ++i) {
    this->ball_positions[i].x     = (random_float() * 2 - 1) * volume->x_min;
    this->ball_positions[i].y     = (random_float() * 2 - 1) * volume->y_min;
    this->ball_positions[i].z     = (random_float() * 2 - 1) * volume->z_min;
    this->ball_positions[i].vx    = random_float() * 1000;
    this->ball_positions[i].vy    = (random_float() * 2 - 1) * 10;
    this->ball_positions[i].vz    = random_float() * 1000;
    this->ball_positions[i].speed = random_float() * 2 + 0.3f;
  }

  metaballs_compute_init(this);
}

static void metaballs_compute_destroy(metaballs_compute_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->tables_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->metaball_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->volume_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->indirect_render_buffer.buffer)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->compute_metaballs_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->compute_marching_cubes_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroup, this->compute_metaballs_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, this->compute_marching_cubes_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, this->vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->normal_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->index_buffer.buffer)
}

static void metaballs_compute_rearrange(metaballs_compute_t* this)
{
  this->subtract_target = 3.0f + random_float() * 3.0f;
  this->strength_target = 3.0f + random_float() * 3.0f;
}

static metaballs_compute_t*
metaballs_compute_update_sim(metaballs_compute_t* this,
                             WGPUComputePassEncoder compute_pass, float time,
                             float time_delta)
{
  UNUSED_VAR(time);

  if (!metaballs_compute_is_ready(this)) {
    return this;
  }

  this->subtract += (this->subtract_target - this->subtract) * time_delta * 4;
  this->strength += (this->strength_target - this->strength) * time_delta * 4;

  const uint32_t numblobs = MAX_METABALLS;

  this->metaball_array_header[0] = MAX_METABALLS;

  for (uint32_t i = 0; i < MAX_METABALLS; i++) {
    imetaball_pos_t* pos = &this->ball_positions[i];

    pos->vx += -pos->x * pos->speed * 20.0f;
    pos->vy += -pos->y * pos->speed * 20.0f;
    pos->vz += -pos->z * pos->speed * 20.0f;

    pos->x += pos->vx * pos->speed * time_delta * 0.0001f;
    pos->y += pos->vy * pos->speed * time_delta * 0.0001f;
    pos->z += pos->vz * pos->speed * time_delta * 0.0001f;

    const float padding = 0.9f;
    const float width   = fabs(this->volume.x_min) - padding;
    const float height  = fabs(this->volume.y_min) - padding;
    const float depth   = fabs(this->volume.z_min) - padding;

    if (pos->x > width) {
      pos->x = width;
      pos->vx *= -1.0f;
    }
    else if (pos->x < -width) {
      pos->x = -width;
      pos->vx *= -1.0f;
    }

    if (pos->y > height) {
      pos->y = height;
      pos->vy *= -1.0f;
    }
    else if (pos->y < -height) {
      pos->y = -height;
      pos->vy *= -1.0f;
    }

    if (pos->z > depth) {
      pos->z = depth;
      pos->vz *= -1.0f;
    }
    else if (pos->z < -depth) {
      pos->z = -depth;
      pos->vz *= -1.0f;
    }
  }

  for (uint32_t i = 0; i < numblobs; i++) {
    imetaball_pos_t* position = &this->ball_positions[i];
    metaball_t* metaball      = &this->metaball_array_balls[i];
    metaball->position[0]     = position->x;
    metaball->position[1]     = position->y;
    metaball->position[2]     = position->z;
    metaball->radius          = sqrt(this->strength / this->subtract);
    metaball->strength        = this->strength;
    metaball->subtract        = this->subtract;
  }

  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->metaball_buffer.buffer, 0,
                          &this->metaball_array, sizeof(this->metaball_array));
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->indirect_render_buffer.buffer, 0,
    &this->indirect_render_array, sizeof(this->indirect_render_array));

  const uint32_t dispatch_size[3] = {
    this->volume.width / METABALLS_COMPUTE_WORKGROUP_SIZE[0],  //
    this->volume.height / METABALLS_COMPUTE_WORKGROUP_SIZE[1], //
    this->volume.depth / METABALLS_COMPUTE_WORKGROUP_SIZE[2],  //
  };

  /* Update metaballs */
  if (this->compute_metaballs_pipeline) {
    wgpuComputePassEncoderSetPipeline(compute_pass,
                                      this->compute_metaballs_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 0, this->compute_metaballs_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass, dispatch_size[0], dispatch_size[1], dispatch_size[2]);
  }

  /* Update marching cubes */
  if (this->compute_marching_cubes_pipeline) {
    wgpuComputePassEncoderSetPipeline(compute_pass,
                                      this->compute_marching_cubes_pipeline);
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 0, this->compute_marching_cubes_bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass, dispatch_size[0], dispatch_size[1], dispatch_size[2]);
  }

  this->has_calced_once = true;

  return this;
}

/* -------------------------------------------------------------------------- *
 * Cube Geometry
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/geometry/create-box.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  struct {
    float* data[18 * 6];
    size_t data_size;
    size_t count;
  } positions;
  struct {
    float data[18 * 6];
    size_t data_size;
    size_t count;
  } normals;
  struct {
    float* data[12 * 6];
    size_t data_size;
    size_t count;
  } uvs;
} cube_geometry_t;

static void cube_geometry_init_defaults(cube_geometry_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void cube_geometry_create_cube(cube_geometry_t* this, vec3 dimensions)
{
  cube_geometry_init_defaults(this);

  /* Cube vertex positions */
  const vec3 position
    = {-dimensions[0] / 2.0f, -dimensions[1] / 2.0f, -dimensions[2] / 2.0f};
  const float x      = position[0];
  const float y      = position[1];
  const float z      = position[2];
  const float width  = dimensions[0];
  const float height = dimensions[1];
  const float depth  = dimensions[2];

  const vec3 fbl = {x, y, z + depth};
  const vec3 fbr = {x + width, y, z + depth};
  const vec3 ftl = {x, y + height, z + depth};
  const vec3 ftr = {x + width, y + height, z + depth};
  const vec3 bbl = {x, y, z};
  const vec3 bbr = {x + width, y, z};
  const vec3 btl = {x, y + height, z};
  const vec3 btr = {x + width, y + height, z};

  const float positions[18 * 6] = {
    // clang-format off
    /* front */
    fbl[0],
    fbl[1],
    fbl[2],
    fbr[0],
    fbr[1],
    fbr[2],
    ftl[0],
    ftl[1],
    ftl[2],
    ftl[0],
    ftl[1],
    ftl[2],
    fbr[0],
    fbr[1],
    fbr[2],
    ftr[0],
    ftr[1],
    ftr[2],

    /* right */
    fbr[0],
    fbr[1],
    fbr[2],
    bbr[0],
    bbr[1],
    bbr[2],
    ftr[0],
    ftr[1],
    ftr[2],
    ftr[0],
    ftr[1],
    ftr[2],
    bbr[0],
    bbr[1],
    bbr[2],
    btr[0],
    btr[1],
    btr[2],

    /* back */
    fbr[0],
    bbr[1],
    bbr[2],
    bbl[0],
    bbl[1],
    bbl[2],
    btr[0],
    btr[1],
    btr[2],
    btr[0],
    btr[1],
    btr[2],
    bbl[0],
    bbl[1],
    bbl[2],
    btl[0],
    btl[1],
    btl[2],

    /* left */
    bbl[0],
    bbl[1],
    bbl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    btl[0],
    btl[1],
    btl[2],
    btl[0],
    btl[1],
    btl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    ftl[0],
    ftl[1],
    ftl[2],

    /* top */
    ftl[0],
    ftl[1],
    ftl[2],
    ftr[0],
    ftr[1],
    ftr[2],
    btl[0],
    btl[1],
    btl[2],
    btl[0],
    btl[1],
    btl[2],
    ftr[0],
    ftr[1],
    ftr[2],
    btr[0],
    btr[1],
    btr[2],

    /* bottom */
    bbl[0],
    bbl[1],
    bbl[2],
    bbr[0],
    bbr[1],
    bbr[2],
    fbl[0],
    fbl[1],
    fbl[2],
    fbl[0],
    fbl[1],
    fbl[2],
    bbr[0],
    bbr[1],
    bbr[2],
    fbr[0],
    fbr[1],
    fbr[2],
    // clang-format on
  };
  memcpy(this->positions.data, positions, sizeof(positions));
  this->positions.data_size = sizeof(positions);
  this->positions.count     = (size_t)ARRAY_SIZE(positions);

  /* Cube vertex uvs */
  static const float uvs[12 * 6] = {
    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* front */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* right */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* back */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* left */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* top */

    0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, /* bottom */
  };
  memcpy(this->uvs.data, uvs, sizeof(uvs));
  this->uvs.data_size = sizeof(uvs);
  this->uvs.count     = (size_t)ARRAY_SIZE(uvs);

  /* Cube vertex normals */
  static const float normals[18 * 6] = {
    0,  0,  1,  0,  0,  1,  0,  0,  1,
    0,  0,  1,  0,  0,  1,  0,  0,  1, /* front */

    1,  0,  0,  1,  0,  0,  1,  0,  0,
    1,  0,  0,  1,  0,  0,  1,  0,  0, /* right */

    0,  0,  -1, 0,  0,  -1, 0,  0,  -1,
    0,  0,  -1, 0,  0,  -1, 0,  0,  -1, /* back */

    -1, 0,  0,  -1, 0,  0,  -1, 0,  0,
    -1, 0,  0,  -1, 0,  0,  -1, 0,  0, /* left */

    0,  1,  0,  0,  1,  0,  0,  1,  0,
    0,  1,  0,  0,  1,  0,  0,  1,  0, /* top */

    0,  -1, 0,  0,  -1, 0,  0,  -1, 0,
    0,  -1, 0,  0,  -1, 0,  0,  -1, 0, /* bottom */
  };
  memcpy(this->normals.data, normals, sizeof(normals));
  this->normals.data_size = sizeof(normals);
  this->normals.count     = (size_t)ARRAY_SIZE(normals);
}

/* -------------------------------------------------------------------------- *
 * Deg2Rad
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/helpers/deg-to-rad.ts
 * -------------------------------------------------------------------------- */

static float deg_to_rad(float deg)
{
  return (deg * PI) / 180.0f;
}

/* -------------------------------------------------------------------------- *
 * Point Lights
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/lighting/point-lights.ts
 * -------------------------------------------------------------------------- */

#define MAX_POINT_LIGHTS_COUNT 256u

typedef struct {
  vec4 position;
  vec4 velocity;
  vec3 color;
  float range;
  float intensity;
} input_point_light_t;

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUBindGroupLayout lights_buffer_compute_bind_group_layout;
  WGPUBindGroup lights_buffer_compute_bind_group;
  WGPUPipelineLayout update_compute_pipeline_layout;
  WGPUComputePipeline update_compute_pipeline;
  wgpu_buffer_t lights_buffer;
  wgpu_buffer_t lights_config_uniform_buffer;
  int32_t lights_count;
} point_lights_t;

static bool point_lights_is_ready(point_lights_t* this)
{
  return this->update_compute_pipeline != NULL;
}

static void point_lights_set_lights_count(point_lights_t* this, uint32_t v)
{
  this->lights_count = v;
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->lights_config_uniform_buffer.buffer, 0, &v,
                          sizeof(uint32_t));
}

static void point_lights_init(point_lights_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->lights_buffer_compute_bind_group_layout, // Group 0
      this->renderer->bind_group_layouts.frame,      // Group 1
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
      .label                = "point light update compute pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->update_compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &compute_pipeline_layout_desc);
    ASSERT(this->update_compute_pipeline_layout != NULL);
  }

  /* Compute pipeline */
  {
    wgpu_shader_t comp_shader = wgpu_shader_create(
      wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .label = "update point lights compute shader wgsl",
        .file
        = "shaders/compute_metaballs/update_point_lights_compute_shader.wgsl",
        .entry = "main",
      });
    this->update_compute_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "point light update compute pipeline",
        .layout  = this->update_compute_pipeline_layout,
        .compute = comp_shader.programmable_stage_descriptor,
      });
    wgpu_shader_release(&comp_shader);
  }
}

static void point_lights_init_defaults(point_lights_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void point_lights_create(point_lights_t* this,
                                webgpu_renderer_t* renderer)
{
  point_lights_init_defaults(this);
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Lights uniform buffer */
  {
    input_point_light_t lights_data[MAX_POINT_LIGHTS_COUNT] = {0};
    float x, y, z, vel_x, vel_y, vel_z, r, g, b, radius, intensity;
    for (uint32_t i = 0; i < MAX_POINT_LIGHTS_COUNT; ++i) {
      input_point_light_t* light_data = &lights_data[i];

      x = (random_float() * 2 - 1) * 20;
      y = -2.0f;
      z = (random_float() * 2 - 1) * 20;

      vel_x = random_float() * 4 - 2;
      vel_y = random_float() * 4 - 2;
      vel_z = random_float() * 4 - 2;

      r = random_float();
      g = random_float();
      b = random_float();

      radius    = 5 + random_float() * 3;
      intensity = 10 + random_float() * 10;

      // position
      light_data->position[0] = x;
      light_data->position[1] = y;
      light_data->position[2] = z;
      light_data->position[3] = 0.0f;
      // velocity
      light_data->velocity[0] = vel_x;
      light_data->velocity[1] = vel_y;
      light_data->velocity[2] = vel_z;
      light_data->velocity[3] = 0.0f;
      // color
      light_data->color[0] = r;
      light_data->color[1] = g;
      light_data->color[2] = b;
      // radius
      light_data->range = radius;
      // intensity
      light_data->intensity = intensity;
    }
    this->lights_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_Storage,
                                           .size  = sizeof(lights_data),
                                           .initial.data = lights_data,
                                         });
  }

  /* Lights config uniform buffer */
  {
    this->lights_count = settings_get_quality_level().point_lights_count;
    this->lights_config_uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size         = 1,
                                           .initial.data = &this->lights_count,
                                         });
  }

  /* Lights buffer compute bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
           .type           = WGPUBufferBindingType_Storage,
           .minBindingSize = this->lights_buffer.size,
         },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
             .type           = WGPUBufferBindingType_Uniform,
             .minBindingSize = this->lights_config_uniform_buffer.size,
         },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "lights update compute bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->lights_buffer_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->lights_buffer_compute_bind_group_layout != NULL);
  }

  /* Lights buffer compute bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->lights_buffer.buffer,
        .size    = this->lights_buffer.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->lights_config_uniform_buffer.buffer,
        .size    = this->lights_config_uniform_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "lights buffer compute bind group",
      .layout     = this->lights_buffer_compute_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->lights_buffer_compute_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->lights_buffer_compute_bind_group != NULL);
  }

  point_lights_init(this);
}

static void point_lights_destroy(point_lights_t* this)
{
  WGPU_RELEASE_RESOURCE(BindGroupLayout,
                        this->lights_buffer_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->lights_buffer_compute_bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->update_compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->update_compute_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->lights_config_uniform_buffer.buffer)
}

static point_lights_t*
point_lights_update_sim(point_lights_t* this,
                        WGPUComputePassEncoder compute_pass)
{
  if (!point_lights_is_ready(this)) {
    return this;
  }
  wgpuComputePassEncoderSetPipeline(compute_pass,
                                    this->update_compute_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->lights_buffer_compute_bind_group, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->renderer->bind_groups.frame, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(settings_get_quality_level().point_lights_count / 64.0f), 1,
    1);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Spot Light
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/lighting/spot-light.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  vec4 position;
  vec4 direction;
  vec3 color;
  float cut_off;
  float outer_cut_off;
  float intensity;
  vec2 padding; /* padding required for struct alignment: size % 8 == 0 */
} spot_light_info_t;

typedef struct {
  webgpu_renderer_t* renderer;
  perspective_camera_t camera;

  vec3 _position;
  vec3 _direction;
  vec3 _color;
  float _cut_off;
  float _outer_cut_off;
  float _intensity;

  struct {
    wgpu_buffer_t light_info;
    wgpu_buffer_t projection;
    wgpu_buffer_t view;
  } ubos;
  struct {
    spot_light_info_t light_info;
    projection_uniforms_t projection;
    view_uniforms_t view;
  } ubos_data;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth_texture;
  struct {
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  struct {
    WGPUBindGroupLayout ubos;
    WGPUBindGroupLayout depth_texture;
  } bind_group_layouts;

  struct {
    WGPUBindGroup ubos;
    WGPUBindGroup depth_texture;
  } bind_groups;
} spot_light_t;

static void spot_light_get_position(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_position, *dest);
}

static void spot_light_set_position(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_position);
  glm_vec3_copy(v, this->ubos_data.light_info.position);
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));

  perspective_camera_set_position(&this->camera,
                                  (vec3){-v[0] * 15, v[1], -v[2] * 15});
  perspective_camera_update_view_matrix(&this->camera);

  view_uniforms_t* view_uniforms = &this->ubos_data.view;
  glm_mat4_copy(this->camera.view_matrix, view_uniforms->matrix);
  glm_mat4_copy(this->camera.view_inv_matrix, view_uniforms->inverse_matrix);
  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->ubos.view.buffer,
                          0, &view_uniforms, sizeof(view_uniforms_t));
}

static void spot_light_get_direction(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_direction, *dest);
}

static void spot_light_set_direction(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_direction);
  glm_vec3_copy(v, this->ubos_data.light_info.direction);
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));

  glm_vec3_copy((vec3){v[0], v[1], v[2]}, this->camera.look_at_position);
  perspective_camera_update_view_matrix(&this->camera);

  view_uniforms_t* view_uniforms = &this->ubos_data.view;
  glm_mat4_copy(this->camera.view_matrix, view_uniforms->matrix);
  glm_mat4_copy(this->camera.view_inv_matrix, view_uniforms->inverse_matrix);
  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->ubos.view.buffer,
                          0, &view_uniforms, sizeof(view_uniforms_t));
}

static void spot_light_get_color(spot_light_t* this, vec3* dest)
{
  glm_vec3_copy(this->_color, *dest);
}

static void spot_light_set_color(spot_light_t* this, vec3 v)
{
  glm_vec3_copy(v, this->_color);
  glm_vec3_copy(v, this->ubos_data.light_info.color);
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));
}

static float spot_light_get_cut_off(spot_light_t* this)
{
  return this->_cut_off;
}

static void spot_light_set_cut_off(spot_light_t* this, float v)
{
  this->_cut_off                     = v;
  this->ubos_data.light_info.cut_off = cosf(v);
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));
}

static float spot_light_get_outer_cut_off(spot_light_t* this)
{
  return this->_outer_cut_off;
}

static void spot_light_set_outer_cut_off(spot_light_t* this, float v)
{
  this->_outer_cut_off                     = v;
  this->ubos_data.light_info.outer_cut_off = cosf(v);
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));

  this->camera.field_of_view = v * 1.5f;
  perspective_camera_update_projection_matrix(&this->camera);

  projection_uniforms_t* projection_uniforms = &this->ubos_data.projection;
  glm_mat4_copy(this->camera.projection_matrix, projection_uniforms->matrix);
  glm_mat4_copy(this->camera.projection_inv_matrix,
                projection_uniforms->inverse_matrix);
  wgpu_queue_write_buffer(this->renderer->wgpu_context,
                          this->ubos.projection.buffer, 0, &projection_uniforms,
                          sizeof(projection_uniforms_t));
}

static float spot_light_get_intensity(spot_light_t* this)
{
  return this->_intensity;
}

static void spot_light_set_intensity(spot_light_t* this, float v)
{
  this->_intensity                     = v;
  this->ubos_data.light_info.intensity = v;
  wgpu_queue_write_buffer(
    this->renderer->wgpu_context, this->ubos.light_info.buffer, 0,
    &this->ubos_data.light_info, sizeof(spot_light_info_t));
}

static void spot_light_init_defaults(spot_light_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void spot_light_create(spot_light_t* this, webgpu_renderer_t* renderer,
                              ispot_light_t* ispot_light)
{
  spot_light_init_defaults(this);

  this->renderer = renderer;
  perspective_camera_init(&this->camera, deg_to_rad(56.0f), 1.0f, 0.1f, 120.0f);
  perspective_camera_update_view_matrix(&this->camera);
  perspective_camera_update_projection_matrix(&this->camera);

  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Depth texture */
  WGPUExtent3D texture_extent = {
    .width              = settings_get_quality_level().shadow_res,
    .height             = settings_get_quality_level().shadow_res,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "spot light depth texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_Depth32Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->depth_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->depth_texture.texture != NULL);

  /* Depth texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "spot light depth texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->depth_texture.view
    = wgpuTextureCreateView(this->depth_texture.texture, &texture_view_dec);
  ASSERT(this->depth_texture.view != NULL);

  /* Light info UBO */
  this->ubos.light_info = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(spot_light_info_t),
                  });

  /* Projection UBO */
  this->ubos.projection = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(projection_uniforms_t),
                  });
  projection_uniforms_t* projection_uniforms = &this->ubos_data.projection;
  glm_mat4_copy(this->camera.projection_matrix, projection_uniforms->matrix);
  glm_mat4_copy(this->camera.projection_inv_matrix,
                projection_uniforms->inverse_matrix);
  glm_vec2_copy((vec2){(float)settings_get_quality_level().shadow_res,
                       (float)settings_get_quality_level().shadow_res},
                projection_uniforms->output_size);
  projection_uniforms->z_near = this->camera.near;
  projection_uniforms->z_far  = this->camera.near;
  wgpu_queue_write_buffer(wgpu_context, this->ubos.projection.buffer, 0,
                          &projection_uniforms, sizeof(projection_uniforms_t));

  /* View UBO */
  this->ubos.view = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(view_uniforms_t),
                  });
  view_uniforms_t* view_uniforms = &this->ubos_data.view;
  glm_mat4_copy(this->camera.view_matrix, view_uniforms->matrix);
  glm_mat4_copy(this->camera.view_inv_matrix, view_uniforms->inverse_matrix);
  glm_vec3_copy(this->camera.position, view_uniforms->position);
  view_uniforms->time       = 0.0f;
  view_uniforms->delta_time = 0.0f;
  wgpu_queue_write_buffer(wgpu_context, this->ubos.view.buffer, 0,
                          &view_uniforms, sizeof(view_uniforms_t));

  /* Splot light properties */
  spot_light_set_position(this, ispot_light->position);
  spot_light_set_direction(this, ispot_light->direction);
  spot_light_set_color(this, ispot_light->color);
  spot_light_set_cut_off(this, ispot_light->cut_off);
  spot_light_set_outer_cut_off(this, ispot_light->outer_cut_off);
  spot_light_set_intensity(this, ispot_light->intensity);

  /* Render pass descriptor */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view         = this->depth_texture.view,
      .depthLoadOp  = WGPULoadOp_Clear,
      .depthStoreOp = WGPUStoreOp_Store,
    };
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 0,
    .colorAttachments       = NULL,
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
    .occlusionQuerySet      = NULL,
  };

  /* Spot light ubos bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->ubos.light_info.size,
        },
        .sampler = {0},
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->ubos.projection.size,
        },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type             = WGPUBufferBindingType_Uniform,
          .minBindingSize   = this->ubos.view.size,
        },
        .sampler = {0},
      }
    };
    this->bind_group_layouts.ubos = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "spot light ubos bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layouts.ubos != NULL);
  }

  /* Spot light depth texture bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Texture view
          .binding    = 0,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .texture = (WGPUTextureBindingLayout) {
            .sampleType    = WGPUTextureSampleType_Depth,
            .viewDimension = WGPUTextureViewDimension_2D,
            .multisampled  = false,
          },
        .storageTexture = {0},
      },
    };
    this->bind_group_layouts.depth_texture = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = "spot light depth texture bind group layout",
        .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
        .entries    = bgl_entries,
      });
    ASSERT(this->bind_group_layouts.depth_texture != NULL);
  }

  /* Spot light ubos bind group */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->ubos.light_info.buffer,
        .size    = this->ubos.light_info.size,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->ubos.projection.buffer,
        .size    = this->ubos.projection.size,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->ubos.view.buffer,
        .size    = this->ubos.view.size,
      },
    };
    this->bind_groups.ubos = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "spot light ubos bind group",
                              .layout     = this->bind_group_layouts.ubos,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_groups.ubos != NULL);
  }

  /* Spot light depth texture bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->depth_texture.view,
        },
      };
    this->bind_groups.depth_texture = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "spot light depth texture bind group",
                              .layout = this->bind_group_layouts.depth_texture,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_groups.depth_texture != NULL);
  }
}

static void spot_light_destroy(spot_light_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.light_info.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.projection.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubos.view.buffer)
  WGPU_RELEASE_RESOURCE(Texture, this->depth_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->depth_texture.view)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.ubos)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layouts.depth_texture)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.ubos)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_groups.depth_texture)
}

/* -------------------------------------------------------------------------- *
 * Box Outline
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/box-outline.ts
 * -------------------------------------------------------------------------- */

#define BOX_OUTLINE_RADIUS 2.5f
#define BOX_OUTLINE_SIDE_COUNT 13u

typedef struct {
  webgpu_renderer_t* renderer;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
    wgpu_buffer_t instance_buffer;
  } buffers;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
} box_outline_t;

static void box_outline_init(box_outline_t* this)
{
  /* Box outline render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->renderer->bind_group_layouts.frame, // Group 0
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "box outline render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /*  Box outline render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_LineStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
        }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 4 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 8 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 12 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x4,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 16 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 4,
        .attributes     = &attributes[1],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "box outline vertex shader wgsl",
            .file  = "shaders/compute_metaballs/box_outline_vertex_shader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "box outline fragment shader wgsl",
          .file  = "shaders/compute_metaballs/box_outline_fragment_shader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline
      = wgpuDeviceCreateRenderPipeline(this->renderer->wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label = "box outline render pipeline",
                                         .layout       = this->pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void box_outline_init_defaults(box_outline_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void box_outline_create(box_outline_t* this, webgpu_renderer_t* renderer)
{
  box_outline_init_defaults(this);
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  const float vertices[2 * 3] = {
    -BOX_OUTLINE_RADIUS, 0.0f, 0.0f, //
    BOX_OUTLINE_RADIUS,  0.0f, 0.0f  //
  };

  const uint16_t indices[16] = {
    0,  1,  2,  3,  //
    4,  5,  6,  7,  //
    8,  9,  10, 11, //
    12, 13, 14, 15, //
  };

  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertices),
                    .initial.data = vertices,
                  });

  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  float instance_matrices[BOX_OUTLINE_SIDE_COUNT * 16] = {0};

  mat4 instance_matrix = GLM_MAT4_IDENTITY_INIT;

  /* Top rig */
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 0});
  memcpy(&instance_matrices[0 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[1 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[2 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[3 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* bottom rig */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 0});
  memcpy(&instance_matrices[4 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 1, 0});
  memcpy(&instance_matrices[5 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, -1, 0});
  memcpy(&instance_matrices[6 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){0, -BOX_OUTLINE_RADIUS, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  memcpy(&instance_matrices[7 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  /* Sides */
  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[9 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[10 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){-BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[11 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  glm_mat4_identity(instance_matrix);
  glm_translate(instance_matrix,
                (vec3){BOX_OUTLINE_RADIUS, 0, -BOX_OUTLINE_RADIUS});
  glm_rotate(instance_matrix, PI, (vec3){0, 1, 0});
  glm_rotate(instance_matrix, PI_2, (vec3){0, 0, 1});
  memcpy(&instance_matrices[12 * 16], &instance_matrix[0],
         sizeof(instance_matrix));

  this->buffers.instance_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "box outline instance matrices buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_matrices),
                    .initial.data = instance_matrices,
                  });

  /* Init render pipeline */
  box_outline_init(this);
}

static void box_outline_destroy(box_outline_t* this)
{
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_buffer.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
}

static void box_outline_render(box_outline_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 2, BOX_OUTLINE_SIDE_COUNT, 0, 0,
                                   0);
}

/* -------------------------------------------------------------------------- *
 * Ground
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/ground.ts
 * -------------------------------------------------------------------------- */

#define GROUND_WORLD_Y -7.5f
#define GROUND_WIDTH 100u
#define GROUND_HEIGHT 100u
#define GROUND_COUNT 100u
#define GROUND_SPACING 0

typedef struct {
  webgpu_renderer_t* renderer;
  spot_light_t* spot_light;

  struct {
    WGPUPipelineLayout render_pipeline;
    WGPUPipelineLayout render_shadow_pipeline;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline render_pipeline;
    WGPURenderPipeline render_shadow_pipeline;
  } render_pipelines;

  WGPUBindGroupLayout model_bind_group_layout;
  WGPUBindGroup model_bind_group;

  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t normal_buffer;
    wgpu_buffer_t instance_offsets_buffer;
    wgpu_buffer_t instance_material_buffer;
    wgpu_buffer_t uniform_buffer;
  } buffers;

  uint32_t instance_count;
  mat4 model_matrix;
} ground_t;

static void ground_init(ground_t* this)
{
  /* Ground render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_group_layouts.frame, // Group 0
      this->model_bind_group_layout,            // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "ground render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_pipeline = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);
  }

  /* Ground render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[5] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [2] = (WGPUVertexAttribute){
        .shaderLocation = 2,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [3] = (WGPUVertexAttribute){
        .shaderLocation = 3,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32,
      },
      [4] = (WGPUVertexAttribute){
        .shaderLocation = 4,
        .offset         = 1 * sizeof(float),
        .format         = WGPUVertexFormat_Float32,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[4] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
      [2] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &attributes[2],
      },
      [3] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 2,
        .attributes     = &attributes[3],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "ground vertex shader wgsl",
            .file  = "shaders/compute_metaballs/ground_vertex_shader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "ground fragment shader wgsl",
          .file  = "shaders/compute_metaballs/ground_fragment_shader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = "ground render pipeline",
        .layout       = this->pipeline_layouts.render_pipeline,
        .primitive    = primitive_state,
        .vertex       = vertex_state,
        .fragment     = &fragment_state,
        .depthStencil = &depth_stencil_state,
        .multisample  = multisample_state,
      });
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Ground shadow render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->spot_light->bind_group_layouts.ubos, // Group 0
      this->model_bind_group_layout,             // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "ground shadow rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_shadow_pipeline
      = wgpuDeviceCreatePipelineLayout(this->renderer->wgpu_context->device,
                                       &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);
  }

  /* Ground shadow render pipeline layout */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[2] = {
      [0] = (WGPUVertexAttribute){
        .shaderLocation = 0,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute){
        .shaderLocation = 1,
        .offset         = 0 * sizeof(float),
        .format         = WGPUVertexFormat_Float32x3,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Instance,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "ground shadow vertex shader wgsl",
            .file  = "shaders/compute_metaballs/ground_shadow_vertex_shader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_shadow_pipeline
      = wgpuDeviceCreateRenderPipeline(
        this->renderer->wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label        = "ground render pipeline",
          .layout       = this->pipeline_layouts.render_shadow_pipeline,
          .primitive    = primitive_state,
          .vertex       = vertex_state,
          .fragment     = NULL,
          .depthStencil = &depth_stencil_state,
          .multisample  = multisample_state,
        });
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  }
}

static void ground_init_defaults(ground_t* this)
{
  memset(this, 0, sizeof(*this));
  glm_mat4_identity(this->model_matrix);
}

static void ground_create(ground_t* this, webgpu_renderer_t* renderer,
                          spot_light_t* spot_light)
{
  ground_init_defaults(this);

  this->renderer   = renderer;
  this->spot_light = spot_light;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Create cube */
  cube_geometry_t cube_geometry;
  vec3 cube_dimensions = GLM_VEC3_ONE_INIT;
  cube_geometry_create_cube(&cube_geometry, cube_dimensions);

  /* Ground vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = cube_geometry.positions.data_size,
                    .initial.data = cube_geometry.positions.data,
                  });

  /* Ground normal buffer */
  this->buffers.normal_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground normal buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = cube_geometry.normals.data_size,
                    .initial.data = cube_geometry.normals.data,
                  });

  /* Ground instance buffers */
  float instance_offsets[GROUND_WIDTH * GROUND_HEIGHT * 3]           = {0};
  float instance_metallic_rougness[GROUND_WIDTH * GROUND_HEIGHT * 2] = {0};

  const float spacing_x = GROUND_WIDTH / GROUND_COUNT + GROUND_SPACING;
  const float spacing_y = GROUND_HEIGHT / GROUND_COUNT + GROUND_SPACING;

  float x_pos = 0.0f, y_pos = 0.0f;
  for (uint32_t x = 0, i = 0; x < GROUND_COUNT; x++) {
    for (uint32_t y = 0; y < GROUND_COUNT; y++) {
      x_pos = x * spacing_x;
      y_pos = y * spacing_y;

      // xyz offset
      instance_offsets[i * 3 + 0] = x_pos - GROUND_WIDTH / 2.0f;
      instance_offsets[i * 3 + 1] = y_pos - GROUND_HEIGHT / 2.0f;
      instance_offsets[i * 3 + 2] = random_float() * 3 + 1;

      // metallic
      instance_metallic_rougness[i * 2 + 0] = 1.0f;

      // roughness
      instance_metallic_rougness[i * 2 + 1] = 0.5f;

      ++i;
    }
  }

  this->instance_count = ARRAY_SIZE(instance_offsets) / 3;

  this->buffers.instance_offsets_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground instance xyz buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_offsets),
                    .initial.data = instance_offsets,
                  });

  this->buffers.instance_material_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground instance material buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(instance_metallic_rougness),
                    .initial.data = instance_metallic_rougness,
                  });

  /* Ground uniform buffer */
  glm_translate(this->model_matrix, (vec3){0, GROUND_WORLD_Y, 0});
  this->buffers.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "ground uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(this->model_matrix),
                    .initial.data = this->model_matrix,
                  });

  /* Ground bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->buffers.uniform_buffer.size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "ground bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->model_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->model_bind_group_layout != NULL);
  }

  /* Ground bind group*/
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->buffers.uniform_buffer.buffer,
        .size    = this->buffers.uniform_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "ground bind group",
      .layout     = this->model_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->model_bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->model_bind_group != NULL);
  }

  /* Init render pipeline */
  ground_init(this);
}

static void ground_destroy(ground_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layouts.render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        this->pipeline_layouts.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipelines.render_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline,
                        this->render_pipelines.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->model_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->model_bind_group)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.normal_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_offsets_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.instance_material_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.uniform_buffer.buffer)
}

static ground_t* ground_render_shadow(ground_t* this,
                                      WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_shadow_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(
    render_pass, this->render_pipelines.render_shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->spot_light->bind_groups.ubos, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.instance_offsets_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

static ground_t* ground_render(ground_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipelines.render_pipeline) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass,
                                   this->render_pipelines.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->model_bind_group, 0,
                                    0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->buffers.normal_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 2, this->buffers.instance_offsets_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 3, this->buffers.instance_material_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(render_pass, 36, this->instance_count, 0, 0);
  return this;
}

/* --------------------------------------------------------------------------
 * Metaballs
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/meshesmetaballs.ts
 * --------------------------------------------------------------------------
 */

typedef struct {
  vec3 color_rgb;
  float roughness;
  float metallic;
} metaballs_material_t;

typedef struct {
  webgpu_renderer_t* renderer;
  ivolume_settings_t volume;
  spot_light_t* spot_light;

  metaballs_compute_t metaballs_compute;

  struct {
    WGPUPipelineLayout render_pipeline;
    WGPUPipelineLayout render_shadow_pipeline;
  } pipeline_layouts;

  struct {
    WGPURenderPipeline render_pipeline;
    WGPURenderPipeline render_shadow_pipeline;
  } render_pipelines;

  wgpu_buffer_t ubo;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  metaballs_material_t material, target_material;
} metaballs_t;

static bool metaballs_is_ready(metaballs_t* this)
{
  return (metaballs_compute_is_ready(&this->metaballs_compute) && //
          (this->render_pipelines.render_pipeline != NULL) &&     //
          (this->render_pipelines.render_shadow_pipeline != NULL) //
  );
}

static bool metaballs_has_updated_at_least_once(metaballs_t* this)
{
  return this->metaballs_compute.has_calced_once;
}

static void metaballs_init(metaballs_t* this)
{
  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_group_layouts.frame, // Group 0
      this->bind_group_layout,                  // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "metaball rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_pipeline = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attributes[2] = {
      [0] = (WGPUVertexAttribute) {
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
      [1] = (WGPUVertexAttribute) {
        .shaderLocation = 1,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
    };
    WGPUVertexBufferLayout vertex_buffers[2] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[0],
      },
      [1] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attributes[1],
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "metaballs vertex shader wgsl",
            .file  = "shaders/compute_metaballs/metaballs_vertex_shader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
          // Fragment shader WGSL
          .label = "metaballs fragment shader wgsl",
          .file  = "shaders/compute_metaballs/metaballs_fragment_shader.wgsl",
          .entry = "main",
        },
        .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
        .targets = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label        = "metaball rendering pipeline",
        .layout       = this->pipeline_layouts.render_pipeline,
        .primitive    = primitive_state,
        .vertex       = vertex_state,
        .fragment     = &fragment_state,
        .depthStencil = &depth_stencil_state,
        .multisample  = multisample_state,
      });
    ASSERT(this->pipeline_layouts.render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }

  /* Metaballs shadow render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[1] = {
      this->spot_light->bind_group_layouts.ubos, // Group 0
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "metaballs shadow rendering pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layouts.render_shadow_pipeline
      = wgpuDeviceCreatePipelineLayout(this->renderer->wgpu_context->device,
                                       &pipeline_layout_desc);
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);
  }

  /* Metaballs shadow render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexAttribute attribute = {
      .shaderLocation = 0,
      .offset         = 0,
      .format         = WGPUVertexFormat_Float32x3,
    };
    WGPUVertexBufferLayout vertex_buffers[1] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 3 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute,
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "metaballs shadow vertex shader wgsl",
            .file  = "shaders/compute_metaballs/metaballs_shadow_vertex_shader.wgsl",
            .entry = "main",
          },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipelines.render_shadow_pipeline
      = wgpuDeviceCreateRenderPipeline(
        this->renderer->wgpu_context->device,
        &(WGPURenderPipelineDescriptor){
          .label        = "metaballs shadow rendering pipeline",
          .layout       = this->pipeline_layouts.render_shadow_pipeline,
          .primitive    = primitive_state,
          .vertex       = vertex_state,
          .fragment     = NULL,
          .depthStencil = &depth_stencil_state,
          .multisample  = multisample_state,
        });
    ASSERT(this->pipeline_layouts.render_shadow_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  }
}

static void metaballs_init_defaults(metaballs_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_one(this->material.color_rgb);
  glm_vec3_one(this->target_material.color_rgb);

  this->material.roughness        = 0.3f;
  this->target_material.roughness = this->material.roughness;
  this->material.metallic         = 0.1;
  this->target_material.metallic  = this->material.metallic;
}

static void metaballs_create(metaballs_t* this, webgpu_renderer_t* renderer,
                             ivolume_settings_t* volume,
                             spot_light_t* spot_light)
{
  metaballs_init_defaults(this);

  this->renderer = renderer;
  memcpy(&this->volume, volume, sizeof(ivolume_settings_t));
  this->spot_light = spot_light;
  metaballs_compute_create(&this->metaballs_compute, renderer, volume);

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Metaballs ubo */
  {
    /* padding required for struct alignment: size % 8 == 0 */
    const float metaballs_ubo_data[5] = {1.0f, 1.0f, 1.0f, 0.3f, 0.1f};
    this->ubo
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "metaballs ubo",
                                           .usage = WGPUBufferUsage_Uniform
                                                    | WGPUBufferUsage_CopyDst,
                                           .size         = 8 * sizeof(float),
                                           .initial.data = metaballs_ubo_data,
                                         });
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->ubo.size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "metaballs bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Bind group*/
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = this->ubo.buffer,
        .size    = this->ubo.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "metaballs bind group",
      .layout     = this->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
  }

  /* Init render pipeline */
  metaballs_init(this);
}

static void metaballs_destroy(metaballs_t* this)
{
  metaballs_compute_destroy(&this->metaballs_compute);
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layouts.render_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        this->pipeline_layouts.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipelines.render_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline,
                        this->render_pipelines.render_shadow_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->ubo.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void metaballs_rearrange(metaballs_t* this)
{
  this->target_material.color_rgb[0] = random_float();
  this->target_material.color_rgb[1] = random_float();
  this->target_material.color_rgb[2] = random_float();

  this->target_material.metallic  = 0.08f + random_float() * 0.92f;
  this->target_material.roughness = 0.08f + random_float() * 0.92f;

  metaballs_compute_rearrange(&this->metaballs_compute);
}

static metaballs_t* metaballs_update_sim(metaballs_t* this,
                                         WGPUComputePassEncoder compute_pass,
                                         float time, float time_delta)
{
  const float color_speed = time_delta * 2.0f;
  this->material.color_rgb[0]
    += (this->target_material.color_rgb[0] - this->material.color_rgb[0])
       * color_speed;
  this->material.color_rgb[1]
    += (this->target_material.color_rgb[1] - this->material.color_rgb[1])
       * color_speed;
  this->material.color_rgb[2]
    += (this->target_material.color_rgb[2] - this->material.color_rgb[2])
       * color_speed;

  const float material_speed = time_delta * 3;
  this->material.metallic
    += (this->target_material.metallic - this->material.metallic)
       * material_speed;
  this->material.roughness
    += (this->target_material.roughness - this->material.roughness)
       * material_speed;

  wgpu_queue_write_buffer(this->renderer->wgpu_context, this->ubo.buffer, 0,
                          &this->material, sizeof(metaballs_material_t));

  metaballs_compute_update_sim(&this->metaballs_compute, compute_pass, time,
                               time_delta);
  return this;
}

static metaballs_t* metaballs_render_shadow(metaballs_t* this,
                                            WGPURenderPassEncoder render_pass)
{
  if (!metaballs_is_ready(this)) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(
    render_pass, this->render_pipelines.render_shadow_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->spot_light->bind_groups.ubos, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->metaballs_compute.vertex_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->metaballs_compute.index_buffer.buffer,
    WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, this->metaballs_compute.index_count, 1, 0, 0, 0);
  return this;
}

static metaballs_t* metaballs_render(metaballs_t* this,
                                     WGPURenderPassEncoder render_pass)
{
  if (!metaballs_is_ready(this)) {
    return this;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass,
                                   this->render_pipelines.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->metaballs_compute.vertex_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 1, this->metaballs_compute.normal_buffer.buffer, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->metaballs_compute.index_buffer.buffer,
    WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, this->metaballs_compute.index_count, 1, 0, 0, 0);
  return this;
}

/* -------------------------------------------------------------------------- *
 * Particles
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/meshes/particles.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
} particles_t;

static void particles_init(particles_t* this)
{
  /* Render pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->renderer->bind_group_layouts.frame, // Group 0
      this->bind_group_layout,                  // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "particles render pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_states[2] = {
      [0] = (WGPUColorTargetState){
        // normal + material id
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
      [1] = (WGPUColorTargetState){
        // albedo
        .format    = WGPUTextureFormat_BGRA8Unorm,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      }
    };

    // Depth stencil state
    WGPUDepthStencilState depth_stencil_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "particles vertex shader wgsl",
            .file  = "shaders/compute_metaballs/particles_vertex_shader.wgsl",
            .entry = "main",
           },
          .buffer_count = 0,
          .buffers = NULL,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Fragment shader WGSL
            .label = "particles fragment shader wgsl",
            .file  = "shaders/compute_metaballs/particles_fragment_shader.wgsl",
            .entry = "main",
          },
          .target_count = (uint32_t)ARRAY_SIZE(color_target_states),
          .targets      = color_target_states,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline
      = wgpuDeviceCreateRenderPipeline(this->renderer->wgpu_context->device,
                                       &(WGPURenderPipelineDescriptor){
                                         .label  = "particles render pipeline",
                                         .layout = this->pipeline_layout,
                                         .primitive    = primitive_state,
                                         .vertex       = vertex_state,
                                         .fragment     = &fragment_state,
                                         .depthStencil = &depth_stencil_state,
                                         .multisample  = multisample_state,
                                       });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void particles_init_defaults(particles_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void particles_create(particles_t* this, webgpu_renderer_t* renderer,
                             wgpu_buffer_t* lights_buffer)
{
  particles_init_defaults(this);

  this->renderer               = renderer;
  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Particles bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = lights_buffer->size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "particles bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Particles bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = lights_buffer->buffer,
        .size    = lights_buffer->size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "particles bind group",
      .layout     = this->bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->bind_group
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->bind_group != NULL);
  }

  /* Init render pipeline */
  particles_init(this);
}

static void particles_destroy(particles_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
}

static void particles_render(particles_t* this,
                             WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 0,
                                    this->renderer->bind_groups.frame, 0, 0);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1, this->bind_group, 0, 0);
  wgpuRenderPassEncoderDrawIndexed(
    render_pass, 6, settings_get_quality_level().point_lights_count, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Effect
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/effect.ts
 * -------------------------------------------------------------------------- */

#define EFFECT_MAX_BIND_GROUP_COUNT 5u

typedef struct {
  webgpu_renderer_t* renderer;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  struct {
    WGPUBindGroup items[EFFECT_MAX_BIND_GROUP_COUNT];
    uint32_t item_count;
  } bind_groups;
  WGPUTextureFormat presentation_format;
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
  } buffers;
} effect_t;

static void effect_init(effect_t* this, const char* fragment_shader_file,
                        WGPUBindGroupLayout* bind_group_layouts,
                        uint32_t bind_group_layout_count, const char* label)
{
  /* Render pipeline layout */
  {
    char pipeline_layout_lbl[STRMAX] = {0};
    snprintf(pipeline_layout_lbl, strlen(label) + 7 + 1, "%s layout", label);
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = pipeline_layout_lbl,
      .bindGroupLayoutCount = bind_group_layout_count,
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->pipeline_layout != NULL);
  }

  /* Render pipeline */
  {
    // Primitive state
    WGPUPrimitiveState primitive_state = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint16,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_None,
    };

    // Color target state
    WGPUBlendState blend_state              = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target_state = (WGPUColorTargetState){
      .format    = this->presentation_format,
      .blend     = &blend_state,
      .writeMask = WGPUColorWriteMask_All,
    };

    // Vertex state
    WGPUVertexAttribute attribute = {
      .shaderLocation = 0,
      .offset         = 0 * sizeof(float),
      .format         = WGPUVertexFormat_Float32x2,
    };
    WGPUVertexBufferLayout vertex_buffers[1] = {
      [0] = (WGPUVertexBufferLayout){
        .arrayStride    = 2 * sizeof(float),
        .stepMode       = WGPUVertexStepMode_Vertex,
        .attributeCount = 1,
        .attributes     = &attribute,
      },
    };

    // Vertex state
    WGPUVertexState vertex_state = wgpu_create_vertex_state(
        this->renderer->wgpu_context, &(wgpu_vertex_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Vertex shader WGSL
            .label = "effect vertex shader wgsl",
            .file  = "shaders/compute_metaballs/effect_vertex_shader.wgsl",
            .entry = "main",
           },
          .buffer_count = (uint32_t)ARRAY_SIZE(vertex_buffers),
          .buffers      = vertex_buffers,
        });

    // Fragment state
    WGPUFragmentState fragment_state = wgpu_create_fragment_state(
        this->renderer->wgpu_context, &(wgpu_fragment_state_t){
        .shader_desc = (wgpu_shader_desc_t){
            // Fragment shader WGSL
            .label = "effect fragment shader wgsl",
            .file  = fragment_shader_file,
            .entry = "main",
          },
          .target_count = 1,
          .targets      = &color_target_state,
        });

    // Multisample state
    WGPUMultisampleState multisample_state
      = wgpu_create_multisample_state_descriptor(
        &(create_multisample_state_desc_t){
          .sample_count = 1,
        });

    // Create rendering pipeline using the specified states
    this->render_pipeline = wgpuDeviceCreateRenderPipeline(
      this->renderer->wgpu_context->device, &(WGPURenderPipelineDescriptor){
                                              .label  = label,
                                              .layout = this->pipeline_layout,
                                              .primitive   = primitive_state,
                                              .vertex      = vertex_state,
                                              .fragment    = &fragment_state,
                                              .multisample = multisample_state,
                                            });
    ASSERT(this->render_pipeline != NULL);

    // Partial cleanup
    WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
    WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
  }
}

static void effect_set_bind_groups(effect_t* this,
                                   iscreen_effect_t* screen_effect)
{
  const uint32_t max_bind_groups = (uint32_t)EFFECT_MAX_BIND_GROUP_COUNT;
  for (uint32_t i = 0;
       i < screen_effect->bind_groups.item_count && i < max_bind_groups; i++) {
    this->bind_groups.items[i] = screen_effect->bind_groups.items[i];
  }
  this->bind_groups.item_count
    = MIN(screen_effect->bind_groups.item_count, max_bind_groups);
}

static void effect_create(effect_t* this, webgpu_renderer_t* renderer,
                          iscreen_effect_t* screen_effect)
{
  this->renderer = renderer;
  effect_set_bind_groups(this, screen_effect);
  this->presentation_format = screen_effect->presentation_format;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Vertex data & indices */
  const float vertex_data[2 * 4] = {
    -1.0f, 1.0f,  //
    -1.0f, -1.0f, //
    1.0f,  -1.0f, //
    1.0f,  1.0f,  //
  };

  const uint16_t indices[3 * 2] = {
    3, 2, 1, //
    3, 1, 0, //
  };

  /* Effect vertex buffer */
  this->buffers.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "fullscreen effect vertex buffer",
                    .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(vertex_data),
                    .initial.data = vertex_data,
                  });

  /* Effect index buffer */
  this->buffers.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "fullscreen effect index buffer",
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(indices),
                    .initial.data = indices,
                  });

  /* Init render pipeline */
  effect_init(this, screen_effect->fragment_shader_file,
              screen_effect->bind_group_layouts.items,
              screen_effect->bind_group_layouts.item_count,
              screen_effect->label);
}

static void effect_destroy(effect_t* this)
{
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, this->render_pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffers.index_buffer.buffer)
}

static void effect_pre_render(effect_t* this, WGPURenderPassEncoder render_pass)
{
  if (!this->render_pipeline) {
    return;
  }

  wgpuRenderPassEncoderSetPipeline(render_pass, this->render_pipeline);
  for (uint32_t i = 0; i < this->bind_groups.item_count; ++i) {
    wgpuRenderPassEncoderSetBindGroup(render_pass, i,
                                      this->bind_groups.items[i], 0, 0);
  }
  wgpuRenderPassEncoderSetVertexBuffer(
    render_pass, 0, this->buffers.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    render_pass, this->buffers.index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
}

/* -------------------------------------------------------------------------- *
 * Copy Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/copy-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } copy_texture;
} copy_pass_t;

static bool copy_pass_is_ready(copy_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void copy_pass_init_defaults(copy_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void copy_pass_create(copy_pass_t* this, webgpu_renderer_t* renderer)
{
  copy_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Copy texture */
  WGPUExtent3D texture_extent = {
    .width              = renderer->output_size[0],
    .height             = renderer->output_size[1],
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "copy pass texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->copy_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->copy_texture.texture != NULL);

  /* Copy texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "copy pass texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->copy_texture.view
    = wgpuTextureCreateView(this->copy_texture.texture, &texture_view_dec);
  ASSERT(this->copy_texture.view != NULL);

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
    // Binding 0: Texture view
    .binding    = 0,
    .visibility = WGPUShaderStage_Fragment,
    .texture = (WGPUTextureBindingLayout) {
      .sampleType    = WGPUTextureSampleType_Float,
      .viewDimension = WGPUTextureViewDimension_2D,
      .multisampled  = false,
    },
    .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "copy pass bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = this->copy_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "copy pass bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->bind_group_layouts.frame,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->bind_groups.frame,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_file
      = "shaders/compute_metaballs/copy_pass_fragment_shader.wgsl",
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = WGPUTextureFormat_RGBA16Float,
      .label                         = "copy pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Frame buffer Color attachments */
  this->framebuffer.color_attachments[0] =
    (WGPURenderPassColorAttachment) {
      .view       = this->copy_texture.view,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };
}

static void copy_pass_destroy(copy_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->copy_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->copy_texture.view)
}

static void copy_pass_render(copy_pass_t* this,
                             WGPURenderPassEncoder render_pass)
{
  if (!copy_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Bloom Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/bloom-pass.ts
 * -------------------------------------------------------------------------- */

#define BLOOM_PASS_TILE_DIM 128u
#define BLOOM_PASS_BATCH {4, 4}
#define BLOOM_PASS_FILTER_SIZE 10u
#define BLOOM_PASS_ITERATIONS 2u

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t* point_lights;
  spot_light_t* spot_light;
  WGPURenderPassDescriptor framebuffer_descriptor;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } bloom_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } input_texture;
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } blur_textures[2];

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  WGPUPipelineLayout blur_pipeline_layout;
  WGPUComputePipeline blur_pipeline;

  WGPUBindGroupLayout blur_constants_bind_group_layout;
  WGPUBindGroup blur_compute_constants_bindGroup;

  WGPUBindGroupLayout blur_compute_bind_group_layout;
  WGPUBindGroup blur_compute_bind_groups[3];

  wgpu_buffer_t blur_params_buffer;
  wgpu_buffer_t buffer_0;
  wgpu_buffer_t buffer_1;

  struct {
    WGPURenderPassColorAttachment color_attachments[1];
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUSampler sampler;
  uint32_t block_dim;
} bloom_pass_t;

static bool bloom_pass_is_ready(bloom_pass_t* this)
{
  return (this->effect.render_pipeline != NULL)
         && (this->blur_pipeline != NULL);
}

static void bloom_pass_init_compute_pipeline(bloom_pass_t* this)
{
  wgpu_context_t* wgpu_context = this->renderer->wgpu_context;

  /* Bloom pass blur pipeline layout */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->blur_constants_bind_group_layout, // Group 0
      this->blur_compute_bind_group_layout,   // Group 1
    };
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = "bloom pass blur pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    };
    this->blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      this->renderer->wgpu_context->device, &pipeline_layout_desc);
    ASSERT(this->blur_pipeline_layout != NULL);
  }

  /* Bloom pass blur pipeline */
  {
    wgpu_shader_t comp_shader = wgpu_shader_create(
      wgpu_context,
      &(wgpu_shader_desc_t){
        // Compute shader WGSL
        .file  = "shaders/compute_metaballs/bloom_blur_compute_shader.wgsl",
        .entry = "main",
      });
    this->blur_pipeline = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = "bloom pass blur pipeline",
        .layout  = this->blur_pipeline_layout,
        .compute = comp_shader.programmable_stage_descriptor,
      });
    wgpu_shader_release(&comp_shader);
  }

  /* Horizontal flip */
  const uint32_t horizontal_flip_data[1] = {
    0 //
  };
  this->buffer_0 = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Horizontal flip buffer",
                    .usage        = WGPUBufferUsage_Uniform,
                    .size         = sizeof(horizontal_flip_data),
                    .initial.data = &horizontal_flip_data[0],
                  });

  /* Vertical flip */
  const uint32_t vertical_flip_data[1] = {
    1 //
  };
  this->buffer_1
    = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                         .label = "Vertical flip buffer",
                                         .usage = WGPUBufferUsage_Uniform,
                                         .size  = sizeof(vertical_flip_data),
                                         .initial.data = &vertical_flip_data[0],
                                       });

  /* Blur compute bind group 0 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->bloom_texture.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_0.buffer,
        .size    = this->buffer_0.size,
      },
    };
    this->blur_compute_bind_groups[0] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "blur compute bind group 0",
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[0] != NULL);
  }

  /* Blur compute bind group 1 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->blur_textures[0].view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[1].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_1.buffer,
        .size    = this->buffer_1.size,
      },
    };
    this->blur_compute_bind_groups[1] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "blur compute bind group 1",
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[1] != NULL);
  }

  /* Blur compute bind group 2 */
  {
    WGPUBindGroupEntry bg_entries[3] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = this->blur_textures[1].view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding     = 1,
        .textureView = this->blur_textures[0].view,
      },
      [2] = (WGPUBindGroupEntry) {
        .binding = 2,
        .buffer  = this->buffer_0.buffer,
        .size    = this->buffer_0.size,
      },
    };
    this->blur_compute_bind_groups[2] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = "blur compute bind group 2",
                              .layout = this->blur_compute_bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->blur_compute_bind_groups[2] != NULL);
  }
}

static void bloom_pass_init_defaults(bloom_pass_t* this)
{
  memset(this, 0, sizeof(*this));
  this->block_dim = 0;
}

static void bloom_pass_create(bloom_pass_t* this, webgpu_renderer_t* renderer,
                              copy_pass_t* copy_pass)
{
  bloom_pass_init_defaults(this);
  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Bloom texture */
  WGPUExtent3D texture_extent = {
    .width              = renderer->output_size[0],
    .height             = renderer->output_size[1],
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "bloom texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_RGBA16Float,
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
  };
  this->bloom_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->bloom_texture.texture != NULL);

  /* Bloom texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "bloom texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->bloom_texture.view
    = wgpuTextureCreateView(this->bloom_texture.texture, &texture_view_dec);
  ASSERT(this->bloom_texture.view != NULL);

  /* Bloom pass bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        // Binding 0: Texture view
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = (WGPUTextureBindingLayout) {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      .storageTexture = {0},
      },
    };
    this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "bloom pass bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(this->bind_group_layout != NULL);
  }

  /* Blur texture and blur texture views */
  for (uint8_t i = 0; i < 2; ++i) {
    /* Blur texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->output_size[0],
      .height             = renderer->output_size[1],
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "blur texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_StorageBinding
               | WGPUTextureUsage_TextureBinding,
    };
    this->blur_textures[i].texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->blur_textures[i].texture != NULL);

    /* Blur texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "blur texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->blur_textures[i].view = wgpuTextureCreateView(
      this->blur_textures[i].texture, &texture_view_dec);
    ASSERT(this->blur_textures[i].view != NULL);
  }

  /* G-buffer bind group */
  {
    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = copy_pass->copy_texture.view,
      },
    };
    this->bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = "gbuffer bind group",
                              .layout     = this->bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(this->bind_group != NULL);
  }

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->bind_group_layouts.frame,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->bind_groups.frame,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_file
      = "shaders/compute_metaballs/bloom_pass_fragment_shader.wgsl",
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = WGPUTextureFormat_RGBA16Float,
      .label                         = "bloom pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Input texture */
  this->input_texture.texture = copy_pass->copy_texture.texture;
  this->input_texture.view    = copy_pass->copy_texture.view;

  /* Frame buffer descriptor */
  this->framebuffer.color_attachments[0] =
    (WGPURenderPassColorAttachment) {
      .view       = this->bloom_texture.view,
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = NULL,
  };

  /* Blur params buffer */
  this->block_dim = BLOOM_PASS_TILE_DIM - (BLOOM_PASS_FILTER_SIZE - 1);
  const uint32_t blur_params[2] = {BLOOM_PASS_FILTER_SIZE, this->block_dim};
  this->blur_params_buffer      = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                         .label = "blur params buffer",
                         .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                         .size  = sizeof(blur_params),
                         .initial.data = &blur_params[0],
                  });

  /* Bloom sampler */
  this->sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = "bloom sampler",
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(this->sampler != NULL);

  /* Blur constants bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[2] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .sampler = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = this->blur_params_buffer.size,
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "blur constants bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->blur_constants_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->blur_constants_bind_group_layout != NULL);
  }

  /* Blur constants bind group */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .sampler = this->sampler,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .buffer  = this->blur_params_buffer.buffer,
        .size    = this->blur_params_buffer.size,
      },
    };
    WGPUBindGroupDescriptor bg_desc = {
      .label      = "blur constants bind group",
      .layout     = this->blur_constants_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
      .entries    = bg_entries,
    };
    this->blur_compute_constants_bindGroup
      = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
    ASSERT(this->blur_compute_constants_bindGroup != NULL);
  }

  /* Blur compute bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Compute,
        .texture = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = (WGPUBindGroupLayoutEntry) {
        .binding    = 1,
        .visibility = WGPUShaderStage_Compute,
        .storageTexture = (WGPUStorageTextureBindingLayout) {
           .access        = WGPUStorageTextureAccess_WriteOnly,
           .format        = WGPUTextureFormat_RGBA8Unorm,
           .viewDimension = WGPUTextureViewDimension_2D,
         },
        .sampler = {0},
      },
      [2] = (WGPUBindGroupLayoutEntry) {
        .binding    = 2,
        .visibility = WGPUShaderStage_Compute,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(float),
        },
        .sampler = {0},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = "blur compute bind group layout",
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    this->blur_compute_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(this->blur_compute_bind_group_layout != NULL);
  }

  /* Init compute pipeline */
  bloom_pass_init_compute_pipeline(this);
}

static void bloom_pass_destroy(bloom_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(Texture, this->bloom_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->bloom_texture.view)
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(this->blur_textures); ++i) {
    WGPU_RELEASE_RESOURCE(Texture, this->blur_textures[i].texture)
    WGPU_RELEASE_RESOURCE(TextureView, this->blur_textures[i].view)
  }
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, this->blur_pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, this->blur_pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->blur_constants_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_constants_bindGroup)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->blur_compute_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(BindGroup, this->blur_compute_bind_groups[2])
  WGPU_RELEASE_RESOURCE(Buffer, this->blur_params_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer_0.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, this->buffer_1.buffer)
  WGPU_RELEASE_RESOURCE(Sampler, this->sampler)
}

static void bloom_pass_update_bloom(bloom_pass_t* this,
                                    WGPUComputePassEncoder compute_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  const webgpu_renderer_t* renderer = this->renderer;
  const uint32_t block_dim          = this->block_dim;
  const uint32_t batch[2]           = BLOOM_PASS_BATCH;
  const uint32_t src_width          = renderer->output_size[0];
  const uint32_t src_height         = renderer->output_size[1];

  wgpuComputePassEncoderSetPipeline(compute_pass, this->blur_pipeline);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 0, this->blur_compute_constants_bindGroup, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->blur_compute_bind_groups[0], 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_width / (float)block_dim), // workgroupCountX
    (uint32_t)ceil(src_height / (float)batch[1]), // workgroupCountY
    1                                             // workgroupCountZ
  );
  wgpuComputePassEncoderSetBindGroup(
    compute_pass, 1, this->blur_compute_bind_groups[1], 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass,
    (uint32_t)ceil(src_height / (float)block_dim), // workgroupCountX
    (uint32_t)ceil(src_width / (float)batch[1]),   // workgroupCountY
    1                                              // workgroupCountZ
  );
  for (uint32_t i = 0; i < BLOOM_PASS_ITERATIONS - 1; ++i) {
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_groups[2], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_width / (float)block_dim), // workgroupCountX
      (uint32_t)ceil(src_height / (float)batch[1]), // workgroupCountY
      1                                             // workgroupCountZ
    );
    wgpuComputePassEncoderSetBindGroup(
      compute_pass, 1, this->blur_compute_bind_groups[1], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(
      compute_pass,
      (uint32_t)ceil(src_height / (float)block_dim), // workgroupCountX
      (uint32_t)ceil(src_width / (float)batch[1]),   // workgroupCountY
      1                                              // workgroupCountZ
    );
  }
}

static void bloom_pass_render(bloom_pass_t* this,
                              WGPURenderPassEncoder render_pass)
{
  if (!bloom_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Deffered Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/deferred-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  point_lights_t point_lights;
  spot_light_t spot_light;

  struct {
    WGPURenderPassColorAttachment color_attachments[2];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } framebuffer;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_normal;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } g_buffer_texture_diffuse;

  vec3 spot_light_target;
  vec3 spot_light_color_target;
} deferred_pass_t;

static bool deferred_pass_is_ready(deferred_pass_t* this)
{
  return point_lights_is_ready(&this->point_lights)
         && (this->effect.render_pipeline != NULL);
}

static void deferred_pass_init_defaults(deferred_pass_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec3_copy((vec3){0.0f, 80.0f, 0.0f}, this->spot_light_target);
  glm_vec3_one(this->spot_light_color_target);
}

static void deferred_pass_create(deferred_pass_t* this,
                                 webgpu_renderer_t* renderer)
{
  deferred_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Point light */
  point_lights_create(&this->point_lights, renderer);

  /* Spot light */
  ispot_light_t ispot_light = {
    .position      = {0.0f, 80.0f, 1.0f},
    .direction     = {0.0f, 1.0f, 0.0f},
    .color         = GLM_VEC3_ONE_INIT,
    .cut_off       = deg_to_rad(1.0f),
    .outer_cut_off = deg_to_rad(4.0f),
    .intensity     = 40.0f,
  };
  spot_light_create(&this->spot_light, renderer, &ispot_light);

  /* G-Buffer normal texture */
  {
    /* G-Buffer normal texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->output_size[0],
      .height             = renderer->output_size[1],
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "gbuffer normal texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_normal.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_normal.texture != NULL);

    /* G-Buffer normal texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "gbuffer normal texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_normal.view = wgpuTextureCreateView(
      this->g_buffer_texture_normal.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_normal.view != NULL);
  }

  /* G-Buffer diffuse texture */
  {
    /* G-Buffer diffuse texture */
    WGPUExtent3D texture_extent = {
      .width              = renderer->output_size[0],
      .height             = renderer->output_size[1],
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .label         = "gbuffer diffuse texture",
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_BGRA8Unorm,
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    };
    this->g_buffer_texture_diffuse.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(this->g_buffer_texture_diffuse.texture != NULL);

    /* G-Buffer diffuse texture view */
    WGPUTextureViewDescriptor texture_view_dec = {
      .label           = "gbuffer diffuse texture view",
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    };
    this->g_buffer_texture_diffuse.view = wgpuTextureCreateView(
      this->g_buffer_texture_diffuse.texture, &texture_view_dec);
    ASSERT(this->g_buffer_texture_diffuse.view != NULL);
  }

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[5] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = this->point_lights.lights_buffer.size,
       },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout) {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = this->point_lights.lights_config_uniform_buffer.size,
       },
      .sampler = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
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
        .sampleType    = WGPUTextureSampleType_Depth,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "gbuffer bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[5] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = this->point_lights.lights_buffer.buffer,
      .size    = this->point_lights.lights_buffer.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .buffer  = this->point_lights.lights_config_uniform_buffer.buffer,
      .size    = this->point_lights.lights_config_uniform_buffer.size,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = this->g_buffer_texture_normal.view,
    },
    [3] = (WGPUBindGroupEntry) {
      .binding     = 3,
      .textureView = this->g_buffer_texture_diffuse.view,
    },
    [4] = (WGPUBindGroupEntry) {
      .binding     = 4,
      .textureView = renderer->textures.depth_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "gbuffer bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[4] = {
      this->bind_group_layout,
      this->renderer->bind_group_layouts.frame,
      this->spot_light.bind_group_layouts.ubos,
      this->spot_light.bind_group_layouts.depth_texture,
    };
    WGPUBindGroup bind_groups[4] = {
      this->bind_group,
      this->renderer->bind_groups.frame,
      this->spot_light.bind_groups.ubos,
      this->spot_light.bind_groups.depth_texture,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_file
      = "shaders/compute_metaballs/deferred_pass_fragment_shader.wgsl",
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format = settings_get_quality_level().bloom_toggle ?
                               WGPUTextureFormat_RGBA16Float :
                               WGPUTextureFormat_BGRA8Unorm,
      .label               = "deferred pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }

  /* Frame buffer Color attachments */
  {
    this->framebuffer.color_attachments[0] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_normal.view,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };
  }
  {
    this->framebuffer.color_attachments[1] =
      (WGPURenderPassColorAttachment) {
        .view       = this->g_buffer_texture_diffuse.view,
        .depthSlice = ~0,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = (WGPUColor) {
          .r = 0.0f,
          .g = 0.0f,
          .b = 0.0f,
          .a = 1.0f,
        },
      };
  }

  /* Frame buffer depth stencil attachment */
  this->framebuffer.depth_stencil_attachment
    = (WGPURenderPassDepthStencilAttachment){
      .view            = renderer->textures.depth_texture.view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };

  /* Frame buffer descriptor */
  this->framebuffer.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 2,
    .colorAttachments       = &this->framebuffer.color_attachments[0],
    .depthStencilAttachment = &this->framebuffer.depth_stencil_attachment,
  };
}

static void deferred_pass_destroy(deferred_pass_t* this)
{
  effect_destroy(&this->effect);
  point_lights_destroy(&this->point_lights);
  spot_light_destroy(&this->spot_light);

  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_normal.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_normal.view)
  WGPU_RELEASE_RESOURCE(Texture, this->g_buffer_texture_diffuse.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->g_buffer_texture_diffuse.view)
}

static void deferred_pass_rearrange(deferred_pass_t* this)
{
  this->spot_light_target[0]       = (random_float() * 2 - 1) * 3;
  this->spot_light_target[2]       = (random_float() * 2 - 1) * 3;
  this->spot_light_color_target[0] = random_float();
  this->spot_light_color_target[1] = random_float();
  this->spot_light_color_target[2] = random_float();
}

static void deferred_pass_update_lights_sim(deferred_pass_t* this,
                                            WGPUComputePassEncoder compute_pass,
                                            float _time, float time_delta)
{
  UNUSED_VAR(_time);

  point_lights_update_sim(&this->point_lights, compute_pass);
  const float speed = time_delta * 2.0f;
  glm_vec3_copy(
    (vec3){
      this->spot_light._position[0]
        + (this->spot_light_target[0] - this->spot_light._position[0])
            * speed, // x
      this->spot_light._position[1]
        + (this->spot_light_target[1] - this->spot_light._position[1])
            * speed, // y
      this->spot_light._position[2]
        + (this->spot_light_target[2] - this->spot_light._position[2])
            * speed, // z
    },
    this->spot_light._position);

  glm_vec3_copy(
    (vec3){
      (this->spot_light_color_target[0] - this->spot_light._color[0]) * speed
        * 4, // r
      (this->spot_light_color_target[1] - this->spot_light._color[1]) * speed
        * 4, // g
      (this->spot_light_color_target[2] - this->spot_light._color[2]) * speed
        * 4, // b
    },
    this->spot_light._color);
}

static void deferred_pass_render(deferred_pass_t* this,
                                 WGPURenderPassEncoder render_pass)
{
  if (!deferred_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Result Pass
 *
 * Ref:
 * https://github.com/gnikoloff/webgpu-compute-metaballs/blob/master/src/postfx/result-pass.ts
 * -------------------------------------------------------------------------- */

typedef struct {
  webgpu_renderer_t* renderer;
  effect_t effect;

  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } empty_texture;
} result_pass_t;

static bool result_pass_is_ready(result_pass_t* this)
{
  return this->effect.render_pipeline != NULL;
}

static void result_pass_init_defaults(result_pass_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void result_pass_create(result_pass_t* this, webgpu_renderer_t* renderer,
                               copy_pass_t* copy_pass, bloom_pass_t* bloom_pass)
{
  result_pass_init_defaults(this);

  this->renderer = renderer;

  wgpu_context_t* wgpu_context = renderer->wgpu_context;

  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
  };
  this->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = "result pass bind group layout",
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(this->bind_group_layout != NULL);

  /* G-Buffer normal texture */
  WGPUExtent3D texture_extent = {
    .width              = 1,
    .height             = 1,
    .depthOrArrayLayers = 1,
  };
  WGPUTextureDescriptor texture_desc = {
    .label         = "empty texture",
    .size          = texture_extent,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .dimension     = WGPUTextureDimension_2D,
    .format        = WGPUTextureFormat_BGRA8Unorm,
    .usage         = WGPUTextureUsage_TextureBinding,
  };
  this->empty_texture.texture
    = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
  ASSERT(this->empty_texture.texture != NULL);

  /* G-Buffer normal texture view */
  WGPUTextureViewDescriptor texture_view_dec = {
    .label           = "empty texture view",
    .dimension       = WGPUTextureViewDimension_2D,
    .format          = texture_desc.format,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  this->empty_texture.view
    = wgpuTextureCreateView(this->empty_texture.texture, &texture_view_dec);
  ASSERT(this->empty_texture.view != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry) {
      .binding     = 0,
      .textureView = copy_pass->copy_texture.view,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = bloom_pass ? bloom_pass->blur_textures[1].view
                                  : this->empty_texture.view,
    },
  };
  this->bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = "result pass bind group",
                            .layout     = this->bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(this->bind_group != NULL);

  /* Initialize effect */
  {
    WGPUBindGroupLayout bind_group_layouts[2] = {
      this->bind_group_layout,
      this->renderer->bind_group_layouts.frame,
    };
    WGPUBindGroup bind_groups[2] = {
      this->bind_group,
      this->renderer->bind_groups.frame,
    };
    iscreen_effect_t screen_effect = {
      .fragment_shader_file
      = "shaders/compute_metaballs/result_pass_fragment_shader.wgsl",
      .bind_group_layouts.items      = bind_group_layouts,
      .bind_group_layouts.item_count = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bind_groups.items             = bind_groups,
      .bind_groups.item_count        = (uint32_t)ARRAY_SIZE(bind_groups),
      .presentation_format           = renderer->presentation_format,
      .label                         = "result pass effect",
    };
    effect_create(&this->effect, renderer, &screen_effect);
  }
}

static void result_pass_destroy(result_pass_t* this)
{
  effect_destroy(&this->effect);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, this->bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, this->bind_group)
  WGPU_RELEASE_RESOURCE(Texture, this->empty_texture.texture)
  WGPU_RELEASE_RESOURCE(TextureView, this->empty_texture.view)
}

static void result_pass_render(result_pass_t* this,
                               WGPURenderPassEncoder render_pass)
{
  if (!result_pass_is_ready(this)) {
    return;
  }

  effect_pre_render(&this->effect, render_pass);
  wgpuRenderPassEncoderSetBindGroup(render_pass, 1,
                                    this->renderer->bind_groups.frame, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(render_pass, 6, 1, 0, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Compute Metaballs example
 * -------------------------------------------------------------------------- */

// Example state object
static struct {
  webgpu_renderer_t renderer;
  camera_controller_t camera_controller;
  perspective_camera_t persp_camera;
  ivolume_settings_t volume;
  // Render passes
  deferred_pass_t deferred_pass;
  copy_pass_t copy_pass;
  bloom_pass_t bloom_pass;
  result_pass_t result_pass;
  // Geometry
  metaballs_t metaballs;
  ground_t ground;
  box_outline_t box_outline;
  particles_t particles;
  // Time-related state
  float last_frame_time;     // has seconds unit
  float dt;                  // has seconds unit
  float rearrange_countdown; // has seconds unit
} example_state = {
  .last_frame_time     = 0.0f,
  .dt                  = 0.0f,
  .rearrange_countdown = 5.0f,
};

// Other variables
static const char* example_title = "Compute Metaballs";
static bool prepared             = false;

static void example_rearrange(void)
{
  deferred_pass_rearrange(&example_state.deferred_pass);
  metaballs_rearrange(&example_state.metaballs);
}

static void init_example_state(wgpu_context_t* wgpu_context)
{
  const uint32_t inner_width  = wgpu_context->surface.width;
  const uint32_t inner_height = wgpu_context->surface.height;

  /* WebGPU renderer */
  webgpu_renderer_t* renderer = &example_state.renderer;
  webgpu_renderer_create(renderer, wgpu_context);
  renderer->device_pixel_ratio = 1.0f;
  renderer->output_size[0]
    = inner_width * settings_get_quality_level().output_scale;
  renderer->output_size[1]
    = inner_height * settings_get_quality_level().output_scale;

  /* Perspective camera */
  perspective_camera_t* persp_camera = &example_state.persp_camera;
  perspective_camera_init(persp_camera, (45.0f * PI) / 180.0f,
                          (float)inner_width / (float)inner_height, 0.1f,
                          100.0f);
  perspective_camera_set_position(persp_camera, (vec3){10.0f, 2.0f, 16.0f});
  perspective_camera_look_at(persp_camera, GLM_VEC3_ZERO);

  /* Camera controller */
  camera_controller_create(&example_state.camera_controller, persp_camera,
                           false, 0.1f);
  camera_controller_look_at(&example_state.camera_controller,
                            (vec3){0.0f, 1.0f, 0.0f});

  /* Initialize WebGPU renderer */
  webgpu_renderer_init(&example_state.renderer);

  /* Projection UBO */
  projection_uniforms_t* projection_ubo = &renderer->ubos_data.projection_ubo;
  glm_mat4_copy(persp_camera->projection_matrix, projection_ubo->matrix);
  glm_mat4_copy(persp_camera->projection_inv_matrix,
                projection_ubo->inverse_matrix);
  glm_vec2_copy(
    (vec2){(float)renderer->output_size[0], (float)renderer->output_size[1]},
    projection_ubo->output_size);
  projection_ubo->z_near = persp_camera->near;
  projection_ubo->z_far  = persp_camera->far;
  wgpu_queue_write_buffer(wgpu_context, renderer->ubos.projection_ubo.buffer, 0,
                          projection_ubo, sizeof(*projection_ubo));

  /* View UBO */
  view_uniforms_t* view_ubo = &renderer->ubos_data.view_ubo;
  glm_mat4_copy(persp_camera->view_matrix, view_ubo->matrix);
  glm_mat4_copy(persp_camera->view_inv_matrix, view_ubo->inverse_matrix);
  glm_vec3_copy(persp_camera->position, view_ubo->position);
  wgpu_queue_write_buffer(wgpu_context, renderer->ubos.view_ubo.buffer, 0,
                          view_ubo, sizeof(*view_ubo));

  /* Volume settings */
  example_state.volume = (ivolume_settings_t){
    .x_min = -3.0f,
    .y_min = -3.0f,
    .z_min = -3.0f,

    .width  = 100,
    .height = 100,
    .depth  = 80,

    .x_step = 0.075f,
    .y_step = 0.075f,
    .z_step = 0.075f,

    .iso_level = 20.0f,
  };

  /* Deferred pass, copy pass, bloom pass & result pass */
  deferred_pass_t* deferred_pass = &example_state.deferred_pass;
  deferred_pass_create(deferred_pass, renderer);
  copy_pass_t* copy_pass = &example_state.copy_pass;
  copy_pass_create(copy_pass, renderer);

  bloom_pass_t* bloom_pass = NULL;
  if (settings_get_quality_level().bloom_toggle) {
    bloom_pass = &example_state.bloom_pass;
    bloom_pass_create(bloom_pass, renderer, copy_pass);
  }

  result_pass_t* result_pass = &example_state.result_pass;
  result_pass_create(result_pass, renderer, copy_pass, bloom_pass);

  /* Metaballs, ground, box outline & particles */
  metaballs_create(&example_state.metaballs, renderer, &example_state.volume,
                   &deferred_pass->spot_light);
  ground_create(&example_state.ground, renderer, &deferred_pass->spot_light);
  box_outline_create(&example_state.box_outline, renderer);
  particles_create(&example_state.particles, renderer,
                   &deferred_pass->point_lights.lights_buffer);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  if (context->paused) {
    camera_controller_pause(&example_state.camera_controller);
  }
  else if (example_state.camera_controller._paused) {
    camera_controller_start(&example_state.camera_controller);
  }

  if (!example_state.camera_controller._paused) {
    camera_controller_tick(&example_state.camera_controller);
  }

  const float frame_timestamp_sec
    = context->frame.timestamp_millis * 0.001; // s
  example_state.dt = frame_timestamp_sec - example_state.last_frame_time;
  example_state.last_frame_time = frame_timestamp_sec;

  wgpu_context_t* wgpu_context       = context->wgpu_context;
  webgpu_renderer_t* renderer        = &example_state.renderer;
  perspective_camera_t* persp_camera = &example_state.persp_camera;

  /* Rearrange geometry  */
  if (example_state.rearrange_countdown < 0.0f) {
    example_rearrange();
    example_state.rearrange_countdown = 5.0f;
  }
  example_state.rearrange_countdown -= example_state.dt;

  /* Update view UBO */
  view_uniforms_t* view_ubo = &renderer->ubos_data.view_ubo;
  glm_mat4_copy(persp_camera->view_matrix, view_ubo->matrix);
  glm_mat4_copy(persp_camera->view_inv_matrix, view_ubo->inverse_matrix);
  glm_vec3_copy(persp_camera->position, view_ubo->position);
  view_ubo->time       = frame_timestamp_sec;
  view_ubo->delta_time = example_state.dt;
  wgpu_queue_write_buffer(wgpu_context, renderer->ubos.view_ubo.buffer, 0,
                          view_ubo, sizeof(*view_ubo));
}

static void suppress_unused_functions(void)
{
  UNUSED_VAR(SHADOW_MAP_SIZE);

  UNUSED_FUNCTION(settings_get_quality);
  UNUSED_FUNCTION(settings_set_quality);
  UNUSED_FUNCTION(orthographic_camera_set_position);
  UNUSED_FUNCTION(orthographic_camera_look_at);
  UNUSED_FUNCTION(orthographic_camera_init);
  UNUSED_FUNCTION(webgpu_renderer_set_output_size);
  UNUSED_FUNCTION(webgpu_renderer_get_output_size);
  UNUSED_FUNCTION(spot_light_get_position);
  UNUSED_FUNCTION(spot_light_get_direction);
  UNUSED_FUNCTION(spot_light_get_color);
  UNUSED_FUNCTION(spot_light_get_cut_off);
  UNUSED_FUNCTION(spot_light_get_intensity);
  UNUSED_FUNCTION(spot_light_get_outer_cut_off);
  UNUSED_FUNCTION(copy_pass_render);
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    suppress_unused_functions();
    init_example_state(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_slider_int(
          context->imgui_overlay, "Point Lights Count",
          &example_state.deferred_pass.point_lights.lights_count, 0,
          MAX_POINT_LIGHTS_COUNT)) {
      point_lights_set_lights_count(
        &example_state.deferred_pass.point_lights,
        example_state.deferred_pass.point_lights.lights_count);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  webgpu_renderer_on_render(&example_state.renderer);

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Run compute shaders */
  {
    WGPUComputePassEncoder compute_pass
      = wgpuCommandEncoderBeginComputePass(wgpu_context->cmd_enc, NULL);
    if (settings_get_quality_level().update_metaballs) {
      metaballs_update_sim(&example_state.metaballs, compute_pass,
                           example_state.last_frame_time, example_state.dt);
    }
    else {
      if (!metaballs_has_updated_at_least_once(&example_state.metaballs)) {
        metaballs_update_sim(&example_state.metaballs, compute_pass,
                             example_state.last_frame_time, example_state.dt);
      }
    }

    deferred_pass_update_lights_sim(&example_state.deferred_pass, compute_pass,
                                    example_state.last_frame_time,
                                    example_state.dt);
    if (settings_get_quality_level().bloom_toggle) {
      bloom_pass_update_bloom(&example_state.bloom_pass, compute_pass);
    }
    wgpuComputePassEncoderEnd(compute_pass);
    WGPU_RELEASE_RESOURCE(ComputePassEncoder, compute_pass)
  }

  /* Render scene from spot light POV */
  {
    example_state.deferred_pass.spot_light.framebuffer.descriptor.label
      = "spot light 0 shadow map render pass";
    WGPURenderPassEncoder spot_light_shadow_pass
      = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc,
        &example_state.deferred_pass.spot_light.framebuffer.descriptor);
    metaballs_render_shadow(&example_state.metaballs, spot_light_shadow_pass);
    ground_render_shadow(&example_state.ground, spot_light_shadow_pass);
    wgpuRenderPassEncoderEnd(spot_light_shadow_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, spot_light_shadow_pass)
  }

  /* Deferred pass */
  {
    example_state.deferred_pass.framebuffer.descriptor.label = "gbuffer";
    WGPURenderPassEncoder g_buffer_pass = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc,
      &example_state.deferred_pass.framebuffer.descriptor);
    metaballs_render(&example_state.metaballs, g_buffer_pass);
    box_outline_render(&example_state.box_outline, g_buffer_pass);
    ground_render(&example_state.ground, g_buffer_pass);
    particles_render(&example_state.particles, g_buffer_pass);
    wgpuRenderPassEncoderEnd(g_buffer_pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, g_buffer_pass)
  }

  /* Bloom pass */
  if (settings_get_quality_level().bloom_toggle) {
    /* Copy pass */
    {
      example_state.copy_pass.framebuffer.descriptor.label = "copy pass";
      WGPURenderPassEncoder copy_render_pass
        = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc,
          &example_state.copy_pass.framebuffer.descriptor);
      deferred_pass_render(&example_state.deferred_pass, copy_render_pass);
      wgpuRenderPassEncoderEnd(copy_render_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, copy_render_pass)
    }

    /* Bloom pass */
    {
      example_state.bloom_pass.framebuffer.descriptor.label = "bloom pass";
      WGPURenderPassEncoder bloom_render_pass
        = wgpuCommandEncoderBeginRenderPass(
          wgpu_context->cmd_enc,
          &example_state.bloom_pass.framebuffer.descriptor);
      bloom_pass_render(&example_state.bloom_pass, bloom_render_pass);
      wgpuRenderPassEncoderEnd(bloom_render_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, bloom_render_pass)
    }

    /* Final composite pass */
    {
      example_state.renderer.framebuffer.descriptor.label
        = "draw default framebuffer";
      WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc, &example_state.renderer.framebuffer.descriptor);
      result_pass_render(&example_state.result_pass, render_pass);
      wgpuRenderPassEncoderEnd(render_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
    }
  }
  else {
    /* Final composite pass */
    {
      example_state.renderer.framebuffer.descriptor.label
        = "draw default framebuffer";
      WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
        wgpu_context->cmd_enc, &example_state.renderer.framebuffer.descriptor);
      deferred_pass_render(&example_state.deferred_pass, render_pass);
      wgpuRenderPassEncoderEnd(render_pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass)
    }
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL)
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  const int draw_result = example_draw(context);
  camera_controller_handle_input_events(&example_state.camera_controller,
                                        context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  webgpu_renderer_destroy(&example_state.renderer);
  deferred_pass_destroy(&example_state.deferred_pass);
  copy_pass_destroy(&example_state.copy_pass);
  if (settings_get_quality_level().bloom_toggle) {
    bloom_pass_destroy(&example_state.bloom_pass);
  }
  result_pass_destroy(&example_state.result_pass);
  metaballs_destroy(&example_state.metaballs);
  ground_destroy(&example_state.ground);
  box_outline_destroy(&example_state.box_outline);
  particles_destroy(&example_state.particles);
}

void example_compute_metaballs(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
