#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pristine Grid
 *
 * A simple WebGPU implementation of the "Pristine Grid" technique described in
 * this wonderful little blog post:
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 *
 * Ref:
 * https://github.com/toji/pristine-grid-webgpu
 * https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// A WebGPU implementation of the "Pristine Grid" shader described at
// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
static const char* grid_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Math functions
 * @ref https://github.com/toji/gl-matrix
 * -------------------------------------------------------------------------- */

/**
 * @brief Generates a perspective projection matrix with the given bounds.
 * The near/far clip planes correspond to a normalized device coordinate Z range
 * of [-1, 1], which matches WebGL/OpenGL's clip volume. Passing
 * null/undefined/no value for far will generate infinite projection matrix.
 *
 * @param {mat4} out mat4 frustum matrix will be written into
 * @param {number} fovy Vertical field of view in radians
 * @param {number} aspect Aspect ratio. typically viewport width/height
 * @param {number} near Near bound of the frustum
 * @param {number} far Far bound of the frustum, can be null or Infinity
 * @returns {mat4} out
 */
mat4* glm_mat4_perspective_zo(mat4* out, float fovy, float aspect, float near,
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

/**
 * @brief Rotates the given 4-by-4 matrix around the x-axis by the given angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix m.
 */
static mat4* glm_mat4_rotate_x(mat4* m, float angle_in_radians)
{
  const float s   = sin(angle_in_radians);
  const float c   = cos(angle_in_radians);
  const float m10 = (*m)[1][0];
  const float m11 = (*m)[1][1];
  const float m12 = (*m)[1][2];
  const float m13 = (*m)[1][3];
  const float m20 = (*m)[2][0];
  const float m21 = (*m)[2][1];
  const float m22 = (*m)[2][2];
  const float m23 = (*m)[2][3];
  /* Perform axis-specific matrix multiplication */
  (*m)[1][0] = m10 * c + m20 * s;
  (*m)[1][1] = m11 * c + m21 * s;
  (*m)[1][2] = m12 * c + m22 * s;
  (*m)[1][3] = m13 * c + m23 * s;
  (*m)[2][0] = m20 * c - m10 * s;
  (*m)[2][1] = m21 * c - m11 * s;
  (*m)[2][2] = m22 * c - m12 * s;
  (*m)[2][3] = m23 * c - m13 * s;
  return m;
}

/**
 * @brief Rotates the given 4-by-4 matrix around the y-axis by the given angle.
 * @param m - The matrix.
 * @param angleInRadians - The angle by which to rotate (in radians).
 * @returns The rotated matrix m.
 */
static mat4* glm_mat4_rotate_y(mat4* m, float angle_in_radians)
{
  const float s   = sin(angle_in_radians);
  const float c   = cos(angle_in_radians);
  const float m00 = (*m)[0][0];
  const float m01 = (*m)[0][1];
  const float m02 = (*m)[0][2];
  const float m03 = (*m)[0][3];
  const float m20 = (*m)[2][0];
  const float m21 = (*m)[2][1];
  const float m22 = (*m)[2][2];
  const float m23 = (*m)[2][3];
  /* Perform axis-specific matrix multiplication */
  (*m)[0][0] = m00 * c - m20 * s;
  (*m)[0][1] = m01 * c - m21 * s;
  (*m)[0][2] = m02 * c - m22 * s;
  (*m)[0][3] = m03 * c - m23 * s;
  (*m)[2][0] = m00 * s + m20 * c;
  (*m)[2][1] = m01 * s + m21 * c;
  (*m)[2][2] = m02 * s + m22 * c;
  (*m)[2][3] = m03 * s + m23 * c;
  return m;
}

/**
 * @brief Transform vec3 by 4x4 matrix.
 * @param v - the vector
 * @param m - The matrix.
 * @param dst - vec3 to store result.
 * @returns the transformed vector dst
 */
static vec3* glm_vec3_transform_mat4(vec3 v, mat4 m, vec3* dst)
{
  const float x = v[0];
  const float y = v[1];
  const float z = v[2];
  const float w = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3];
  (*dst)[0]     = (m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0]) / w;
  (*dst)[1]     = (m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1]) / w;
  (*dst)[2]     = (m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2]) / w;
  return dst;
}

/* -------------------------------------------------------------------------- *
 * Orbit camera
 * -------------------------------------------------------------------------- */

typedef struct orbit_camera_t {
  vec2 orbit;
  vec2 min_orbit;
  vec2 max_orbit;
  bool constrain_orbit[2];
  float max_distance;
  float min_distance;
  float distance_step;
  bool constrain_distance;
  vec3 distance;
  vec3 target;
  mat4 view_mat;
  mat4 camera_mat;
  vec3 position;
  bool dirty;
  struct {
    bool moving;
    vec2 move_delta;
    vec2 prev_position;
  } mouse;
} orbit_camera_t;

static void orbit_camera_init_defaults(orbit_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_vec2_zero(this->orbit);
  glm_vec2_copy((vec2){-PI_2, -PI}, this->min_orbit);
  glm_vec2_copy((vec2){PI_2, PI}, this->max_orbit);
  this->constrain_orbit[0] = true;
  this->constrain_orbit[1] = false;

  this->max_distance       = 10.0f;
  this->min_distance       = 1.0f;
  this->distance_step      = 0.005f;
  this->constrain_distance = true;

  glm_vec3_copy((vec3){0.0f, 0.0f, 1.0f}, this->distance);
  glm_vec3_zero(this->target);
  glm_mat4_identity(this->view_mat);
  glm_mat4_identity(this->camera_mat);
  glm_vec3_zero(this->position);
  this->dirty = false;
}

/* Construtor */
static void orbit_camera_init(orbit_camera_t* this)
{
  orbit_camera_init_defaults(this);
}

static void orbit_camera_orbit(orbit_camera_t* this, float x_delta,
                               float y_delta)
{
  if (x_delta || y_delta) {
    this->orbit[1] += x_delta;
    if (this->constrain_orbit[1]) {
      this->orbit[1]
        = MIN(MAX(this->orbit[1], this->min_orbit[1]), this->max_orbit[1]);
    }
    else {
      while (this->orbit[1] < -PI) {
        this->orbit[1] += PI2;
      }
      while (this->orbit[1] >= PI) {
        this->orbit[1] -= PI2;
      }
    }

    this->orbit[0] += y_delta;
    if (this->constrain_orbit[0]) {
      this->orbit[0]
        = MIN(MAX(this->orbit[0], this->min_orbit[0]), this->max_orbit[0]);
    }
    else {
      while (this->orbit[0] < -PI) {
        this->orbit[0] += PI2;
      }
      while (this->orbit[0] >= PI) {
        this->orbit[0] -= PI2;
      }
    }

    this->dirty = true;
  }
}

static vec3* orbit_camera_get_target(orbit_camera_t* this)
{
  return &this->target;
}

static void orbit_camera_set_target(orbit_camera_t* this, vec3 value)
{
  glm_vec3_copy(value, this->target);
  this->dirty = true;
}

static float orbit_camera_get_distance(orbit_camera_t* this)
{
  return this->distance[2];
}

static void orbit_camera_set_distance(orbit_camera_t* this, float value)
{
  this->distance[2] = value;
  if (this->constrain_distance) {
    this->distance[2]
      = MIN(MAX(this->distance[2], this->min_distance), this->max_distance);
  }
  this->dirty = true;
}

static void orbit_camera_update_matrices(orbit_camera_t* this)
{
  if (this->dirty) {
    glm_mat4_identity(this->camera_mat);

    glm_translate(this->camera_mat, this->target);
    glm_mat4_rotate_y(&this->camera_mat, -this->orbit[1]);
    glm_mat4_rotate_x(&this->camera_mat, -this->orbit[0]);
    glm_translate(this->camera_mat, this->distance);
    glm_mat4_inv(this->camera_mat, this->view_mat);

    this->dirty = false;
  }
}

static vec3* orbit_camera_get_position(orbit_camera_t* this)
{
  orbit_camera_update_matrices(this);
  glm_vec3_zero(this->position);
  glm_vec3_transform_mat4(this->position, this->camera_mat, &this->position);
  return &this->position;
}

static mat4* orbit_camera_get_view_matrix(orbit_camera_t* this)
{
  orbit_camera_update_matrices(this);
  return &this->view_mat;
}

/* -------------------------------------------------------------------------- *
 * Pristine Grid example
 * -------------------------------------------------------------------------- */

static const bool use_msaa              = false;
static const uint32_t msaa_sample_count = use_msaa ? 4u : 1u;
static WGPUTextureFormat depth_format   = WGPUTextureFormat_Depth24PlusStencil8;

/* Vertex layout used in this example */
typedef struct {
  vec3 position;
  vec2 uv;
} vertex_t;

static struct {
  mat4 projection_matrix;
  mat4 view_matrix;
  vec3 camera_position;
  float time;
} camera_uniforms = {0};

static struct {
  vec4 line_color;
  vec4 base_color;
  vec2 line_width;
  vec4 padding;
} uniform_array = {0};

static struct {
  vec4 clear_color;
  vec4 line_color;
  vec4 base_color;
  float line_width_x;
  float line_width_y;
} grid_options = {
  .clear_color = {
    /* .r = */ 0.0f,
    /* .g = */ 0.0f,
    /* .b = */ 0.2f,
    /* .a = */ 1.0f,
  },
  .line_color = {
    /* .r = */ 1.0f,
    /* .g = */ 1.0f,
    /* .b = */ 1.0f,
    /* .a = */ 1.0f,
  },
  .base_color = {
    /* .r = */ 0.0f,
    /* .g = */ 0.0f,
    /* .b = */ 0.0f,
    /* .a = */ 1.0f,
  },
  .line_width_x = 0.05f,
  .line_width_y = 0.05f,
};

static wgpu_buffer_t vertex_buffer        = {0};
static wgpu_buffer_t index_buffer         = {0};
static wgpu_buffer_t frame_uniform_buffer = {0};
static wgpu_buffer_t uniform_buffer       = {0};

static WGPUBindGroupLayout frame_bind_group_layout = NULL;
static WGPUBindGroupLayout bind_group_layout       = NULL;

static WGPUBindGroup frame_bind_group = NULL;
static WGPUBindGroup bind_group       = NULL;

static WGPUPipelineLayout pipeline_layout = NULL;
static WGPURenderPipeline pipeline        = NULL;

static struct {
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } msaa_color, depth;
} textures = {0};

/* Render pass descriptor for frame buffer writes */
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

/* The orbit camera */
static struct {
  float fov;
  float z_near;
  float z_far;
} camera_parms = {
  .fov    = PI_2,
  .z_near = 0.01f,
  .z_far  = 128.0f,
};

static orbit_camera_t camera;

/* Other variables */
static const char* example_title = "Pristine Grid";
static bool prepared             = false;

static void prepare_vertex_and_index_buffers(wgpu_context_t* wgpu_context)
{
  /* Setup vertices (x, y, z, u,v) */
  {
    static const vertex_t vertex_array[4] = {
      {
        .position = {-20.0f, -0.5f, -20.0f},
        .uv       = {0.0f, 0.0f},
      },
      {
        .position = {20.0f, -0.5f, -20.0f},
        .uv       = {200.0f, 0.0f},
      },
      {
        .position = {-20.0f, -0.5f, 20.0f},
        .uv       = {0.0f, 200.0f},
      },
      {
        .position = {20.0f, -0.5f, 20.0f},
        .uv       = {200.0f, 200.0f},
      },
    };
    vertex_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Vertex buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                      .size  = sizeof(vertex_array),
                      .count = (uint32_t)ARRAY_SIZE(vertex_array),
                      .initial.data = vertex_array,
                    });
  }

  /* Setup indices */
  {
    static const uint16_t index_array[6] = {
      0, 1, 2, /* */
      1, 2, 3  /* */
    };
    index_buffer = wgpu_create_buffer(
      wgpu_context, &(wgpu_buffer_desc_t){
                      .label = "Index buffer",
                      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                      .size  = sizeof(index_array),
                      .count = (uint32_t)ARRAY_SIZE(index_array),
                      .initial.data = index_array,
                    });
  }
}

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  /* Update color attachment clear color */
  render_pass.color_attachments[0].clearValue = (WGPUColor){
    .r = grid_options.clear_color[0], /* Red   */
    .g = grid_options.clear_color[1], /* Green */
    .b = grid_options.clear_color[2], /* Blue  */
    .a = grid_options.clear_color[3], /* Alpha */
  };

  /* Update uniforms data */
  memcpy(&uniform_array.line_color, &grid_options.line_color,
         sizeof(WGPUColor));
  memcpy(&uniform_array.base_color, &grid_options.base_color,
         sizeof(WGPUColor));
  glm_vec2_copy((vec2){grid_options.line_width_x, grid_options.line_width_y},
                uniform_array.line_width);

  /* Update uniform buffer */
  wgpu_queue_write_buffer(wgpu_context, uniform_buffer.buffer, 0,
                          &uniform_array, sizeof(uniform_array));
}

static void update_projection(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;
  // Using mat4.perspectiveZO instead of mat4.perpective because WebGPU's
  // normalized device coordinates Z range is [0, 1], instead of WebGL's [-1, 1]
  glm_mat4_perspective_zo(&camera_uniforms.projection_matrix, camera_parms.fov,
                          aspect_ratio, camera_parms.z_near,
                          &camera_parms.z_far);
}

static void update_mouse_state(wgpu_example_context_t* context)
{
  if (!camera.mouse.moving && context->mouse_buttons.left) {
    /* pointerdown -> downCallback */
    glm_vec2_copy(context->mouse_position, camera.mouse.prev_position);
    camera.mouse.moving = true;
  }
  else if (camera.mouse.moving && context->mouse_buttons.left) {
    /* pointermove -> moveCallback */
    glm_vec2_sub(context->mouse_position, camera.mouse.prev_position,
                 camera.mouse.move_delta);
    glm_vec2_copy(context->mouse_position, camera.mouse.prev_position);
    camera.dirty = camera.dirty
                   || ((fabs(camera.mouse.move_delta[0]) > 1.0f)
                       || (fabs(camera.mouse.move_delta[1]) > 1.0f));
  }
  else if (camera.mouse.moving && !context->mouse_buttons.left) {
    /* pointerup -> upCallback */
    camera.mouse.moving = false;
  }
}

static void update_frame_uniforms(wgpu_example_context_t* context)
{
  /* Update mouse state */
  update_mouse_state(context);

  /* Handle mouse movement */
  if (camera.dirty) {
    orbit_camera_orbit(&camera, camera.mouse.move_delta[0] * 0.025f,
                       camera.mouse.move_delta[1] * 0.025f);
  }

  /* Update frame uniforms data */
  glm_mat4_copy(*orbit_camera_get_view_matrix(&camera),
                camera_uniforms.view_matrix);
  glm_vec3_copy(*orbit_camera_get_position(&camera),
                camera_uniforms.camera_position);
  camera_uniforms.time = context->frame.timestamp_millis;

  /* Update uniform buffer */
  wgpu_queue_write_buffer(context->wgpu_context, frame_uniform_buffer.buffer, 0,
                          &camera_uniforms, sizeof(camera_uniforms));
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  /* Frame uniform buffer */
  {
    frame_uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Frame - Uniform buffer",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = sizeof(camera_uniforms),
                                         });
    ASSERT(frame_uniform_buffer.buffer != NULL);
  }

  /* Update uniform buffer */
  update_projection(wgpu_context);
  update_frame_uniforms(context);

  /* Uniform buffer */
  {
    uniform_buffer
      = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                           .label = "Uniform buffer",
                                           .usage = WGPUBufferUsage_CopyDst
                                                    | WGPUBufferUsage_Uniform,
                                           .size = sizeof(uniform_array),
                                         });
    ASSERT(uniform_buffer.buffer != NULL);
  }

  /* Update uniform buffer */
  update_uniforms(wgpu_context);
}

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Frame bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0 : Camera/Frame uniforms */
        .binding    = 0, /* Camera/Frame uniforms */
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(camera_uniforms),
        },
        .sampler = {0},
      },
    };

    frame_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Frame - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(frame_bind_group_layout != NULL);
  }

  /* Pristine Grid bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        /* Binding 0 : uniform array */
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(uniform_array),
        },
        .sampler = {0},
      },
    };

    bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = "Pristine Grid - Bind group layout",
                              .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                              .entries    = bgl_entries,
                            });
    ASSERT(bind_group_layout != NULL);
  }
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayout bind_group_layouts[2] = {
    frame_bind_group_layout, /* Group 0 */
    bind_group_layout,       /* Group 1 */
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = "Pristine Grid - Pipeline layout",
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bind_group_layouts),
      .bindGroupLayouts     = bind_group_layouts,
    });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Frame bind group */
  {
    frame_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = "Frame - Bind group",
        .layout     = frame_bind_group_layout,
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          /* Binding 0 : Camera uniforms */
          .binding = 0, // Camera uniforms
          .buffer  = frame_uniform_buffer.buffer,
          .offset  = 0,
          .size    = frame_uniform_buffer.size,
        },
      }
      );
    ASSERT(frame_bind_group != NULL);
  }

  /* Pristine Grid bind group */
  {
    bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor) {
        .label      = "Pristine Grid - Bind group",
        .layout     = bind_group_layout,
        .entryCount = 1,
        .entries    = &(WGPUBindGroupEntry) {
          /* Binding 0 : Uniform buffer */
          .binding = 0,
          .buffer  = uniform_buffer.buffer,
          .offset  = 0,
          .size    = uniform_buffer.size,
        },
      }
      );
    ASSERT(bind_group != NULL);
  }
}

static void allocate_render_targets(wgpu_context_t* wgpu_context,
                                    WGPUExtent2D size)
{
  WGPU_RELEASE_RESOURCE(TextureView, textures.msaa_color.view)
  WGPU_RELEASE_RESOURCE(Texture, textures.msaa_color.texture)

  /* Multi-sampled color render target */
  if (msaa_sample_count > 1) {
    /* Create the multi-sampled texture */
    WGPUTextureDescriptor multisampled_frame_desc = {
      .label         = "Multi-sampled texture",
      .size          = (WGPUExtent3D){
         .width              = size.width,
         .height             = size.height,
         .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = msaa_sample_count,
      .dimension     = WGPUTextureDimension_2D,
      .format        = wgpu_context->swap_chain.format,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    textures.msaa_color.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
    ASSERT(textures.msaa_color.texture != NULL);

    /* Create the multi-sampled texture view */
    textures.msaa_color.view = wgpuTextureCreateView(
      textures.msaa_color.texture, &(WGPUTextureViewDescriptor){
                                     .label  = "Multi-sampled texture view",
                                     .format = wgpu_context->swap_chain.format,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                   });
    ASSERT(textures.msaa_color.view != NULL);
  }

  WGPU_RELEASE_RESOURCE(TextureView, textures.depth.view)
  WGPU_RELEASE_RESOURCE(Texture, textures.depth.texture)

  /* Multi-sampled color render target */
  {
    /* Create the multi-sampled texture */
    WGPUTextureDescriptor multisampled_frame_desc = {
      .label         = "Depth texture",
      .size          = (WGPUExtent3D){
         .width              = size.width,
         .height             = size.height,
         .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = msaa_sample_count,
      .dimension     = WGPUTextureDimension_2D,
      .format        = depth_format,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    textures.depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &multisampled_frame_desc);
    ASSERT(textures.depth.texture != NULL);

    /* Create the multi-sampled texture view */
    textures.depth.view = wgpuTextureCreateView(
      textures.depth.texture, &(WGPUTextureViewDescriptor){
                                .label           = "Multi-sampled texture view",
                                .format          = depth_format,
                                .dimension       = WGPUTextureViewDimension_2D,
                                .baseMipLevel    = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = 0,
                                .arrayLayerCount = 1,
                              });
    ASSERT(textures.depth.view != NULL);
  }
}

static void setup_render_pass(void)
{
  /* Color attachment */
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment){
    /* Appropriate target will be populated in onFrame */
    .view          = msaa_sample_count > 1 ? textures.msaa_color.view : NULL,
    .depthSlice    = ~0,
    .resolveTarget = NULL,
    .clearValue    = {
      .r = grid_options.clear_color[0],
      .g = grid_options.clear_color[1],
      .b = grid_options.clear_color[2],
      .a = grid_options.clear_color[3],
    },
    .loadOp        = WGPULoadOp_Clear,
    .storeOp = msaa_sample_count > 1 ? WGPUStoreOp_Discard : WGPUStoreOp_Store,
  };

  /* Depth-stencil attachment */
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view              = textures.depth.view,
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Discard,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  };

  /* Render pass descriptor */
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
  };
}

static void prepare_render_pipeline(wgpu_context_t* wgpu_context)
{
  /* Primitive state */
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  /* Color target state */
  WGPUBlendState blend_state              = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = depth_format,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_LessEqual;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    triangle, 20,
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
    /* Attribute location 1: UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, sizeof(float) * 3))

  /* Vertex state */
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      /* Vertex shader WGSL */
      .label             = "Grid - Vertex shader WGSL",
      .wgsl_code.source  = grid_shader_wgsl,
      .entry             = "vertexMain",
    },
    .buffer_count = 1,
    .buffers      = &triangle_vertex_buffer_layout,
  });

  /* Fragment state */
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
    .shader_desc = (wgpu_shader_desc_t){
      /* Fragment shader WGSL */
      .label             = "Grid - Fragment shader WGSL",
      .wgsl_code.source  = grid_shader_wgsl,
      .entry             = "fragmentMain",
    },
    .target_count = 1,
    .targets      = &color_target_state,
  });

  /* Multisample state */
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = msaa_sample_count,
      });

  /* Create rendering pipeline using the specified states */
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "Pristine Grid - Render pipeline",
                            .layout       = pipeline_layout,
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });

  /* Partial cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void example_on_resize(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  if (wgpu_context->surface.width == 0 || wgpu_context->surface.height == 0) {
    return;
  }

  update_projection(context->wgpu_context);

  if (wgpu_context->device) {
    allocate_render_targets(wgpu_context,
                            (WGPUExtent2D){
                              .width  = wgpu_context->surface.width,
                              .height = wgpu_context->surface.height,
                            });
  }
}

static int example_initialize(wgpu_example_context_t* context)
{
  UNUSED_FUNCTION(orbit_camera_get_target);
  UNUSED_FUNCTION(orbit_camera_set_target);
  UNUSED_FUNCTION(orbit_camera_get_distance);
  UNUSED_FUNCTION(orbit_camera_set_distance);
  UNUSED_FUNCTION(example_on_resize);

  if (context) {
    wgpu_context_t* wgpu_context = context->wgpu_context;
    orbit_camera_init(&camera);
    prepare_vertex_and_index_buffers(wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_group_layouts(wgpu_context);
    setup_pipeline_layout(wgpu_context);
    prepare_render_pipeline(wgpu_context);
    setup_bind_groups(wgpu_context);
    allocate_render_targets(wgpu_context,
                            (WGPUExtent2D){
                              .width  = wgpu_context->surface.width,
                              .height = wgpu_context->surface.height,
                            });
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPURenderPassDescriptor const*
get_default_render_pass_descriptor(wgpu_context_t* wgpu_context)
{
  const WGPUTextureView color_texture = wgpu_context->swap_chain.frame_buffer;
  if (msaa_sample_count > 1) {
    render_pass.color_attachments[0].resolveTarget = color_texture;
  }
  else {
    render_pass.color_attachments[0].view = color_texture;
  }
  return &render_pass.descriptor;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    if (imgui_overlay_color_edit4(context->imgui_overlay, "clearColor",
                                  grid_options.clear_color)) {
      update_uniforms(context->wgpu_context);
    }
    if (imgui_overlay_color_edit4(context->imgui_overlay, "lineColor",
                                  grid_options.line_color)) {
      update_uniforms(context->wgpu_context);
    }
    if (imgui_overlay_color_edit4(context->imgui_overlay, "baseColor",
                                  grid_options.base_color)) {
      update_uniforms(context->wgpu_context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "lineWidthX",
                                   &grid_options.line_width_x, 0.0f, 1.0f,
                                   "%.001f")) {
      update_uniforms(context->wgpu_context);
    }
    if (imgui_overlay_slider_float(context->imgui_overlay, "lineWidthY",
                                   &grid_options.line_width_y, 0.0f, 1.0f,
                                   "%.001f")) {
      update_uniforms(context->wgpu_context);
    }
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  /* Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Create render pass encoder for encoding drawing commands */
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, get_default_render_pass_descriptor(wgpu_context));

  if (pipeline) {
    /* Bind the rendering pipeline */
    wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

    /* Set viewport */
    wgpuRenderPassEncoderSetViewport(
      wgpu_context->rpass_enc, 0.0f, 0.0f, (float)wgpu_context->surface.width,
      (float)wgpu_context->surface.height, 0.0f, 1.0f);

    /* Set scissor rectangle */
    wgpuRenderPassEncoderSetScissorRect(wgpu_context->rpass_enc, 0u, 0u,
                                        wgpu_context->surface.width,
                                        wgpu_context->surface.height);

    /* Set the bind groups */
    wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                      frame_bind_group, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(
      wgpu_context->rpass_enc, 1, bind_group, 0,
      0); /* Assumes the camera bind group is already set. */

    /* Bind vertex buffer (contains positions & uvs) */
    wgpuRenderPassEncoderSetVertexBuffer(
      wgpu_context->rpass_enc, 0, vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);

    /* Bind index buffer */
    wgpuRenderPassEncoderSetIndexBuffer(
      wgpu_context->rpass_enc, index_buffer.buffer, WGPUIndexFormat_Uint16, 0,
      WGPU_WHOLE_SIZE);

    /* Draw quad */
    wgpuRenderPassEncoderDrawIndexed(wgpu_context->rpass_enc,
                                     index_buffer.count, 1, 0, 0, 0);
  }

  /* End render pass */
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  /* Draw ui overlay */
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  /* Get command buffer */
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_context_t* wgpu_context)
{
  /* Get next image in the swap chain (back/front buffer) */
  wgpu_swap_chain_get_current_image(wgpu_context);

  /* Create command buffer */
  WGPUCommandBuffer command_buffer = build_command_buffer(wgpu_context);
  ASSERT(command_buffer != NULL);

  /* Submit command buffer to the queue */
  wgpu_flush_command_buffers(wgpu_context, &command_buffer, 1);

  /* Present the current buffer to the swap chain */
  wgpu_swap_chain_present(wgpu_context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }

  if (!context->paused) {
    /* Update the frame uniforms */
    update_frame_uniforms(context);
  }

  return example_draw(context->wgpu_context);
}

/* Clean up used resources */
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);

  WGPU_RELEASE_RESOURCE(Buffer, vertex_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, index_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, frame_uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, frame_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, frame_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(Texture, textures.msaa_color.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.msaa_color.view)
  WGPU_RELEASE_RESOURCE(Texture, textures.depth.texture)
  WGPU_RELEASE_RESOURCE(TextureView, textures.depth.view)
}

void example_pristine_grid(int argc, char* argv[])
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

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* grid_shader_wgsl = CODE(
  // grid function from Best Darn Grid article
  fn PristineGrid(uv: vec2f, lineWidth: vec2f) -> f32 {
    let uvDDXY = vec4f(dpdx(uv), dpdy(uv));
    let uvDeriv = vec2f(length(uvDDXY.xz), length(uvDDXY.yw));
    let invertLine: vec2<bool> = lineWidth > vec2f(0.5);
    let targetWidth: vec2f = select(lineWidth, 1 - lineWidth, invertLine);
    let drawWidth: vec2f = clamp(targetWidth, uvDeriv, vec2f(0.5));
    let lineAA: vec2f = uvDeriv * 1.5;
    var gridUV: vec2f = abs(fract(uv) * 2.0 - 1.0);
    gridUV = select(1 - gridUV, gridUV, invertLine);
    var grid2: vec2f = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = mix(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2 = select(grid2, 1.0 - grid2, invertLine);
    return mix(grid2.x, 1.0, grid2.y);
  }

  struct VertexIn {
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
  }

  struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
  }

  struct Camera {
    projection: mat4x4f,
    view: mat4x4f,
    position: vec3f,
    time: f32,
  }
  @group(0) @binding(0) var<uniform> camera: Camera;

  struct GridArgs {
    lineColor: vec4f,
    baseColor: vec4f,
    lineWidth: vec2f,
  }
  @group(1) @binding(0) var<uniform> gridArgs: GridArgs;

  @vertex
  fn vertexMain(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.pos = camera.projection * camera.view * in.pos;
    out.uv = in.uv;
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
    var grid = PristineGrid(in.uv, gridArgs.lineWidth);

    // lerp between base and line color
    return mix(gridArgs.baseColor, gridArgs.lineColor, grid * gridArgs.lineColor.a);
  }
);
// clang-format on
