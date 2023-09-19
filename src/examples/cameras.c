#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"
#include "meshes.h"

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cameras
 *
 * This example provides example camera implementations.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/cameras
 * https://github.com/pr0g/c-polymorphism
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* cube_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Math functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Calculates the square root of the sum of squares of its arguments.
 * @param a argument 1
 * @param b argument 2
 * @param c argument 3
 * @return the square root of the sum of squares of its arguments
 */
static float math_hypot3(float a, float b, float c)
{
  return sqrt(a * a + b * b + c * c);
}

/**
 * @brief Calculates the length of a vec3.
 *
 * @param {ReadonlyVec3} a vector to calculate length of
 * @returns {Number} length of a
 */
static float glm_vec3_length(vec3 v)
{
  return math_hypot3(v[0], v[1], v[2]);
}

/* -------------------------------------------------------------------------- *
 * The input event handling
 * -------------------------------------------------------------------------- */

typedef struct input_handler_t {
  struct {
    vec2 prev_mouse_position;
    vec2 current_mouse_position;
    vec2 mouse_drag_distance;
    bool mouse_down;
  } mouse_state;
} input_handler_t;

static void input_handler_init_defaults(input_handler_t* this)
{
  memset(this, 0, sizeof(*this));
}

static void input_handler_init(input_handler_t* this)
{
  input_handler_init_defaults(this);
}

static void update_mouse_state(input_handler_t* this,
                               wgpu_example_context_t* context)
{
  context->mouse_position[1]
    = context->wgpu_context->surface.height - context->mouse_position[1];
  if (!this->mouse_state.mouse_down && context->mouse_buttons.left) {
    glm_vec2_copy(context->mouse_position,
                  this->mouse_state.prev_mouse_position);
    this->mouse_state.mouse_down = true;
  }
  else if (this->mouse_state.mouse_down && context->mouse_buttons.left) {
    glm_vec2_sub(context->mouse_position, this->mouse_state.prev_mouse_position,
                 this->mouse_state.mouse_drag_distance);
    glm_vec2_add(this->mouse_state.current_mouse_position,
                 this->mouse_state.mouse_drag_distance,
                 this->mouse_state.current_mouse_position);
    glm_vec2_copy(context->mouse_position,
                  this->mouse_state.prev_mouse_position);
  }
  else if (this->mouse_state.mouse_down && !context->mouse_buttons.left) {
    this->mouse_state.mouse_down = false;
  }
}

/* -------------------------------------------------------------------------- *
 * The common functionality between camera implementations
 * -------------------------------------------------------------------------- */

struct camera_base_t;

typedef struct camera_base_vtbl_t {
  mat4* (*update)(struct camera_base_t*, float, input_handler_t*);
} camera_base_vtbl_t;

typedef struct camera_base_t {
  camera_base_vtbl_t _vtbl;
  /* The camera matrix */
  mat4 _matrix;
  /* The calculated view matrix */
  mat4 _view;
} camera_base_t;

static void camera_base_init_defaults(camera_base_t* this)
{
  memset(this, 0, sizeof(*this));

  glm_mat4_identity(this->_matrix);
}

static void camera_base_init(camera_base_t* this)
{
  camera_base_init_defaults(this);
}

/* Returns the camera matrix */
static mat4* camera_base_get_matrix(camera_base_t* this)
{
  return &this->_matrix;
}

/* Assigns `mat` to the camera matrix */
static void camera_base_set_matrix(camera_base_t* this, mat4 mat)
{
  glm_mat4_copy(mat, this->_matrix);
}

/* Returns the camera view matrix */
static mat4* camera_base_view(camera_base_t* this)
{
  return &this->_view;
}

/* Assigns `mat` to the camera view */
static void camera_base_set_view(camera_base_t* this, mat4 mat)
{
  glm_mat4_copy(mat, this->_view);
}

/* Returns column vector 0 of the camera matrix */
static vec4* camera_base_get_right(camera_base_t* this)
{
  return &this->_matrix[0];
}

/* Assigns `vec` to the first 3 elements of column vector 0 of the camera matrix
 */
static void camera_base_set_right(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[0]);
}

/* Returns column vector 1 of the camera matrix */
static vec4* camera_base_get_up(camera_base_t* this)
{
  return &this->_matrix[1];
}

/* Assigns `vec` to the first 3 elements of column vector 1 of the camera matrix
 */
static void camera_base_set_up(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[1]);
}

/* Returns column vector 2 of the camera matrix */
static vec4* camera_base_get_back(camera_base_t* this)
{
  return &this->_matrix[2];
}

/* Assigns `vec` to the first 3 elements of column vector 2 of the camera matrix
 */
static void camera_base_set_back(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[2]);
}

/* Returns column vector 3 of the camera matrix  */
static vec4* camera_base_get_position(camera_base_t* this)
{
  return &this->_matrix[3];
}

/* Assigns `vec` to the first 3 elements of column vector 3 of the camera matrix
 */
static void camera_base_set_position(camera_base_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_matrix[3]);
}

/* -------------------------------------------------------------------------- *
 * WASDCamera is a camera implementation that behaves similar to
 * first-person-shooter PC games.
 * -------------------------------------------------------------------------- */

typedef struct wasd_camera_t {
  /* The camera bass class */
  camera_base_t super;
  /* The camera absolute pitch angle */
  float pitch;
  /* The camera absolute yaw angle */
  float yaw;
  /* The movement veloicty */
  vec3 _velocity;
  /* Speed multiplier for camera movement */
  float movement_speed;
  /* Speed multiplier for camera rotation */
  float rotation_speed;
  /* Movement velocity drag coeffient [0 .. 1] */
  /* 0: Instantly stops moving                 */
  /* 1: Continues forever                      */
  float friction_coefficient;
} wasd_camera_t;

static void wasd_camera_recalculate_angles(wasd_camera_t* this, vec3 dir);
static mat4* wasd_camera_update(camera_base_t* this, float delta_time,
                                input_handler_t* input);

static void wasd_camera_init_defaults(wasd_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->pitch = 0.0f;
  this->yaw   = 0.0f;

  glm_vec3_zero(this->_velocity);

  this->movement_speed       = 10.0f;
  this->rotation_speed       = 1.0f;
  this->friction_coefficient = 0.01f;
}

static void wasd_camera_init_virtual_method_table(wasd_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->update = wasd_camera_update;
}

/* Construtor */
static void wasd_camera_init(wasd_camera_t* this,
                             /* The initial position of the camera */
                             vec3* iposition,
                             /* The initial target of the camera */
                             vec3* itarget)
{
  wasd_camera_init_defaults(this);

  camera_base_init(&this->super);
  wasd_camera_init_virtual_method_table(this);

  if ((iposition != NULL) || (itarget != NULL)) {
    vec3 position, target, forward;
    glm_vec3_copy((iposition == NULL) ? (vec3){0.0f, 0.0f, -5.0f} : *iposition,
                  position);
    glm_vec3_copy((itarget == NULL) ? (vec3){0.0f, 0.0f, 0.0f} : *itarget,
                  target);
    glm_vec3_sub(target, position, forward);
    glm_vec3_normalize(forward);
    wasd_camera_recalculate_angles(this, forward);
    camera_base_set_position(&this->super, position);
  }
}

/* Returns velocity vector */
static vec3* wasd_camera_get_velocity(wasd_camera_t* this)
{
  return &this->_velocity;
}

/* Assigns `vec` to the velocity vector */
static vec3* wasd_camera_set_velocity(wasd_camera_t* this, vec3 vec)
{
  glm_vec3_copy(vec, this->_velocity);
}

/* Returns the camera matrix */
static mat4* wasd_camera_get_matrix(wasd_camera_t* this)
{
  return camera_base_get_matrix(&this->super);
}

/* Assigns `mat` to the camera matrix, and recalcuates the camera angles */
static void wasd_camera_set_matrix(wasd_camera_t* this, mat4 mat)
{
  camera_base_set_matrix(&this->super, mat);
  wasd_camera_recalculate_angles(this, *camera_base_get_back(&this->super));
}

static mat4* wasd_camera_update(camera_base_t* this, float delta_time,
                                input_handler_t* input)
{
  return NULL;
}

/* Recalculates the yaw and pitch values from a directional vector */
static void wasd_camera_recalculate_angles(wasd_camera_t* this, vec3 dir)
{
  this->yaw   = atan2(dir[0], dir[2]);
  this->pitch = -asin(dir[1]);
}

/* -------------------------------------------------------------------------- *
 * ArcballCamera implements a basic orbiting camera around the world origin
 * -------------------------------------------------------------------------- */

typedef struct arcball_camera_t {
  /* The camera bass class */
  camera_base_t super;
  /* The camera distance from the target */
  float distance;
  /* The current angular velocity  */
  float angular_velocity;
  /* The current rotation axis */
  vec3 _axis;
  /* Speed multiplier for camera rotation */
  float rotation_speed;
  /* Speed multiplier for camera zoom */
  float zoom_speed;
  /* Movement velocity drag coeffient [0 .. 1] */
  /* 0: Instantly stops spinning               */
  /* 1: Spins forever                          */
  float friction_coefficient;
} arcball_camera_t;

static mat4* arcball_camera_update(camera_base_t* this, float delta_time,
                                   input_handler_t* input);
static void arcball_camera_recalcuate_right(arcball_camera_t* this);
static void arcball_camera_recalcuate_up(arcball_camera_t* this);

static void arcball_camera_init_defaults(arcball_camera_t* this)
{
  memset(this, 0, sizeof(*this));

  this->distance         = 0.0f;
  this->angular_velocity = 0.0f;

  glm_vec3_zero(this->_axis);

  this->rotation_speed       = 1.0f;
  this->zoom_speed           = 0.1f;
  this->friction_coefficient = 0.0001f;
}

static void arcball_camera_init_virtual_method_table(arcball_camera_t* this)
{
  camera_base_vtbl_t* vtbl = &this->super._vtbl;

  vtbl->update = arcball_camera_update;
}

/* Construtor */
static void arcball_camera_init(arcball_camera_t* this,
                                /* The initial position of the camera */
                                vec3* iposition)
{
  arcball_camera_init_virtual_method_table(this);

  camera_base_init(&this->super);
  arcball_camera_init_virtual_method_table(this);

  if (iposition != NULL) {
    camera_base_set_position(&this->super, *iposition);
    this->distance = glm_vec3_length(*camera_base_get_position(&this->super));
    glm_vec3_normalize_to(*camera_base_get_position(&this->super),
                          *camera_base_get_back(&this->super));
    arcball_camera_recalcuate_right(this);
    arcball_camera_recalcuate_up(this);
  }
}

static mat4* arcball_camera_update(camera_base_t* this, float delta_time,
                                   input_handler_t* input)
{
  return NULL;
}

/* Assigns `this.right` with the cross product of `this.up` and `this.back` */
static void arcball_camera_recalcuate_right(arcball_camera_t* this)
{
}

/* Assigns `this.up` with the cross product of `this.back` and `this.right` */
static void arcball_camera_recalcuate_up(arcball_camera_t* this)
{
}

/* --------------------------------------------------------------------------
 * Cameras example.
 * -------------------------------------------------------------------------- */

/* Cube mesh */
static cube_mesh_t cube_mesh = {0};

// Cube struct
static struct {
  WGPUBindGroup uniform_buffer_bind_group;
  WGPUBindGroupLayout bind_group_layout;
  struct {
    mat4 model_view_projection;
  } view_mtx;
} cube = {0};

/* Cube vertex buffer */
static wgpu_buffer_t vertices = {0};

/* Uniform buffer block object */
static wgpu_buffer_t uniform_buffer_vs = {0};

static struct {
  mat4 projection;
  mat4 view;
} view_matrices = {0};

// The pipeline layout
static WGPUPipelineLayout pipeline_layout = NULL;

/* Cube render pipeline */
static WGPURenderPipeline pipeline = NULL;

/* Render pass descriptor for frame buffer writes */
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Texture and sampler
static struct {
  texture_t cube;
  texture_t depth;
  WGPUSampler sampler;
} textures = {0};

/* Camera parameters */
typedef enum camera_type_t {
  CameraType_Arcball,
  Renderer_WASD,
} camera_type_t;

static struct {
  vec3 initial_camera_position;
  camera_type_t camera_type;
} example_parms = {
  .initial_camera_position = {3.0f, 2.0f, 5.0f},
  .camera_type             = CameraType_Arcball,
};

/* The camera types */
static camera_type_t old_camera_type = CameraType_Arcball;
static struct {
  arcball_camera_t arcball;
  wasd_camera_t wasd;
} cameras = {0};

/* GUI */
static const char* camera_type_names[2] = {"arcball", "WASD"};

/* Input handling */
static input_handler_t input_handler = {0};

// Other variables
static const char* example_title = "Cameras";
static bool prepared             = false;

static void initialize_cameras(void)
{
  arcball_camera_init(&cameras.arcball, &example_parms.initial_camera_position);
  wasd_camera_init(&cameras.wasd, &example_parms.initial_camera_position, NULL);
}

/* Prepare the cube geometry */
static void prepare_cube_mesh(void)
{
  cube_mesh_init(&cube_mesh);
}

/* Create a vertex buffer from the cube data. */
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  vertices = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Cube vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(cube_mesh.vertex_array),
                    .initial.data = cube_mesh.vertex_array,
                  });
}

static void prepare_texture(wgpu_context_t* wgpu_context)
{
  /* Create a depth/stencil texture for the color rendering pipeline */
  {
    WGPUExtent3D texture_extent = {
      .width              = wgpu_context->surface.width,
      .height             = wgpu_context->surface.height,
      .depthOrArrayLayers = 1,
    };
    WGPUTextureDescriptor texture_desc = {
      .size          = texture_extent,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage         = WGPUTextureUsage_RenderAttachment,
    };
    textures.depth.texture
      = wgpuDeviceCreateTexture(wgpu_context->device, &texture_desc);
    ASSERT(textures.depth.texture != NULL);

    // Create the texture view
    WGPUTextureViewDescriptor texture_view_dec = {
      .dimension       = WGPUTextureViewDimension_2D,
      .format          = texture_desc.format,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
    };
    textures.depth.view
      = wgpuTextureCreateView(textures.depth.texture, &texture_view_dec);
    ASSERT(textures.depth.view != NULL);
  }

  /* Cube texture */
  {
    const char* file = "textures/Di-3d.png";
    textures.cube    = wgpu_create_texture_from_file(
      wgpu_context, file,
      &(struct wgpu_texture_load_options_t){
           .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });
  }

  /* Create a sampler with linear filtering for smooth interpolation. */
  {
    textures.sampler = wgpuDeviceCreateSampler(
      wgpu_context->device, &(WGPUSamplerDescriptor){
                              .addressModeU  = WGPUAddressMode_ClampToEdge,
                              .addressModeV  = WGPUAddressMode_ClampToEdge,
                              .addressModeW  = WGPUAddressMode_ClampToEdge,
                              .minFilter     = WGPUFilterMode_Linear,
                              .magFilter     = WGPUFilterMode_Linear,
                              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                              .lodMinClamp   = 0.0f,
                              .lodMaxClamp   = 1.0f,
                              .maxAnisotropy = 1,
                            });
    ASSERT(textures.sampler != NULL);
  }
}

static void prepare_view_matrices(wgpu_context_t* wgpu_context)
{
  const float aspect_ratio
    = (float)wgpu_context->surface.width / (float)wgpu_context->surface.height;

  // Projection matrix
  glm_mat4_identity(view_matrices.projection);
  glm_perspective(PI2 / 5.0f, aspect_ratio, 1.0f, 100.0f,
                  view_matrices.projection);
}

static void prepare_uniform_buffers(wgpu_example_context_t* context)
{
  // Setup the view matrices for the camera
  prepare_view_matrices(context->wgpu_context);

  /* Uniform buffer */
  uniform_buffer_vs = wgpu_create_buffer(
    context->wgpu_context,
    &(wgpu_buffer_desc_t){
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(mat4), // 4x4 matrix
    });
  ASSERT(uniform_buffer_vs.buffer != NULL);
}

static void update_transformation_matrix(wgpu_example_context_t* context)
{
  const float now = context->frame.timestamp_millis / 1000.0f;

  // View matrix
  glm_mat4_identity(view_matrices.view);
  glm_translate(view_matrices.view, (vec3){0.0f, 0.0f, -4.0f});
  glm_rotate(view_matrices.view, 1.0f, (vec3){sin(now), cos(now), 0.0f});

  // Model view projection matrix
  glm_mat4_identity(cube.view_mtx.model_view_projection);
  glm_mat4_mul(view_matrices.projection, view_matrices.view,
               cube.view_mtx.model_view_projection);
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  // Update the model-view-projection matrix
  update_transformation_matrix(context);

  // Map uniform buffer and update it
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer_vs.buffer, 0,
                          &cube.view_mtx.model_view_projection,
                          uniform_buffer_vs.size);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Transform
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), // 4x4 matrix
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      // Sampler
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      // Texture view
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    }
  };
  cube.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(cube.bind_group_layout != NULL);

  // Create the pipeline layout that is used to generate the rendering pipelines
  // that are based on this bind group layout
  pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &cube.bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);
}

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding = 1,
      .sampler = textures.sampler,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding     = 2,
      .textureView = textures.cube.view,
    }
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = "cube_uniform_buffer_bind_group",
    .layout     = cube.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  cube.uniform_buffer_bind_group
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(cube.uniform_buffer_bind_group != NULL);
}

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_Back,
  };

  // Color target state
  WGPUBlendState blend_state              = wgpu_create_blend_state(true);
  WGPUColorTargetState color_target_state = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Depth stencil state
  // Enable depth testing so that the fragment closest to the camera
  // is rendered in front.
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  // Vertex buffer layout
  WGPU_VERTEX_BUFFER_LAYOUT(
    textured_cube, cube_mesh.vertex_size,
    // Attribute location 0: Position
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                       cube_mesh.position_offset),
    // Attribute location 1: UV
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, cube_mesh.uv_offset))

  // Vertex state
  WGPUVertexState vertex_state = wgpu_create_vertex_state(
    wgpu_context, &(wgpu_vertex_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Vertex shader WGSL
                      .label            = "cube_shader_wgsl",
                      .wgsl_code.source = cube_shader_wgsl,
                      .entry            = "vertex_main",
                    },
                    .buffer_count = 1,
                    .buffers = &textured_cube_vertex_buffer_layout,
                  });

  // Fragment state
  WGPUFragmentState fragment_state = wgpu_create_fragment_state(
    wgpu_context, &(wgpu_fragment_state_t){
                    .shader_desc = (wgpu_shader_desc_t){
                      // Fragment shader WGSL
                      .label            = "cube_shader_wgsl",
                      .wgsl_code.source = cube_shader_wgsl,
                      .entry            = "fragment_main",
                    },
                    .target_count = 1,
                    .targets = &color_target_state,
                  });

  // Multisample state
  WGPUMultisampleState multisample_state
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline using the specified states
  pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
                            .label        = "textured_cube_render_pipeline",
                            .primitive    = primitive_state,
                            .vertex       = vertex_state,
                            .fragment     = &fragment_state,
                            .depthStencil = &depth_stencil_state,
                            .multisample  = multisample_state,
                          });
  ASSERT(pipeline != NULL);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state.module);
}

static void setup_render_pass(void)
{
  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, // Assigned later
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.5f,
        .g = 0.5f,
        .b = 0.5f,
        .a = 1.0f,
      },
  };

  // Depth attachment
  render_pass.depth_stencil_attachment = (WGPURenderPassDepthStencilAttachment){
    .view            = textures.depth.view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = &render_pass.depth_stencil_attachment,
  };
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    input_handler_init(&input_handler);
    initialize_cameras();
    prepare_cube_mesh();
    prepare_vertex_buffer(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_texture(context->wgpu_context);
    prepare_uniform_buffers(context);
    setup_bind_groups(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass();
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Create render pass
  wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
    wgpu_context->cmd_enc, &render_pass.descriptor);

  // Bind the rendering pipeline
  wgpuRenderPassEncoderSetPipeline(wgpu_context->rpass_enc, pipeline);

  // Bind cube vertex buffer (contains position and colors)
  wgpuRenderPassEncoderSetVertexBuffer(wgpu_context->rpass_enc, 0,
                                       vertices.buffer, 0, WGPU_WHOLE_SIZE);

  // Set the bind group
  wgpuRenderPassEncoderSetBindGroup(wgpu_context->rpass_enc, 0,
                                    cube.uniform_buffer_bind_group, 0, 0);

  // Draw textured cube
  wgpuRenderPassEncoderDraw(wgpu_context->rpass_enc, cube_mesh.vertex_count, 1,
                            0, 0);

  // End render pass
  wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
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

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }

  update_mouse_state(&input_handler, context);

  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(BindGroup, cube.uniform_buffer_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, cube.bind_group_layout)
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  wgpu_destroy_texture(&textures.cube);
  wgpu_destroy_texture(&textures.depth);
  WGPU_RELEASE_RESOURCE(Sampler, textures.sampler)
}

void example_cameras(int argc, char* argv[])
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

static const char* cube_shader_wgsl = CODE(
  struct Uniforms {
   modelViewProjectionMatrix : mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var mySampler: sampler;
  @group(0) @binding(2) var myTexture: texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) Position : vec4f,
    @location(0) fragUV : vec2f,
  }

  @vertex
  fn vertex_main(
  @location(0) position : vec4f,
    @location(1) uv : vec2f
  ) -> VertexOutput {
    return VertexOutput(uniforms.modelViewProjectionMatrix * position, uv);
  }

  @fragment
  fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
    return textureSample(myTexture, mySampler, fragUV);
  }
);
// clang-format on
