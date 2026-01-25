#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Water Simulation
 *
 * This example demonstrates real-time water simulation using WebGPU.
 * It renders an interactive water surface with realistic physics, lighting,
 * and reflections. Based on Evan Wallace's WebGL Water demo.
 *
 * Features:
 * - Interactive water ripples (click/drag on water surface)
 * - Draggable sphere with physics (gravity, buoyancy)
 * - Orbit camera controls (drag on empty space)
 * - Dynamic lighting
 * - Pause/resume simulation (spacebar)
 * - Black background (no skybox)
 *
 * Ref:
 * https://github.com/jeantimex/webgpu-water
 * https://madebyevan.com/webgl-water/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders - Forward declarations
 * -------------------------------------------------------------------------- */

static const char* drop_shader_wgsl;
static const char* update_shader_wgsl;
static const char* normal_shader_wgsl;
static const char* sphere_move_shader_wgsl;
static const char* caustics_shader_wgsl;
static const char* water_surface_above_shader_wgsl;
static const char* water_surface_under_shader_wgsl;
static const char* pool_shader_wgsl;
static const char* sphere_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants and Enums
 * -------------------------------------------------------------------------- */

#define WATER_WIDTH 256
#define WATER_HEIGHT 256
#define WATER_SURFACE_DETAIL 200
#define CAUSTICS_SIZE 1024
#define SPHERE_DETAIL 10
#define SPHERE_RADIUS 0.25f

typedef enum interaction_mode_t {
  INTERACTION_MODE_NONE = 0,
  INTERACTION_MODE_ADD_DROPS,
  INTERACTION_MODE_MOVE_SPHERE,
  INTERACTION_MODE_ORBIT_CAMERA,
} interaction_mode_t;

/* -------------------------------------------------------------------------- *
 * Water simulation structures
 * -------------------------------------------------------------------------- */

typedef struct water_t {
  /* Ping-pong textures for double buffered simulation */
  wgpu_texture_t texture_a;
  wgpu_texture_t texture_b;
  wgpu_texture_t caustics_texture;
  WGPUSampler sampler;

  /* Simulation render pipelines (using render passes for fullscreen quad) */
  WGPURenderPipeline drop_pipeline;
  WGPURenderPipeline update_pipeline;
  WGPURenderPipeline normal_pipeline;
  WGPURenderPipeline sphere_move_pipeline;
  WGPURenderPipeline caustics_pipeline;

  /* Surface render pipelines */
  WGPURenderPipeline surface_above_pipeline;
  WGPURenderPipeline surface_under_pipeline;

  /* Uniform buffers for simulation */
  wgpu_buffer_t drop_uniform_buffer;
  wgpu_buffer_t update_uniform_buffer;
  wgpu_buffer_t sphere_move_uniform_buffer;

  /* Geometry for water surface */
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;

  /* Current active texture (A or B) */
  bool use_texture_a;
} water_t;

typedef struct pool_t {
  WGPURenderPipeline pipeline;
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;
} pool_t;

typedef struct sphere_t {
  WGPURenderPipeline pipeline;
  wgpu_buffer_t vertex_buffer;
  wgpu_buffer_t index_buffer;
  uint32_t index_count;
} sphere_t;

static struct {
  /* Core WebGPU resources */
  wgpu_texture_t tiles_texture;
  WGPUSampler tile_sampler;

  /* Water simulation */
  water_t water;

  /* Scene objects */
  pool_t pool;
  sphere_t sphere;

  /* Uniform buffers */
  wgpu_buffer_t uniform_buffer;        /* View-projection + eye position */
  wgpu_buffer_t light_uniform_buffer;  /* Light direction */
  wgpu_buffer_t sphere_uniform_buffer; /* Sphere position/radius */
  wgpu_buffer_t shadow_uniform_buffer; /* Shadow flags */

  /* Camera state */
  struct {
    float angle_x; /* Pitch */
    float angle_y; /* Yaw */
    mat4 view;
    mat4 projection;
    vec3 eye_position;
  } camera;

  /* Sphere physics */
  struct {
    vec3 center;
    vec3 old_center;
    vec3 velocity;
    float radius;
    bool physics_enabled;
  } sphere_physics;

  /* Lighting */
  struct {
    vec3 direction;
  } light;

  /* Interaction state */
  struct {
    interaction_mode_t mode;
    float old_x, old_y;
    vec3 prev_hit;
    vec3 plane_normal;
    bool mouse_down;
  } interaction;

  /* File loading */
  struct {
    uint8_t file_buffer[512 * 512 * 4];
    size_t loaded_data_size;
    WGPUBool tiles_loaded;
  } file_loading;

  /* GUI settings */
  struct {
    bool show_sphere;
    bool gravity_enabled;
    bool follow_camera;
    bool paused;
  } settings;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  uint64_t last_frame_time;
  WGPUBool initialized;
  WGPUBool resources_ready;
} state = {
  .camera = {
    .angle_x = -25.0f,
    .angle_y = -200.5f,
  },
  .sphere_physics = {
    .center = {-0.4f, -0.75f, 0.2f},
    .radius = SPHERE_RADIUS,
    .physics_enabled = false,
  },
  .light = {
    .direction = {2.0f, 2.0f, -1.0f},
  },
  .interaction = {
    .mode = INTERACTION_MODE_NONE,
    .mouse_down = false,
  },
  .settings = {
    .show_sphere = true,
    .gravity_enabled = false,
    .follow_camera = false,
    .paused = false,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f}, /* Black background */
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .initialized = false,
  .resources_ready = false,
};

/* -------------------------------------------------------------------------- *
 * Forward declarations
 * -------------------------------------------------------------------------- */

static void init_water(wgpu_context_t* wgpu_context);
static void init_pool(wgpu_context_t* wgpu_context);
static void init_sphere(wgpu_context_t* wgpu_context);
static void cleanup_water(void);
static void cleanup_pool(void);
static void cleanup_sphere(void);
static void water_add_drop(wgpu_context_t* wgpu_context, float x, float y,
                           float radius, float strength);
static void water_step_simulation(wgpu_context_t* wgpu_context);
static void water_update_normals(wgpu_context_t* wgpu_context);
static void water_move_sphere(wgpu_context_t* wgpu_context, vec3 old_center,
                              vec3 new_center, float radius);
static void water_update_caustics(wgpu_context_t* wgpu_context);
static void swap_water_textures(void);
static void render_gui(wgpu_context_t* wgpu_context);
static void update_shadow_uniforms(wgpu_context_t* wgpu_context);

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

static void update_camera_matrices(wgpu_context_t* wgpu_context)
{
  // Calculate aspect ratio
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  // Build projection matrix
  glm_perspective(glm_rad(45.0f), aspect, 0.01f, 100.0f,
                  state.camera.projection);

  // Build view matrix
  glm_mat4_identity(state.camera.view);
  glm_translate(state.camera.view,
                (vec3){0.0f, 0.0f, -4.0f}); // Camera distance
  glm_rotate_x(state.camera.view, glm_rad(-state.camera.angle_x),
               state.camera.view); // Pitch
  glm_rotate_y(state.camera.view, glm_rad(-state.camera.angle_y),
               state.camera.view); // Yaw
  glm_translate(state.camera.view,
                (vec3){0.0f, 0.5f, 0.0f}); // Look slightly above center

  // Calculate eye position from inverse view matrix
  mat4 inv_view;
  glm_mat4_inv(state.camera.view, inv_view);
  glm_vec3_copy((vec3){0.0f, 0.0f, 0.0f}, state.camera.eye_position);
  glm_mat4_mulv3(inv_view, state.camera.eye_position, 1.0f,
                 state.camera.eye_position);
}

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  update_camera_matrices(wgpu_context);

  // Calculate view-projection matrix
  mat4 view_projection;
  glm_mat4_mul(state.camera.projection, state.camera.view, view_projection);

  // Pack uniform data: mat4 (16 floats) + vec3 (3 floats) + padding (1 float) +
  // vec3 (3 floats) + padding (1 float)
  float uniform_data[24];
  memcpy(uniform_data, view_projection, sizeof(mat4));
  memcpy(&uniform_data[16], state.camera.eye_position, sizeof(vec3));
  uniform_data[19] = 0.0f; // padding

  // Normalize light direction and add to uniform data
  glm_vec3_normalize(state.light.direction);
  memcpy(&uniform_data[20], state.light.direction, sizeof(vec3));
  uniform_data[23] = 0.0f; // padding

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       uniform_data, sizeof(uniform_data));
}

static void update_light_uniforms(wgpu_context_t* wgpu_context)
{

  // Normalize light direction
  glm_vec3_normalize(state.light.direction);

  float light_data[4] = {
    state.light.direction[0], state.light.direction[1],
    state.light.direction[2],
    0.0f // padding
  };

  wgpuQueueWriteBuffer(wgpu_context->queue, state.light_uniform_buffer.buffer,
                       0, light_data, sizeof(light_data));
}

static void update_sphere_uniforms(wgpu_context_t* wgpu_context)
{

  float sphere_data[4]
    = {state.sphere_physics.center[0], state.sphere_physics.center[1],
       state.sphere_physics.center[2], state.sphere_physics.radius};

  wgpuQueueWriteBuffer(wgpu_context->queue, state.sphere_uniform_buffer.buffer,
                       0, sphere_data, sizeof(sphere_data));
}

static void update_shadow_uniforms(wgpu_context_t* wgpu_context)
{

  float shadow_data[4] = {
    1.0f,                                     // rim lighting enabled
    state.settings.show_sphere ? 1.0f : 0.0f, // sphere shadows
    1.0f,                                     // ambient occlusion enabled
    0.0f                                      // padding
  };

  wgpuQueueWriteBuffer(wgpu_context->queue, state.shadow_uniform_buffer.buffer,
                       0, shadow_data, sizeof(shadow_data));
}

/* -------------------------------------------------------------------------- *
 * Resource initialization
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Main uniform buffer: view-projection matrix + eye position + light
   * direction */
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Water - Main uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 96, /* 16*4 + 3*4 + 4 + 3*4 + 4 bytes */
                  });

  /* Light direction buffer */
  state.light_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Water - Light uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 16, /* vec3 + padding */
                  });

  /* Sphere position/radius buffer */
  state.sphere_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Water - Sphere uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 16, /* vec3 + float */
                  });

  /* Shadow flags buffer */
  state.shadow_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Water - Shadow uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 16, /* 3 floats + padding */
                  });
}

/* -------------------------------------------------------------------------- *
 * File loading callbacks
 * -------------------------------------------------------------------------- */

static void tiles_texture_loaded(const sfetch_response_t* response)
{
  if (response->fetched) {
    if (response->data.size <= sizeof(state.file_loading.file_buffer)) {
      memcpy(state.file_loading.file_buffer, response->data.ptr,
             response->data.size);
      state.file_loading.loaded_data_size = response->data.size;
      state.file_loading.tiles_loaded     = true;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Main functions
 * -------------------------------------------------------------------------- */

static int example_init(wgpu_context_t* wgpu_context)
{
  /* Initialize sokol libraries */
  sfetch_setup(&(sfetch_desc_t){
    .num_channels = 1,
    .num_lanes    = 4,
    .logger.func  = slog_func,
  });
  stm_setup();

  /* Initialize ImGui overlay */
  imgui_overlay_init(wgpu_context);

  /* Initialize uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Normalize light direction and update uniforms */
  glm_vec3_normalize(state.light.direction);
  update_uniforms(wgpu_context);
  update_light_uniforms(wgpu_context);
  update_sphere_uniforms(wgpu_context);
  update_shadow_uniforms(wgpu_context);

  /* Initialize sphere physics state */
  glm_vec3_copy(state.sphere_physics.center, state.sphere_physics.old_center);
  glm_vec3_zero(state.sphere_physics.velocity);

  /* Start loading tiles texture */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/tiles.jpg",
    .callback = tiles_texture_loaded,
    .buffer   = SFETCH_RANGE(state.file_loading.file_buffer),
  });

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static void example_cleanup(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown ImGui overlay */
  imgui_overlay_shutdown();

  /* Shutdown sokol */
  sfetch_shutdown();

  /* Cleanup scene objects */
  if (state.resources_ready) {
    cleanup_water();
    cleanup_pool();
    cleanup_sphere();
  }

  /* Cleanup buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.light_uniform_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_uniform_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.shadow_uniform_buffer.buffer);

  /* Cleanup textures */
  wgpu_destroy_texture(&state.tiles_texture);

  /* Cleanup samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.tile_sampler);
}

static int example_frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_SUCCESS;
  }

  /* Process sokol-fetch requests */
  sfetch_dowork();

  /* Create tiles texture if data is loaded but texture not yet created */
  if (state.file_loading.tiles_loaded && !state.tiles_texture.handle) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load_from_memory(
      state.file_loading.file_buffer, (int)state.file_loading.loaded_data_size,
      &width, &height, &channels, STBI_rgb_alpha);

    if (pixels) {
      state.tiles_texture = wgpu_create_texture(
        wgpu_context,
        &(wgpu_texture_desc_t){
          .extent = {(uint32_t)width, (uint32_t)height, 1},
          .format = WGPUTextureFormat_RGBA8Unorm,
          .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
          .pixels = {pixels, (uint32_t)(width * height * 4)},
        });

      state.tile_sampler = wgpuDeviceCreateSampler(
        wgpu_context->device, &(WGPUSamplerDescriptor){
                                .label         = STRVIEW("Tile sampler"),
                                .magFilter     = WGPUFilterMode_Linear,
                                .minFilter     = WGPUFilterMode_Linear,
                                .addressModeU  = WGPUAddressMode_Repeat,
                                .addressModeV  = WGPUAddressMode_Repeat,
                                .maxAnisotropy = 1,
                              });

      stbi_image_free(pixels);
    }
    else {
      /* Create fallback texture if loading fails */
      state.tiles_texture = wgpu_create_color_bars_texture(
        wgpu_context,
        &(wgpu_texture_desc_t){
          .extent = {256, 256, 1},
          .format = WGPUTextureFormat_RGBA8Unorm,
          .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        });
      state.tile_sampler = wgpuDeviceCreateSampler(
        wgpu_context->device, &(WGPUSamplerDescriptor){
                                .label         = STRVIEW("Tile sampler"),
                                .magFilter     = WGPUFilterMode_Linear,
                                .minFilter     = WGPUFilterMode_Linear,
                                .addressModeU  = WGPUAddressMode_Repeat,
                                .addressModeV  = WGPUAddressMode_Repeat,
                                .maxAnisotropy = 1,
                              });
    }
  }

  /* Initialize scene objects after texture is ready */
  if (!state.resources_ready && state.tiles_texture.handle) {
    init_water(wgpu_context);
    init_pool(wgpu_context);
    init_sphere(wgpu_context);

    /* Add initial random ripples */
    for (int i = 0; i < 20; i++) {
      float x        = random_float() * 2.0f - 1.0f;
      float y        = random_float() * 2.0f - 1.0f;
      float strength = (i & 1) ? 0.01f : -0.01f;
      water_add_drop(wgpu_context, x, y, 0.03f, strength);
    }

    state.resources_ready = true;
  }

  if (!state.resources_ready) {
    return EXIT_SUCCESS;
  }

  /* Calculate delta time */
  uint64_t current_time = stm_now();
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;
  if (delta_time > 1.0f) {
    delta_time = 1.0f; /* Cap delta time */
  }

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI controls */
  render_gui(wgpu_context);

  /* Update light direction if follow camera is enabled */
  if (state.settings.follow_camera) {
    float yaw_rad            = glm_rad(90.0f - state.camera.angle_y);
    float pitch_rad          = glm_rad(-state.camera.angle_x);
    state.light.direction[0] = cosf(yaw_rad) * cosf(pitch_rad);
    state.light.direction[1] = sinf(pitch_rad);
    state.light.direction[2] = sinf(yaw_rad) * cosf(pitch_rad);
    glm_vec3_normalize(state.light.direction);
    update_light_uniforms(wgpu_context);
  }

  if (!state.settings.paused) {
    /* Update sphere physics */
    if (state.interaction.mode != INTERACTION_MODE_MOVE_SPHERE
        && state.sphere_physics.physics_enabled) {
      /* Calculate buoyancy */
      float percent_underwater
        = fmaxf(0.0f, fminf(1.0f, (state.sphere_physics.radius
                                   - state.sphere_physics.center[1])
                                    / (2.0f * state.sphere_physics.radius)));

      /* Apply gravity reduced by buoyancy */
      vec3 gravity = {0.0f, -4.0f, 0.0f};
      vec3 buoyancy_force;
      glm_vec3_scale(gravity,
                     delta_time - 1.1f * delta_time * percent_underwater,
                     buoyancy_force);
      glm_vec3_add(state.sphere_physics.velocity, buoyancy_force,
                   state.sphere_physics.velocity);

      /* Water drag */
      if (percent_underwater > 0.0f) {
        float velocity_mag = glm_vec3_norm(state.sphere_physics.velocity);
        if (velocity_mag > 0.001f) {
          vec3 velocity_unit;
          glm_vec3_normalize_to(state.sphere_physics.velocity, velocity_unit);
          float drag_factor
            = percent_underwater * delta_time * velocity_mag * velocity_mag;
          vec3 drag;
          glm_vec3_scale(velocity_unit, drag_factor, drag);
          glm_vec3_sub(state.sphere_physics.velocity, drag,
                       state.sphere_physics.velocity);
        }
      }

      /* Update position */
      vec3 displacement;
      glm_vec3_scale(state.sphere_physics.velocity, delta_time, displacement);
      glm_vec3_add(state.sphere_physics.center, displacement,
                   state.sphere_physics.center);

      /* Floor collision */
      if (state.sphere_physics.center[1] < state.sphere_physics.radius - 1.0f) {
        state.sphere_physics.center[1] = state.sphere_physics.radius - 1.0f;
        state.sphere_physics.velocity[1]
          = fabsf(state.sphere_physics.velocity[1]) * 0.7f;
      }

      update_sphere_uniforms(wgpu_context);
    }

    /* Update water displacement from sphere movement */
    if (state.settings.show_sphere) {
      water_move_sphere(wgpu_context, state.sphere_physics.old_center,
                        state.sphere_physics.center,
                        state.sphere_physics.radius);
    }
    glm_vec3_copy(state.sphere_physics.center, state.sphere_physics.old_center);

    /* Run water simulation (twice per frame for smoother waves) */
    water_step_simulation(wgpu_context);
    water_step_simulation(wgpu_context);
    water_update_normals(wgpu_context);
    water_update_caustics(wgpu_context);
  }

  /* Update camera uniforms */
  update_uniforms(wgpu_context);

  /* Render frame */
  WGPUCommandEncoder cmd_encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("Water - Command encoder"),
                          });

  /* Update render pass attachments */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(
    cmd_encoder, &state.render_pass_descriptor);

  /* Render pool walls */
  if (state.pool.pipeline) {
    WGPUTextureView water_view = state.water.use_texture_a ?
                                   state.water.texture_a.view :
                                   state.water.texture_b.view;
    WGPUBindGroup pool_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Pool bind group"),
        .layout     = wgpuRenderPipelineGetBindGroupLayout(state.pool.pipeline, 0),
        .entryCount = 9,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .buffer = state.uniform_buffer.buffer, .size = 80},
          {.binding = 1, .sampler = state.tile_sampler},
          {.binding = 2, .textureView = state.tiles_texture.view},
          {.binding = 3, .buffer = state.light_uniform_buffer.buffer, .size = 16},
          {.binding = 4, .buffer = state.sphere_uniform_buffer.buffer, .size = 16},
          {.binding = 5, .sampler = state.water.sampler},
          {.binding = 6, .textureView = water_view},
          {.binding = 7, .textureView = state.water.caustics_texture.view},
          {.binding = 8, .buffer = state.shadow_uniform_buffer.buffer, .size = 16},
        },
      });

    wgpuRenderPassEncoderSetPipeline(render_pass, state.pool.pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, pool_bind_group, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 0, state.pool.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      render_pass, state.pool.index_buffer.buffer, WGPUIndexFormat_Uint32, 0,
      WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(render_pass, state.pool.index_count, 1, 0,
                                     0, 0);
    wgpuBindGroupRelease(pool_bind_group);
  }

  /* Render sphere if visible */
  if (state.settings.show_sphere && state.sphere.pipeline) {
    WGPUTextureView water_view = state.water.use_texture_a ?
                                   state.water.texture_a.view :
                                   state.water.texture_b.view;
    WGPUBindGroup sphere_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Sphere bind group"),
        .layout     = wgpuRenderPipelineGetBindGroupLayout(state.sphere.pipeline, 0),
        .entryCount = 6,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .buffer = state.uniform_buffer.buffer, .size = 80},
          {.binding = 1, .buffer = state.light_uniform_buffer.buffer, .size = 16},
          {.binding = 2, .buffer = state.sphere_uniform_buffer.buffer, .size = 16},
          {.binding = 3, .sampler = state.water.sampler},
          {.binding = 4, .textureView = water_view},
          {.binding = 5, .textureView = state.water.caustics_texture.view},
        },
      });

    wgpuRenderPassEncoderSetPipeline(render_pass, state.sphere.pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, sphere_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 0, state.sphere.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      render_pass, state.sphere.index_buffer.buffer, WGPUIndexFormat_Uint32, 0,
      WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(render_pass, state.sphere.index_count, 1,
                                     0, 0, 0);
    wgpuBindGroupRelease(sphere_bind_group);
  }

  /* Render water surface (above and below) */
  if (state.water.surface_above_pipeline) {
    WGPUTextureView water_view = state.water.use_texture_a ?
                                   state.water.texture_a.view :
                                   state.water.texture_b.view;
    WGPUBindGroup water_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Water surface bind group"),
        .layout     = wgpuRenderPipelineGetBindGroupLayout(state.water.surface_above_pipeline, 0),
        .entryCount = 9,
        .entries = (WGPUBindGroupEntry[]){
          {.binding = 0, .buffer = state.uniform_buffer.buffer, .size = 80},
          {.binding = 1, .buffer = state.light_uniform_buffer.buffer, .size = 16},
          {.binding = 2, .buffer = state.sphere_uniform_buffer.buffer, .size = 16},
          {.binding = 3, .sampler = state.tile_sampler},
          {.binding = 4, .textureView = state.tiles_texture.view},
          {.binding = 5, .sampler = state.water.sampler},
          {.binding = 6, .textureView = water_view},
          {.binding = 7, .textureView = state.water.caustics_texture.view},
          {.binding = 8, .buffer = state.shadow_uniform_buffer.buffer, .size = 16},
        },
      });

    /* Render from above (cull front faces) */
    wgpuRenderPassEncoderSetPipeline(render_pass,
                                     state.water.surface_above_pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, water_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      render_pass, 0, state.water.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(
      render_pass, state.water.index_buffer.buffer, WGPUIndexFormat_Uint32, 0,
      WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(render_pass, state.water.index_count, 1, 0,
                                     0, 0);

    /* Render from below (cull back faces) */
    if (state.water.surface_under_pipeline) {
      WGPUBindGroup under_bind_group = wgpuDeviceCreateBindGroup(
        wgpu_context->device,
        &(WGPUBindGroupDescriptor){
          .label      = STRVIEW("Water surface under bind group"),
          .layout     = wgpuRenderPipelineGetBindGroupLayout(state.water.surface_under_pipeline, 0),
          .entryCount = 9,
          .entries = (WGPUBindGroupEntry[]){
            {.binding = 0, .buffer = state.uniform_buffer.buffer, .size = 80},
            {.binding = 1, .buffer = state.light_uniform_buffer.buffer, .size = 16},
            {.binding = 2, .buffer = state.sphere_uniform_buffer.buffer, .size = 16},
            {.binding = 3, .sampler = state.tile_sampler},
            {.binding = 4, .textureView = state.tiles_texture.view},
            {.binding = 5, .sampler = state.water.sampler},
            {.binding = 6, .textureView = water_view},
            {.binding = 7, .textureView = state.water.caustics_texture.view},
            {.binding = 8, .buffer = state.shadow_uniform_buffer.buffer, .size = 16},
          },
        });

      wgpuRenderPassEncoderSetPipeline(render_pass,
                                       state.water.surface_under_pipeline);
      wgpuRenderPassEncoderSetBindGroup(render_pass, 0, under_bind_group, 0,
                                        NULL);
      wgpuRenderPassEncoderDrawIndexed(render_pass, state.water.index_count, 1,
                                       0, 0, 0);
      wgpuBindGroupRelease(under_bind_group);
    }

    wgpuBindGroupRelease(water_bind_group);
  }

  wgpuRenderPassEncoderEnd(render_pass);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(
    cmd_encoder, &(WGPUCommandBufferDescriptor){
                   .label = STRVIEW("Water - Command buffer"),
                 });

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, render_pass);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * GUI rendering
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Water Simulation Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Rendering settings */
  if (igCollapsingHeader_TreeNodeFlags("Rendering",
                                       ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igCheckbox("Show Sphere", &state.settings.show_sphere)) {
      /* Update shadow buffer when sphere visibility changes */
      update_shadow_uniforms(wgpu_context);
    }
    igCheckbox("Light Follows Camera", &state.settings.follow_camera);
  }

  /* Physics settings */
  if (igCollapsingHeader_TreeNodeFlags("Physics",
                                       ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igCheckbox("Enable Gravity", &state.settings.gravity_enabled)) {
      state.sphere_physics.physics_enabled = state.settings.gravity_enabled;
    }
    igCheckbox("Pause Simulation", &state.settings.paused);
  }

  /* Controls help */
  if (igCollapsingHeader_TreeNodeFlags("Controls", ImGuiTreeNodeFlags_None)) {
    igText("Mouse drag on water: Add ripples");
    igText("Mouse drag elsewhere: Orbit camera");
    igText("G key: Toggle gravity");
    igText("Space: Pause/resume");
  }

  /* Camera info */
  if (igCollapsingHeader_TreeNodeFlags("Camera Info",
                                       ImGuiTreeNodeFlags_None)) {
    igText("Pitch: %.1f", state.camera.angle_x);
    igText("Yaw: %.1f", state.camera.angle_y);
  }

  igEnd();
}

static void example_on_input_event(wgpu_context_t* wgpu_context,
                                   const input_event_t* event)
{
  /* Pass input events to ImGui */
  imgui_overlay_handle_input(wgpu_context, event);

  /* Check if ImGui wants to capture input */
  ImGuiIO* io            = igGetIO();
  bool imgui_wants_input = io->WantCaptureMouse || io->WantCaptureKeyboard;

  /* Handle resize events always */
  if (event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Framework handles depth stencil recreation, just update camera */
    update_camera_matrices(wgpu_context);
    return;
  }

  /* Skip scene interaction if ImGui wants input */
  if (imgui_wants_input) {
    return;
  }

  if (event->type == INPUT_EVENT_TYPE_KEY_DOWN) {
    switch (event->key_code) {
      case KEY_G:
        /* Toggle gravity */
        state.settings.gravity_enabled       = !state.settings.gravity_enabled;
        state.sphere_physics.physics_enabled = state.settings.gravity_enabled;
        break;
      case KEY_SPACE:
        /* Toggle pause */
        state.settings.paused = !state.settings.paused;
        break;
      default:
        break;
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_MOUSE_DOWN) {
    if (event->mouse_button == BUTTON_LEFT) {
      state.interaction.mouse_down = true;
      state.interaction.old_x      = event->mouse_x;
      state.interaction.old_y      = event->mouse_y;

      /* Determine interaction mode based on where user clicked */
      /* For now, default to orbit camera - sphere/water detection can be added
       */
      state.interaction.mode = INTERACTION_MODE_ORBIT_CAMERA;

      /* Check if clicking on water surface (simple check) */
      /* If y coordinate is in lower part of screen, assume water click */
      if (event->mouse_y > (float)wgpu_context->height * 0.3f) {
        state.interaction.mode = INTERACTION_MODE_ADD_DROPS;
        /* Convert screen coords to water coords */
        float x = (event->mouse_x / (float)wgpu_context->width) * 2.0f - 1.0f;
        float y = (event->mouse_y / (float)wgpu_context->height) * 2.0f - 1.0f;
        water_add_drop(wgpu_context, x, y, 0.03f, 0.01f);
      }
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_MOUSE_UP) {
    if (event->mouse_button == BUTTON_LEFT) {
      state.interaction.mouse_down = false;
      state.interaction.mode       = INTERACTION_MODE_NONE;
    }
  }
  else if (event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (state.interaction.mouse_down) {
      if (state.interaction.mode == INTERACTION_MODE_ORBIT_CAMERA) {
        /* Rotate camera */
        state.camera.angle_y -= event->mouse_dx * 0.5f;
        state.camera.angle_x -= event->mouse_dy * 0.5f;
        state.camera.angle_x
          = fmaxf(-89.999f, fminf(89.999f, state.camera.angle_x));
      }
      else if (state.interaction.mode == INTERACTION_MODE_ADD_DROPS) {
        /* Add ripples while dragging */
        float x = (event->mouse_x / (float)wgpu_context->width) * 2.0f - 1.0f;
        float y = (event->mouse_y / (float)wgpu_context->height) * 2.0f - 1.0f;
        if (fabsf(x) < 1.0f && fabsf(y) < 1.0f) {
          water_add_drop(wgpu_context, x, y, 0.03f, 0.01f);
        }
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Water simulation implementation
 * -------------------------------------------------------------------------- */

static void swap_water_textures(void)
{
  state.water.use_texture_a = !state.water.use_texture_a;
}

static void create_water_textures(wgpu_context_t* wgpu_context)
{
  /* Create ping-pong textures for simulation */
  WGPUTextureFormat format = WGPUTextureFormat_RGBA16Float;

  state.water.texture_a = wgpu_create_texture(
    wgpu_context, &(wgpu_texture_desc_t){
                    .extent = {WATER_WIDTH, WATER_HEIGHT, 1},
                    .format = format,
                    .usage  = WGPUTextureUsage_TextureBinding
                             | WGPUTextureUsage_RenderAttachment,
                  });

  state.water.texture_b = wgpu_create_texture(
    wgpu_context, &(wgpu_texture_desc_t){
                    .extent = {WATER_WIDTH, WATER_HEIGHT, 1},
                    .format = format,
                    .usage  = WGPUTextureUsage_TextureBinding
                             | WGPUTextureUsage_RenderAttachment,
                  });

  /* Caustics texture (higher resolution for detail) */
  state.water.caustics_texture = wgpu_create_texture(
    wgpu_context, &(wgpu_texture_desc_t){
                    .extent = {CAUSTICS_SIZE, CAUSTICS_SIZE, 1},
                    .format = WGPUTextureFormat_RGBA8Unorm,
                    .usage  = WGPUTextureUsage_TextureBinding
                             | WGPUTextureUsage_RenderAttachment,
                  });

  /* Create sampler */
  state.water.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Water sampler"),
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .maxAnisotropy = 1,
                          });
}

static void create_water_surface_mesh(wgpu_context_t* wgpu_context)
{
  const int detail       = WATER_SURFACE_DETAIL;
  const int vertex_count = (detail + 1) * (detail + 1) * 3;
  const int index_count  = detail * detail * 6;

  float* positions  = malloc(vertex_count * sizeof(float));
  uint32_t* indices = malloc(index_count * sizeof(uint32_t));

  if (!positions || !indices) {
    free(positions);
    free(indices);
    return;
  }

  /* Generate vertex grid */
  int pos_idx = 0;
  for (int z = 0; z <= detail; z++) {
    float t = (float)z / detail;
    for (int x = 0; x <= detail; x++) {
      float s              = (float)x / detail;
      positions[pos_idx++] = 2.0f * s - 1.0f; /* X */
      positions[pos_idx++] = 2.0f * t - 1.0f; /* Y (mapped to Z in shader) */
      positions[pos_idx++] = 0.0f;            /* Z (height from texture) */
    }
  }

  /* Generate triangle indices */
  int idx_idx = 0;
  for (int z = 0; z < detail; z++) {
    for (int x = 0; x < detail; x++) {
      int i = x + z * (detail + 1);
      /* First triangle */
      indices[idx_idx++] = i;
      indices[idx_idx++] = i + 1;
      indices[idx_idx++] = i + detail + 1;
      /* Second triangle */
      indices[idx_idx++] = i + detail + 1;
      indices[idx_idx++] = i + 1;
      indices[idx_idx++] = i + detail + 2;
    }
  }

  state.water.index_count = index_count;

  /* Create vertex buffer */
  state.water.vertex_buffer
    = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                         .label = "Water surface vertices",
                                         .usage = WGPUBufferUsage_Vertex,
                                         .size  = vertex_count * sizeof(float),
                                         .initial.data = positions,
                                       });

  /* Create index buffer */
  state.water.index_buffer
    = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                         .label = "Water surface indices",
                                         .usage = WGPUBufferUsage_Index,
                                         .size = index_count * sizeof(uint32_t),
                                         .initial.data = indices,
                                       });

  free(positions);
  free(indices);
}

static void create_water_simulation_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat format = WGPUTextureFormat_RGBA16Float;

  /* Drop pipeline */
  WGPUShaderModule drop_module
    = wgpu_create_shader_module(wgpu_context->device, drop_shader_wgsl);

  state.water.drop_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Drop pipeline"),
      .layout = NULL,
      .vertex = {
        .module     = drop_module,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = drop_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(drop_module);

  /* Update pipeline */
  WGPUShaderModule update_module
    = wgpu_create_shader_module(wgpu_context->device, update_shader_wgsl);

  state.water.update_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Update pipeline"),
      .layout = NULL,
      .vertex = {
        .module     = update_module,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = update_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(update_module);

  /* Normal pipeline */
  WGPUShaderModule normal_module
    = wgpu_create_shader_module(wgpu_context->device, normal_shader_wgsl);

  state.water.normal_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Normal pipeline"),
      .layout = NULL,
      .vertex = {
        .module     = normal_module,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = normal_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(normal_module);

  /* Sphere move pipeline */
  WGPUShaderModule sphere_move_module
    = wgpu_create_shader_module(wgpu_context->device, sphere_move_shader_wgsl);

  state.water.sphere_move_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Sphere move pipeline"),
      .layout = NULL,
      .vertex = {
        .module     = sphere_move_module,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = sphere_move_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(sphere_move_module);

  /* Create uniform buffers for simulation */
  state.water.drop_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Drop uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 32, /* center(2) + radius + strength + padding */
                  });

  state.water.update_uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Update uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = 16, /* delta(2) + padding */
                  });

  state.water.sphere_move_uniform_buffer = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label = "Sphere move uniform buffer",
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = 32, /* old_center(3) + radius + new_center(3) + padding */
    });
}

static void create_water_surface_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat canvas_format = wgpu_context->render_format;

  /* Water surface above pipeline */
  WGPUShaderModule surface_above_module = wgpu_create_shader_module(
    wgpu_context->device, water_surface_above_shader_wgsl);

  state.water.surface_above_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Water surface above pipeline"),
      .layout = NULL,
      .vertex = {
        .module      = surface_above_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 3 * sizeof(float),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &(WGPUVertexAttribute){
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = 0,
            .shaderLocation = 0,
          },
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = surface_above_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = canvas_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_Front,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = true,
        .depthCompare        = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(surface_above_module);

  /* Water surface under pipeline */
  WGPUShaderModule surface_under_module = wgpu_create_shader_module(
    wgpu_context->device, water_surface_under_shader_wgsl);

  state.water.surface_under_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Water surface under pipeline"),
      .layout = NULL,
      .vertex = {
        .module      = surface_under_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 3 * sizeof(float),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &(WGPUVertexAttribute){
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = 0,
            .shaderLocation = 0,
          },
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = surface_under_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = canvas_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = true,
        .depthCompare        = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(surface_under_module);

  /* Caustics pipeline */
  WGPUShaderModule caustics_module
    = wgpu_create_shader_module(wgpu_context->device, caustics_shader_wgsl);

  state.water.caustics_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Caustics pipeline"),
      .layout = NULL,
      .vertex = {
        .module      = caustics_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 3 * sizeof(float),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &(WGPUVertexAttribute){
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = 0,
            .shaderLocation = 0,
          },
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = caustics_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
          .blend     = &(WGPUBlendState){
            .color = {
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
            },
            .alpha = {
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
            },
          },
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(caustics_module);
}

static void init_water(wgpu_context_t* wgpu_context)
{
  create_water_textures(wgpu_context);
  create_water_surface_mesh(wgpu_context);
  create_water_simulation_pipelines(wgpu_context);
  create_water_surface_pipelines(wgpu_context);
  state.water.use_texture_a = true;
}

/* Water simulation helper - runs a simulation pass */
static void run_simulation_pass(wgpu_context_t* wgpu_context,
                                WGPURenderPipeline pipeline,
                                WGPUBuffer uniform_buffer,
                                uint32_t uniform_size)
{
  WGPUTextureView input_view  = state.water.use_texture_a ?
                                  state.water.texture_a.view :
                                  state.water.texture_b.view;
  WGPUTextureView output_view = state.water.use_texture_a ?
                                  state.water.texture_b.view :
                                  state.water.texture_a.view;

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Simulation bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0),
      .entryCount = 3,
      .entries = (WGPUBindGroupEntry[]){
        {.binding = 0, .textureView = input_view},
        {.binding = 1, .sampler = state.water.sampler},
        {.binding = 2, .buffer = uniform_buffer, .size = uniform_size},
      },
    });

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){0});

  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder,
    &(WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = &(WGPURenderPassColorAttachment){
        .view       = output_view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
    });

  wgpuRenderPassEncoderSetPipeline(pass, pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
  wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);
  wgpuBindGroupRelease(bind_group);

  swap_water_textures();
}

static void water_add_drop(wgpu_context_t* wgpu_context, float x, float y,
                           float radius, float strength)
{
  if (!state.water.drop_pipeline)
    return;

  float uniform_data[8] = {x, y, radius, strength, 0, 0, 0, 0};
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.water.drop_uniform_buffer.buffer, 0, uniform_data,
                       sizeof(uniform_data));

  run_simulation_pass(wgpu_context, state.water.drop_pipeline,
                      state.water.drop_uniform_buffer.buffer, 32);
}

static void water_step_simulation(wgpu_context_t* wgpu_context)
{
  if (!state.water.update_pipeline)
    return;

  float uniform_data[4] = {1.0f / WATER_WIDTH, 1.0f / WATER_HEIGHT, 0, 0};
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.water.update_uniform_buffer.buffer, 0,
                       uniform_data, sizeof(uniform_data));

  run_simulation_pass(wgpu_context, state.water.update_pipeline,
                      state.water.update_uniform_buffer.buffer, 16);
}

static void water_update_normals(wgpu_context_t* wgpu_context)
{
  if (!state.water.normal_pipeline)
    return;

  float uniform_data[4] = {1.0f / WATER_WIDTH, 1.0f / WATER_HEIGHT, 0, 0};
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.water.update_uniform_buffer.buffer, 0,
                       uniform_data, sizeof(uniform_data));

  run_simulation_pass(wgpu_context, state.water.normal_pipeline,
                      state.water.update_uniform_buffer.buffer, 16);
}

static void water_move_sphere(wgpu_context_t* wgpu_context, vec3 old_center,
                              vec3 new_center, float radius)
{
  if (!state.water.sphere_move_pipeline)
    return;

  float uniform_data[8] = {old_center[0], old_center[1], old_center[2], radius,
                           new_center[0], new_center[1], new_center[2], 0};
  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.water.sphere_move_uniform_buffer.buffer, 0,
                       uniform_data, sizeof(uniform_data));

  run_simulation_pass(wgpu_context, state.water.sphere_move_pipeline,
                      state.water.sphere_move_uniform_buffer.buffer, 32);
}

static void water_update_caustics(wgpu_context_t* wgpu_context)
{
  if (!state.water.caustics_pipeline)
    return;

  WGPUTextureView water_view = state.water.use_texture_a ?
                                 state.water.texture_a.view :
                                 state.water.texture_b.view;

  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    &(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Caustics bind group"),
      .layout     = wgpuRenderPipelineGetBindGroupLayout(state.water.caustics_pipeline, 0),
      .entryCount = 5,
      .entries = (WGPUBindGroupEntry[]){
        {.binding = 0, .buffer = state.light_uniform_buffer.buffer, .size = 16},
        {.binding = 1, .buffer = state.sphere_uniform_buffer.buffer, .size = 16},
        {.binding = 2, .sampler = state.water.sampler},
        {.binding = 3, .textureView = water_view},
        {.binding = 4, .buffer = state.shadow_uniform_buffer.buffer, .size = 16},
      },
    });

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){0});

  WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
    encoder,
    &(WGPURenderPassDescriptor){
      .colorAttachmentCount = 1,
      .colorAttachments     = &(WGPURenderPassColorAttachment){
        .view       = state.water.caustics_texture.view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
    });

  wgpuRenderPassEncoderSetPipeline(pass, state.water.caustics_pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(
    pass, 0, state.water.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass, state.water.index_buffer.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(pass, state.water.index_count, 1, 0, 0, 0);
  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);
  wgpuBindGroupRelease(bind_group);
}

static void init_pool(wgpu_context_t* wgpu_context)
{
  /* Create cube geometry without top face (open pool) */
  /* Using octant picking technique from TypeScript version */
  float positions[5 * 4 * 3]; /* 5 faces, 4 vertices each, 3 components */
  uint32_t indices[5 * 6];    /* 5 faces, 6 indices each */

  int v_idx = 0;
  int i_idx = 0;

  /* Cube face definitions: [v0, v1, v2, v3] indices into octants */
  const int cube_faces[5][4] = {
    {0, 4, 2, 6}, /* -x (left wall) */
    {1, 3, 5, 7}, /* +x (right wall) */
    {2, 6, 3, 7}, /* +y (floor) */
    {0, 2, 1, 3}, /* -z (front wall) */
    {4, 5, 6, 7}, /* +z (back wall) */
  };

  for (int face = 0; face < 5; face++) {
    int vertex_offset = v_idx / 3;

    for (int j = 0; j < 4; j++) {
      int octant = cube_faces[face][j];
      /* Pick octant: bit 0 = X, bit 1 = Y, bit 2 = Z */
      float x            = (octant & 1) * 2.0f - 1.0f;
      float y            = ((octant & 2) >> 1) * 2.0f - 1.0f;
      float z            = ((octant & 4) >> 2) * 2.0f - 1.0f;
      positions[v_idx++] = x;
      positions[v_idx++] = y;
      positions[v_idx++] = z;
    }

    /* Two triangles per face */
    indices[i_idx++] = vertex_offset + 0;
    indices[i_idx++] = vertex_offset + 1;
    indices[i_idx++] = vertex_offset + 2;
    indices[i_idx++] = vertex_offset + 2;
    indices[i_idx++] = vertex_offset + 1;
    indices[i_idx++] = vertex_offset + 3;
  }

  state.pool.index_count = 30; /* 5 faces * 6 indices */

  /* Create vertex buffer */
  state.pool.vertex_buffer
    = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                         .label        = "Pool vertices",
                                         .usage        = WGPUBufferUsage_Vertex,
                                         .size         = sizeof(positions),
                                         .initial.data = positions,
                                       });

  /* Create index buffer */
  state.pool.index_buffer
    = wgpu_create_buffer(wgpu_context, &(wgpu_buffer_desc_t){
                                         .label        = "Pool indices",
                                         .usage        = WGPUBufferUsage_Index,
                                         .size         = sizeof(indices),
                                         .initial.data = indices,
                                       });

  /* Create render pipeline */
  WGPUShaderModule pool_module
    = wgpu_create_shader_module(wgpu_context->device, pool_shader_wgsl);

  state.pool.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Pool pipeline"),
      .layout = NULL,
      .vertex = {
        .module      = pool_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 3 * sizeof(float),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &(WGPUVertexAttribute){
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = 0,
            .shaderLocation = 0,
          },
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = pool_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = true,
        .depthCompare        = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(pool_module);
}

static void init_sphere(wgpu_context_t* wgpu_context)
{
  /* Generate sphere geometry using octahedron subdivision */
  /* (Similar to TypeScript version) */
  const int detail       = SPHERE_DETAIL;
  const int max_vertices = 8 * (detail + 1) * (detail + 2) / 2;
  const int max_indices  = 8 * detail * detail * 3;

  float* positions    = malloc(max_vertices * 3 * sizeof(float));
  uint32_t* indices   = malloc(max_indices * sizeof(uint32_t));
  int* unique_map     = malloc(max_vertices * sizeof(int));
  int unique_count    = 0;
  int final_idx_count = 0;

  if (!positions || !indices || !unique_map) {
    free(positions);
    free(indices);
    free(unique_map);
    return;
  }

/* Helper to add unique vertex */
#define ADD_VERTEX(px, py, pz)                                                 \
  do {                                                                         \
    int found = -1;                                                            \
    for (int k = 0; k < unique_count && found < 0; k++) {                      \
      if (fabsf(positions[k * 3] - (px)) < 1e-6f                               \
          && fabsf(positions[k * 3 + 1] - (py)) < 1e-6f                        \
          && fabsf(positions[k * 3 + 2] - (pz)) < 1e-6f) {                     \
        found = k;                                                             \
      }                                                                        \
    }                                                                          \
    if (found < 0) {                                                           \
      found                           = unique_count;                          \
      positions[unique_count * 3]     = (px);                                  \
      positions[unique_count * 3 + 1] = (py);                                  \
      positions[unique_count * 3 + 2] = (pz);                                  \
      unique_count++;                                                          \
    }                                                                          \
    unique_map[vertex_idx++] = found;                                          \
  } while (0)

  /* Generate sphere using octahedron subdivision */
  for (int octant = 0; octant < 8; octant++) {
    float sx = (octant & 1) ? 1.0f : -1.0f;
    float sy = (octant & 2) ? 1.0f : -1.0f;
    float sz = (octant & 4) ? 1.0f : -1.0f;
    int flip = (sx * sy * sz > 0.0f) ? 1 : 0;

    int vertex_idx = 0;
    int local_indices[256];
    int local_count = 0;

    /* Generate vertices for this octant */
    for (int i = 0; i <= detail; i++) {
      for (int j = 0; i + j <= detail; j++) {
        float a = (float)i / detail;
        float b = (float)j / detail;
        float c = (float)(detail - i - j) / detail;

        /* Apply smoothing */
        a = a + (a - a * a) / 2.0f;
        b = b + (b - b * b) / 2.0f;
        c = c + (c - c * c) / 2.0f;

        float len = sqrtf(a * a + b * b + c * c);
        float px  = (a / len) * sx;
        float py  = (b / len) * sy;
        float pz  = (c / len) * sz;

        ADD_VERTEX(px, py, pz);
        local_indices[local_count++] = unique_map[vertex_idx - 1];
      }
    }

    /* Generate triangle indices for this octant */
    int k = 0;
    for (int i = 0; i < detail; i++) {
      for (int j = 0; i + j < detail; j++) {
        int a_idx = local_indices[k];
        int b_idx = local_indices[k + detail - i + 1];
        int c_idx = local_indices[k + 1];

        if (flip) {
          indices[final_idx_count++] = a_idx;
          indices[final_idx_count++] = b_idx;
          indices[final_idx_count++] = c_idx;
        }
        else {
          indices[final_idx_count++] = a_idx;
          indices[final_idx_count++] = c_idx;
          indices[final_idx_count++] = b_idx;
        }

        if (i + j < detail - 1) {
          int d_idx = local_indices[k + detail - i + 2];
          if (flip) {
            indices[final_idx_count++] = b_idx;
            indices[final_idx_count++] = d_idx;
            indices[final_idx_count++] = c_idx;
          }
          else {
            indices[final_idx_count++] = b_idx;
            indices[final_idx_count++] = c_idx;
            indices[final_idx_count++] = d_idx;
          }
        }
        k++;
      }
      k++;
    }
  }

#undef ADD_VERTEX

  state.sphere.index_count = final_idx_count;

  /* Create vertex buffer */
  state.sphere.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Sphere vertices",
                    .usage        = WGPUBufferUsage_Vertex,
                    .size         = unique_count * 3 * sizeof(float),
                    .initial.data = positions,
                  });

  /* Create index buffer */
  state.sphere.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label        = "Sphere indices",
                    .usage        = WGPUBufferUsage_Index,
                    .size         = final_idx_count * sizeof(uint32_t),
                    .initial.data = indices,
                  });

  free(positions);
  free(indices);
  free(unique_map);

  /* Create render pipeline */
  WGPUShaderModule sphere_module
    = wgpu_create_shader_module(wgpu_context->device, sphere_shader_wgsl);

  state.sphere.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Sphere pipeline"),
      .layout = NULL,
      .vertex = {
        .module      = sphere_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &(WGPUVertexBufferLayout){
          .arrayStride    = 3 * sizeof(float),
          .stepMode       = WGPUVertexStepMode_Vertex,
          .attributeCount = 1,
          .attributes     = &(WGPUVertexAttribute){
            .format         = WGPUVertexFormat_Float32x3,
            .offset         = 0,
            .shaderLocation = 0,
          },
        },
      },
      .fragment = &(WGPUFragmentState){
        .module      = sphere_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = true,
        .depthCompare        = WGPUCompareFunction_Less,
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });

  wgpuShaderModuleRelease(sphere_module);
}

static void cleanup_water(void)
{
  /* Cleanup textures */
  wgpu_destroy_texture(&state.water.texture_a);
  wgpu_destroy_texture(&state.water.texture_b);
  wgpu_destroy_texture(&state.water.caustics_texture);

  /* Cleanup buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.water.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.water.index_buffer.buffer);

  /* Cleanup simulation pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.drop_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.update_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.normal_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.sphere_move_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.caustics_pipeline);

  /* Cleanup surface pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.surface_above_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water.surface_under_pipeline);

  /* Cleanup sampler */
  WGPU_RELEASE_RESOURCE(Sampler, state.water.sampler);
}

static void cleanup_pool(void)
{
  /* Cleanup buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.pool.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.pool.index_buffer.buffer);

  /* Cleanup pipeline */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pool.pipeline);
}

static void cleanup_sphere(void)
{
  /* Cleanup buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere.vertex_buffer.buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere.index_buffer.buffer);

  /* Cleanup pipeline */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.sphere.pipeline);
}

/* -------------------------------------------------------------------------- *
 * Main function
 * -------------------------------------------------------------------------- */

int main(int argc, char** argv)
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "WebGPU - Water Simulation",
    .width          = 1200,
    .height         = 800,
    .init_cb        = example_init,
    .frame_cb       = example_frame,
    .shutdown_cb    = example_cleanup,
    .input_event_cb = example_on_input_event,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

/* clang-format off */

/* ----------------------------- Drop Shader -------------------------------- */
static const char* drop_shader_wgsl = CODE(
  struct DropUniforms {
    center : vec2f,
    radius : f32,
    strength : f32,
  }
  @group(0) @binding(0) var waterTexture : texture_2d<f32>;
  @group(0) @binding(1) var waterSampler : sampler;
  @group(0) @binding(2) var<uniform> u : DropUniforms;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) coord : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) idx : u32) -> VertexOutput {
    var pos = array<vec2f, 6>(
      vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
      vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    var out : VertexOutput;
    out.position = vec4f(pos[idx], 0.0, 1.0);
    out.coord = (pos[idx] + 1.0) * 0.5;
    return out;
  }

  @fragment
  fn fs_main(@location(0) coord : vec2f) -> @location(0) vec4f {
    var info = textureSample(waterTexture, waterSampler, coord);
    let drop = max(0.0, 1.0 - length((u.center * 0.5 + 0.5) - coord) / u.radius);
    let dropVal = 0.5 - cos(drop * 3.14159265) * 0.5;
    info.r += dropVal * u.strength;
    return info;
  }
);

/* --------------------------- Update Shader -------------------------------- */
static const char* update_shader_wgsl = CODE(
  struct UpdateUniforms {
    delta : vec2f,
  }
  @group(0) @binding(0) var waterTexture : texture_2d<f32>;
  @group(0) @binding(1) var waterSampler : sampler;
  @group(0) @binding(2) var<uniform> u : UpdateUniforms;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) coord : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) idx : u32) -> VertexOutput {
    var pos = array<vec2f, 6>(
      vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
      vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    var out : VertexOutput;
    out.position = vec4f(pos[idx], 0.0, 1.0);
    out.coord = (pos[idx] + 1.0) * 0.5;
    return out;
  }

  @fragment
  fn fs_main(@location(0) coord : vec2f) -> @location(0) vec4f {
    var info = textureSample(waterTexture, waterSampler, coord);
    let dx = vec2f(u.delta.x, 0.0);
    let dy = vec2f(0.0, u.delta.y);
    let avg = (
      textureSample(waterTexture, waterSampler, coord - dx).r +
      textureSample(waterTexture, waterSampler, coord - dy).r +
      textureSample(waterTexture, waterSampler, coord + dx).r +
      textureSample(waterTexture, waterSampler, coord + dy).r
    ) * 0.25;
    info.g += (avg - info.r) * 2.0;
    info.g *= 0.995;
    info.r += info.g;
    return info;
  }
);

/* --------------------------- Normal Shader -------------------------------- */
static const char* normal_shader_wgsl = CODE(
  struct NormalUniforms {
    delta : vec2f,
  }
  @group(0) @binding(0) var waterTexture : texture_2d<f32>;
  @group(0) @binding(1) var waterSampler : sampler;
  @group(0) @binding(2) var<uniform> u : NormalUniforms;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) coord : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) idx : u32) -> VertexOutput {
    var pos = array<vec2f, 6>(
      vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
      vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    var out : VertexOutput;
    out.position = vec4f(pos[idx], 0.0, 1.0);
    out.coord = (pos[idx] + 1.0) * 0.5;
    return out;
  }

  @fragment
  fn fs_main(@location(0) coord : vec2f) -> @location(0) vec4f {
    var info = textureSample(waterTexture, waterSampler, coord);
    let dx = vec2f(u.delta.x, 0.0);
    let dy = vec2f(0.0, u.delta.y);
    let val_dx = textureSample(waterTexture, waterSampler, coord + dx).r;
    let val_dy = textureSample(waterTexture, waterSampler, coord + dy).r;
    let tangX = vec3f(u.delta.x, val_dx - info.r, 0.0);
    let tangY = vec3f(0.0, val_dy - info.r, u.delta.y);
    let norm = normalize(cross(tangY, tangX));
    info.b = norm.x;
    info.a = norm.z;
    return info;
  }
);

/* ------------------------ Sphere Move Shader ------------------------------ */
static const char* sphere_move_shader_wgsl = CODE(
  struct SphereMoveUniforms {
    oldCenter : vec3f,
    radius : f32,
    newCenter : vec3f,
    _pad : f32,
  }
  @group(0) @binding(0) var waterTexture : texture_2d<f32>;
  @group(0) @binding(1) var waterSampler : sampler;
  @group(0) @binding(2) var<uniform> u : SphereMoveUniforms;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) coord : vec2f,
  }

  @vertex
  fn vs_main(@builtin(vertex_index) idx : u32) -> VertexOutput {
    var pos = array<vec2f, 6>(
      vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
      vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    var out : VertexOutput;
    out.position = vec4f(pos[idx], 0.0, 1.0);
    out.coord = vec2f((pos[idx].x + 1.0) * 0.5, (1.0 - pos[idx].y) * 0.5);
    return out;
  }

  // Calculates the volume of sphere intersecting the water at a UV position
  fn volumeInSphere(center : vec3f, uv : vec2f, radius : f32) -> f32 {
    let p = vec3f(uv.x * 2.0 - 1.0, 0.0, uv.y * 2.0 - 1.0);
    let dist = length(p - center);
    let t = dist / radius;
    // Gaussian-like falloff for smooth interaction
    let dy = exp(-pow(t * 1.5, 6.0));
    let ymin = min(0.0, center.y - dy);
    let ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);
    return (ymax - ymin) * 0.1;
  }

  @fragment
  fn fs_main(@location(0) coord : vec2f) -> @location(0) vec4f {
    var info = textureSample(waterTexture, waterSampler, coord);
    // Water rises where sphere was, falls where sphere is now
    info.r += volumeInSphere(u.oldCenter, coord, u.radius);
    info.r -= volumeInSphere(u.newCenter, coord, u.radius);
    return info;
  }
);

/* --------------------------- Caustics Shader ------------------------------ */
static const char* caustics_shader_wgsl = CODE(
  struct LightUniforms {
    direction : vec3f,
    _pad : f32,
  }
  struct SphereUniforms {
    center : vec3f,
    radius : f32,
  }
  struct ShadowUniforms {
    rim : f32,
    sphere : f32,
    ao : f32,
    _pad : f32,
  }
  @group(0) @binding(0) var<uniform> light : LightUniforms;
  @group(0) @binding(1) var<uniform> sphere : SphereUniforms;
  @group(0) @binding(2) var waterSampler : sampler;
  @group(0) @binding(3) var waterTexture : texture_2d<f32>;
  @group(0) @binding(4) var<uniform> shadows : ShadowUniforms;

  const IOR_AIR : f32 = 1.0;
  const IOR_WATER : f32 = 1.333;
  const poolHeight : f32 = 1.0;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) oldPos : vec3f,
    @location(1) newPos : vec3f,
    @location(2) ray : vec3f,
  }

  fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
  }

  fn project(origin: vec3f, ray: vec3f, refractedLight: vec3f) -> vec3f {
    var point = origin;
    let tcube = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
    point += ray * tcube.y;
    let tplane = (-point.y - 1.0) / refractedLight.y;
    return point + refractedLight * tplane;
  }

  @vertex
  fn vs_main(@location(0) position : vec3f) -> VertexOutput {
    var out : VertexOutput;
    let uv = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    // Reconstruct normal (scaled down for stability)
    let ba = info.ba * 0.5;
    let normal = vec3f(ba.x, sqrt(max(0.0, 1.0 - dot(ba, ba))), ba.y);
    // Calculate refracted light directions
    let lightDir = normalize(light.direction);
    // Flat water refraction (reference)
    let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    // Displaced water refraction (actual)
    let ray = refract(-lightDir, normal, IOR_AIR / IOR_WATER);
    // Water surface position
    let pos = vec3f(position.x, 0.0, position.y);
    // Project both rays to pool floor
    out.oldPos = project(pos, refractedLight, refractedLight);
    out.newPos = project(pos + vec3f(0.0, info.r, 0.0), ray, refractedLight);
    out.ray = ray;
    // Position in caustics texture space
    let projectedPos = 0.75 * (out.newPos.xz - out.newPos.y * refractedLight.xz / refractedLight.y);
    out.position = vec4f(projectedPos.x, -projectedPos.y, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(@location(0) oldPos : vec3f, @location(1) newPos : vec3f, @location(2) ray : vec3f) -> @location(0) vec4f {
    // Calculate intensity from area ratio using screen-space derivatives
    let oldArea = length(dpdx(oldPos)) * length(dpdy(oldPos));
    let newArea = length(dpdx(newPos)) * length(dpdy(newPos));
    var intensity = oldArea / newArea * 0.2;

    // Calculate sphere shadow
    let lightDir = normalize(light.direction);
    let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    // Sphere shadow using distance to ray
    let dir = (sphere.center - newPos) / sphere.radius;
    let area = cross(dir, refractedLight);
    var shadow = dot(area, area);
    let dist = dot(dir, -refractedLight);
    shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
    shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
    shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
    shadow = mix(1.0, shadow, shadows.sphere);

    // Rim shadow at pool edges
    let t = intersectCube(newPos, -refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
    let rimShadow = 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (newPos.y - refractedLight.y * t.y - 2.0 / 12.0)));
    intensity *= mix(1.0, rimShadow, shadows.rim);

    // R = caustic intensity, G = sphere shadow factor
    return vec4f(intensity, shadow, 0.0, 1.0);
  }
);

/* --------------------- Water Surface Above Shader ------------------------- */
static const char* water_surface_above_shader_wgsl = CODE(
  struct CameraUniforms {
    viewProjectionMatrix : mat4x4f,
    eyePosition : vec3f,
    _pad : f32,
  }
  struct LightUniforms {
    direction : vec3f,
    _pad : f32,
  }
  struct SphereUniforms {
    center : vec3f,
    radius : f32,
  }
  struct ShadowUniforms {
    rim : f32,
    sphere : f32,
    ao : f32,
    _pad : f32,
  }
  @group(0) @binding(0) var<uniform> commonUniforms : CameraUniforms;
  @group(0) @binding(1) var<uniform> light : LightUniforms;
  @group(0) @binding(2) var<uniform> sphere : SphereUniforms;
  @group(0) @binding(3) var tileSampler : sampler;
  @group(0) @binding(4) var tileTexture : texture_2d<f32>;
  @group(0) @binding(5) var waterSampler : sampler;
  @group(0) @binding(6) var waterTexture : texture_2d<f32>;
  @group(0) @binding(7) var causticTexture : texture_2d<f32>;
  @group(0) @binding(8) var<uniform> shadows : ShadowUniforms;

  const IOR_AIR : f32 = 1.0;
  const IOR_WATER : f32 = 1.333;
  const poolHeight : f32 = 1.0;
  const aboveWaterColor : vec3f = vec3f(0.25, 1.0, 1.25);

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPos : vec3f,
  }

  @vertex
  fn vs_main(@location(0) position : vec3f) -> VertexOutput {
    var out : VertexOutput;
    let uv = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    var pos = position.xzy;
    pos.y = info.r;
    out.worldPos = pos;
    out.position = commonUniforms.viewProjectionMatrix * vec4f(pos, 1.0);
    return out;
  }

  fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
  }

  fn intersectSphere(origin: vec3f, ray: vec3f, sphereCenter: vec3f, sphereRadius: f32) -> f32 {
    let toSphere = origin - sphereCenter;
    let a = dot(ray, ray);
    let b = 2.0 * dot(toSphere, ray);
    let c = dot(toSphere, toSphere) - sphereRadius * sphereRadius;
    let discriminant = b*b - 4.0*a*c;
    if (discriminant > 0.0) {
      let t = (-b - sqrt(discriminant)) / (2.0 * a);
      if (t > 0.0) { return t; }
    }
    return 1.0e6;
  }

  fn getSphereColor(point: vec3f) -> vec3f {
    var color = vec3f(0.5);
    let sphereRadius = sphere.radius;
    color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.x)) / sphereRadius, 3.0);
    color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.z)) / sphereRadius, 3.0);
    color *= 1.0 - 0.9 / pow((point.y + 1.0 + sphereRadius) / sphereRadius, 3.0);
    let sphereNormal = (point - sphere.center) / sphereRadius;
    let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;
    let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
    if (point.y < info.r) {
      let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      diffuse *= caustic.r * 4.0;
    }
    color += diffuse;
    return color;
  }

  fn getWallColor(point: vec3f) -> vec3f {
    var wallColor : vec3f;
    var normal = vec3f(0.0, 1.0, 0.0);
    if (abs(point.x) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
      normal = vec3f(-point.x, 0.0, 0.0);
    } else if (abs(point.z) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
      normal = vec3f(0.0, 0.0, -point.z);
    } else {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
    }
    var scale = 0.5;
    scale /= length(point);
    scale *= mix(1.0, 1.0 - 0.9 / pow(length(point - sphere.center) / sphere.radius, 4.0), shadows.sphere);
    let refractedLight = -refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(refractedLight, normal));
    let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
    if (point.y < info.r) {
      let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      scale += diffuse * caustic.r * 2.0 * caustic.g;
    } else {
      let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
      scale += diffuse * 0.5;
    }
    return wallColor * scale;
  }

  fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f) -> vec3f {
    var color : vec3f;
    var q = 1.0e6;
    if (shadows.sphere > 0.5) {
      q = intersectSphere(origin, ray, sphere.center, sphere.radius);
    }
    if (q < 1.0e6) {
      color = getSphereColor(origin + ray * q);
    } else if (ray.y < 0.0) {
      let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      color = getWallColor(origin + ray * t.y);
    } else {
      let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      let hit = origin + ray * t.y;
      if (hit.y < 2.0 / 12.0) {
        color = getWallColor(hit);
      } else {
        // Black background instead of skybox
        color = vec3f(0.0);
      }
    }
    if (ray.y < 0.0) {
      color *= waterColor;
    }
    return color;
  }

  @fragment
  fn fs_main(@location(0) worldPos : vec3f) -> @location(0) vec4f {
    var uv = worldPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    for (var i = 0; i < 5; i++) {
      uv += info.ba * 0.005;
      info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    }
    let ba = vec2f(info.b, info.a);
    var normal = vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);
    let incomingRay = normalize(worldPos - commonUniforms.eyePosition);
    let reflectedRay = reflect(incomingRay, normal);
    let refractedRay = refract(incomingRay, normal, IOR_AIR / IOR_WATER);
    let fresnel = mix(0.25, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));
    let reflectedColor = getSurfaceRayColor(worldPos, reflectedRay, aboveWaterColor);
    let refractedColor = getSurfaceRayColor(worldPos, refractedRay, aboveWaterColor);
    let finalColor = mix(refractedColor, reflectedColor, fresnel);
    return vec4f(finalColor, 1.0);
  }
);

/* --------------------- Water Surface Under Shader ------------------------- */
static const char* water_surface_under_shader_wgsl = CODE(
  struct CameraUniforms {
    viewProjectionMatrix : mat4x4f,
    eyePosition : vec3f,
    _pad : f32,
  }
  struct LightUniforms {
    direction : vec3f,
    _pad : f32,
  }
  struct SphereUniforms {
    center : vec3f,
    radius : f32,
  }
  struct ShadowUniforms {
    rim : f32,
    sphere : f32,
    ao : f32,
    _pad : f32,
  }
  @group(0) @binding(0) var<uniform> commonUniforms : CameraUniforms;
  @group(0) @binding(1) var<uniform> light : LightUniforms;
  @group(0) @binding(2) var<uniform> sphere : SphereUniforms;
  @group(0) @binding(3) var tileSampler : sampler;
  @group(0) @binding(4) var tileTexture : texture_2d<f32>;
  @group(0) @binding(5) var waterSampler : sampler;
  @group(0) @binding(6) var waterTexture : texture_2d<f32>;
  @group(0) @binding(7) var causticTexture : texture_2d<f32>;
  @group(0) @binding(8) var<uniform> shadows : ShadowUniforms;

  const IOR_AIR : f32 = 1.0;
  const IOR_WATER : f32 = 1.333;
  const poolHeight : f32 = 1.0;
  const underWaterColor : vec3f = vec3f(0.4, 0.9, 1.0);

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) worldPos : vec3f,
  }

  @vertex
  fn vs_main(@location(0) position : vec3f) -> VertexOutput {
    var out : VertexOutput;
    let uv = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    var pos = position.xzy;
    pos.y = info.r;
    out.worldPos = pos;
    out.position = commonUniforms.viewProjectionMatrix * vec4f(pos, 1.0);
    return out;
  }

  fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
  }

  fn intersectSphere(origin: vec3f, ray: vec3f, sphereCenter: vec3f, sphereRadius: f32) -> f32 {
    let toSphere = origin - sphereCenter;
    let a = dot(ray, ray);
    let b = 2.0 * dot(toSphere, ray);
    let c = dot(toSphere, toSphere) - sphereRadius * sphereRadius;
    let discriminant = b*b - 4.0*a*c;
    if (discriminant > 0.0) {
      let t = (-b - sqrt(discriminant)) / (2.0 * a);
      if (t > 0.0) { return t; }
    }
    return 1.0e6;
  }

  fn getSphereColor(point: vec3f) -> vec3f {
    var color = vec3f(0.5);
    let sphereRadius = sphere.radius;
    color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.x)) / sphereRadius, 3.0);
    color *= 1.0 - 0.9 / pow((1.0 + sphereRadius - abs(point.z)) / sphereRadius, 3.0);
    color *= 1.0 - 0.9 / pow((point.y + 1.0 + sphereRadius) / sphereRadius, 3.0);
    let sphereNormal = (point - sphere.center) / sphereRadius;
    let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;
    let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
    if (point.y < info.r) {
      let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      diffuse *= caustic.r * 4.0;
    }
    color += diffuse;
    return color;
  }

  fn getWallColor(point: vec3f) -> vec3f {
    var wallColor : vec3f;
    var normal = vec3f(0.0, 1.0, 0.0);
    if (abs(point.x) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
      normal = vec3f(-point.x, 0.0, 0.0);
    } else if (abs(point.z) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
      normal = vec3f(0.0, 0.0, -point.z);
    } else {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
    }
    var scale = 0.5;
    scale /= length(point);
    scale *= mix(1.0, 1.0 - 0.9 / pow(length(point - sphere.center) / sphere.radius, 4.0), shadows.sphere);
    let refractedLight = -refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(refractedLight, normal));
    let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
    if (point.y < info.r) {
      let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      scale += diffuse * caustic.r * 2.0 * caustic.g;
    } else {
      let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
      scale += diffuse * 0.5;
    }
    return wallColor * scale;
  }

  fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f) -> vec3f {
    var color : vec3f;
    var q = 1.0e6;
    if (shadows.sphere > 0.5) {
      q = intersectSphere(origin, ray, sphere.center, sphere.radius);
    }
    if (q < 1.0e6) {
      color = getSphereColor(origin + ray * q);
    } else if (ray.y < 0.0) {
      let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      color = getWallColor(origin + ray * t.y);
    } else {
      let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      let hit = origin + ray * t.y;
      if (hit.y < 2.0 / 12.0) {
        color = getWallColor(hit);
      } else {
        color = vec3f(0.0);
      }
    }
    if (ray.y < 0.0) {
      color *= waterColor;
    }
    return color;
  }

  @fragment
  fn fs_main(@location(0) worldPos : vec3f) -> @location(0) vec4f {
    var uv = worldPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    for (var i = 0; i < 5; i++) {
      uv += info.ba * 0.005;
      info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
    }
    let ba = vec2f(info.b, info.a);
    var normal = vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);
    // UNDERWATER VIEW: Looking up at water surface
    normal = -normal; // Flip normal for underwater
    let incomingRay = normalize(worldPos - commonUniforms.eyePosition);
    let reflectedRay = reflect(incomingRay, normal);
    let refractedRay = refract(incomingRay, normal, IOR_WATER / IOR_AIR);
    let fresnel = mix(0.5, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));
    let reflectedColor = getSurfaceRayColor(worldPos, reflectedRay, underWaterColor);
    let refractedColor = getSurfaceRayColor(worldPos, refractedRay, vec3f(1.0)) * vec3f(0.8, 1.0, 1.1);
    let finalColor = mix(reflectedColor, refractedColor, (1.0 - fresnel) * length(refractedRay));
    return vec4f(finalColor, 1.0);
  }
);

/* ----------------------------- Pool Shader -------------------------------- */
static const char* pool_shader_wgsl = CODE(
  struct Uniforms {
    viewProjectionMatrix : mat4x4f,
    eyePosition : vec3f,
    _pad : f32,
  }
  struct LightUniforms {
    direction : vec3f,
    _pad : f32,
  }
  struct SphereUniforms {
    center : vec3f,
    radius : f32,
  }
  struct ShadowUniforms {
    rim : f32,
    sphere : f32,
    ao : f32,
    _pad : f32,
  }
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var tileSampler : sampler;
  @group(0) @binding(2) var tileTexture : texture_2d<f32>;
  @group(0) @binding(3) var<uniform> light : LightUniforms;
  @group(0) @binding(4) var<uniform> sphere : SphereUniforms;
  @group(0) @binding(5) var waterSampler : sampler;
  @group(0) @binding(6) var waterTexture : texture_2d<f32>;
  @group(0) @binding(7) var causticTexture : texture_2d<f32>;
  @group(0) @binding(8) var<uniform> shadows : ShadowUniforms;

  const IOR_AIR : f32 = 1.0;
  const IOR_WATER : f32 = 1.333;
  const poolHeight : f32 = 1.0;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) localPos : vec3f,
  }

  fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
  }

  @vertex
  fn vs_main(@location(0) position : vec3f) -> VertexOutput {
    var out : VertexOutput;
    // Transform Y coordinate to create pool depth
    var transformedPos = position;
    transformedPos.y = ((1.0 - position.y) * (7.0 / 12.0) - 1.0);
    out.position = uniforms.viewProjectionMatrix * vec4f(transformedPos, 1.0);
    out.localPos = transformedPos;
    return out;
  }

  @fragment
  fn fs_main(@location(0) localPos : vec3f) -> @location(0) vec4f {
    var wallColor : vec3f;
    let point = localPos;

    // Sample tile texture based on which face we're rendering
    if (abs(point.x) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
    } else if (abs(point.z) > 0.999) {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
    } else {
      wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
    }

    // Determine surface normal based on face
    var normal = vec3f(0.0, 1.0, 0.0);
    if (abs(point.x) > 0.999) { normal = vec3f(-point.x, 0.0, 0.0); }
    else if (abs(point.z) > 0.999) { normal = vec3f(0.0, 0.0, -point.z); }

    // Ambient occlusion
    var scale = 0.5;
    scale /= length(point);
    scale *= mix(1.0, 1.0 - 0.9 / pow(length(point - sphere.center) / sphere.radius, 4.0), shadows.sphere);

    // Lighting with caustics or rim shadow
    let refractedLight = -refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(refractedLight, normal));

    let info = textureSampleLevel(waterTexture, waterSampler, point.xz * 0.5 + 0.5, 0.0);
    if (point.y < info.r) {
      // Underwater: sample caustics
      let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      scale += diffuse * caustic.r * 2.0 * caustic.g;
    } else {
      // Above water: apply rim shadow
      let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
      diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
      scale += diffuse * 0.5;
    }

    return vec4f(wallColor * scale, 1.0);
  }
);

/* --------------------------- Sphere Shader -------------------------------- */
static const char* sphere_shader_wgsl = CODE(
  struct CameraUniforms {
    viewProjectionMatrix : mat4x4f,
    eyePosition : vec3f,
    _pad : f32,
  }
  struct LightUniforms {
    direction : vec3f,
    _pad : f32,
  }
  struct SphereUniforms {
    center : vec3f,
    radius : f32,
  }
  @group(0) @binding(0) var<uniform> camera : CameraUniforms;
  @group(0) @binding(1) var<uniform> light : LightUniforms;
  @group(0) @binding(2) var<uniform> sphere : SphereUniforms;
  @group(0) @binding(3) var waterSampler : sampler;
  @group(0) @binding(4) var waterTexture : texture_2d<f32>;
  @group(0) @binding(5) var causticTexture : texture_2d<f32>;

  const IOR_AIR : f32 = 1.0;
  const IOR_WATER : f32 = 1.333;
  const underwaterColor : vec3f = vec3f(0.4, 0.9, 1.0);

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) localPos : vec3f,
    @location(1) worldPos : vec3f,
  }

  @vertex
  fn vs_main(@location(0) position : vec3f) -> VertexOutput {
    var out : VertexOutput;
    let worldPos = sphere.center + position * sphere.radius;
    out.position = camera.viewProjectionMatrix * vec4f(worldPos, 1.0);
    out.localPos = position;
    out.worldPos = worldPos;
    return out;
  }

  @fragment
  fn fs_main(@location(0) localPos : vec3f, @location(1) worldPos : vec3f) -> @location(0) vec4f {
    var color = vec3f(0.5);

    // Distance-based darkening near pool walls
    color *= 1.0 - 0.9 / pow((1.0 + sphere.radius - abs(worldPos.x)) / sphere.radius, 3.0);
    color *= 1.0 - 0.9 / pow((1.0 + sphere.radius - abs(worldPos.z)) / sphere.radius, 3.0);
    color *= 1.0 - 0.9 / pow((worldPos.y + 1.0 + sphere.radius) / sphere.radius, 3.0);

    // Diffuse lighting with caustics
    let sphereNormal = normalize(localPos);
    let refractedLight = refract(-light.direction, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    var diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;

    let info = textureSampleLevel(waterTexture, waterSampler, worldPos.xz * 0.5 + 0.5, 0.0);
    if (worldPos.y < info.r) {
      // Underwater: apply caustics
      let causticUV = 0.75 * (worldPos.xz - worldPos.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
      let caustic = textureSampleLevel(causticTexture, waterSampler, causticUV, 0.0);
      diffuse *= caustic.r * 4.0;
    }
    color += diffuse;

    // Apply underwater tint
    if (worldPos.y < info.r) {
      color *= underwaterColor;
    }

    return vec4f(color, 1.0);
  }
);

/* clang-format on */
