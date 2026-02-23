#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

/* WebGPU uses depth range [0,1] (zero-to-one), not OpenGL's [-1,1] */
#define CGLM_FORCE_DEPTH_ZERO_TO_ONE
#include <cglm/cglm.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

/* Async file loading */
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Clustered Forward Shading
 *
 * A simple clustered forward shading renderer with WebGPU. Uses a compute
 * shader to build a cluster grid and assign lights to clusters. Renders the
 * scene with PBR (Physically Based Rendering) materials loaded from a glTF
 * model (Sponza). Supports debug visualization modes for depth, depth slices,
 * cluster distances, and lights per cluster.
 *
 * Ref:
 * https://github.com/toji/webgpu-clustered-shading
 * http://www.aortiz.me/2018/12/21/CG.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define SAMPLE_COUNT (4)
#define DEPTH_FORMAT WGPUTextureFormat_Depth24Plus

/* Clustered shading grid dimensions */
#define TILE_COUNT_X (32)
#define TILE_COUNT_Y (18)
#define TILE_COUNT_Z (48)
#define TOTAL_TILES (TILE_COUNT_X * TILE_COUNT_Y * TILE_COUNT_Z)

/* Compute shader workgroup sizes */
#define WORKGROUP_SIZE_X (4)
#define WORKGROUP_SIZE_Y (2)
#define WORKGROUP_SIZE_Z (4)

/* Dispatch sizes = ceil(tile_count / workgroup_size) */
#define DISPATCH_SIZE_X                                                        \
  ((TILE_COUNT_X + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X)
#define DISPATCH_SIZE_Y                                                        \
  ((TILE_COUNT_Y + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y)
#define DISPATCH_SIZE_Z                                                        \
  ((TILE_COUNT_Z + WORKGROUP_SIZE_Z - 1) / WORKGROUP_SIZE_Z)

/* Light constants */
#define MAX_LIGHT_COUNT (1024)
#define MAX_LIGHTS_PER_CLUSTER (100)
#define LIGHT_FLOAT_COUNT (8)
#define LIGHT_BYTE_SIZE (LIGHT_FLOAT_COUNT * 4)

/* Cluster lights buffer size:
 * (8 * TOTAL_TILES) + (4 * MAX_LIGHTS_PER_CLUSTER * TOTAL_TILES) + 4 */
#define CLUSTER_LIGHTS_SIZE                                                    \
  ((8u * TOTAL_TILES) + (4u * MAX_LIGHTS_PER_CLUSTER * TOTAL_TILES) + 4u)

/* Uniform sizes (matching JS version) */
#define PROJECTION_UNIFORMS_SIZE (144)
#define VIEW_UNIFORMS_SIZE (80)

/* GL type constants for glTF parsing (reuse from gl.h included via GLFW) */
#ifndef GL_BYTE
#define GL_BYTE 0x1400
#define GL_UNSIGNED_BYTE 0x1401
#define GL_SHORT 0x1402
#define GL_UNSIGNED_SHORT 0x1403
#define GL_UNSIGNED_INT 0x1405
#define GL_FLOAT 0x1406
#define GL_TRIANGLES 0x0004
#define GL_TRIANGLE_STRIP 0x0005
#define GL_NEAREST 0x2600
#define GL_LINEAR 0x2601
#define GL_LINEAR_MIPMAP_NEAREST 0x2701
#define GL_NEAREST_MIPMAP_LINEAR 0x2702
#define GL_LINEAR_MIPMAP_LINEAR 0x2703
#define GL_REPEAT 0x2901
#define GL_MIRRORED_REPEAT 0x8370
#endif

/* glTF attribute locations */
#define ATTRIB_MAP_POSITION (1)
#define ATTRIB_MAP_NORMAL (2)
#define ATTRIB_MAP_TANGENT (3)
#define ATTRIB_MAP_TEXCOORD_0 (4)
#define ATTRIB_MAP_COLOR_0 (5)

/* Max scene limits */
#define MAX_PRIMITIVES (512)
#define MAX_MATERIALS (64)
#define MAX_IMAGES (128)
#define MAX_SAMPLERS (16)
#define MAX_BUFFER_VIEWS (512)
#define MAX_VERTEX_BUFFERS_PER_PRIM (4)
#define MAX_CACHED_PIPELINES (128)

/* Async image loading */
#define IMG_FETCH_BUFFER_SIZE (6 * 1024 * 1024) /* 6 MB fetch buffer */

/* Output type enum */
typedef enum output_type_t {
  OUTPUT_NAIVE_FORWARD   = 0,
  OUTPUT_DEPTH           = 1,
  OUTPUT_DEPTH_SLICE     = 2,
  OUTPUT_CLUSTER_DIST    = 3,
  OUTPUT_LIGHTS_PER_CLST = 4,
  OUTPUT_CLUSTERED_FWD   = 5,
  OUTPUT_COUNT           = 6,
} output_type_t;

/* Loading phase enum — controls progressive startup */
typedef enum load_phase_t {
  LOAD_PHASE_INIT     = 0, /* First frames: show light sprites + GUI */
  LOAD_PHASE_GLTF     = 1, /* Load GLTF model on this frame */
  LOAD_PHASE_TEXTURES = 2, /* GLTF loaded, textures loading progressively */
  LOAD_PHASE_DONE     = 3, /* Everything loaded */
} load_phase_t;

/* -------------------------------------------------------------------------- *
 * GLTF types
 * -------------------------------------------------------------------------- */

typedef struct gltf_image_t {
  WGPUTexture texture;
  WGPUTextureView view;
  bool loaded;
} gltf_image_t;

typedef struct gltf_sampler_t {
  WGPUSampler gpu_sampler;
} gltf_sampler_t;

typedef struct gltf_material_t {
  float base_color_factor[4];
  float metallic_factor;
  float roughness_factor;
  float emissive_factor[3];
  float occlusion_strength;
  int32_t base_color_tex_idx;
  int32_t normal_tex_idx;
  int32_t metallic_roughness_tex_idx;
  int32_t occlusion_tex_idx;
  int32_t emissive_tex_idx;
  int32_t sampler_idx;
  bool blend;
  bool double_sided;
  WGPUBuffer uniform_buffer;
  WGPUBindGroup bind_group;
} gltf_material_t;

typedef struct vertex_attribute_t {
  WGPUVertexFormat format;
  uint32_t shader_location;
  uint32_t offset;
  uint32_t byte_offset; /* Original byte offset in buffer view */
} vertex_attribute_t;

typedef struct vertex_buffer_layout_t {
  uint32_t buffer_view_idx;
  uint32_t array_stride;
  uint32_t min_byte_offset;
  vertex_attribute_t attributes[8];
  uint32_t attribute_count;
} vertex_buffer_layout_t;

typedef struct gltf_primitive_t {
  vertex_buffer_layout_t vertex_buffers[MAX_VERTEX_BUFFERS_PER_PRIM];
  uint32_t vertex_buffer_count;
  uint32_t index_buffer_view_idx;
  WGPUIndexFormat index_format;
  uint32_t index_byte_offset;
  uint32_t element_count;
  int32_t material_idx;
  mat4 world_matrix;
  WGPUBindGroup model_bind_group;
  /* Attribute flags */
  bool has_normals;
  bool has_tangents;
  bool has_texcoords;
  bool has_colors;
  WGPUPrimitiveTopology topology;
} gltf_primitive_t;

/* Pipeline cache key: identifies a unique render pipeline configuration */
typedef struct pipeline_cache_key_t {
  uint32_t vertex_buffer_count;
  uint32_t attribute_locations; /* bitmask of shader locations present */
  uint32_t array_strides[MAX_VERTEX_BUFFERS_PER_PRIM];
  bool blend;
  bool double_sided;
  WGPUPrimitiveTopology topology;
  output_type_t output_type;
} pipeline_cache_key_t;

typedef struct pipeline_cache_entry_t {
  pipeline_cache_key_t key;
  WGPURenderPipeline pipeline;
  bool valid;
} pipeline_cache_entry_t;

typedef struct gltf_buffer_view_t {
  WGPUBuffer gpu_buffer;
  uint32_t byte_length;
  uint32_t byte_stride;
  bool is_vertex;
  bool is_index;
} gltf_buffer_view_t;

/* Light struct (mirrors JS Light class) */
typedef struct light_t {
  float position[4]; /* xyz + range in w */
  float color[4];    /* rgb + intensity */
  float velocity[3];
  float destination[3];
  float travel_time;
} light_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */

static struct {
  /* GLTF scene data */
  struct {
    gltf_buffer_view_t buffer_views[MAX_BUFFER_VIEWS];
    uint32_t buffer_view_count;
    gltf_image_t images[MAX_IMAGES];
    uint32_t image_count;
    gltf_sampler_t samplers[MAX_SAMPLERS];
    uint32_t sampler_count;
    gltf_material_t materials[MAX_MATERIALS];
    uint32_t material_count;
    gltf_primitive_t primitives[MAX_PRIMITIVES];
    uint32_t primitive_count;
    bool loaded;
  } gltf;

  /* Lights */
  struct {
    float uniform_array[4 + LIGHT_FLOAT_COUNT * MAX_LIGHT_COUNT];
    light_t lights[MAX_LIGHT_COUNT];
    uint32_t light_count;
    uint32_t max_light_count;
    bool render_sprites;
  } light_mgr;

  /* Camera */
  struct {
    vec3 position;
    vec3 angles; /* pitch, yaw */
    mat4 view_mat;
    mat4 rot_mat;
    bool dirty;
    float speed;
    bool moving;
    float last_x;
    float last_y;
    bool key_w, key_s, key_a, key_d, key_space, key_shift;
  } camera;

  /* Frame uniforms (projection + view) */
  float frame_uniforms[16 + 16 + 2 + 2 + 16 + 4];
  /* Layout:
   *  [0..15]  projection matrix
   * [16..31]  inverse projection matrix
   * [32..33]  output size
   * [34..35]  z range (near, far)
   * [36..51]  view matrix
   * [52..54]  camera position
   */

  /* Bind group layouts */
  struct {
    WGPUBindGroupLayout frame;
    WGPUBindGroupLayout material;
    WGPUBindGroupLayout primitive;
    WGPUBindGroupLayout cluster;
  } bind_group_layouts;

  /* Bind groups */
  struct {
    WGPUBindGroup frame;
    WGPUBindGroup cluster;
  } bind_groups;

  /* GPU buffers */
  struct {
    WGPUBuffer projection;
    WGPUBuffer view;
    WGPUBuffer lights;
    WGPUBuffer cluster_lights;
    WGPUBuffer cluster;
  } buffers;

  /* Uniform buffers for primitives */
  WGPUBuffer model_buffers[MAX_PRIMITIVES];

  /* Default texture views */
  struct {
    WGPUTexture black_tex;
    WGPUTexture white_tex;
    WGPUTexture blue_tex;
    WGPUTextureView black;
    WGPUTextureView white;
    WGPUTextureView blue;
  } default_textures;

  WGPUSampler default_sampler;

  /* Pipelines */
  WGPUPipelineLayout pipeline_layout;
  WGPUPipelineLayout cluster_dist_pipeline_layout;
  WGPURenderPipeline light_sprite_pipeline;
  WGPUComputePipeline cluster_bounds_pipeline;
  WGPUComputePipeline cluster_lights_pipeline;

  /* Render bundles per output type */
  WGPURenderBundle render_bundles[OUTPUT_COUNT];
  bool render_bundle_valid[OUTPUT_COUNT];

  /* Pipeline cache */
  pipeline_cache_entry_t pipeline_cache[MAX_CACHED_PIPELINES];
  uint32_t pipeline_cache_count;

  /* Render pass textures for MSAA and depth */
  WGPUTexture msaa_texture;
  WGPUTextureView msaa_view;
  WGPUTexture depth_texture;
  WGPUTextureView depth_view;

  /* Cluster compute bind groups */
  WGPUBindGroup cluster_storage_bind_group;
  WGPUBindGroupLayout cluster_storage_bgl;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    output_type_t output_type;
    bool render_light_sprites;
    int32_t light_count;
    float max_light_range;
  } settings;

  /* Timing */
  uint64_t last_frame_time;
  int32_t frame_count;

  /* Loading state machine */
  load_phase_t load_phase;

  /* Async image loading */
  struct {
    char paths[MAX_IMAGES][512];
    uint32_t total;
    uint32_t loaded_count;
    bool all_done;
    wgpu_context_t* wgpu_context;
  } pending_images;
  uint8_t fetch_buffer[IMG_FETCH_BUFFER_SIZE];

  /* Render bundle descriptor */
  WGPURenderBundleEncoderDescriptor render_bundle_desc;
  WGPUTextureFormat context_format;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Discard,
    .clearValue = {0.0, 0.0, 0.5, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Discard,
    .depthClearValue = 1.0f,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .output_type          = OUTPUT_CLUSTERED_FWD,
    .render_light_sprites = true,
    .light_count          = 128,
    .max_light_range      = 2.0f,
  },
  .camera = {
    .position = {0.0f, 1.5f, 0.0f},
    .speed    = 3.0f,
    .dirty    = true,
  },
  .light_mgr = {
    .light_count     = 128,
    .max_light_count = MAX_LIGHT_COUNT,
    .render_sprites  = true,
  },
  .frame_count = -1,
};

/* -------------------------------------------------------------------------- *
 * Forward declarations for shader strings
 * -------------------------------------------------------------------------- */

static const char* pbr_vertex_shader_wgsl;
static const char* pbr_vertex_shader_no_tangent_wgsl;
static const char* pbr_fragment_naive_wgsl     = NULL;
static const char* pbr_fragment_clustered_wgsl = NULL;
static void build_pbr_fragment_shaders(void);
static const char* light_sprite_vertex_shader_wgsl;
static const char* light_sprite_fragment_shader_wgsl;
static const char* cluster_bounds_compute_shader_wgsl;
static const char* cluster_lights_compute_shader_wgsl;
static const char* depth_viz_fragment_wgsl;
static const char* depth_slice_viz_fragment_wgsl;
static const char* cluster_dist_viz_vertex_wgsl;
static const char* cluster_dist_viz_fragment_wgsl;
static const char* lights_per_cluster_viz_fragment_wgsl;
static const char* simple_vertex_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Helper: random float in range [min, max]
 * -------------------------------------------------------------------------- */

static float rand_float(float min_val, float max_val)
{
  return min_val + ((float)rand() / (float)RAND_MAX) * (max_val - min_val);
}

/* -------------------------------------------------------------------------- *
 * Camera
 * -------------------------------------------------------------------------- */

static void camera_init(void)
{
  glm_vec3_copy((vec3){0.0f, 1.5f, 0.0f}, state.camera.position);
  glm_vec3_zero(state.camera.angles);
  glm_mat4_identity(state.camera.view_mat);
  glm_mat4_identity(state.camera.rot_mat);
  state.camera.dirty     = true;
  state.camera.speed     = 3.0f;
  state.camera.key_w     = false;
  state.camera.key_s     = false;
  state.camera.key_a     = false;
  state.camera.key_d     = false;
  state.camera.key_space = false;
  state.camera.key_shift = false;

  /* Initial rotation: face -Z (rotate yaw by -PI/2) */
  state.camera.angles[1] = -(float)GLM_PI * 0.5f;

  /* Build initial rotation matrix */
  glm_mat4_identity(state.camera.rot_mat);
  glm_rotate_y(state.camera.rot_mat, -state.camera.angles[1],
               state.camera.rot_mat);
  glm_rotate_x(state.camera.rot_mat, -state.camera.angles[0],
               state.camera.rot_mat);
}

static void camera_rotate_view(float x_delta, float y_delta)
{
  if (x_delta == 0.0f && y_delta == 0.0f) {
    return;
  }

  state.camera.angles[1] += x_delta;
  /* Keep yaw in [0, 2*PI] */
  while (state.camera.angles[1] < 0.0f) {
    state.camera.angles[1] += (float)GLM_PI * 2.0f;
  }
  while (state.camera.angles[1] >= (float)GLM_PI * 2.0f) {
    state.camera.angles[1] -= (float)GLM_PI * 2.0f;
  }

  state.camera.angles[0] += y_delta;
  /* Clamp pitch to [-PI/2, PI/2] */
  if (state.camera.angles[0] < -(float)GLM_PI * 0.5f) {
    state.camera.angles[0] = -(float)GLM_PI * 0.5f;
  }
  if (state.camera.angles[0] > (float)GLM_PI * 0.5f) {
    state.camera.angles[0] = (float)GLM_PI * 0.5f;
  }

  /* Update rotation matrix */
  glm_mat4_identity(state.camera.rot_mat);
  glm_rotate_y(state.camera.rot_mat, -state.camera.angles[1],
               state.camera.rot_mat);
  glm_rotate_x(state.camera.rot_mat, -state.camera.angles[0],
               state.camera.rot_mat);

  state.camera.dirty = true;
}

static void camera_get_view_matrix(mat4 dest)
{
  if (state.camera.dirty) {
    glm_mat4_identity(state.camera.view_mat);
    glm_rotate_x(state.camera.view_mat, state.camera.angles[0],
                 state.camera.view_mat);
    glm_rotate_y(state.camera.view_mat, state.camera.angles[1],
                 state.camera.view_mat);
    vec3 neg_pos;
    glm_vec3_negate_to(state.camera.position, neg_pos);
    glm_translate(state.camera.view_mat, neg_pos);
    state.camera.dirty = false;
  }
  glm_mat4_copy(state.camera.view_mat, dest);
}

static void camera_update(float frame_time_ms)
{
  const float speed = (state.camera.speed / 1000.0f) * frame_time_ms;
  vec3 dir          = {0.0f, 0.0f, 0.0f};

  if (state.camera.key_w) {
    dir[2] -= speed;
  }
  if (state.camera.key_s) {
    dir[2] += speed;
  }
  if (state.camera.key_a) {
    dir[0] -= speed;
  }
  if (state.camera.key_d) {
    dir[0] += speed;
  }
  if (state.camera.key_space) {
    dir[1] += speed;
  }
  if (state.camera.key_shift) {
    dir[1] -= speed;
  }

  if (dir[0] != 0.0f || dir[1] != 0.0f || dir[2] != 0.0f) {
    vec3 transformed;
    glm_mat4_mulv3(state.camera.rot_mat, dir, 1.0f, transformed);
    glm_vec3_add(state.camera.position, transformed, state.camera.position);
    state.camera.dirty = true;
  }
}

/* -------------------------------------------------------------------------- *
 * Light Manager
 * -------------------------------------------------------------------------- */

static void light_manager_init(void)
{
  memset(state.light_mgr.uniform_array, 0,
         sizeof(state.light_mgr.uniform_array));

  /* Ambient color: (0.002, 0.002, 0.002) */
  state.light_mgr.uniform_array[0] = 0.002f;
  state.light_mgr.uniform_array[1] = 0.002f;
  state.light_mgr.uniform_array[2] = 0.002f;

  /* Light count in uniform_array[3] (as uint32) */
  uint32_t count = state.light_mgr.light_count;
  memcpy(&state.light_mgr.uniform_array[3], &count, sizeof(uint32_t));

  /* Initialize fixed corner lights (0-3) */
  light_t* l = &state.light_mgr.lights[0];
  glm_vec3_copy((vec3){8.95f, 1.0f, -3.55f}, l->position);
  l->position[3] = 4.0f; /* range */
  glm_vec3_copy((vec3){5.0f, 1.0f, 1.0f}, l->color);

  l = &state.light_mgr.lights[1];
  glm_vec3_copy((vec3){8.95f, 1.0f, 3.2f}, l->position);
  l->position[3] = 4.0f;
  glm_vec3_copy((vec3){5.0f, 1.0f, 1.0f}, l->color);

  l = &state.light_mgr.lights[2];
  glm_vec3_copy((vec3){-9.65f, 1.0f, -3.55f}, l->position);
  l->position[3] = 4.0f;
  glm_vec3_copy((vec3){1.0f, 1.0f, 5.0f}, l->color);

  l = &state.light_mgr.lights[3];
  glm_vec3_copy((vec3){-9.65f, 1.0f, 3.2f}, l->position);
  l->position[3] = 4.0f;
  glm_vec3_copy((vec3){1.0f, 1.0f, 5.0f}, l->color);

  /* Light 4: large, bright wandering light */
  l = &state.light_mgr.lights[4];
  glm_vec3_copy((vec3){0.0f, 1.5f, 0.0f}, l->position);
  l->position[3] = 5.0f;
  glm_vec3_copy((vec3){5.0f, 5.0f, 5.0f}, l->color);

  /* Initialize remaining lights randomly */
  for (uint32_t i = 5; i < MAX_LIGHT_COUNT; ++i) {
    l              = &state.light_mgr.lights[i];
    l->position[0] = rand_float(-11.0f, 10.0f);
    l->position[1] = rand_float(0.2f, 6.5f);
    l->position[2] = rand_float(-4.5f, 4.0f);
    l->position[3] = 2.0f; /* range */
    l->color[0]    = rand_float(0.1f, 1.0f);
    l->color[1]    = rand_float(0.1f, 1.0f);
    l->color[2]    = rand_float(0.1f, 1.0f);
    l->color[3]    = 1.0f;
  }
}

/* Sync lights[] into uniform_array for GPU upload */
static void light_manager_update_uniform_array(void)
{
  uint32_t count = state.light_mgr.light_count;
  memcpy(&state.light_mgr.uniform_array[3], &count, sizeof(uint32_t));

  for (uint32_t i = 0; i < count; ++i) {
    light_t* l = &state.light_mgr.lights[i];
    float* dst = &state.light_mgr.uniform_array[4 + i * LIGHT_FLOAT_COUNT];
    dst[0]     = l->position[0];
    dst[1]     = l->position[1];
    dst[2]     = l->position[2];
    dst[3]     = l->position[3]; /* range */
    dst[4]     = l->color[0];
    dst[5]     = l->color[1];
    dst[6]     = l->color[2];
    dst[7]     = l->color[3]; /* intensity */
  }
}

/* Update wandering lights */
static void update_wandering_lights(float time_delta)
{
  for (uint32_t i = 4; i < state.light_mgr.light_count; ++i) {
    light_t* l = &state.light_mgr.lights[i];
    l->travel_time -= time_delta;

    if (l->travel_time <= 0.0f) {
      l->travel_time    = rand_float(500.0f, 2000.0f);
      l->destination[0] = rand_float(-11.0f, 10.0f);
      l->destination[1] = rand_float(0.2f, 6.5f);
      l->destination[2] = rand_float(-4.5f, 4.0f);
    }

    l->velocity[0]
      += (l->destination[0] - l->position[0]) * 0.000005f * time_delta;
    l->velocity[1]
      += (l->destination[1] - l->position[1]) * 0.000005f * time_delta;
    l->velocity[2]
      += (l->destination[2] - l->position[2]) * 0.000005f * time_delta;

    /* Clamp velocity */
    float vel_len = glm_vec3_norm(l->velocity);
    if (vel_len > 0.05f) {
      glm_vec3_scale(l->velocity, 0.05f / vel_len, l->velocity);
    }

    glm_vec3_add(l->position, l->velocity, l->position);
  }
}

static void update_light_range(float range)
{
  for (uint32_t i = 5; i < state.light_mgr.max_light_count; ++i) {
    state.light_mgr.lights[i].position[3] = range;
  }
}

/* -------------------------------------------------------------------------- *
 * Default textures (1x1 pixel solid colors)
 * -------------------------------------------------------------------------- */

static void create_default_textures(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Helper to create a 1x1 texture */
  uint8_t black_pixel[4] = {0, 0, 0, 0};
  uint8_t white_pixel[4] = {255, 255, 255, 255};
  uint8_t blue_pixel[4]  = {0, 0, 255, 0};

  WGPUTextureDescriptor tex_desc = {
    .usage         = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {1, 1, 1},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };

  state.default_textures.black_tex = wgpuDeviceCreateTexture(device, &tex_desc);
  state.default_textures.white_tex = wgpuDeviceCreateTexture(device, &tex_desc);
  state.default_textures.blue_tex  = wgpuDeviceCreateTexture(device, &tex_desc);

  WGPUExtent3D size = {1, 1, 1};

  wgpu_image_to_texure(wgpu_context, state.default_textures.black_tex,
                       black_pixel, size, 4);
  wgpu_image_to_texure(wgpu_context, state.default_textures.white_tex,
                       white_pixel, size, 4);
  wgpu_image_to_texure(wgpu_context, state.default_textures.blue_tex,
                       blue_pixel, size, 4);

  state.default_textures.black
    = wgpuTextureCreateView(state.default_textures.black_tex, NULL);
  state.default_textures.white
    = wgpuTextureCreateView(state.default_textures.white_tex, NULL);
  state.default_textures.blue
    = wgpuTextureCreateView(state.default_textures.blue_tex, NULL);

  /* Default sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .magFilter     = WGPUFilterMode_Linear,
    .minFilter     = WGPUFilterMode_Linear,
    .mipmapFilter  = WGPUMipmapFilterMode_Linear,
    .addressModeU  = WGPUAddressMode_Repeat,
    .addressModeV  = WGPUAddressMode_Repeat,
    .addressModeW  = WGPUAddressMode_Repeat,
    .maxAnisotropy = 1,
  };
  state.default_sampler = wgpuDeviceCreateSampler(device, &sampler_desc);
}

/* -------------------------------------------------------------------------- *
 * GLTF Loading
 * -------------------------------------------------------------------------- */

static WGPUVertexFormat get_vertex_format(int component_type,
                                          int component_count, bool normalized)
{
  if (component_type == GL_FLOAT) {
    switch (component_count) {
      case 1:
        return WGPUVertexFormat_Float32;
      case 2:
        return WGPUVertexFormat_Float32x2;
      case 3:
        return WGPUVertexFormat_Float32x3;
      case 4:
        return WGPUVertexFormat_Float32x4;
    }
  }
  else if (component_type == GL_UNSIGNED_BYTE) {
    if (normalized) {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Unorm8x2;
        case 4:
          return WGPUVertexFormat_Unorm8x4;
      }
    }
    else {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Uint8x2;
        case 4:
          return WGPUVertexFormat_Uint8x4;
      }
    }
  }
  else if (component_type == GL_SHORT) {
    if (normalized) {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Snorm16x2;
        case 4:
          return WGPUVertexFormat_Snorm16x4;
      }
    }
    else {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Sint16x2;
        case 4:
          return WGPUVertexFormat_Sint16x4;
      }
    }
  }
  else if (component_type == GL_UNSIGNED_SHORT) {
    if (normalized) {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Unorm16x2;
        case 4:
          return WGPUVertexFormat_Unorm16x4;
      }
    }
    else {
      switch (component_count) {
        case 2:
          return WGPUVertexFormat_Uint16x2;
        case 4:
          return WGPUVertexFormat_Uint16x4;
      }
    }
  }
  else if (component_type == GL_UNSIGNED_INT) {
    switch (component_count) {
      case 1:
        return WGPUVertexFormat_Uint32;
      case 2:
        return WGPUVertexFormat_Uint32x2;
      case 3:
        return WGPUVertexFormat_Uint32x3;
      case 4:
        return WGPUVertexFormat_Uint32x4;
    }
  }
  return WGPUVertexFormat_Float32x3; /* fallback */
}

static int get_component_count(cgltf_type type)
{
  switch (type) {
    case cgltf_type_scalar:
      return 1;
    case cgltf_type_vec2:
      return 2;
    case cgltf_type_vec3:
      return 3;
    case cgltf_type_vec4:
      return 4;
    case cgltf_type_mat2:
      return 4;
    case cgltf_type_mat3:
      return 9;
    case cgltf_type_mat4:
      return 16;
    default:
      return 0;
  }
}

static int get_component_type_gl(cgltf_component_type ct)
{
  switch (ct) {
    case cgltf_component_type_r_8:
      return GL_BYTE;
    case cgltf_component_type_r_8u:
      return GL_UNSIGNED_BYTE;
    case cgltf_component_type_r_16:
      return GL_SHORT;
    case cgltf_component_type_r_16u:
      return GL_UNSIGNED_SHORT;
    case cgltf_component_type_r_32u:
      return GL_UNSIGNED_INT;
    case cgltf_component_type_r_32f:
      return GL_FLOAT;
    default:
      return GL_FLOAT;
  }
}

static uint32_t get_component_type_size(int gl_type)
{
  switch (gl_type) {
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
      return 1;
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
      return 2;
    case GL_UNSIGNED_INT:
    case GL_FLOAT:
      return 4;
    default:
      return 0;
  }
}

static int get_attrib_location(const char* name)
{
  if (strcmp(name, "POSITION") == 0)
    return ATTRIB_MAP_POSITION;
  if (strcmp(name, "NORMAL") == 0)
    return ATTRIB_MAP_NORMAL;
  if (strcmp(name, "TANGENT") == 0)
    return ATTRIB_MAP_TANGENT;
  if (strcmp(name, "TEXCOORD_0") == 0)
    return ATTRIB_MAP_TEXCOORD_0;
  if (strcmp(name, "COLOR_0") == 0)
    return ATTRIB_MAP_COLOR_0;
  return -1;
}

static const char* get_attrib_name(cgltf_attribute_type type, int index)
{
  switch (type) {
    case cgltf_attribute_type_position:
      return "POSITION";
    case cgltf_attribute_type_normal:
      return "NORMAL";
    case cgltf_attribute_type_tangent:
      return "TANGENT";
    case cgltf_attribute_type_texcoord:
      return (index == 0) ? "TEXCOORD_0" : NULL;
    case cgltf_attribute_type_color:
      return (index == 0) ? "COLOR_0" : NULL;
    default:
      return NULL;
  }
}

static WGPUAddressMode get_address_mode(int wrap)
{
  switch (wrap) {
    case GL_REPEAT:
      return WGPUAddressMode_Repeat;
    case GL_MIRRORED_REPEAT:
      return WGPUAddressMode_MirrorRepeat;
    default:
      return WGPUAddressMode_ClampToEdge;
  }
}

/* Forward declaration for node processing */
static void process_gltf_node(wgpu_context_t* wgpu_context, cgltf_data* data,
                              cgltf_node* node, mat4 parent_matrix);

/* (Re)create a material's bind group using currently loaded textures.
 * Called initially with default textures, then again when images arrive. */
static void recreate_material_bind_group(WGPUDevice device, uint32_t mat_idx)
{
  gltf_material_t* g_mat = &state.gltf.materials[mat_idx];

  /* Get sampler */
  WGPUSampler sampler = state.default_sampler;
  if (g_mat->sampler_idx >= 0
      && g_mat->sampler_idx < (int32_t)state.gltf.sampler_count) {
    sampler = state.gltf.samplers[g_mat->sampler_idx].gpu_sampler;
  }

  /* Get texture views — use default if image not yet loaded */
  WGPUTextureView base_color_view = state.default_textures.white;
  if (g_mat->base_color_tex_idx >= 0
      && g_mat->base_color_tex_idx < (int32_t)state.gltf.image_count
      && state.gltf.images[g_mat->base_color_tex_idx].loaded) {
    base_color_view = state.gltf.images[g_mat->base_color_tex_idx].view;
  }

  WGPUTextureView normal_view = state.default_textures.blue;
  if (g_mat->normal_tex_idx >= 0
      && g_mat->normal_tex_idx < (int32_t)state.gltf.image_count
      && state.gltf.images[g_mat->normal_tex_idx].loaded) {
    normal_view = state.gltf.images[g_mat->normal_tex_idx].view;
  }

  WGPUTextureView mr_view = state.default_textures.white;
  if (g_mat->metallic_roughness_tex_idx >= 0
      && g_mat->metallic_roughness_tex_idx < (int32_t)state.gltf.image_count
      && state.gltf.images[g_mat->metallic_roughness_tex_idx].loaded) {
    mr_view = state.gltf.images[g_mat->metallic_roughness_tex_idx].view;
  }

  WGPUTextureView occ_view = state.default_textures.white;
  if (g_mat->occlusion_tex_idx >= 0
      && g_mat->occlusion_tex_idx < (int32_t)state.gltf.image_count
      && state.gltf.images[g_mat->occlusion_tex_idx].loaded) {
    occ_view = state.gltf.images[g_mat->occlusion_tex_idx].view;
  }

  WGPUTextureView em_view = state.default_textures.black;
  if (g_mat->emissive_tex_idx >= 0
      && g_mat->emissive_tex_idx < (int32_t)state.gltf.image_count
      && state.gltf.images[g_mat->emissive_tex_idx].loaded) {
    em_view = state.gltf.images[g_mat->emissive_tex_idx].view;
  }

  WGPUBindGroupEntry entries[7] = {
    {.binding = 0,
     .buffer  = g_mat->uniform_buffer,
     .size    = 48}, /* 12 floats × 4 bytes */
    {.binding = 1, .sampler = sampler},
    {.binding = 2, .textureView = base_color_view},
    {.binding = 3, .textureView = normal_view},
    {.binding = 4, .textureView = mr_view},
    {.binding = 5, .textureView = occ_view},
    {.binding = 6, .textureView = em_view},
  };

  /* Release old bind group if any */
  if (g_mat->bind_group) {
    wgpuBindGroupRelease(g_mat->bind_group);
  }

  g_mat->bind_group = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .layout     = state.bind_group_layouts.material,
              .entryCount = 7,
              .entries    = entries,
            });
}

/* Load a GLTF file and create all GPU resources */
static void load_gltf(wgpu_context_t* wgpu_context, const char* path)
{
  cgltf_options options = {0};
  cgltf_data* data      = NULL;
  cgltf_result result   = cgltf_parse_file(&options, path, &data);
  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to parse GLTF: %s (error %d)\n", path, result);
    return;
  }

  result = cgltf_load_buffers(&options, data, path);
  if (result != cgltf_result_success) {
    fprintf(stderr, "Failed to load GLTF buffers: %s\n", path);
    cgltf_free(data);
    return;
  }

  WGPUDevice device = wgpu_context->device;

  /* --- Load buffer views --- */
  state.gltf.buffer_view_count
    = MIN((uint32_t)data->buffer_views_count, MAX_BUFFER_VIEWS);
  for (uint32_t i = 0; i < state.gltf.buffer_view_count; ++i) {
    cgltf_buffer_view* bv                  = &data->buffer_views[i];
    state.gltf.buffer_views[i].byte_length = (uint32_t)bv->size;
    state.gltf.buffer_views[i].byte_stride = (uint32_t)bv->stride;
    state.gltf.buffer_views[i].is_vertex   = false;
    state.gltf.buffer_views[i].is_index    = false;
  }

  /* Determine usage from accessors */
  for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
    cgltf_mesh* mesh = &data->meshes[mi];
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi) {
      cgltf_primitive* prim = &mesh->primitives[pi];
      for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
        cgltf_accessor* acc = prim->attributes[ai].data;
        if (acc->buffer_view) {
          uint32_t bv_idx = (uint32_t)(acc->buffer_view - data->buffer_views);
          if (bv_idx < state.gltf.buffer_view_count) {
            state.gltf.buffer_views[bv_idx].is_vertex = true;
          }
        }
      }
      if (prim->indices && prim->indices->buffer_view) {
        uint32_t bv_idx
          = (uint32_t)(prim->indices->buffer_view - data->buffer_views);
        if (bv_idx < state.gltf.buffer_view_count) {
          state.gltf.buffer_views[bv_idx].is_index = true;
        }
      }
    }
  }

  /* Create GPU buffers for buffer views */
  for (uint32_t i = 0; i < state.gltf.buffer_view_count; ++i) {
    gltf_buffer_view_t* gbv = &state.gltf.buffer_views[i];
    WGPUBufferUsage usage   = 0;
    if (gbv->is_vertex)
      usage |= WGPUBufferUsage_Vertex;
    if (gbv->is_index)
      usage |= WGPUBufferUsage_Index;
    if (!usage)
      continue;

    cgltf_buffer_view* bv = &data->buffer_views[i];
    uint32_t aligned_len  = ((uint32_t)bv->size + 3u) & ~3u;

    gbv->gpu_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .usage            = usage | WGPUBufferUsage_CopyDst,
                .size             = aligned_len,
                .mappedAtCreation = true,
              });

    uint8_t* src = (uint8_t*)bv->buffer->data + bv->offset;
    void* mapped = wgpuBufferGetMappedRange(gbv->gpu_buffer, 0, aligned_len);
    memcpy(mapped, src, bv->size);
    if (aligned_len > bv->size) {
      memset((uint8_t*)mapped + bv->size, 0, aligned_len - bv->size);
    }
    wgpuBufferUnmap(gbv->gpu_buffer);
  }

  /* --- Prepare images for async loading --- */
  state.gltf.image_count     = MIN((uint32_t)data->images_count, MAX_IMAGES);
  state.pending_images.total = 0;
  state.pending_images.loaded_count = 0;
  state.pending_images.all_done     = false;

  for (uint32_t i = 0; i < state.gltf.image_count; ++i) {
    cgltf_image* img            = &data->images[i];
    state.gltf.images[i].loaded = false;

    /* Embedded images (buffer view): load immediately */
    if (img->buffer_view != NULL) {
      void* raw_data
        = (uint8_t*)img->buffer_view->buffer->data + img->buffer_view->offset;
      size_t raw_size = img->buffer_view->size;
      int img_w, img_h, img_channels;
      stbi_uc* pixels
        = stbi_load_from_memory((const stbi_uc*)raw_data, (int)raw_size, &img_w,
                                &img_h, &img_channels, 4);
      if (pixels) {
        WGPUExtent3D tex_size        = {(uint32_t)img_w, (uint32_t)img_h, 1};
        state.gltf.images[i].texture = wgpuDeviceCreateTexture(
          device,
          &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
            .dimension     = WGPUTextureDimension_2D,
            .size          = tex_size,
            .format        = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1,
            .sampleCount   = 1,
          });
        wgpu_image_to_texure(wgpu_context, state.gltf.images[i].texture, pixels,
                             tex_size, 4);
        state.gltf.images[i].view
          = wgpuTextureCreateView(state.gltf.images[i].texture, NULL);
        state.gltf.images[i].loaded = true;
        stbi_image_free(pixels);
      }
      continue;
    }

    /* External URI images: save path for async loading */
    if (img->uri != NULL) {
      char* img_path       = state.pending_images.paths[i];
      const char* last_sep = strrchr(path, '/');
      if (last_sep) {
        size_t dir_len = (size_t)(last_sep - path + 1);
        memcpy(img_path, path, dir_len);
        img_path[dir_len] = '\0';
        strcat(img_path, img->uri);
      }
      else {
        strcpy(img_path, img->uri);
      }
      state.pending_images.total++;
    }
  }

  /* --- Load samplers --- */
  state.gltf.sampler_count = MIN((uint32_t)data->samplers_count, MAX_SAMPLERS);
  for (uint32_t i = 0; i < state.gltf.sampler_count; ++i) {
    cgltf_sampler* s = &data->samplers[i];

    WGPUSamplerDescriptor desc = {
      .maxAnisotropy = 1,
    };

    /* magFilter: default (0 or GL_LINEAR) → linear, GL_NEAREST → nearest */
    if (!s->mag_filter || s->mag_filter == GL_LINEAR) {
      desc.magFilter = WGPUFilterMode_Linear;
    }

    /* minFilter: match JS Sampler.gpuSamplerDescriptor exactly */
    switch (s->min_filter) {
      case 0: /* undefined in GLTF → linear + mipmapLinear */
        desc.minFilter    = WGPUFilterMode_Linear;
        desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
        break;
      case GL_LINEAR:
      case GL_LINEAR_MIPMAP_NEAREST:
        desc.minFilter = WGPUFilterMode_Linear;
        break;
      case GL_NEAREST_MIPMAP_LINEAR:
        desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
        /* minFilter stays nearest (default 0) */
        break;
      case GL_LINEAR_MIPMAP_LINEAR:
        desc.minFilter    = WGPUFilterMode_Linear;
        desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
        break;
      case GL_NEAREST:
      default:
        /* minFilter stays nearest (default 0) */
        break;
    }

    desc.addressModeU = get_address_mode(s->wrap_s);
    desc.addressModeV = get_address_mode(s->wrap_t);

    state.gltf.samplers[i].gpu_sampler = wgpuDeviceCreateSampler(device, &desc);
  }

  /* --- Load materials --- */
  state.gltf.material_count
    = MIN((uint32_t)data->materials_count, MAX_MATERIALS);
  for (uint32_t i = 0; i < state.gltf.material_count; ++i) {
    cgltf_material* mat    = &data->materials[i];
    gltf_material_t* g_mat = &state.gltf.materials[i];
    memset(g_mat, 0, sizeof(*g_mat));

    g_mat->base_color_factor[0]       = 1.0f;
    g_mat->base_color_factor[1]       = 1.0f;
    g_mat->base_color_factor[2]       = 1.0f;
    g_mat->base_color_factor[3]       = 1.0f;
    g_mat->metallic_factor            = 1.0f;
    g_mat->roughness_factor           = 1.0f;
    g_mat->occlusion_strength         = 1.0f;
    g_mat->base_color_tex_idx         = -1;
    g_mat->normal_tex_idx             = -1;
    g_mat->metallic_roughness_tex_idx = -1;
    g_mat->occlusion_tex_idx          = -1;
    g_mat->emissive_tex_idx           = -1;
    g_mat->sampler_idx                = -1;
    g_mat->double_sided               = mat->double_sided;

    if (mat->has_pbr_metallic_roughness) {
      cgltf_pbr_metallic_roughness* pbr = &mat->pbr_metallic_roughness;
      memcpy(g_mat->base_color_factor, pbr->base_color_factor,
             4 * sizeof(float));
      g_mat->metallic_factor  = pbr->metallic_factor;
      g_mat->roughness_factor = pbr->roughness_factor;

      if (pbr->base_color_texture.texture) {
        cgltf_texture* tex = pbr->base_color_texture.texture;
        if (tex->image) {
          g_mat->base_color_tex_idx = (int32_t)(tex->image - data->images);
        }
        if (tex->sampler) {
          g_mat->sampler_idx = (int32_t)(tex->sampler - data->samplers);
        }
      }
      if (pbr->metallic_roughness_texture.texture) {
        cgltf_texture* tex = pbr->metallic_roughness_texture.texture;
        if (tex->image) {
          g_mat->metallic_roughness_tex_idx
            = (int32_t)(tex->image - data->images);
        }
      }
    }

    if (mat->normal_texture.texture) {
      cgltf_texture* tex = mat->normal_texture.texture;
      if (tex->image) {
        g_mat->normal_tex_idx = (int32_t)(tex->image - data->images);
      }
    }

    if (mat->occlusion_texture.texture) {
      cgltf_texture* tex = mat->occlusion_texture.texture;
      if (tex->image) {
        g_mat->occlusion_tex_idx = (int32_t)(tex->image - data->images);
      }
      g_mat->occlusion_strength = mat->occlusion_texture.scale;
    }

    memcpy(g_mat->emissive_factor, mat->emissive_factor, 3 * sizeof(float));
    if (mat->emissive_texture.texture) {
      cgltf_texture* tex = mat->emissive_texture.texture;
      if (tex->image) {
        g_mat->emissive_tex_idx = (int32_t)(tex->image - data->images);
      }
    }

    if (mat->alpha_mode == cgltf_alpha_mode_blend
        || mat->alpha_mode == cgltf_alpha_mode_mask) {
      g_mat->blend = true;
    }
  }

  /* --- Create material bind groups --- */
  for (uint32_t i = 0; i < state.gltf.material_count; ++i) {
    gltf_material_t* g_mat = &state.gltf.materials[i];

    /* Material uniform data: baseColorFactor(4) + metallicRoughnessFactor(2) +
     * pad(2) + emissiveFactor(3) + occlusionStrength(1) = 12 floats = 48 bytes
     */
    float mat_uniforms[12] = {0};
    memcpy(&mat_uniforms[0], g_mat->base_color_factor, 4 * sizeof(float));
    mat_uniforms[4] = g_mat->metallic_factor;
    mat_uniforms[5] = g_mat->roughness_factor;
    /* mat_uniforms[6..7] padding */
    memcpy(&mat_uniforms[8], g_mat->emissive_factor, 3 * sizeof(float));
    mat_uniforms[11] = g_mat->occlusion_strength;

    WGPUBuffer mat_buf = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(mat_uniforms),
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, mat_buf, 0, mat_uniforms,
                         sizeof(mat_uniforms));
    g_mat->uniform_buffer = mat_buf;

    /* Create bind group with currently loaded textures (may use defaults) */
    recreate_material_bind_group(device, i);
  }

  /* Process nodes -> handled by process_gltf_node below */
  /* Process scene */
  cgltf_scene* scene = data->scene ? data->scene : &data->scenes[0];
  mat4 identity;
  glm_mat4_identity(identity);
  for (cgltf_size ni = 0; ni < scene->nodes_count; ++ni) {
    process_gltf_node(wgpu_context, data, scene->nodes[ni], identity);
  }

  state.gltf.loaded = true;
  cgltf_free(data);
}

/* -------------------------------------------------------------------------- *
 * Async image loading via sokol_fetch
 * -------------------------------------------------------------------------- */

/* Callback fired by sokol_fetch when an image file has been loaded */
static void image_fetch_callback(const sfetch_response_t* response)
{
  if (response->finished && !response->fetched) {
    fprintf(stderr, "Image fetch failed: error %d\n", response->error_code);
    return;
  }
  if (!response->fetched)
    return;

  uint32_t img_idx = *(const uint32_t*)response->user_data;
  if (img_idx >= state.gltf.image_count)
    return;

  wgpu_context_t* wgpu_context = state.pending_images.wgpu_context;
  WGPUDevice device            = wgpu_context->device;

  int img_w, img_h, img_channels;
  stbi_uc* pixels = stbi_load_from_memory((const stbi_uc*)response->data.ptr,
                                          (int)response->data.size, &img_w,
                                          &img_h, &img_channels, 4);
  if (!pixels) {
    fprintf(stderr, "Image decode failed for image %u\n", img_idx);
    return;
  }

  WGPUExtent3D tex_size              = {(uint32_t)img_w, (uint32_t)img_h, 1};
  state.gltf.images[img_idx].texture = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .usage     = WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size      = tex_size,
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });
  wgpu_image_to_texure(wgpu_context, state.gltf.images[img_idx].texture, pixels,
                       tex_size, 4);
  state.gltf.images[img_idx].view
    = wgpuTextureCreateView(state.gltf.images[img_idx].texture, NULL);
  state.gltf.images[img_idx].loaded = true;
  stbi_image_free(pixels);

  state.pending_images.loaded_count++;

  /* Rebuild material bind groups that reference this image */
  for (uint32_t m = 0; m < state.gltf.material_count; ++m) {
    gltf_material_t* mat = &state.gltf.materials[m];
    if (mat->base_color_tex_idx == (int32_t)img_idx
        || mat->normal_tex_idx == (int32_t)img_idx
        || mat->metallic_roughness_tex_idx == (int32_t)img_idx
        || mat->occlusion_tex_idx == (int32_t)img_idx
        || mat->emissive_tex_idx == (int32_t)img_idx) {
      recreate_material_bind_group(device, m);
    }
  }

  /* Invalidate render bundles so they pick up new textures */
  for (int i = 0; i < OUTPUT_COUNT; ++i) {
    if (state.render_bundle_valid[i] && state.render_bundles[i]) {
      wgpuRenderBundleRelease(state.render_bundles[i]);
      state.render_bundles[i] = NULL;
    }
    state.render_bundle_valid[i] = false;
  }
  /* Keep pipeline cache — pipelines are still valid, only bundles need
   * rebuilding because the bind groups changed */

  if (state.pending_images.loaded_count >= state.pending_images.total) {
    state.pending_images.all_done = true;
  }
}

/* Image indices for user_data (must persist until callback fires) */
static uint32_t image_indices[MAX_IMAGES];

/* Start async loading for all pending images */
static void start_async_image_loading(void)
{
  for (uint32_t i = 0; i < state.gltf.image_count; ++i) {
    if (state.gltf.images[i].loaded)
      continue;
    if (state.pending_images.paths[i][0] == '\0')
      continue;

    image_indices[i] = i;
    sfetch_send(&(sfetch_request_t){
      .path      = state.pending_images.paths[i],
      .callback  = image_fetch_callback,
      .buffer    = {.ptr = state.fetch_buffer, .size = IMG_FETCH_BUFFER_SIZE},
      .user_data = {.ptr = &image_indices[i], .size = sizeof(uint32_t)},
    });
  }
}

/* Forward declared above load_gltf, defined here for readability */
static void process_gltf_node(wgpu_context_t* wgpu_context, cgltf_data* data,
                              cgltf_node* node, mat4 parent_matrix)
{
  WGPUDevice device = wgpu_context->device;
  mat4 world_matrix;
  mat4 local_matrix;

  if (node->has_matrix) {
    memcpy(local_matrix, node->matrix, sizeof(mat4));
    /* cgltf stores column-major, same as cglm */
  }
  else {
    float translation[3] = {0, 0, 0};
    float rotation[4]    = {0, 0, 0, 1};
    float scale[3]       = {1, 1, 1};
    if (node->has_translation)
      memcpy(translation, node->translation, 3 * sizeof(float));
    if (node->has_rotation)
      memcpy(rotation, node->rotation, 4 * sizeof(float));
    if (node->has_scale)
      memcpy(scale, node->scale, 3 * sizeof(float));

    glm_mat4_identity(local_matrix);
    /* TRS: Translation * Rotation * Scale */
    glm_translate(local_matrix, translation);
    versor q;
    glm_vec4_copy(rotation, q);
    mat4 rot_m;
    glm_quat_mat4(q, rot_m);
    glm_mat4_mul(local_matrix, rot_m, local_matrix);
    glm_scale(local_matrix, scale);
  }

  if (node->has_matrix || node->has_translation || node->has_rotation
      || node->has_scale) {
    glm_mat4_mul(parent_matrix, local_matrix, world_matrix);
  }
  else {
    glm_mat4_copy(parent_matrix, world_matrix);
  }

  if (node->mesh) {
    cgltf_mesh* mesh = node->mesh;
    for (cgltf_size pi = 0; pi < mesh->primitives_count; ++pi) {
      if (state.gltf.primitive_count >= MAX_PRIMITIVES)
        break;
      cgltf_primitive* prim = &mesh->primitives[pi];

      gltf_primitive_t* g_prim
        = &state.gltf.primitives[state.gltf.primitive_count];
      memset(g_prim, 0, sizeof(*g_prim));
      glm_mat4_copy(world_matrix, g_prim->world_matrix);

      /* Material index */
      if (prim->material) {
        g_prim->material_idx = (int32_t)(prim->material - data->materials);
      }
      else {
        g_prim->material_idx = -1;
      }

      /* Topology */
      switch (prim->type) {
        case cgltf_primitive_type_triangles:
          g_prim->topology = WGPUPrimitiveTopology_TriangleList;
          break;
        case cgltf_primitive_type_triangle_strip:
          g_prim->topology = WGPUPrimitiveTopology_TriangleStrip;
          break;
        case cgltf_primitive_type_lines:
          g_prim->topology = WGPUPrimitiveTopology_LineList;
          break;
        case cgltf_primitive_type_line_strip:
          g_prim->topology = WGPUPrimitiveTopology_LineStrip;
          break;
        case cgltf_primitive_type_points:
          g_prim->topology = WGPUPrimitiveTopology_PointList;
          break;
        default:
          g_prim->topology = WGPUPrimitiveTopology_TriangleList;
          break;
      }

      /* Process attributes - group by buffer view */
      /* Map: buffer_view_index -> vertex_buffer slot */
      int32_t bv_to_slot[MAX_BUFFER_VIEWS];
      memset(bv_to_slot, -1, sizeof(bv_to_slot));

      for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
        cgltf_attribute* attr = &prim->attributes[ai];
        cgltf_accessor* acc   = attr->data;
        if (!acc || !acc->buffer_view)
          continue;

        const char* attr_name = get_attrib_name(attr->type, attr->index);
        if (!attr_name)
          continue;

        int shader_loc = get_attrib_location(attr_name);
        if (shader_loc < 0)
          continue;

        uint32_t bv_idx = (uint32_t)(acc->buffer_view - data->buffer_views);
        if (bv_idx >= MAX_BUFFER_VIEWS)
          continue;
        int slot = bv_to_slot[bv_idx];
        if (slot < 0) {
          slot               = (int)g_prim->vertex_buffer_count++;
          bv_to_slot[bv_idx] = slot;
          g_prim->vertex_buffers[slot].buffer_view_idx = bv_idx;
          g_prim->vertex_buffers[slot].array_stride
            = (uint32_t)acc->buffer_view->stride;
          g_prim->vertex_buffers[slot].min_byte_offset = (uint32_t)acc->offset;
        }
        else {
          if (acc->offset < g_prim->vertex_buffers[slot].min_byte_offset) {
            g_prim->vertex_buffers[slot].min_byte_offset
              = (uint32_t)acc->offset;
          }
        }

        int comp_count  = get_component_count(acc->type);
        int comp_type   = get_component_type_gl(acc->component_type);
        bool normalized = acc->normalized;

        vertex_buffer_layout_t* vbl            = &g_prim->vertex_buffers[slot];
        uint32_t a_idx                         = vbl->attribute_count++;
        vbl->attributes[a_idx].shader_location = (uint32_t)shader_loc;
        vbl->attributes[a_idx].format
          = get_vertex_format(comp_type, comp_count, normalized);
        vbl->attributes[a_idx].byte_offset = (uint32_t)acc->offset;

        if (!acc->buffer_view->stride) {
          vbl->array_stride
            += get_component_type_size(comp_type) * (uint32_t)comp_count;
        }

        g_prim->element_count = (uint32_t)acc->count;

        if (attr->type == cgltf_attribute_type_normal)
          g_prim->has_normals = true;
        if (attr->type == cgltf_attribute_type_tangent)
          g_prim->has_tangents = true;
        if (attr->type == cgltf_attribute_type_texcoord)
          g_prim->has_texcoords = true;
        if (attr->type == cgltf_attribute_type_color)
          g_prim->has_colors = true;
      }

      /* Compute attribute offsets relative to min_byte_offset */
      for (uint32_t s = 0; s < g_prim->vertex_buffer_count; ++s) {
        vertex_buffer_layout_t* vbl = &g_prim->vertex_buffers[s];
        for (uint32_t a = 0; a < vbl->attribute_count; ++a) {
          vbl->attributes[a].offset
            = vbl->attributes[a].byte_offset - vbl->min_byte_offset;
        }
      }

      /* Indices */
      if (prim->indices) {
        cgltf_accessor* idx_acc = prim->indices;
        g_prim->element_count   = (uint32_t)idx_acc->count;
        g_prim->index_buffer_view_idx
          = (uint32_t)(idx_acc->buffer_view - data->buffer_views);
        g_prim->index_byte_offset = (uint32_t)idx_acc->offset;
        g_prim->index_format
          = (idx_acc->component_type == cgltf_component_type_r_16u) ?
              WGPUIndexFormat_Uint16 :
              WGPUIndexFormat_Uint32;
      }
      else {
        g_prim->index_buffer_view_idx = UINT32_MAX; /* No index buffer */
      }

      /* Create model uniform buffer and bind group */
      WGPUBuffer model_buf = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                  .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                  .size  = 64,
                  .mappedAtCreation = false,
                });
      wgpuQueueWriteBuffer(wgpu_context->queue, model_buf, 0,
                           g_prim->world_matrix, 64);
      state.model_buffers[state.gltf.primitive_count] = model_buf;

      WGPUBindGroupEntry model_entry = {
        .binding = 0,
        .buffer  = model_buf,
        .size    = 64,
      };
      g_prim->model_bind_group = wgpuDeviceCreateBindGroup(
        device, &(WGPUBindGroupDescriptor){
                  .layout     = state.bind_group_layouts.primitive,
                  .entryCount = 1,
                  .entries    = &model_entry,
                });

      state.gltf.primitive_count++;
    }
  }

  for (cgltf_size ci = 0; ci < node->children_count; ++ci) {
    process_gltf_node(wgpu_context, data, node->children[ci], world_matrix);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts and pipeline layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Frame bind group layout (group 0) */
  WGPUBindGroupLayoutEntry frame_entries[4] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                    | WGPUShaderStage_Compute,
      .buffer = {.type = WGPUBufferBindingType_Uniform},
    },
    {
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Uniform},
    },
    {
      .binding    = 2,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment
                    | WGPUShaderStage_Compute,
      .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
    },
    {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Storage},
    },
  };
  state.bind_group_layouts.frame
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("frame-bgl"),
                                                .entryCount = 4,
                                                .entries    = frame_entries,
                                              });

  /* Material bind group layout (group 1) */
  WGPUBindGroupLayoutEntry mat_entries[7] = {
    {.binding    = 0,
     .visibility = WGPUShaderStage_Fragment,
     .buffer     = {.type = WGPUBufferBindingType_Uniform}},
    {.binding    = 1,
     .visibility = WGPUShaderStage_Fragment,
     .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
    {.binding    = 2,
     .visibility = WGPUShaderStage_Fragment,
     .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                    .viewDimension = WGPUTextureViewDimension_2D}},
    {.binding    = 3,
     .visibility = WGPUShaderStage_Fragment,
     .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                    .viewDimension = WGPUTextureViewDimension_2D}},
    {.binding    = 4,
     .visibility = WGPUShaderStage_Fragment,
     .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                    .viewDimension = WGPUTextureViewDimension_2D}},
    {.binding    = 5,
     .visibility = WGPUShaderStage_Fragment,
     .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                    .viewDimension = WGPUTextureViewDimension_2D}},
    {.binding    = 6,
     .visibility = WGPUShaderStage_Fragment,
     .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                    .viewDimension = WGPUTextureViewDimension_2D}},
  };
  state.bind_group_layouts.material = wgpuDeviceCreateBindGroupLayout(
    device, &(WGPUBindGroupLayoutDescriptor){
              .label      = STRVIEW("Material - Bind group layout"),
              .entryCount = 7,
              .entries    = mat_entries,
            });

  /* Primitive/model bind group layout (group 2) */
  WGPUBindGroupLayoutEntry prim_entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Vertex,
    .buffer     = {.type = WGPUBufferBindingType_Uniform},
  };
  state.bind_group_layouts.primitive = wgpuDeviceCreateBindGroupLayout(
    device, &(WGPUBindGroupLayoutDescriptor){
              .label      = STRVIEW("Primitive - Bind group layout"),
              .entryCount = 1,
              .entries    = &prim_entry,
            });

  /* Cluster bind group layout */
  WGPUBindGroupLayoutEntry cluster_entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Compute,
    .buffer     = {.type = WGPUBufferBindingType_ReadOnlyStorage},
  };
  state.bind_group_layouts.cluster = wgpuDeviceCreateBindGroupLayout(
    device, &(WGPUBindGroupLayoutDescriptor){
              .label      = STRVIEW("Cluster - Bind group layout"),
              .entryCount = 1,
              .entries    = &cluster_entry,
            });

  /* Main pipeline layout: frame + material + primitive */
  WGPUBindGroupLayout layouts[3] = {
    state.bind_group_layouts.frame,
    state.bind_group_layouts.material,
    state.bind_group_layouts.primitive,
  };
  state.pipeline_layout
    = wgpuDeviceCreatePipelineLayout(device, &(WGPUPipelineLayoutDescriptor){
                                               .bindGroupLayoutCount = 3,
                                               .bindGroupLayouts     = layouts,
                                             });

  /* Cluster distance pipeline layout: frame + material + primitive + cluster */
  WGPUBindGroupLayout cluster_dist_layouts[4] = {
    state.bind_group_layouts.frame,
    state.bind_group_layouts.material,
    state.bind_group_layouts.primitive,
    state.bind_group_layouts.cluster,
  };
  state.cluster_dist_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .bindGroupLayoutCount = 4,
              .bindGroupLayouts     = cluster_dist_layouts,
            });
}

/* -------------------------------------------------------------------------- *
 * GPU Buffers
 * -------------------------------------------------------------------------- */

static void init_gpu_buffers(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.buffers.projection = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
              .size  = PROJECTION_UNIFORMS_SIZE,
            });

  state.buffers.view = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
              .size  = VIEW_UNIFORMS_SIZE,
            });

  state.buffers.lights = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
              .size  = sizeof(state.light_mgr.uniform_array),
            });

  state.buffers.cluster_lights = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
              .size  = CLUSTER_LIGHTS_SIZE,
            });

  /* Frame bind group */
  WGPUBindGroupEntry frame_entries[4] = {
    {.binding = 0,
     .buffer  = state.buffers.projection,
     .size    = PROJECTION_UNIFORMS_SIZE},
    {.binding = 1, .buffer = state.buffers.view, .size = VIEW_UNIFORMS_SIZE},
    {.binding = 2,
     .buffer  = state.buffers.lights,
     .size    = sizeof(state.light_mgr.uniform_array)},
    {.binding = 3,
     .buffer  = state.buffers.cluster_lights,
     .size    = CLUSTER_LIGHTS_SIZE},
  };
  state.bind_groups.frame = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .layout     = state.bind_group_layouts.frame,
              .entryCount = 4,
              .entries    = frame_entries,
            });
}

/* -------------------------------------------------------------------------- *
 * Pipeline creation
 * -------------------------------------------------------------------------- */

static void init_light_sprite_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  WGPUBindGroupLayout sprite_layouts[1] = {state.bind_group_layouts.frame};
  WGPUPipelineLayout sprite_pl          = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
                       .bindGroupLayoutCount = 1,
                       .bindGroupLayouts     = sprite_layouts,
            });

  WGPUShaderModule vs_module
    = wgpu_create_shader_module(device, light_sprite_vertex_shader_wgsl);
  WGPUShaderModule fs_module
    = wgpu_create_shader_module(device, light_sprite_fragment_shader_wgsl);

  WGPUBlendState blend_state = {
    .color = {
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_One,
      .operation = WGPUBlendOperation_Add,
    },
    .alpha = {
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_One,
      .operation = WGPUBlendOperation_Add,
    },
  };

  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUDepthStencilState depth_state = {
    .format            = DEPTH_FORMAT,
    .depthWriteEnabled = false,
    .depthCompare      = WGPUCompareFunction_Less,
  };

  state.light_sprite_pipeline = wgpuDeviceCreateRenderPipeline(device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("light-sprite-pipeline"),
      .layout = sprite_pl,
      .vertex = {
        .module     = vs_module,
        .entryPoint = STRVIEW("vertexMain"),
      },
      .fragment = &(WGPUFragmentState){
        .module      = fs_module,
        .entryPoint  = STRVIEW("fragmentMain"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .primitive = {
        .topology         = WGPUPrimitiveTopology_TriangleStrip,
        .stripIndexFormat  = WGPUIndexFormat_Uint32,
      },
      .depthStencil = &depth_state,
      .multisample = {
        .count = SAMPLE_COUNT,
        .mask  = 0xFFFFFFFF,
      },
    });

  wgpuShaderModuleRelease(vs_module);
  wgpuShaderModuleRelease(fs_module);
  wgpuPipelineLayoutRelease(sprite_pl);
}

/* -------------------------------------------------------------------------- *
 * Compute pipelines (cluster bounds + cluster lights)
 * -------------------------------------------------------------------------- */

static void init_cluster_compute(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Cluster storage bind group layout */
  WGPUBindGroupLayoutEntry storage_entry = {
    .binding    = 0,
    .visibility = WGPUShaderStage_Compute,
    .buffer     = {.type = WGPUBufferBindingType_Storage},
  };
  state.cluster_storage_bgl
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 1,
                                                .entries    = &storage_entry,
                                              });

  /* Cluster bounds pipeline */
  WGPUBindGroupLayout bounds_layouts[2] = {
    state.bind_group_layouts.frame,
    state.cluster_storage_bgl,
  };
  WGPUPipelineLayout bounds_pl = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .bindGroupLayoutCount = 2,
              .bindGroupLayouts     = bounds_layouts,
            });

  WGPUShaderModule bounds_sm
    = wgpu_create_shader_module(device, cluster_bounds_compute_shader_wgsl);
  state.cluster_bounds_pipeline = wgpuDeviceCreateComputePipeline(device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("cluster-bounds"),
      .layout  = bounds_pl,
      .compute = {
        .module     = bounds_sm,
        .entryPoint = STRVIEW("main"),
      },
    });
  wgpuShaderModuleRelease(bounds_sm);
  wgpuPipelineLayoutRelease(bounds_pl);

  /* Cluster buffer (stores AABB bounds for each cluster) */
  state.buffers.cluster
    = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
                                       .usage = WGPUBufferUsage_Storage,
                                       .size  = TOTAL_TILES * 32,
                                     });

  /* Cluster storage bind group */
  WGPUBindGroupEntry cluster_storage_entry = {
    .binding = 0,
    .buffer  = state.buffers.cluster,
    .size    = TOTAL_TILES * 32,
  };
  state.cluster_storage_bind_group
    = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                          .layout = state.cluster_storage_bgl,
                                          .entryCount = 1,
                                          .entries    = &cluster_storage_entry,
                                        });

  /* Cluster read-only bind group (for rendering) */
  state.bind_groups.cluster = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .layout     = state.bind_group_layouts.cluster,
              .entryCount = 1,
              .entries    = &cluster_storage_entry,
            });

  /* Cluster lights pipeline */
  WGPUBindGroupLayout lights_layouts[2] = {
    state.bind_group_layouts.frame,
    state.bind_group_layouts.cluster,
  };
  WGPUPipelineLayout lights_pl = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .bindGroupLayoutCount = 2,
              .bindGroupLayouts     = lights_layouts,
            });

  WGPUShaderModule lights_sm
    = wgpu_create_shader_module(device, cluster_lights_compute_shader_wgsl);
  state.cluster_lights_pipeline = wgpuDeviceCreateComputePipeline(device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("cluster-lights"),
      .layout  = lights_pl,
      .compute = {
        .module     = lights_sm,
        .entryPoint = STRVIEW("main"),
      },
    });
  wgpuShaderModuleRelease(lights_sm);
  wgpuPipelineLayoutRelease(lights_pl);
}

/* -------------------------------------------------------------------------- *
 * MSAA and depth textures for rendering
 * -------------------------------------------------------------------------- */

static void create_render_textures(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Release old textures */
  if (state.msaa_view) {
    wgpuTextureViewRelease(state.msaa_view);
    state.msaa_view = NULL;
  }
  if (state.msaa_texture) {
    wgpuTextureRelease(state.msaa_texture);
    state.msaa_texture = NULL;
  }
  if (state.depth_view) {
    wgpuTextureViewRelease(state.depth_view);
    state.depth_view = NULL;
  }
  if (state.depth_texture) {
    wgpuTextureRelease(state.depth_texture);
    state.depth_texture = NULL;
  }

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;

  /* MSAA color texture */
  state.msaa_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = wgpu_context->render_format,
              .sampleCount   = SAMPLE_COUNT,
              .mipLevelCount = 1,
            });
  state.msaa_view = wgpuTextureCreateView(state.msaa_texture, NULL);

  /* Depth texture */
  state.depth_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = DEPTH_FORMAT,
              .sampleCount   = SAMPLE_COUNT,
              .mipLevelCount = 1,
            });
  state.depth_view = wgpuTextureCreateView(state.depth_texture, NULL);
}

/* -------------------------------------------------------------------------- *
 * Render bundle creation
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * Pipeline cache helpers
 * -------------------------------------------------------------------------- */

static pipeline_cache_key_t make_pipeline_key(const gltf_primitive_t* prim,
                                              output_type_t output_type)
{
  pipeline_cache_key_t key = {0};
  key.vertex_buffer_count  = prim->vertex_buffer_count;
  key.topology             = prim->topology;
  key.output_type          = output_type;

  /* Build attribute location bitmask and record strides */
  for (uint32_t i = 0; i < prim->vertex_buffer_count; ++i) {
    const vertex_buffer_layout_t* vbl = &prim->vertex_buffers[i];
    key.array_strides[i]              = vbl->array_stride;
    for (uint32_t a = 0; a < vbl->attribute_count; ++a) {
      key.attribute_locations |= (1u << vbl->attributes[a].shader_location);
    }
  }

  if (prim->material_idx >= 0) {
    key.blend        = state.gltf.materials[prim->material_idx].blend;
    key.double_sided = state.gltf.materials[prim->material_idx].double_sided;
  }

  /* For non-PBR visualization modes, material blend and double-sided are
   * irrelevant (fragment shader ignores material data). Override to maximize
   * pipeline cache hits — reduces unique pipelines from ~50 to ~5 per mode,
   * preventing multi-second freezes when switching output modes. */
  if (output_type != OUTPUT_NAIVE_FORWARD
      && output_type != OUTPUT_CLUSTERED_FWD) {
    key.blend        = false;
    key.double_sided = false;
  }

  return key;
}

static bool pipeline_keys_equal(const pipeline_cache_key_t* a,
                                const pipeline_cache_key_t* b)
{
  return a->vertex_buffer_count == b->vertex_buffer_count
         && a->attribute_locations == b->attribute_locations
         && a->blend == b->blend && a->double_sided == b->double_sided
         && a->topology == b->topology && a->output_type == b->output_type
         && memcmp(a->array_strides, b->array_strides,
                   sizeof(uint32_t) * a->vertex_buffer_count)
              == 0;
}

static WGPURenderPipeline cache_find_pipeline(const pipeline_cache_key_t* key)
{
  for (uint32_t i = 0; i < state.pipeline_cache_count; ++i) {
    if (state.pipeline_cache[i].valid
        && pipeline_keys_equal(&state.pipeline_cache[i].key, key)) {
      return state.pipeline_cache[i].pipeline;
    }
  }
  return NULL;
}

static void cache_store_pipeline(const pipeline_cache_key_t* key,
                                 WGPURenderPipeline pipeline)
{
  if (state.pipeline_cache_count < MAX_CACHED_PIPELINES) {
    state.pipeline_cache[state.pipeline_cache_count].key      = *key;
    state.pipeline_cache[state.pipeline_cache_count].pipeline = pipeline;
    state.pipeline_cache[state.pipeline_cache_count].valid    = true;
    state.pipeline_cache_count++;
  }
}

static void cache_clear_pipelines(void)
{
  for (uint32_t i = 0; i < state.pipeline_cache_count; ++i) {
    if (state.pipeline_cache[i].valid && state.pipeline_cache[i].pipeline) {
      wgpuRenderPipelineRelease(state.pipeline_cache[i].pipeline);
    }
    state.pipeline_cache[i].valid = false;
  }
  state.pipeline_cache_count = 0;
}

static WGPURenderPipeline create_pbr_pipeline(wgpu_context_t* wgpu_context,
                                              const gltf_primitive_t* prim,
                                              WGPUShaderModule vs_module,
                                              WGPUShaderModule fs_module,
                                              WGPUPipelineLayout layout)
{
  WGPUDevice device = wgpu_context->device;

  /* Build vertex buffer layouts */
  WGPUVertexBufferLayout vb_layouts[MAX_VERTEX_BUFFERS_PER_PRIM];
  WGPUVertexAttribute vb_attrs[MAX_VERTEX_BUFFERS_PER_PRIM * 8];
  uint32_t attr_offset = 0;

  for (uint32_t i = 0; i < prim->vertex_buffer_count; ++i) {
    const vertex_buffer_layout_t* vbl = &prim->vertex_buffers[i];
    WGPUVertexAttribute* attrs        = &vb_attrs[attr_offset];

    uint32_t valid_count = 0;
    for (uint32_t a = 0; a < vbl->attribute_count; ++a) {
      attrs[valid_count] = (WGPUVertexAttribute){
        .shaderLocation = vbl->attributes[a].shader_location,
        .format         = vbl->attributes[a].format,
        .offset         = vbl->attributes[a].offset,
      };
      valid_count++;
    }

    vb_layouts[i] = (WGPUVertexBufferLayout){
      .arrayStride    = vbl->array_stride,
      .attributeCount = valid_count,
      .attributes     = attrs,
    };
    attr_offset += valid_count;
  }

  /* Determine blend */
  bool blend = false;
  if (prim->material_idx >= 0) {
    blend = state.gltf.materials[prim->material_idx].blend;
  }

  /* Only set blend state for transparent materials (matching JS: undefined for
   * opaque). Using {0} gives WGPUBlendFactor_Undefined which makes pipeline
   * creation fail. */
  WGPUBlendState blend_state = {
    .color = {.srcFactor = WGPUBlendFactor_SrcAlpha,
              .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
              .operation = WGPUBlendOperation_Add},
    .alpha = {.srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_One,
              .operation = WGPUBlendOperation_Add},
  };

  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = blend ? &blend_state : NULL,
    .writeMask = WGPUColorWriteMask_All,
  };

  bool cull = true;
  if (prim->material_idx >= 0) {
    cull = !state.gltf.materials[prim->material_idx].double_sided;
  }

  WGPUDepthStencilState depth_stencil = {
    .format            = DEPTH_FORMAT,
    .depthWriteEnabled = true,
    .depthCompare      = WGPUCompareFunction_Less,
  };

  WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device,
    &(WGPURenderPipelineDescriptor){
      .layout = layout,
      .vertex = {
        .module      = vs_module,
        .entryPoint  = STRVIEW("main"),
        .bufferCount = prim->vertex_buffer_count,
        .buffers     = vb_layouts,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fs_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .primitive = {
        .topology = prim->topology,
        .cullMode = cull ? WGPUCullMode_Back : WGPUCullMode_None,
      },
      .depthStencil = &depth_stencil,
      .multisample = {
        .count = SAMPLE_COUNT,
        .mask  = 0xFFFFFFFF,
      },
    });

  return pipeline;
}

/* Create a render bundle for the current output type */
static void create_render_bundle(wgpu_context_t* wgpu_context,
                                 output_type_t output_type)
{
  if (!state.gltf.loaded)
    return;
  if (state.render_bundle_valid[output_type])
    return;

  /* Determine which shaders to use */
  const char* frag_shader = NULL;
  const char* vert_shader = NULL;

  switch (output_type) {
    case OUTPUT_NAIVE_FORWARD:
      frag_shader = pbr_fragment_naive_wgsl;
      vert_shader = pbr_vertex_shader_wgsl;
      break;
    case OUTPUT_CLUSTERED_FWD:
      frag_shader = pbr_fragment_clustered_wgsl;
      vert_shader = pbr_vertex_shader_wgsl;
      break;
    case OUTPUT_DEPTH:
      frag_shader = depth_viz_fragment_wgsl;
      vert_shader = simple_vertex_shader_wgsl;
      break;
    case OUTPUT_DEPTH_SLICE:
      frag_shader = depth_slice_viz_fragment_wgsl;
      vert_shader = simple_vertex_shader_wgsl;
      break;
    case OUTPUT_CLUSTER_DIST:
      frag_shader = cluster_dist_viz_fragment_wgsl;
      vert_shader = cluster_dist_viz_vertex_wgsl;
      break;
    case OUTPUT_LIGHTS_PER_CLST:
      frag_shader = lights_per_cluster_viz_fragment_wgsl;
      vert_shader = simple_vertex_shader_wgsl;
      break;
    default:
      return;
  }

  /* Select pipeline layout based on output type */
  WGPUPipelineLayout pl_layout = (output_type == OUTPUT_CLUSTER_DIST) ?
                                   state.cluster_dist_pipeline_layout :
                                   state.pipeline_layout;

  /* Pre-create shader modules once (avoids re-parsing WGSL per primitive) */
  WGPUDevice device       = wgpu_context->device;
  bool is_pbr             = (output_type == OUTPUT_NAIVE_FORWARD
                 || output_type == OUTPUT_CLUSTERED_FWD);
  WGPUShaderModule vs_mod = wgpu_create_shader_module(device, vert_shader);
  WGPUShaderModule vs_no_tan_mod = NULL;
  if (is_pbr) {
    vs_no_tan_mod
      = wgpu_create_shader_module(device, pbr_vertex_shader_no_tangent_wgsl);
  }
  WGPUShaderModule fs_mod = wgpu_create_shader_module(device, frag_shader);

  WGPUTextureFormat color_fmts[1] = {wgpu_context->render_format};
  WGPURenderBundleEncoder encoder = wgpuDeviceCreateRenderBundleEncoder(
    wgpu_context->device, &(WGPURenderBundleEncoderDescriptor){
                            .colorFormatCount   = 1,
                            .colorFormats       = color_fmts,
                            .depthStencilFormat = DEPTH_FORMAT,
                            .sampleCount        = SAMPLE_COUNT,
                          });

  /* Set frame bind group */
  wgpuRenderBundleEncoderSetBindGroup(encoder, 0, state.bind_groups.frame, 0,
                                      NULL);

  /* For cluster distance visualization, also set cluster bind group */
  if (output_type == OUTPUT_CLUSTER_DIST) {
    wgpuRenderBundleEncoderSetBindGroup(encoder, 3, state.bind_groups.cluster,
                                        0, NULL);
  }

  /* Draw all primitives, using cached pipelines for efficiency */
  for (uint32_t i = 0; i < state.gltf.primitive_count; ++i) {
    gltf_primitive_t* prim = &state.gltf.primitives[i];

    /* Select vertex shader: use no-tangent variant when primitive lacks
     * tangent data (avoids "attribute slot 3 not present" validation error) */
    WGPUShaderModule prim_vs
      = (is_pbr && !prim->has_tangents) ? vs_no_tan_mod : vs_mod;

    /* Look up or create pipeline for this primitive */
    pipeline_cache_key_t key    = make_pipeline_key(prim, output_type);
    WGPURenderPipeline pipeline = cache_find_pipeline(&key);
    if (!pipeline) {
      pipeline
        = create_pbr_pipeline(wgpu_context, prim, prim_vs, fs_mod, pl_layout);
      cache_store_pipeline(&key, pipeline);
    }

    wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);

    /* Set material bind group */
    if (prim->material_idx >= 0
        && prim->material_idx < (int32_t)state.gltf.material_count) {
      wgpuRenderBundleEncoderSetBindGroup(
        encoder, 1, state.gltf.materials[prim->material_idx].bind_group, 0,
        NULL);
    }

    /* Set model bind group */
    wgpuRenderBundleEncoderSetBindGroup(encoder, 2, prim->model_bind_group, 0,
                                        NULL);

    /* Set vertex buffers */
    for (uint32_t vb = 0; vb < prim->vertex_buffer_count; ++vb) {
      uint32_t bv_idx = prim->vertex_buffers[vb].buffer_view_idx;
      if (bv_idx < state.gltf.buffer_view_count
          && state.gltf.buffer_views[bv_idx].gpu_buffer) {
        wgpuRenderBundleEncoderSetVertexBuffer(
          encoder, vb, state.gltf.buffer_views[bv_idx].gpu_buffer,
          prim->vertex_buffers[vb].min_byte_offset, WGPU_WHOLE_SIZE);
      }
    }

    /* Draw */
    if (prim->index_buffer_view_idx != UINT32_MAX) {
      uint32_t ibv_idx = prim->index_buffer_view_idx;
      if (ibv_idx < state.gltf.buffer_view_count
          && state.gltf.buffer_views[ibv_idx].gpu_buffer) {
        wgpuRenderBundleEncoderSetIndexBuffer(
          encoder, state.gltf.buffer_views[ibv_idx].gpu_buffer,
          prim->index_format, prim->index_byte_offset, WGPU_WHOLE_SIZE);
        wgpuRenderBundleEncoderDrawIndexed(encoder, prim->element_count, 1, 0,
                                           0, 0);
      }
    }
    else {
      wgpuRenderBundleEncoderDraw(encoder, prim->element_count, 1, 0, 0);
    }
  }

  state.render_bundles[output_type]
    = wgpuRenderBundleEncoderFinish(encoder, NULL);
  state.render_bundle_valid[output_type] = true;

  wgpuRenderBundleEncoderRelease(encoder);

  /* Release pre-created shader modules */
  wgpuShaderModuleRelease(vs_mod);
  wgpuShaderModuleRelease(fs_mod);
  if (vs_no_tan_mod) {
    wgpuShaderModuleRelease(vs_no_tan_mod);
  }
}

/* -------------------------------------------------------------------------- *
 * Compute cluster bounds (called on resize)
 * -------------------------------------------------------------------------- */

static void compute_cluster_bounds(wgpu_context_t* wgpu_context)
{
  /* Update projection uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.projection, 0,
                       state.frame_uniforms, PROJECTION_UNIFORMS_SIZE);

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPUComputePassEncoder pass
    = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);

  wgpuComputePassEncoderSetPipeline(pass, state.cluster_bounds_pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, state.bind_groups.frame, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(pass, 1, state.cluster_storage_bind_group,
                                     0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, DISPATCH_SIZE_X,
                                           DISPATCH_SIZE_Y, DISPATCH_SIZE_Z);
  wgpuComputePassEncoderEnd(pass);

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);

  wgpuComputePassEncoderRelease(pass);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(cmd_enc);
}

/* -------------------------------------------------------------------------- *
 * Compute cluster lights (called every frame for clustered modes)
 * -------------------------------------------------------------------------- */

static void compute_cluster_lights(wgpu_context_t* wgpu_context,
                                   WGPUCommandEncoder cmd_enc)
{
  /* Reset offset counter to 0 */
  uint32_t zero = 0;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.cluster_lights, 0,
                       &zero, sizeof(uint32_t));

  WGPUComputePassEncoder pass
    = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
  wgpuComputePassEncoderSetPipeline(pass, state.cluster_lights_pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, state.bind_groups.frame, 0, NULL);
  wgpuComputePassEncoderSetBindGroup(pass, 1, state.bind_groups.cluster, 0,
                                     NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, DISPATCH_SIZE_X,
                                           DISPATCH_SIZE_Y, DISPATCH_SIZE_Z);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
}

/* -------------------------------------------------------------------------- *
 * Invalidate render bundles (called on resize / output type change)
 * -------------------------------------------------------------------------- */

static void invalidate_render_bundles(void)
{
  for (int i = 0; i < OUTPUT_COUNT; ++i) {
    if (state.render_bundle_valid[i] && state.render_bundles[i]) {
      wgpuRenderBundleRelease(state.render_bundles[i]);
      state.render_bundles[i] = NULL;
    }
    state.render_bundle_valid[i] = false;
  }
  cache_clear_pipelines();
}

/* -------------------------------------------------------------------------- *
 * Update frame uniforms
 * -------------------------------------------------------------------------- */

static void update_frame_uniforms(wgpu_context_t* wgpu_context)
{
  float* fu = state.frame_uniforms;

  /* Projection matrix [0..15] */
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  float fov    = (float)GLM_PI * 0.5f;
  float z_near = 0.2f;
  float z_far  = 100.0f;

  mat4 proj;
  /* perspectiveZO: clip Z range [0,1] for WebGPU */
  glm_perspective(fov, aspect, z_near, z_far, proj);
  /* cglm glm_perspective already uses [0,1] depth by default with
   * GLM_CLIP_CONTROL_ZO. Check if it needs adjustment. */
  memcpy(&fu[0], proj, 16 * sizeof(float));

  /* Inverse projection matrix [16..31] */
  mat4 inv_proj;
  glm_mat4_inv(proj, inv_proj);
  memcpy(&fu[16], inv_proj, 16 * sizeof(float));

  /* Output size [32..33] */
  fu[32] = (float)wgpu_context->width;
  fu[33] = (float)wgpu_context->height;

  /* Z range [34..35] */
  fu[34] = z_near;
  fu[35] = z_far;

  /* View matrix [36..51] */
  mat4 view_mat;
  camera_get_view_matrix(view_mat);
  memcpy(&fu[36], view_mat, 16 * sizeof(float));

  /* Camera position [52..54] */
  fu[52] = state.camera.position[0];
  fu[53] = state.camera.position[1];
  fu[54] = state.camera.position[2];
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  switch (input_event->type) {
    case INPUT_EVENT_TYPE_KEY_DOWN: {
      keycode_t k = input_event->key_code;
      if (k == KEY_W)
        state.camera.key_w = true;
      if (k == KEY_S)
        state.camera.key_s = true;
      if (k == KEY_A)
        state.camera.key_a = true;
      if (k == KEY_D)
        state.camera.key_d = true;
      if (k == KEY_SPACE)
        state.camera.key_space = true;
      if (k == KEY_LEFT_SHIFT || k == KEY_RIGHT_SHIFT)
        state.camera.key_shift = true;
    } break;
    case INPUT_EVENT_TYPE_KEY_UP: {
      keycode_t k = input_event->key_code;
      if (k == KEY_W)
        state.camera.key_w = false;
      if (k == KEY_S)
        state.camera.key_s = false;
      if (k == KEY_A)
        state.camera.key_a = false;
      if (k == KEY_D)
        state.camera.key_d = false;
      if (k == KEY_SPACE)
        state.camera.key_space = false;
      if (k == KEY_LEFT_SHIFT || k == KEY_RIGHT_SHIFT)
        state.camera.key_shift = false;
    } break;
    case INPUT_EVENT_TYPE_MOUSE_DOWN:
      state.camera.moving = true;
      state.camera.last_x = input_event->mouse_x;
      state.camera.last_y = input_event->mouse_y;
      break;
    case INPUT_EVENT_TYPE_MOUSE_UP:
      state.camera.moving = false;
      break;
    case INPUT_EVENT_TYPE_MOUSE_MOVE:
      if (state.camera.moving) {
        float dx = input_event->mouse_dx;
        float dy = input_event->mouse_dy;
        camera_rotate_view(dx * 0.025f, dy * 0.025f);
      }
      break;
    default:
      break;
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static const char* output_type_names[OUTPUT_COUNT] = {
  "Naive Forward",      "Depth",
  "Depth Slice",        "Cluster Distance",
  "Lights Per Cluster", "Clustered Forward",
};

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);
  igBegin("Clustered Shading", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Show loading status based on phase */
  if (state.load_phase <= LOAD_PHASE_GLTF) {
    igTextColored((ImVec4){1.0f, 1.0f, 0.0f, 1.0f}, "Loading model...");
    igSeparator();
  }

  /* Output mode selector */
  int output_mode = (int)state.settings.output_type;
  if (imgui_overlay_combo_box("Output", &output_mode, output_type_names,
                              OUTPUT_COUNT)) {
    state.settings.output_type = (output_type_t)output_mode;
  }

  /* Render light sprites toggle */
  igCheckbox("Light Sprites", &state.settings.render_light_sprites);
  state.light_mgr.render_sprites = state.settings.render_light_sprites;

  /* Light count slider */
  if (igSliderInt("Light Count", &state.settings.light_count, 5, 1024, "%d")) {
    state.light_mgr.light_count = (uint32_t)MIN(
      state.settings.light_count, (int32_t)state.light_mgr.max_light_count);
  }

  /* Max light range slider */
  if (igSliderFloat("Max Light Range", &state.settings.max_light_range, 0.1f,
                    5.0f, "%.1f", 0)) {
    update_light_range(state.settings.max_light_range);
  }

  /* Loading progress for textures */
  if (state.load_phase == LOAD_PHASE_TEXTURES
      && state.pending_images.total > 0) {
    igSeparator();
    float progress = (float)state.pending_images.loaded_count
                     / (float)state.pending_images.total;
    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%u / %u textures",
             state.pending_images.loaded_count, state.pending_images.total);
    igProgressBar(progress, (ImVec2){-1.0f, 0.0f}, overlay);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context)
    return EXIT_FAILURE;

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = MAX_IMAGES,
    .num_channels = 1,
    .num_lanes    = 1,
  });

  camera_init();
  light_manager_init();
  build_pbr_fragment_shaders();

  state.context_format = wgpu_context->render_format;

  /* Create default textures and bind group layouts first */
  create_default_textures(wgpu_context);
  init_bind_group_layouts(wgpu_context);
  init_gpu_buffers(wgpu_context);

  /* Create pipelines (independent of GLTF data) */
  init_light_sprite_pipeline(wgpu_context);
  init_cluster_compute(wgpu_context);

  /* Create render textures */
  create_render_textures(wgpu_context);

  /* Initial projection + cluster bounds */
  update_frame_uniforms(wgpu_context);
  compute_cluster_bounds(wgpu_context);

  /* Init ImGui */
  imgui_overlay_init(wgpu_context);

  /* GLTF model loading is deferred to frame() so that light sprites and
   * the GUI are visible immediately while the model loads. */
  state.pending_images.wgpu_context = wgpu_context;
  state.load_phase                  = LOAD_PHASE_INIT;

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized)
    return EXIT_FAILURE;

  /* Process async image loading */
  sfetch_dowork();

  /* ---- Deferred loading state machine ---- */
  switch (state.load_phase) {
    case LOAD_PHASE_INIT:
      /* Let a few frames render (light sprites + GUI) before loading GLTF */
      if (state.frame_count >= 2) {
        state.load_phase = LOAD_PHASE_GLTF;
      }
      break;
    case LOAD_PHASE_GLTF:
      /* Load the GLTF model (file I/O + GPU buffer creation). This blocks
       * for a short time, but the user already sees light sprites + GUI. */
      load_gltf(wgpu_context, "assets/models/Sponza/glTF/Sponza.gltf");
      start_async_image_loading();
      state.load_phase = LOAD_PHASE_TEXTURES;
      break;
    case LOAD_PHASE_TEXTURES:
      if (state.pending_images.all_done) {
        state.load_phase = LOAD_PHASE_DONE;
      }
      break;
    case LOAD_PHASE_DONE:
      break;
  }

  /* Timing */
  uint64_t current_time = stm_now();
  float time_delta      = 0.0f;
  float timestamp_ms    = (float)stm_ms(current_time);

  if (state.last_frame_time > 0) {
    time_delta = (float)stm_ms(stm_diff(current_time, state.last_frame_time));
  }
  state.last_frame_time = current_time;

  state.frame_count++;
  /* Skip every 200th frame (matching JS) */
  if (state.frame_count % 200 == 0)
    return EXIT_SUCCESS;

  /* Update camera */
  camera_update(time_delta);

  /* Bob corner lights */
  for (int i = 0; i < 4; ++i) {
    state.light_mgr.lights[i].position[1]
      = 1.25f + sinf((timestamp_ms + (float)i * 250.0f) / 500.0f) * 0.25f;
  }

  /* Update wandering lights */
  update_wandering_lights(time_delta);

  /* Sync uniform array */
  light_manager_update_uniform_array();

  /* Update frame uniforms */
  update_frame_uniforms(wgpu_context);

  /* Detect resize: recreate render textures + recompute clusters */
  static int prev_w = 0, prev_h = 0;
  if (wgpu_context->width != prev_w || wgpu_context->height != prev_h) {
    prev_w = wgpu_context->width;
    prev_h = wgpu_context->height;
    create_render_textures(wgpu_context);
    invalidate_render_bundles();
    compute_cluster_bounds(wgpu_context);
  }

  /* ImGui frame */
  float gui_delta = time_delta / 1000.0f;
  imgui_overlay_new_frame(wgpu_context, gui_delta > 0.0f ? gui_delta : 0.016f);
  render_gui(wgpu_context);

  /* Upload projection uniforms (needed for cluster compute and debug viz) */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.projection, 0,
                       state.frame_uniforms, PROJECTION_UNIFORMS_SIZE);

  /* Upload view uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.view, 0,
                       &state.frame_uniforms[36], VIEW_UNIFORMS_SIZE);

  /* Upload light uniforms */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.buffers.lights, 0,
                       state.light_mgr.uniform_array,
                       sizeof(state.light_mgr.uniform_array));

  /* Create render bundle if needed */
  output_type_t out = state.settings.output_type;
  create_render_bundle(wgpu_context, out);

  /* Set render pass attachments */
  state.color_attachment.view          = state.msaa_view;
  state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view  = state.depth_view;

  WGPUDevice device          = wgpu_context->device;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute cluster lights if needed */
  if (out == OUTPUT_CLUSTERED_FWD || out == OUTPUT_LIGHTS_PER_CLST) {
    compute_cluster_lights(wgpu_context, cmd_enc);
  }

  /* Render pass */
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  if (state.render_bundle_valid[out]) {
    wgpuRenderPassEncoderExecuteBundles(rpass, 1, &state.render_bundles[out]);
  }

  /* Render light sprites */
  if (state.light_mgr.render_sprites) {
    wgpuRenderPassEncoderSetPipeline(rpass, state.light_sprite_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bind_groups.frame, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(rpass, 4, state.light_mgr.light_count, 0, 0);
  }

  wgpuRenderPassEncoderEnd(rpass);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Release render bundles and cached pipelines */
  invalidate_render_bundles();

  /* Release GLTF resources */
  for (uint32_t i = 0; i < state.gltf.buffer_view_count; ++i) {
    WGPU_RELEASE_RESOURCE(Buffer, state.gltf.buffer_views[i].gpu_buffer)
  }
  for (uint32_t i = 0; i < state.gltf.image_count; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, state.gltf.images[i].view)
    WGPU_RELEASE_RESOURCE(Texture, state.gltf.images[i].texture)
  }
  for (uint32_t i = 0; i < state.gltf.sampler_count; ++i) {
    WGPU_RELEASE_RESOURCE(Sampler, state.gltf.samplers[i].gpu_sampler)
  }
  for (uint32_t i = 0; i < state.gltf.material_count; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.gltf.materials[i].bind_group)
    WGPU_RELEASE_RESOURCE(Buffer, state.gltf.materials[i].uniform_buffer)
  }
  for (uint32_t i = 0; i < state.gltf.primitive_count; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.gltf.primitives[i].model_bind_group)
    WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers[i])
  }

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.projection)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.view)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.lights)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.cluster_lights)
  WGPU_RELEASE_RESOURCE(Buffer, state.buffers.cluster)

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.frame)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.cluster)
  WGPU_RELEASE_RESOURCE(BindGroup, state.cluster_storage_bind_group)

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.frame)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.material)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.primitive)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layouts.cluster)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cluster_storage_bgl)

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.cluster_dist_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.light_sprite_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.cluster_bounds_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.cluster_lights_pipeline)

  /* Release default textures */
  WGPU_RELEASE_RESOURCE(TextureView, state.default_textures.black)
  WGPU_RELEASE_RESOURCE(TextureView, state.default_textures.white)
  WGPU_RELEASE_RESOURCE(TextureView, state.default_textures.blue)
  WGPU_RELEASE_RESOURCE(Texture, state.default_textures.black_tex)
  WGPU_RELEASE_RESOURCE(Texture, state.default_textures.white_tex)
  WGPU_RELEASE_RESOURCE(Texture, state.default_textures.blue_tex)
  WGPU_RELEASE_RESOURCE(Sampler, state.default_sampler)

  /* Release render textures */
  WGPU_RELEASE_RESOURCE(TextureView, state.msaa_view)
  WGPU_RELEASE_RESOURCE(Texture, state.msaa_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_texture)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Clustered Forward Shading",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shader strings
 * -------------------------------------------------------------------------- */

/* clang-format off */

/* ---- Common shader structs ---- */

#define PROJECTION_UNIFORMS_WGSL \
  "struct ProjectionUniforms {\n" \
  "  matrix : mat4x4<f32>,\n" \
  "  inverseMatrix : mat4x4<f32>,\n" \
  "  outputSize : vec2<f32>,\n" \
  "  zNear : f32,\n" \
  "  zFar : f32\n" \
  "};\n" \
  "@group(0) @binding(0) var<uniform> projection : ProjectionUniforms;\n"

#define VIEW_UNIFORMS_WGSL \
  "struct ViewUniforms {\n" \
  "  matrix : mat4x4<f32>,\n" \
  "  position : vec3<f32>\n" \
  "};\n" \
  "@group(0) @binding(1) var<uniform> view : ViewUniforms;\n"

#define LIGHT_UNIFORMS_WGSL \
  "struct Light {\n" \
  "  position : vec3<f32>,\n" \
  "  range : f32,\n" \
  "  color : vec3<f32>,\n" \
  "  intensity : f32\n" \
  "};\n" \
  "struct GlobalLightUniforms {\n" \
  "  ambient : vec3<f32>,\n" \
  "  lightCount : u32,\n" \
  "  lights : array<Light>\n" \
  "};\n" \
  "@group(0) @binding(2) var<storage> globalLights : GlobalLightUniforms;\n"

#define MODEL_UNIFORMS_WGSL \
  "struct ModelUniforms {\n" \
  "  matrix : mat4x4<f32>\n" \
  "};\n" \
  "@group(2) @binding(0) var<uniform> model : ModelUniforms;\n"

#define MATERIAL_UNIFORMS_WGSL \
  "struct MaterialUniforms {\n" \
  "  baseColorFactor : vec4<f32>,\n" \
  "  metallicRoughnessFactor : vec2<f32>,\n" \
  "  emissiveFactor : vec3<f32>,\n" \
  "  occlusionStrength : f32\n" \
  "};\n" \
  "@group(1) @binding(0) var<uniform> material : MaterialUniforms;\n" \
  "@group(1) @binding(1) var defaultSampler : sampler;\n" \
  "@group(1) @binding(2) var baseColorTexture : texture_2d<f32>;\n" \
  "@group(1) @binding(3) var normalTexture : texture_2d<f32>;\n" \
  "@group(1) @binding(4) var metallicRoughnessTexture : texture_2d<f32>;\n" \
  "@group(1) @binding(5) var occlusionTexture : texture_2d<f32>;\n" \
  "@group(1) @binding(6) var emissiveTexture : texture_2d<f32>;\n"

#define COLOR_CONVERSIONS_WGSL \
  "fn linearTosRGB(linear : vec3<f32>) -> vec3<f32> {\n" \
  "  if (all(linear <= vec3<f32>(0.0031308, 0.0031308, 0.0031308))) {\n" \
  "    return linear * 12.92;\n" \
  "  }\n" \
  "  return (pow(abs(linear), vec3<f32>(1.0/2.4, 1.0/2.4, 1.0/2.4)) * 1.055) - vec3<f32>(0.055, 0.055, 0.055);\n" \
  "}\n" \
  "fn sRGBToLinear(srgb : vec3<f32>) -> vec3<f32> {\n" \
  "  if (all(srgb <= vec3<f32>(0.04045, 0.04045, 0.04045))) {\n" \
  "    return srgb / vec3<f32>(12.92, 12.92, 12.92);\n" \
  "  }\n" \
  "  return pow((srgb + vec3<f32>(0.055, 0.055, 0.055)) / vec3<f32>(1.055, 1.055, 1.055), vec3<f32>(2.4, 2.4, 2.4));\n" \
  "}\n"

/* Tile / cluster helper functions */
#define TILE_COUNT_WGSL \
  "const tileCount : vec3<u32> = vec3<u32>(32u, 18u, 48u);\n"

#define TILE_FUNCTIONS_WGSL \
  TILE_COUNT_WGSL \
  "fn linearDepth(depthSample : f32) -> f32 {\n" \
  "  return projection.zFar*projection.zNear / fma(depthSample, projection.zNear-projection.zFar, projection.zFar);\n" \
  "}\n" \
  "fn getTile(fragCoord : vec4<f32>) -> vec3<u32> {\n" \
  "  let sliceScale = f32(tileCount.z) / log2(projection.zFar / projection.zNear);\n" \
  "  let sliceBias = -(f32(tileCount.z) * log2(projection.zNear) / log2(projection.zFar / projection.zNear));\n" \
  "  let zTile = u32(max(log2(linearDepth(fragCoord.z)) * sliceScale + sliceBias, 0.0));\n" \
  "  return vec3<u32>(u32(fragCoord.x / (projection.outputSize.x / f32(tileCount.x))),\n" \
  "                   u32(fragCoord.y / (projection.outputSize.y / f32(tileCount.y))),\n" \
  "                   zTile);\n" \
  "}\n" \
  "fn getClusterIndex(fragCoord : vec4<f32>) -> u32 {\n" \
  "  let tile = getTile(fragCoord);\n" \
  "  return tile.x +\n" \
  "         tile.y * tileCount.x +\n" \
  "         tile.z * tileCount.x * tileCount.y;\n" \
  "}\n"

#define CLUSTER_STRUCTS_WGSL \
  "struct ClusterBounds {\n" \
  "  minAABB : vec3<f32>,\n" \
  "  maxAABB : vec3<f32>\n" \
  "};\n" \
  "struct Clusters {\n" \
  "  bounds : array<ClusterBounds, 27648>\n" \
  "};\n"

#define CLUSTER_LIGHTS_STRUCTS_WGSL \
  "struct ClusterLights {\n" \
  "  offset : u32,\n" \
  "  count : u32\n" \
  "};\n" \
  "struct ClusterLightGroup {\n" \
  "  offset : atomic<u32>,\n" \
  "  lights : array<ClusterLights, 27648>,\n" \
  "  indices : array<u32, 2764800>\n" \
  "};\n" \
  "@group(0) @binding(3) var<storage, read_write> clusterLights : ClusterLightGroup;\n"

/* Read-only version for fragment shaders (no atomic<u32>, keep read_write to
 * match bind group layout type 'storage') */
#define CLUSTER_LIGHTS_STRUCTS_FRAG_WGSL \
  "struct ClusterLights {\n" \
  "  offset : u32,\n" \
  "  count : u32\n" \
  "};\n" \
  "struct ClusterLightGroup {\n" \
  "  offset : u32,\n" \
  "  lights : array<ClusterLights, 27648>,\n" \
  "  indices : array<u32, 2764800>\n" \
  "};\n" \
  "@group(0) @binding(3) var<storage, read_write> clusterLights : ClusterLightGroup;\n"

/* ---- PBR Vertex Shader ---- */
/* Note: This is a single combined shader that handles all define variants.
 * We always include all attributes and conditionally use them via material. */
static const char* pbr_vertex_shader_wgsl =
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  MODEL_UNIFORMS_WGSL
  "struct VertexOutput {\n"
  "  @builtin(position) position : vec4<f32>,\n"
  "  @location(0) worldPos : vec3<f32>,\n"
  "  @location(1) view : vec3<f32>,\n"
  "  @location(2) texCoord : vec2<f32>,\n"
  "  @location(3) color : vec4<f32>,\n"
  "  @location(4) normal : vec3<f32>,\n"
  "  @location(5) tangent : vec3<f32>,\n"
  "  @location(6) bitangent : vec3<f32>,\n"
  "};\n"
  "struct VertexInputs {\n"
  "  @location(1) position : vec3<f32>,\n"
  "  @location(2) normal : vec3<f32>,\n"
  "  @location(3) tangent : vec4<f32>,\n"
  "  @location(4) texCoord : vec2<f32>,\n"
  "};\n"
  "@vertex\n"
  "fn main(input : VertexInputs) -> VertexOutput {\n"
  "  var output : VertexOutput;\n"
  "  output.normal = normalize((model.matrix * vec4<f32>(input.normal, 0.0)).xyz);\n"
  "  if (dot(input.tangent.xyz, input.tangent.xyz) > 0.0) {\n"
  "    output.tangent = normalize((model.matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz);\n"
  "    output.bitangent = cross(output.normal, output.tangent) * input.tangent.w;\n"
  "  } else {\n"
  "    output.tangent = vec3<f32>(0.0, 0.0, 0.0);\n"
  "    output.bitangent = vec3<f32>(0.0, 0.0, 0.0);\n"
  "  }\n"
  "  output.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);\n"
  "  output.texCoord = input.texCoord;\n"
  "  let modelPos = model.matrix * vec4<f32>(input.position, 1.0);\n"
  "  output.worldPos = modelPos.xyz;\n"
  "  output.view = view.position - modelPos.xyz;\n"
  "  output.position = projection.matrix * view.matrix * modelPos;\n"
  "  return output;\n"
  "}\n";

/* ---- PBR Vertex Shader (no-tangent variant) ---- */
/* Used for primitives that don't have tangent vertex data to avoid
 * the "Vertex attribute slot 3 used in shader but not present in
 * VertexState" validation error. */
static const char* pbr_vertex_shader_no_tangent_wgsl =
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  MODEL_UNIFORMS_WGSL
  "struct VertexOutput {\n"
  "  @builtin(position) position : vec4<f32>,\n"
  "  @location(0) worldPos : vec3<f32>,\n"
  "  @location(1) view : vec3<f32>,\n"
  "  @location(2) texCoord : vec2<f32>,\n"
  "  @location(3) color : vec4<f32>,\n"
  "  @location(4) normal : vec3<f32>,\n"
  "  @location(5) tangent : vec3<f32>,\n"
  "  @location(6) bitangent : vec3<f32>,\n"
  "};\n"
  "struct VertexInputs {\n"
  "  @location(1) position : vec3<f32>,\n"
  "  @location(2) normal : vec3<f32>,\n"
  "  @location(4) texCoord : vec2<f32>,\n"
  "};\n"
  "@vertex\n"
  "fn main(input : VertexInputs) -> VertexOutput {\n"
  "  var output : VertexOutput;\n"
  "  output.normal = normalize((model.matrix * vec4<f32>(input.normal, 0.0)).xyz);\n"
  "  output.tangent = vec3<f32>(0.0, 0.0, 0.0);\n"
  "  output.bitangent = vec3<f32>(0.0, 0.0, 0.0);\n"
  "  output.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);\n"
  "  output.texCoord = input.texCoord;\n"
  "  let modelPos = model.matrix * vec4<f32>(input.position, 1.0);\n"
  "  output.worldPos = modelPos.xyz;\n"
  "  output.view = view.position - modelPos.xyz;\n"
  "  output.position = projection.matrix * view.matrix * modelPos;\n"
  "  return output;\n"
  "}\n";

/* ---- PBR Functions (shared between naive and clustered) ---- */
#define PBR_SURFACE_AND_FUNCTIONS_WGSL_A \
  "struct VertexOutput {\n" \
  "  @builtin(position) position : vec4<f32>,\n" \
  "  @location(0) worldPos : vec3<f32>,\n" \
  "  @location(1) view : vec3<f32>,\n" \
  "  @location(2) texCoord : vec2<f32>,\n" \
  "  @location(3) color : vec4<f32>,\n" \
  "  @location(4) normal : vec3<f32>,\n" \
  "  @location(5) tangent : vec3<f32>,\n" \
  "  @location(6) bitangent : vec3<f32>,\n" \
  "};\n" \
  "struct SurfaceInfo {\n" \
  "  baseColor : vec4<f32>,\n" \
  "  albedo : vec3<f32>,\n" \
  "  metallic : f32,\n" \
  "  roughness : f32,\n" \
  "  normal : vec3<f32>,\n" \
  "  f0 : vec3<f32>,\n" \
  "  ao : f32,\n" \
  "  emissive : vec3<f32>,\n" \
  "  v : vec3<f32>\n" \
  "};\n" \
  "fn GetSurfaceInfo(input : VertexOutput) -> SurfaceInfo {\n" \
  "  var surface : SurfaceInfo;\n" \
  "  surface.v = normalize(input.view);\n" \
  "  surface.baseColor = material.baseColorFactor * input.color;\n" \
  "  let baseColorMap = textureSample(baseColorTexture, defaultSampler, input.texCoord);\n" \
  "  surface.baseColor = surface.baseColor * baseColorMap;\n" \
  "  surface.albedo = surface.baseColor.rgb;\n" \
  "  surface.metallic = material.metallicRoughnessFactor.x;\n" \
  "  surface.roughness = material.metallicRoughnessFactor.y;\n" \
  "  let metallicRoughness = textureSample(metallicRoughnessTexture, defaultSampler, input.texCoord);\n" \
  "  surface.metallic = surface.metallic * metallicRoughness.b;\n" \
  "  surface.roughness = surface.roughness * metallicRoughness.g;\n" \
  "  let tbn = mat3x3<f32>(input.tangent, input.bitangent, input.normal);\n" \
  "  let N = textureSample(normalTexture, defaultSampler, input.texCoord).rgb;\n" \
  "  let tangentNormal = normalize(tbn * (2.0 * N - vec3<f32>(1.0, 1.0, 1.0)));\n" \
  "  let hasTangent = dot(input.tangent, input.tangent) > 0.001;\n" \
  "  surface.normal = select(normalize(input.normal), tangentNormal, hasTangent);\n" \
  "  let dielectricSpec = vec3<f32>(0.04, 0.04, 0.04);\n" \
  "  surface.f0 = mix(dielectricSpec, surface.albedo, vec3<f32>(surface.metallic, surface.metallic, surface.metallic));\n" \
  "  surface.ao = textureSample(occlusionTexture, defaultSampler, input.texCoord).r * material.occlusionStrength;\n" \
  "  surface.emissive = material.emissiveFactor;\n" \
  "  surface.emissive = surface.emissive * textureSample(emissiveTexture, defaultSampler, input.texCoord).rgb;\n" \
  "  return surface;\n" \
  "}\n"

#define PBR_SURFACE_AND_FUNCTIONS_WGSL_B \
  "const PI = 3.141592653589793;\n" \
  "const LightType_Point = 0u;\n" \
  "const LightType_Spot = 1u;\n" \
  "const LightType_Directional = 2u;\n" \
  "struct PuctualLight {\n" \
  "  lightType : u32,\n" \
  "  pointToLight : vec3<f32>,\n" \
  "  range : f32,\n" \
  "  color : vec3<f32>,\n" \
  "  intensity : f32\n" \
  "};\n" \
  "fn FresnelSchlick(cosTheta : f32, F0 : vec3<f32>) -> vec3<f32> {\n" \
  "  return F0 + (vec3<f32>(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);\n" \
  "}\n" \
  "fn DistributionGGX(N : vec3<f32>, H : vec3<f32>, roughness : f32) -> f32 {\n" \
  "  let a = roughness*roughness;\n" \
  "  let a2 = a*a;\n" \
  "  let NdotH = max(dot(N, H), 0.0);\n" \
  "  let NdotH2 = NdotH*NdotH;\n" \
  "  let num = a2;\n" \
  "  let denom = (NdotH2 * (a2 - 1.0) + 1.0);\n" \
  "  return num / (PI * denom * denom);\n" \
  "}\n" \
  "fn GeometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {\n" \
  "  let r = (roughness + 1.0);\n" \
  "  let k = (r*r) / 8.0;\n" \
  "  let num = NdotV;\n" \
  "  let denom = NdotV * (1.0 - k) + k;\n" \
  "  return num / denom;\n" \
  "}\n" \
  "fn GeometrySmith(N : vec3<f32>, V : vec3<f32>, L : vec3<f32>, roughness : f32) -> f32 {\n" \
  "  let NdotV = max(dot(N, V), 0.0);\n" \
  "  let NdotL = max(dot(N, L), 0.0);\n" \
  "  let ggx2 = GeometrySchlickGGX(NdotV, roughness);\n" \
  "  let ggx1 = GeometrySchlickGGX(NdotL, roughness);\n" \
  "  return ggx1 * ggx2;\n" \
  "}\n" \
  "fn rangeAttenuation(range : f32, distance : f32) -> f32 {\n" \
  "  if (range <= 0.0) {\n" \
  "    return 1.0 / pow(distance, 2.0);\n" \
  "  }\n" \
  "  return clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0) / pow(distance, 2.0);\n" \
  "}\n" \
  "fn lightRadiance(light : PuctualLight, surface : SurfaceInfo) -> vec3<f32> {\n" \
  "  let L = normalize(light.pointToLight);\n" \
  "  let H = normalize(surface.v + L);\n" \
  "  let distance = length(light.pointToLight);\n" \
  "  let NDF = DistributionGGX(surface.normal, H, surface.roughness);\n" \
  "  let G = GeometrySmith(surface.normal, surface.v, L, surface.roughness);\n" \
  "  let F = FresnelSchlick(max(dot(H, surface.v), 0.0), surface.f0);\n" \
  "  let kD = (vec3<f32>(1.0, 1.0, 1.0) - F) * (1.0 - surface.metallic);\n" \
  "  let NdotL = max(dot(surface.normal, L), 0.0);\n" \
  "  let numerator = NDF * G * F;\n" \
  "  let denominator = max(4.0 * max(dot(surface.normal, surface.v), 0.0) * NdotL, 0.001);\n" \
  "  let specular = numerator / vec3<f32>(denominator, denominator, denominator);\n" \
  "  let attenuation = rangeAttenuation(light.range, distance);\n" \
  "  let radiance = light.color * light.intensity * attenuation;\n" \
  "  return (kD * surface.albedo / vec3<f32>(PI, PI, PI) + specular) * radiance * NdotL;\n" \
  "}\n"

/* ---- PBR Naive Forward Fragment Shader (built at runtime) ---- */
static char pbr_fragment_naive_buf[16384];
static char pbr_fragment_clustered_buf[16384];

static void build_pbr_fragment_shaders(void)
{
  /* Naive */
  snprintf(pbr_fragment_naive_buf, sizeof(pbr_fragment_naive_buf),
           "%s%s%s%s%s%s",
           COLOR_CONVERSIONS_WGSL,
           LIGHT_UNIFORMS_WGSL,
           MATERIAL_UNIFORMS_WGSL,
           PBR_SURFACE_AND_FUNCTIONS_WGSL_A,
           PBR_SURFACE_AND_FUNCTIONS_WGSL_B,
           "@fragment\n"
           "fn main(input : VertexOutput) -> @location(0) vec4<f32> {\n"
           "  let surface = GetSurfaceInfo(input);\n"
           "  var Lo = vec3<f32>(0.0, 0.0, 0.0);\n"
           "  for (var i = 0u; i < globalLights.lightCount; i = i + 1u) {\n"
           "    var light : PuctualLight;\n"
           "    light.lightType = LightType_Point;\n"
           "    light.pointToLight = globalLights.lights[i].position.xyz - input.worldPos;\n"
           "    light.range = globalLights.lights[i].range;\n"
           "    light.color = globalLights.lights[i].color;\n"
           "    light.intensity = 1.0;\n"
           "    Lo = Lo + lightRadiance(light, surface);\n"
           "  }\n"
           "  let ambient = globalLights.ambient * surface.albedo * surface.ao;\n"
           "  let color = linearTosRGB(Lo + ambient + surface.emissive);\n"
           "  return vec4<f32>(color, surface.baseColor.a);\n"
           "}\n");
  pbr_fragment_naive_wgsl = pbr_fragment_naive_buf;

  /* Clustered */
  snprintf(pbr_fragment_clustered_buf, sizeof(pbr_fragment_clustered_buf),
           "%s%s%s%s%s%s%s%s",
           COLOR_CONVERSIONS_WGSL,
           PROJECTION_UNIFORMS_WGSL,
           CLUSTER_LIGHTS_STRUCTS_FRAG_WGSL,
           MATERIAL_UNIFORMS_WGSL,
           LIGHT_UNIFORMS_WGSL,
           TILE_FUNCTIONS_WGSL,
           PBR_SURFACE_AND_FUNCTIONS_WGSL_A,
           PBR_SURFACE_AND_FUNCTIONS_WGSL_B);

  /* Append the main function */
  size_t len = strlen(pbr_fragment_clustered_buf);
  snprintf(pbr_fragment_clustered_buf + len,
           sizeof(pbr_fragment_clustered_buf) - len,
           "%s",
           "@fragment\n"
           "fn main(input : VertexOutput) -> @location(0) vec4<f32> {\n"
           "  let surface = GetSurfaceInfo(input);\n"
           "  if (surface.baseColor.a < 0.05) {\n"
           "    discard;\n"
           "  }\n"
           "  var Lo = vec3<f32>(0.0, 0.0, 0.0);\n"
           "  let clusterIndex = getClusterIndex(input.position);\n"
           "  let lightOffset = clusterLights.lights[clusterIndex].offset;\n"
           "  let lightCount = clusterLights.lights[clusterIndex].count;\n"
           "  for (var lightIndex = 0u; lightIndex < lightCount; lightIndex = lightIndex + 1u) {\n"
           "    let i = clusterLights.indices[lightOffset + lightIndex];\n"
           "    var light : PuctualLight;\n"
           "    light.lightType = LightType_Point;\n"
           "    light.pointToLight = globalLights.lights[i].position.xyz - input.worldPos;\n"
           "    light.range = globalLights.lights[i].range;\n"
           "    light.color = globalLights.lights[i].color;\n"
           "    light.intensity = 1.0;\n"
           "    Lo = Lo + lightRadiance(light, surface);\n"
           "  }\n"
           "  let ambient = globalLights.ambient * surface.albedo * surface.ao;\n"
           "  let color = linearTosRGB(Lo + ambient + surface.emissive);\n"
           "  return vec4<f32>(color, surface.baseColor.a);\n"
           "}\n");
  pbr_fragment_clustered_wgsl = pbr_fragment_clustered_buf;
}

/* ---- Light Sprite Vertex Shader ---- */
static const char* light_sprite_vertex_shader_wgsl =
  "var<private> pos : array<vec2<f32>, 4> = array<vec2<f32>, 4>(\n"
  "  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0)\n"
  ");\n"
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  LIGHT_UNIFORMS_WGSL
  "struct VertexInput {\n"
  "  @builtin(vertex_index) vertexIndex : u32,\n"
  "  @builtin(instance_index) instanceIndex : u32\n"
  "};\n"
  "struct VertexOutput {\n"
  "  @builtin(position) position : vec4<f32>,\n"
  "  @location(0) localPos : vec2<f32>,\n"
  "  @location(1) color: vec3<f32>\n"
  "};\n"
  "@vertex\n"
  "fn vertexMain(input : VertexInput) -> VertexOutput {\n"
  "  var output : VertexOutput;\n"
  "  output.localPos = pos[input.vertexIndex];\n"
  "  output.color = globalLights.lights[input.instanceIndex].color;\n"
  "  let worldPos = vec3<f32>(output.localPos, 0.0) * globalLights.lights[input.instanceIndex].range * 0.025;\n"
  "  var bbModelViewMatrix : mat4x4<f32>;\n"
  "  bbModelViewMatrix[3] = vec4<f32>(globalLights.lights[input.instanceIndex].position, 1.0);\n"
  "  bbModelViewMatrix = view.matrix * bbModelViewMatrix;\n"
  "  bbModelViewMatrix[0][0] = 1.0;\n"
  "  bbModelViewMatrix[0][1] = 0.0;\n"
  "  bbModelViewMatrix[0][2] = 0.0;\n"
  "  bbModelViewMatrix[1][0] = 0.0;\n"
  "  bbModelViewMatrix[1][1] = 1.0;\n"
  "  bbModelViewMatrix[1][2] = 0.0;\n"
  "  bbModelViewMatrix[2][0] = 0.0;\n"
  "  bbModelViewMatrix[2][1] = 0.0;\n"
  "  bbModelViewMatrix[2][2] = 1.0;\n"
  "  output.position = projection.matrix * bbModelViewMatrix * vec4<f32>(worldPos, 1.0);\n"
  "  return output;\n"
  "}\n";

/* ---- Light Sprite Fragment Shader ---- */
static const char* light_sprite_fragment_shader_wgsl =
  COLOR_CONVERSIONS_WGSL
  "struct FragmentInput {\n"
  "  @location(0) localPos : vec2<f32>,\n"
  "  @location(1) color: vec3<f32>\n"
  "};\n"
  "@fragment\n"
  "fn fragmentMain(input : FragmentInput) -> @location(0) vec4<f32> {\n"
  "  let distToCenter = length(input.localPos);\n"
  "  let fade = (1.0 - distToCenter) * (1.0 / (distToCenter * distToCenter));\n"
  "  return vec4<f32>(linearTosRGB(input.color * fade), fade);\n"
  "}\n";

/* ---- Cluster Bounds Compute Shader ---- */
static const char* cluster_bounds_compute_shader_wgsl =
  PROJECTION_UNIFORMS_WGSL
  CLUSTER_STRUCTS_WGSL
  "@group(1) @binding(0) var<storage, read_write> clusters : Clusters;\n"
  "fn lineIntersectionToZPlane(a : vec3<f32>, b : vec3<f32>, zDistance : f32) -> vec3<f32> {\n"
  "  let normal = vec3<f32>(0.0, 0.0, 1.0);\n"
  "  let ab = b - a;\n"
  "  let t = (zDistance - dot(normal, a)) / dot(normal, ab);\n"
  "  return a + t * ab;\n"
  "}\n"
  "fn clipToView(clip : vec4<f32>) -> vec4<f32> {\n"
  "  let view = projection.inverseMatrix * clip;\n"
  "  return view / vec4<f32>(view.w, view.w, view.w, view.w);\n"
  "}\n"
  "fn screen2View(screen : vec4<f32>) -> vec4<f32> {\n"
  "  let texCoord = screen.xy / projection.outputSize.xy;\n"
  "  let clip = vec4<f32>(vec2<f32>(texCoord.x, 1.0 - texCoord.y) * 2.0 - vec2<f32>(1.0, 1.0), screen.z, screen.w);\n"
  "  return clipToView(clip);\n"
  "}\n"
  "const tileCount = vec3<u32>(32u, 18u, 48u);\n"
  "const eyePos = vec3<f32>(0.0);\n"
  "@compute @workgroup_size(4, 2, 4)\n"
  "fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {\n"
  "  let tileIndex = global_id.x +\n"
  "                  global_id.y * tileCount.x +\n"
  "                  global_id.z * tileCount.x * tileCount.y;\n"
  "  let tileSize = vec2<f32>(projection.outputSize.x / f32(tileCount.x),\n"
  "                           projection.outputSize.y / f32(tileCount.y));\n"
  "  let maxPoint_sS = vec4<f32>(vec2<f32>(f32(global_id.x+1u), f32(global_id.y+1u)) * tileSize, 0.0, 1.0);\n"
  "  let minPoint_sS = vec4<f32>(vec2<f32>(f32(global_id.x), f32(global_id.y)) * tileSize, 0.0, 1.0);\n"
  "  let maxPoint_vS = screen2View(maxPoint_sS).xyz;\n"
  "  let minPoint_vS = screen2View(minPoint_sS).xyz;\n"
  "  let tileNear = -projection.zNear * pow(projection.zFar/ projection.zNear, f32(global_id.z)/f32(tileCount.z));\n"
  "  let tileFar = -projection.zNear * pow(projection.zFar/ projection.zNear, f32(global_id.z+1u)/f32(tileCount.z));\n"
  "  let minPointNear = lineIntersectionToZPlane(eyePos, minPoint_vS, tileNear);\n"
  "  let minPointFar = lineIntersectionToZPlane(eyePos, minPoint_vS, tileFar);\n"
  "  let maxPointNear = lineIntersectionToZPlane(eyePos, maxPoint_vS, tileNear);\n"
  "  let maxPointFar = lineIntersectionToZPlane(eyePos, maxPoint_vS, tileFar);\n"
  "  clusters.bounds[tileIndex].minAABB = min(min(minPointNear, minPointFar),min(maxPointNear, maxPointFar));\n"
  "  clusters.bounds[tileIndex].maxAABB = max(max(minPointNear, minPointFar),max(maxPointNear, maxPointFar));\n"
  "}\n";

/* ---- Cluster Lights Compute Shader ---- */
static const char* cluster_lights_compute_shader_wgsl =
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  LIGHT_UNIFORMS_WGSL
  CLUSTER_LIGHTS_STRUCTS_WGSL
  CLUSTER_STRUCTS_WGSL
  "@group(1) @binding(0) var<storage> clusters : Clusters;\n"
  TILE_FUNCTIONS_WGSL
  "fn sqDistPointAABB(_point : vec3<f32>, minAABB : vec3<f32>, maxAABB : vec3<f32>) -> f32 {\n"
  "  var sqDist = 0.0;\n"
  "  for(var i = 0; i < 3; i = i + 1) {\n"
  "    let v = _point[i];\n"
  "    if(v < minAABB[i]){\n"
  "      sqDist = sqDist + (minAABB[i] - v) * (minAABB[i] - v);\n"
  "    }\n"
  "    if(v > maxAABB[i]){\n"
  "      sqDist = sqDist + (v - maxAABB[i]) * (v - maxAABB[i]);\n"
  "    }\n"
  "  }\n"
  "  return sqDist;\n"
  "}\n"
  "@compute @workgroup_size(4, 2, 4)\n"
  "fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {\n"
  "  let tileIndex = global_id.x +\n"
  "                  global_id.y * tileCount.x +\n"
  "                  global_id.z * tileCount.x * tileCount.y;\n"
  "  var clusterLightCount = 0u;\n"
  "  var cluserLightIndices : array<u32, 100>;\n"
  "  for (var i = 0u; i < globalLights.lightCount; i = i + 1u) {\n"
  "    let range = globalLights.lights[i].range;\n"
  "    var lightInCluster = range <= 0.0;\n"
  "    if (!lightInCluster) {\n"
  "      let lightViewPos = view.matrix * vec4<f32>(globalLights.lights[i].position, 1.0);\n"
  "      let sqDist = sqDistPointAABB(lightViewPos.xyz, clusters.bounds[tileIndex].minAABB, clusters.bounds[tileIndex].maxAABB);\n"
  "      lightInCluster = sqDist <= (range * range);\n"
  "    }\n"
  "    if (lightInCluster) {\n"
  "      cluserLightIndices[clusterLightCount] = i;\n"
  "      clusterLightCount = clusterLightCount + 1u;\n"
  "    }\n"
  "    if (clusterLightCount == 100u) {\n"
  "      break;\n"
  "    }\n"
  "  }\n"
  "  var offset = atomicAdd(&clusterLights.offset, clusterLightCount);\n"
  "  for(var i = 0u; i < clusterLightCount; i = i + 1u) {\n"
  "    clusterLights.indices[offset + i] = cluserLightIndices[i];\n"
  "  }\n"
  "  clusterLights.lights[tileIndex].offset = offset;\n"
  "  clusterLights.lights[tileIndex].count = clusterLightCount;\n"
  "}\n";

/* ---- Debug Visualization Shaders ---- */

/* Simple vertex shader (for debug modes) */
static const char* simple_vertex_shader_wgsl =
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  MODEL_UNIFORMS_WGSL
  "@vertex\n"
  "fn main(@location(1) POSITION : vec3<f32>) -> @builtin(position) vec4<f32> {\n"
  "  return projection.matrix * view.matrix * model.matrix * vec4<f32>(POSITION, 1.0);\n"
  "}\n";

/* Depth visualization */
static const char* depth_viz_fragment_wgsl =
  "@fragment\n"
  "fn main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {\n"
  "  return vec4<f32>(fragCoord.zzz, 1.0);\n"
  "}\n";

/* Depth slice visualization */
static const char* depth_slice_viz_fragment_wgsl =
  PROJECTION_UNIFORMS_WGSL
  TILE_FUNCTIONS_WGSL
  "var<private> colorSet : array<vec3<f32>, 9> = array<vec3<f32>, 9>(\n"
  "  vec3<f32>(1.0, 0.0, 0.0),\n"
  "  vec3<f32>(1.0, 0.5, 0.0),\n"
  "  vec3<f32>(0.5, 1.0, 0.0),\n"
  "  vec3<f32>(0.0, 1.0, 0.0),\n"
  "  vec3<f32>(0.0, 1.0, 0.5),\n"
  "  vec3<f32>(0.0, 0.5, 1.0),\n"
  "  vec3<f32>(0.0, 0.0, 1.0),\n"
  "  vec3<f32>(0.5, 0.0, 1.0),\n"
  "  vec3<f32>(1.0, 0.0, 0.5)\n"
  ");\n"
  "@fragment\n"
  "fn main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {\n"
  "  var tile : vec3<u32> = getTile(fragCoord);\n"
  "  return vec4<f32>(colorSet[tile.z % 9u], 1.0);\n"
  "}\n";

/* Cluster distance visualization vertex shader */
static const char* cluster_dist_viz_vertex_wgsl =
  PROJECTION_UNIFORMS_WGSL
  VIEW_UNIFORMS_WGSL
  MODEL_UNIFORMS_WGSL
  "struct VertexOutput {\n"
  "  @builtin(position) position : vec4<f32>,\n"
  "  @location(0) viewPosition : vec4<f32>\n"
  "};\n"
  "@vertex\n"
  "fn main(@location(1) inPosition : vec3<f32>) -> VertexOutput {\n"
  "  var output : VertexOutput;\n"
  "  output.viewPosition = view.matrix * model.matrix * vec4<f32>(inPosition, 1.0);\n"
  "  output.position = projection.matrix * output.viewPosition;\n"
  "  return output;\n"
  "}\n";

/* Cluster distance visualization fragment shader */
static const char* cluster_dist_viz_fragment_wgsl =
  PROJECTION_UNIFORMS_WGSL
  TILE_FUNCTIONS_WGSL
  CLUSTER_STRUCTS_WGSL
  "@group(3) @binding(0) var<storage, read> clusters : Clusters;\n"
  "struct FragmentInput {\n"
  "  @builtin(position) fragCoord : vec4<f32>,\n"
  "  @location(0) viewPosition : vec4<f32>\n"
  "};\n"
  "@fragment\n"
  "fn main(input : FragmentInput) -> @location(0) vec4<f32> {\n"
  "  let clusterIndex : u32 = getClusterIndex(input.fragCoord);\n"
  "  let midPoint : vec3<f32> = (clusters.bounds[clusterIndex].maxAABB - clusters.bounds[clusterIndex].minAABB) / vec3<f32>(2.0, 2.0, 2.0);\n"
  "  let center : vec3<f32> = clusters.bounds[clusterIndex].minAABB + midPoint;\n"
  "  let radius : f32 = length(midPoint);\n"
  "  let fragToBoundsCenter : vec3<f32> = input.viewPosition.xyz - center;\n"
  "  let distToBoundsCenter : f32 = length(fragToBoundsCenter);\n"
  "  let normDist : f32 = distToBoundsCenter / radius;\n"
  "  return vec4<f32>(normDist, normDist, normDist, 1.0);\n"
  "}\n";

/* Lights per cluster visualization */
static const char* lights_per_cluster_viz_fragment_wgsl =
  PROJECTION_UNIFORMS_WGSL
  TILE_FUNCTIONS_WGSL
  CLUSTER_LIGHTS_STRUCTS_FRAG_WGSL
  "@fragment\n"
  "fn main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {\n"
  "  let clusterIndex : u32 = getClusterIndex(fragCoord);\n"
  "  let lightCount : u32 = clusterLights.lights[clusterIndex].count;\n"
  "  let lightFactor : f32 = f32(lightCount) / f32(100);\n"
  "  return mix(vec4<f32>(0.0, 0.0, 1.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(lightFactor, lightFactor, lightFactor, lightFactor));\n"
  "}\n";

/* clang-format on */
