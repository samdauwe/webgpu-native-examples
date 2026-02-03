#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Cornell Box
 *
 * Visualizes a classic Cornell box, using a lightmap generated using software
 * ray-tracing.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/cornell
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

/* Forward declare shaders (defined at the bottom of the file) */
static const char* common_shader_wgsl;
static const char* radiosity_shader_wgsl;
static const char* rasterizer_shader_wgsl;
static const char* raytracer_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Lightmap dimensions */
#define LIGHTMAP_WIDTH 256
#define LIGHTMAP_HEIGHT 256
#define LIGHTMAP_FORMAT WGPUTextureFormat_RGBA16Float

/* Radiosity constants */
#define PHOTONS_PER_WORKGROUP 256
#define WORKGROUPS_PER_FRAME 1024
#define PHOTON_ENERGY 100000.0f

/* Accumulation to lightmap workgroup size */
#define ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_X 16
#define ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_Y 16

/* Raytracer workgroup size */
#define RAYTRACER_WORKGROUP_SIZE_X 16
#define RAYTRACER_WORKGROUP_SIZE_Y 16

/* Tonemapper workgroup size */
#define TONEMAPPER_WORKGROUP_SIZE_X 16
#define TONEMAPPER_WORKGROUP_SIZE_Y 16

/* Maximum accumulation mean before scaling */
#define ACCUMULATION_MEAN_MAX 0x10000000

/* -------------------------------------------------------------------------- *
 * Renderer types
 * -------------------------------------------------------------------------- */

typedef enum renderer_type_t {
  RENDERER_RASTERIZER = 0,
  RENDERER_RAYTRACER  = 1,
  RENDERER_COUNT      = 2,
} renderer_type_t;

/* -------------------------------------------------------------------------- *
 * Quad structure for scene geometry
 * -------------------------------------------------------------------------- */

typedef struct quad_t {
  vec3 center;
  vec3 right;
  vec3 up;
  vec3 color;
  float emissive;
} quad_t;

/* -------------------------------------------------------------------------- *
 * Helper functions
 * -------------------------------------------------------------------------- */

static void vec3_reciprocal(vec3 v, vec3 result)
{
  const float s = 1.0f / glm_vec3_norm2(v);
  glm_vec3_scale(v, s, result);
}

static float vec3_len(vec3 v)
{
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/* -------------------------------------------------------------------------- *
 * Box generation (similar to TypeScript version)
 * -------------------------------------------------------------------------- */

typedef enum cube_face_t {
  CUBE_FACE_POSITIVE_X = 0,
  CUBE_FACE_POSITIVE_Y = 1,
  CUBE_FACE_POSITIVE_Z = 2,
  CUBE_FACE_NEGATIVE_X = 3,
  CUBE_FACE_NEGATIVE_Y = 4,
  CUBE_FACE_NEGATIVE_Z = 5,
} cube_face_t;

static void create_box_quads(vec3 center, float width, float height,
                             float depth, float rotation, vec3 colors[6],
                             bool is_concave, quad_t out_quads[6])
{
  /* Calculate basis vectors */
  vec3 x
    = {cosf(rotation) * (width / 2.0f), 0.0f, sinf(rotation) * (depth / 2.0f)};
  vec3 y = {0.0f, height / 2.0f, 0.0f};
  vec3 z
    = {sinf(rotation) * (width / 2.0f), 0.0f, -cosf(rotation) * (depth / 2.0f)};

  vec3 neg_x, neg_y, neg_z;
  glm_vec3_negate_to(x, neg_x);
  glm_vec3_negate_to(y, neg_y);
  glm_vec3_negate_to(z, neg_z);

  UNUSED_VAR(neg_x);
  UNUSED_VAR(neg_y);
  UNUSED_VAR(neg_z);

  /* PositiveX face */
  glm_vec3_add(center, x, out_quads[0].center);
  if (is_concave) {
    glm_vec3_negate_to(z, out_quads[0].right);
  }
  else {
    glm_vec3_copy(z, out_quads[0].right);
  }
  glm_vec3_copy(y, out_quads[0].up);
  glm_vec3_copy(colors[0], out_quads[0].color);
  out_quads[0].emissive = 0.0f;

  /* PositiveY face */
  glm_vec3_add(center, y, out_quads[1].center);
  if (is_concave) {
    glm_vec3_copy(x, out_quads[1].right);
  }
  else {
    glm_vec3_negate_to(x, out_quads[1].right);
  }
  glm_vec3_negate_to(z, out_quads[1].up);
  glm_vec3_copy(colors[1], out_quads[1].color);
  out_quads[1].emissive = 0.0f;

  /* PositiveZ face */
  glm_vec3_add(center, z, out_quads[2].center);
  if (is_concave) {
    glm_vec3_copy(x, out_quads[2].right);
  }
  else {
    glm_vec3_negate_to(x, out_quads[2].right);
  }
  glm_vec3_copy(y, out_quads[2].up);
  glm_vec3_copy(colors[2], out_quads[2].color);
  out_quads[2].emissive = 0.0f;

  /* NegativeX face */
  glm_vec3_sub(center, x, out_quads[3].center);
  if (is_concave) {
    glm_vec3_copy(z, out_quads[3].right);
  }
  else {
    glm_vec3_negate_to(z, out_quads[3].right);
  }
  glm_vec3_copy(y, out_quads[3].up);
  glm_vec3_copy(colors[3], out_quads[3].color);
  out_quads[3].emissive = 0.0f;

  /* NegativeY face */
  glm_vec3_sub(center, y, out_quads[4].center);
  if (is_concave) {
    glm_vec3_copy(x, out_quads[4].right);
  }
  else {
    glm_vec3_negate_to(x, out_quads[4].right);
  }
  glm_vec3_copy(z, out_quads[4].up);
  glm_vec3_copy(colors[4], out_quads[4].color);
  out_quads[4].emissive = 0.0f;

  /* NegativeZ face */
  glm_vec3_sub(center, z, out_quads[5].center);
  if (is_concave) {
    glm_vec3_negate_to(x, out_quads[5].right);
  }
  else {
    glm_vec3_copy(x, out_quads[5].right);
  }
  glm_vec3_copy(y, out_quads[5].up);
  glm_vec3_copy(colors[5], out_quads[5].color);
  out_quads[5].emissive = 0.0f;
}

/* -------------------------------------------------------------------------- *
 * Scene data
 * -------------------------------------------------------------------------- */

#define TOTAL_QUADS (6 + 6 + 6 + 1) /* Room + small box + tall box + light */

/* GPU quad data structure (matches shader) */
typedef struct gpu_quad_t {
  float plane[4]; /* vec4f */
  float right[4]; /* vec4f */
  float up[4];    /* vec4f */
  float color[3]; /* vec3f */
  float emissive; /* f32 */
} gpu_quad_t;

/* Vertex data structure for rasterizer */
typedef struct vertex_t {
  float position[4]; /* vec4f */
  float uv[3];       /* vec3f (u, v, quad_idx) */
  float emissive[3]; /* vec3f */
} vertex_t;

/* -------------------------------------------------------------------------- *
 * State struct
 * -------------------------------------------------------------------------- */

static struct {
  /* Scene data */
  struct {
    quad_t quads[TOTAL_QUADS];
    uint32_t quad_count;
    vec3 light_center;
    float light_width;
    float light_height;
  } scene;

  /* Common resources */
  struct {
    WGPUBuffer uniform_buffer;
    WGPUBuffer quad_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    uint32_t frame;
  } common;

  /* Radiosity */
  struct {
    WGPUTexture lightmap;
    WGPUTextureView lightmap_view;
    WGPUBuffer accumulation_buffer;
    WGPUBuffer uniform_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUComputePipeline radiosity_pipeline;
    WGPUComputePipeline accumulation_to_lightmap_pipeline;
    float accumulation_mean;
  } radiosity;

  /* Rasterizer */
  struct {
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
    uint32_t vertex_count;
    uint32_t index_count;
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPURenderPipeline pipeline;
    WGPUSampler sampler;
  } rasterizer;

  /* Raytracer */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUComputePipeline pipeline;
    WGPUSampler sampler;
  } raytracer;

  /* Tonemapper */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUComputePipeline pipeline;
    WGPUTexture output_texture;
    WGPUTextureView output_view;
  } tonemapper;

  /* Blit (fullscreen copy from tonemapper output to swapchain) */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPURenderPipeline pipeline;
    WGPUSampler sampler;
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDescriptor render_pass_descriptor;
  } blit;

  /* Framebuffer (rgba16float) */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    uint32_t width;
    uint32_t height;
  } framebuffer;

  /* View matrices */
  struct {
    mat4 projection;
    mat4 view;
    mat4 mvp;
    mat4 inv_mvp;
  } matrices;

  /* GUI settings */
  struct {
    int32_t renderer;
    bool rotate_camera;
  } settings;

  const char* renderer_names[RENDERER_COUNT];
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .settings = {
    .renderer      = RENDERER_RASTERIZER,
    .rotate_camera = true,
  },
  .renderer_names = {
    "Rasterizer",
    "Raytracer",
  },
};

/* -------------------------------------------------------------------------- *
 * Scene initialization
 * -------------------------------------------------------------------------- */

static void init_scene(void)
{
  uint32_t quad_idx = 0;

  /* Room (concave box) */
  vec3 room_center    = {0.0f, 5.0f, 0.0f};
  vec3 room_colors[6] = {
    {0.0f, 0.5f, 0.0f}, /* PositiveX - green */
    {0.5f, 0.5f, 0.5f}, /* PositiveY - gray */
    {0.5f, 0.5f, 0.5f}, /* PositiveZ - gray */
    {0.5f, 0.0f, 0.0f}, /* NegativeX - red */
    {0.5f, 0.5f, 0.5f}, /* NegativeY - gray */
    {0.5f, 0.5f, 0.5f}, /* NegativeZ - gray */
  };
  create_box_quads(room_center, 10.0f, 10.0f, 10.0f, 0.0f, room_colors, true,
                   &state.scene.quads[quad_idx]);
  quad_idx += 6;

  /* Small box (convex) */
  vec3 small_box_center = {1.5f, 1.5f, 1.0f};
  vec3 small_box_color  = {0.8f, 0.8f, 0.8f};
  vec3 small_box_colors[6];
  for (int i = 0; i < 6; i++) {
    glm_vec3_copy(small_box_color, small_box_colors[i]);
  }
  create_box_quads(small_box_center, 3.0f, 3.0f, 3.0f, 0.3f, small_box_colors,
                   false, &state.scene.quads[quad_idx]);
  quad_idx += 6;

  /* Tall box (convex) */
  vec3 tall_box_center = {-2.0f, 3.0f, -2.0f};
  vec3 tall_box_color  = {0.8f, 0.8f, 0.8f};
  vec3 tall_box_colors[6];
  for (int i = 0; i < 6; i++) {
    glm_vec3_copy(tall_box_color, tall_box_colors[i]);
  }
  create_box_quads(tall_box_center, 3.0f, 6.0f, 3.0f, -0.4f, tall_box_colors,
                   false, &state.scene.quads[quad_idx]);
  quad_idx += 6;

  /* Light quad */
  state.scene.quads[quad_idx] = (quad_t){
    .center   = {0.0f, 9.95f, 0.0f},
    .right    = {1.0f, 0.0f, 0.0f},
    .up       = {0.0f, 0.0f, 1.0f},
    .color    = {5.0f, 5.0f, 5.0f},
    .emissive = 1.0f,
  };

  /* Store light properties */
  glm_vec3_copy(state.scene.quads[quad_idx].center, state.scene.light_center);
  state.scene.light_width  = vec3_len(state.scene.quads[quad_idx].right) * 2.0f;
  state.scene.light_height = vec3_len(state.scene.quads[quad_idx].up) * 2.0f;

  quad_idx++;
  state.scene.quad_count = quad_idx;
}

/* -------------------------------------------------------------------------- *
 * GPU buffer creation for scene
 * -------------------------------------------------------------------------- */

static void init_scene_buffers(wgpu_context_t* wgpu_context)
{
  const uint32_t quad_count = state.scene.quad_count;

  /* Create GPU quad buffer */
  gpu_quad_t* gpu_quads = (gpu_quad_t*)malloc(quad_count * sizeof(gpu_quad_t));
  ASSERT(gpu_quads != NULL);

  /* Create vertex and index data for rasterizer */
  vertex_t* vertices = (vertex_t*)malloc(quad_count * 4 * sizeof(vertex_t));
  uint16_t* indices  = (uint16_t*)malloc(quad_count * 6 * sizeof(uint16_t));
  ASSERT(vertices != NULL && indices != NULL);

  uint32_t vertex_count  = 0;
  uint32_t index_count   = 0;
  uint32_t vertex_offset = 0;
  uint32_t index_offset  = 0;

  for (uint32_t quad_idx = 0; quad_idx < quad_count; quad_idx++) {
    quad_t* quad = &state.scene.quads[quad_idx];

    /* Calculate normal from cross(right, up) */
    vec3 normal;
    glm_vec3_cross(quad->right, quad->up, normal);
    glm_vec3_normalize(normal);

    /* GPU quad data */
    gpu_quads[quad_idx].plane[0] = normal[0];
    gpu_quads[quad_idx].plane[1] = normal[1];
    gpu_quads[quad_idx].plane[2] = normal[2];
    gpu_quads[quad_idx].plane[3] = -glm_vec3_dot(normal, quad->center);

    /* Calculate reciprocal of right vector */
    vec3 inv_right;
    vec3_reciprocal(quad->right, inv_right);
    gpu_quads[quad_idx].right[0] = inv_right[0];
    gpu_quads[quad_idx].right[1] = inv_right[1];
    gpu_quads[quad_idx].right[2] = inv_right[2];
    gpu_quads[quad_idx].right[3] = -glm_vec3_dot(inv_right, quad->center);

    /* Calculate reciprocal of up vector */
    vec3 inv_up;
    vec3_reciprocal(quad->up, inv_up);
    gpu_quads[quad_idx].up[0] = inv_up[0];
    gpu_quads[quad_idx].up[1] = inv_up[1];
    gpu_quads[quad_idx].up[2] = inv_up[2];
    gpu_quads[quad_idx].up[3] = -glm_vec3_dot(inv_up, quad->center);

    /* Color and emissive */
    gpu_quads[quad_idx].color[0] = quad->color[0];
    gpu_quads[quad_idx].color[1] = quad->color[1];
    gpu_quads[quad_idx].color[2] = quad->color[2];
    gpu_quads[quad_idx].emissive = quad->emissive;

    /* Calculate vertex positions: a, b, c, d corners */
    /* a ----- b
       |       |
       |   m   |
       |       |
       c ----- d */
    vec3 a, b, c, d;
    vec3 temp1;

    /* a = center - right + up */
    glm_vec3_sub(quad->center, quad->right, temp1);
    glm_vec3_add(temp1, quad->up, a);

    /* b = center + right + up */
    glm_vec3_add(quad->center, quad->right, temp1);
    glm_vec3_add(temp1, quad->up, b);

    /* c = center - right - up */
    glm_vec3_sub(quad->center, quad->right, temp1);
    glm_vec3_sub(temp1, quad->up, c);

    /* d = center + right - up */
    glm_vec3_add(quad->center, quad->right, temp1);
    glm_vec3_sub(temp1, quad->up, d);

    /* Calculate emissive color */
    float emissive_r = quad->color[0] * quad->emissive;
    float emissive_g = quad->color[1] * quad->emissive;
    float emissive_b = quad->color[2] * quad->emissive;

    /* Vertex a */
    vertices[vertex_offset].position[0] = a[0];
    vertices[vertex_offset].position[1] = a[1];
    vertices[vertex_offset].position[2] = a[2];
    vertices[vertex_offset].position[3] = 1.0f;
    vertices[vertex_offset].uv[0]       = 0.0f; /* u */
    vertices[vertex_offset].uv[1]       = 1.0f; /* v */
    vertices[vertex_offset].uv[2]       = (float)quad_idx;
    vertices[vertex_offset].emissive[0] = emissive_r;
    vertices[vertex_offset].emissive[1] = emissive_g;
    vertices[vertex_offset].emissive[2] = emissive_b;
    vertex_offset++;

    /* Vertex b */
    vertices[vertex_offset].position[0] = b[0];
    vertices[vertex_offset].position[1] = b[1];
    vertices[vertex_offset].position[2] = b[2];
    vertices[vertex_offset].position[3] = 1.0f;
    vertices[vertex_offset].uv[0]       = 1.0f; /* u */
    vertices[vertex_offset].uv[1]       = 1.0f; /* v */
    vertices[vertex_offset].uv[2]       = (float)quad_idx;
    vertices[vertex_offset].emissive[0] = emissive_r;
    vertices[vertex_offset].emissive[1] = emissive_g;
    vertices[vertex_offset].emissive[2] = emissive_b;
    vertex_offset++;

    /* Vertex c */
    vertices[vertex_offset].position[0] = c[0];
    vertices[vertex_offset].position[1] = c[1];
    vertices[vertex_offset].position[2] = c[2];
    vertices[vertex_offset].position[3] = 1.0f;
    vertices[vertex_offset].uv[0]       = 0.0f; /* u */
    vertices[vertex_offset].uv[1]       = 0.0f; /* v */
    vertices[vertex_offset].uv[2]       = (float)quad_idx;
    vertices[vertex_offset].emissive[0] = emissive_r;
    vertices[vertex_offset].emissive[1] = emissive_g;
    vertices[vertex_offset].emissive[2] = emissive_b;
    vertex_offset++;

    /* Vertex d */
    vertices[vertex_offset].position[0] = d[0];
    vertices[vertex_offset].position[1] = d[1];
    vertices[vertex_offset].position[2] = d[2];
    vertices[vertex_offset].position[3] = 1.0f;
    vertices[vertex_offset].uv[0]       = 1.0f; /* u */
    vertices[vertex_offset].uv[1]       = 0.0f; /* v */
    vertices[vertex_offset].uv[2]       = (float)quad_idx;
    vertices[vertex_offset].emissive[0] = emissive_r;
    vertices[vertex_offset].emissive[1] = emissive_g;
    vertices[vertex_offset].emissive[2] = emissive_b;
    vertex_offset++;

    /* Indices: two triangles per quad (a, c, b) and (b, c, d) */
    indices[index_offset++] = vertex_count + 0; /* a */
    indices[index_offset++] = vertex_count + 2; /* c */
    indices[index_offset++] = vertex_count + 1; /* b */
    indices[index_offset++] = vertex_count + 1; /* b */
    indices[index_offset++] = vertex_count + 2; /* c */
    indices[index_offset++] = vertex_count + 3; /* d */

    index_count += 6;
    vertex_count += 4;
  }

  state.rasterizer.vertex_count = vertex_count;
  state.rasterizer.index_count  = index_count;

  /* Create GPU quad buffer */
  state.common.quad_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Scene - Quad buffer"),
      .usage            = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size             = quad_count * sizeof(gpu_quad_t),
      .mappedAtCreation = true,
    });
  ASSERT(state.common.quad_buffer != NULL);
  memcpy(wgpuBufferGetMappedRange(state.common.quad_buffer, 0,
                                  quad_count * sizeof(gpu_quad_t)),
         gpu_quads, quad_count * sizeof(gpu_quad_t));
  wgpuBufferUnmap(state.common.quad_buffer);

  /* Create vertex buffer */
  state.rasterizer.vertex_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Scene - Vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = vertex_count * sizeof(vertex_t),
      .mappedAtCreation = true,
    });
  ASSERT(state.rasterizer.vertex_buffer != NULL);
  memcpy(wgpuBufferGetMappedRange(state.rasterizer.vertex_buffer, 0,
                                  vertex_count * sizeof(vertex_t)),
         vertices, vertex_count * sizeof(vertex_t));
  wgpuBufferUnmap(state.rasterizer.vertex_buffer);

  /* Create index buffer */
  state.rasterizer.index_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Scene - Index buffer"),
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = index_count * sizeof(uint16_t),
      .mappedAtCreation = true,
    });
  ASSERT(state.rasterizer.index_buffer != NULL);
  memcpy(wgpuBufferGetMappedRange(state.rasterizer.index_buffer, 0,
                                  index_count * sizeof(uint16_t)),
         indices, index_count * sizeof(uint16_t));
  wgpuBufferUnmap(state.rasterizer.index_buffer);

  /* Cleanup temporary arrays */
  free(gpu_quads);
  free(vertices);
  free(indices);
}

/* -------------------------------------------------------------------------- *
 * Common resources initialization
 * -------------------------------------------------------------------------- */

static void init_common_resources(wgpu_context_t* wgpu_context)
{
  /* Common uniform buffer: mvp (4x4) + inv_mvp (4x4) + seed (vec3u) + padding
   */
  const uint64_t uniform_size = 4 * 16 + 4 * 16 + 4 * 4;
  state.common.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Common - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = uniform_size,
    });
  ASSERT(state.common.uniform_buffer != NULL);

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_size,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_ReadOnlyStorage,
        .minBindingSize = state.scene.quad_count * sizeof(gpu_quad_t),
      },
    },
  };
  state.common.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Common - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.common.bind_group_layout != NULL);

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.common.uniform_buffer,
      .size    = uniform_size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.common.quad_buffer,
      .size    = state.scene.quad_count * sizeof(gpu_quad_t),
    },
  };
  state.common.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Common - Bind group"),
                            .layout     = state.common.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.common.bind_group != NULL);

  state.common.frame = 0;
}

/* -------------------------------------------------------------------------- *
 * Framebuffer initialization
 * -------------------------------------------------------------------------- */

static void init_framebuffer(wgpu_context_t* wgpu_context)
{
  /* Destroy existing framebuffer if any */
  WGPU_RELEASE_RESOURCE(TextureView, state.framebuffer.view)
  WGPU_RELEASE_RESOURCE(Texture, state.framebuffer.texture)

  state.framebuffer.width  = wgpu_context->width;
  state.framebuffer.height = wgpu_context->height;

  /* Create framebuffer texture (rgba16float) */
  state.framebuffer.texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Framebuffer - Texture"),
      .size  = (WGPUExtent3D){
        .width              = wgpu_context->width,
        .height             = wgpu_context->height,
        .depthOrArrayLayers = 1,
      },
      .format        = WGPUTextureFormat_RGBA16Float,
      .usage         = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_StorageBinding
                       | WGPUTextureUsage_TextureBinding,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
    });
  ASSERT(state.framebuffer.texture != NULL);

  state.framebuffer.view = wgpuTextureCreateView(
    state.framebuffer.texture, &(WGPUTextureViewDescriptor){
                                 .label         = STRVIEW("Framebuffer - View"),
                                 .format        = WGPUTextureFormat_RGBA16Float,
                                 .dimension     = WGPUTextureViewDimension_2D,
                                 .baseMipLevel  = 0,
                                 .mipLevelCount = 1,
                                 .baseArrayLayer  = 0,
                                 .arrayLayerCount = 1,
                               });
  ASSERT(state.framebuffer.view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Radiosity initialization
 * -------------------------------------------------------------------------- */

static void init_radiosity(wgpu_context_t* wgpu_context)
{
  const uint32_t quad_count = state.scene.quad_count;

  /* Create lightmap texture */
  state.radiosity.lightmap = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Radiosity - Lightmap"),
      .size  = (WGPUExtent3D){
        .width              = LIGHTMAP_WIDTH,
        .height             = LIGHTMAP_HEIGHT,
        .depthOrArrayLayers = quad_count,
      },
      .format        = LIGHTMAP_FORMAT,
      .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
    });
  ASSERT(state.radiosity.lightmap != NULL);

  state.radiosity.lightmap_view = wgpuTextureCreateView(
    state.radiosity.lightmap, &(WGPUTextureViewDescriptor){
                                .label  = STRVIEW("Radiosity - Lightmap view"),
                                .format = LIGHTMAP_FORMAT,
                                .dimension = WGPUTextureViewDimension_2DArray,
                                .baseMipLevel    = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = 0,
                                .arrayLayerCount = quad_count,
                              });
  ASSERT(state.radiosity.lightmap_view != NULL);

  /* Create accumulation buffer */
  const uint64_t accumulation_size
    = (uint64_t)LIGHTMAP_WIDTH * LIGHTMAP_HEIGHT * quad_count * 16;
  state.radiosity.accumulation_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device, &(WGPUBufferDescriptor){
                            .label = STRVIEW("Radiosity - Accumulation buffer"),
                            .usage = WGPUBufferUsage_Storage,
                            .size  = accumulation_size,
                          });
  ASSERT(state.radiosity.accumulation_buffer != NULL);

  /* Create uniform buffer */
  const uint64_t uniform_size    = 8 * 4; /* 8 x f32 */
  state.radiosity.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Radiosity - Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = uniform_size,
    });
  ASSERT(state.radiosity.uniform_buffer != NULL);

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type = WGPUBufferBindingType_Storage,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = LIGHTMAP_FORMAT,
        .viewDimension = WGPUTextureViewDimension_2DArray,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = uniform_size,
      },
    },
  };
  state.radiosity.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Radiosity - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.radiosity.bind_group_layout != NULL);

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.radiosity.accumulation_buffer,
      .size    = accumulation_size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding     = 1,
      .textureView = state.radiosity.lightmap_view,
    },
    [2] = (WGPUBindGroupEntry){
      .binding = 2,
      .buffer  = state.radiosity.uniform_buffer,
      .size    = uniform_size,
    },
  };
  state.radiosity.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Radiosity - Bind group"),
                            .layout     = state.radiosity.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.radiosity.bind_group != NULL);

  /* Create pipeline layout */
  WGPUBindGroupLayout bg_layouts[2] = {
    state.common.bind_group_layout,
    state.radiosity.bind_group_layout,
  };
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Radiosity - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    });
  ASSERT(pipeline_layout != NULL);

  /* Concatenate shaders */
  size_t shader_len
    = strlen(radiosity_shader_wgsl) + strlen(common_shader_wgsl) + 1;
  char* shader_code = (char*)malloc(shader_len);
  ASSERT(shader_code != NULL);
  snprintf(shader_code, shader_len, "%s%s", radiosity_shader_wgsl,
           common_shader_wgsl);

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_code);
  ASSERT(shader_module != NULL);

  /* Create radiosity pipeline */
  WGPUConstantEntry radiosity_constants[2] = {
    {.key = STRVIEW("PhotonsPerWorkgroup"), .value = PHOTONS_PER_WORKGROUP},
    {.key = STRVIEW("PhotonEnergy"), .value = PHOTON_ENERGY},
  };
  state.radiosity.radiosity_pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Radiosity - Radiosity pipeline"),
      .layout  = pipeline_layout,
      .compute = (WGPUComputeState){
        .module        = shader_module,
        .entryPoint    = STRVIEW("radiosity"),
        .constantCount = (uint32_t)ARRAY_SIZE(radiosity_constants),
        .constants     = radiosity_constants,
      },
    });
  ASSERT(state.radiosity.radiosity_pipeline != NULL);

  /* Create accumulation_to_lightmap pipeline */
  WGPUConstantEntry acc_constants[2] = {
    {.key   = STRVIEW("AccumulationToLightmapWorkgroupSizeX"),
     .value = ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_X},
    {.key   = STRVIEW("AccumulationToLightmapWorkgroupSizeY"),
     .value = ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_Y},
  };
  state.radiosity.accumulation_to_lightmap_pipeline
    = wgpuDeviceCreateComputePipeline(
      wgpu_context->device,
      &(WGPUComputePipelineDescriptor){
        .label   = STRVIEW("Radiosity - Accumulation to lightmap pipeline"),
        .layout  = pipeline_layout,
        .compute = (WGPUComputeState){
          .module        = shader_module,
          .entryPoint    = STRVIEW("accumulation_to_lightmap"),
          .constantCount = (uint32_t)ARRAY_SIZE(acc_constants),
          .constants     = acc_constants,
        },
      });
  ASSERT(state.radiosity.accumulation_to_lightmap_pipeline != NULL);

  state.radiosity.accumulation_mean = 0.0f;

  /* Cleanup */
  free(shader_code);
  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

/* -------------------------------------------------------------------------- *
 * Rasterizer initialization
 * -------------------------------------------------------------------------- */

static void init_rasterizer_depth_texture(wgpu_context_t* wgpu_context)
{
  /* Destroy existing depth texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.rasterizer.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.rasterizer.depth_texture)

  /* Create depth texture */
  state.rasterizer.depth_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Rasterizer - Depth texture"),
      .size  = (WGPUExtent3D){
        .width              = state.framebuffer.width,
        .height             = state.framebuffer.height,
        .depthOrArrayLayers = 1,
      },
      .format        = WGPUTextureFormat_Depth24Plus,
      .usage         = WGPUTextureUsage_RenderAttachment,
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
    });
  ASSERT(state.rasterizer.depth_texture != NULL);

  state.rasterizer.depth_view
    = wgpuTextureCreateView(state.rasterizer.depth_texture,
                            &(WGPUTextureViewDescriptor){
                              .label     = STRVIEW("Rasterizer - Depth view"),
                              .format    = WGPUTextureFormat_Depth24Plus,
                              .dimension = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                            });
  ASSERT(state.rasterizer.depth_view != NULL);
}

static void init_rasterizer(wgpu_context_t* wgpu_context)
{
  /* Create sampler */
  state.rasterizer.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Rasterizer - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.rasterizer.sampler != NULL);

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };
  state.rasterizer.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Rasterizer - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.rasterizer.bind_group_layout != NULL);

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.radiosity.lightmap_view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.rasterizer.sampler,
    },
  };
  state.rasterizer.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Rasterizer - Bind group"),
                            .layout     = state.rasterizer.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.rasterizer.bind_group != NULL);

  /* Create pipeline layout */
  WGPUBindGroupLayout bg_layouts[2] = {
    state.common.bind_group_layout,
    state.rasterizer.bind_group_layout,
  };
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Rasterizer - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    });
  ASSERT(pipeline_layout != NULL);

  /* Concatenate shaders */
  size_t shader_len
    = strlen(rasterizer_shader_wgsl) + strlen(common_shader_wgsl) + 1;
  char* shader_code = (char*)malloc(shader_len);
  ASSERT(shader_code != NULL);
  snprintf(shader_code, shader_len, "%s%s", rasterizer_shader_wgsl,
           common_shader_wgsl);

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_code);
  ASSERT(shader_module != NULL);

  /* Vertex buffer layout */
  WGPUVertexAttribute vertex_attributes[3] = {
    [0] = {
      .shaderLocation = 0,
      .offset         = offsetof(vertex_t, position),
      .format         = WGPUVertexFormat_Float32x4,
    },
    [1] = {
      .shaderLocation = 1,
      .offset         = offsetof(vertex_t, uv),
      .format         = WGPUVertexFormat_Float32x3,
    },
    [2] = {
      .shaderLocation = 2,
      .offset         = offsetof(vertex_t, emissive),
      .format         = WGPUVertexFormat_Float32x3,
    },
  };
  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attributes),
    .attributes     = vertex_attributes,
  };

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = WGPUTextureFormat_Depth24Plus,
      .depth_write_enabled = true,
    });
  depth_stencil_state.depthCompare = WGPUCompareFunction_Less;

  /* Create render pipeline */
  state.rasterizer.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Rasterizer - Pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = shader_module,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA16Float,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_Back,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &depth_stencil_state,
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.rasterizer.pipeline != NULL);

  /* Cleanup */
  free(shader_code);
  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

/* -------------------------------------------------------------------------- *
 * Raytracer initialization
 * -------------------------------------------------------------------------- */

static void init_raytracer(wgpu_context_t* wgpu_context)
{
  /* Create sampler */
  state.raytracer.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Raytracer - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.raytracer.sampler != NULL);

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA16Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };
  state.raytracer.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Raytracer - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.raytracer.bind_group_layout != NULL);

  /* Create pipeline layout */
  WGPUBindGroupLayout bg_layouts[2] = {
    state.common.bind_group_layout,
    state.raytracer.bind_group_layout,
  };
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Raytracer - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    });
  ASSERT(pipeline_layout != NULL);

  /* Concatenate shaders */
  size_t shader_len
    = strlen(raytracer_shader_wgsl) + strlen(common_shader_wgsl) + 1;
  char* shader_code = (char*)malloc(shader_len);
  ASSERT(shader_code != NULL);
  snprintf(shader_code, shader_len, "%s%s", raytracer_shader_wgsl,
           common_shader_wgsl);

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_code);
  ASSERT(shader_module != NULL);

  /* Create compute pipeline */
  WGPUConstantEntry constants[2] = {
    {.key = STRVIEW("WorkgroupSizeX"), .value = RAYTRACER_WORKGROUP_SIZE_X},
    {.key = STRVIEW("WorkgroupSizeY"), .value = RAYTRACER_WORKGROUP_SIZE_Y},
  };
  state.raytracer.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Raytracer - Pipeline"),
      .layout  = pipeline_layout,
      .compute = (WGPUComputeState){
        .module        = shader_module,
        .entryPoint    = STRVIEW("main"),
        .constantCount = (uint32_t)ARRAY_SIZE(constants),
        .constants     = constants,
      },
    });
  ASSERT(state.raytracer.pipeline != NULL);

  /* Cleanup */
  free(shader_code);
  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

static void init_raytracer_bind_group(wgpu_context_t* wgpu_context)
{
  /* Destroy existing bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.raytracer.bind_group)

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.radiosity.lightmap_view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.raytracer.sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.framebuffer.view,
    },
  };
  state.raytracer.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Raytracer - Bind group"),
                            .layout     = state.raytracer.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.raytracer.bind_group != NULL);
}

/* -------------------------------------------------------------------------- *
 * Tonemapper initialization
 * -------------------------------------------------------------------------- */

static void init_tonemapper_output_texture(wgpu_context_t* wgpu_context)
{
  /* Destroy existing texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.tonemapper.output_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tonemapper.output_texture)

  /* Create output texture (RGBA8Unorm - storable) */
  state.tonemapper.output_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Tonemapper - Output texture"),
      .size  = (WGPUExtent3D){
        .width              = wgpu_context->width,
        .height             = wgpu_context->height,
        .depthOrArrayLayers = 1,
      },
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .usage         = WGPUTextureUsage_StorageBinding
                       | WGPUTextureUsage_TextureBinding,
    });
  ASSERT(state.tonemapper.output_texture != NULL);

  state.tonemapper.output_view = wgpuTextureCreateView(
    state.tonemapper.output_texture,
    &(WGPUTextureViewDescriptor){
      .label           = STRVIEW("Tonemapper - Output texture view"),
      .format          = WGPUTextureFormat_RGBA8Unorm,
      .dimension       = WGPUTextureViewDimension_2D,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
    });
  ASSERT(state.tonemapper.output_view != NULL);
}

static void init_tonemapper(wgpu_context_t* wgpu_context)
{
  /* Create output texture */
  init_tonemapper_output_texture(wgpu_context);

  /* Create bind group layout - always use rgba8unorm for storage */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .storageTexture = (WGPUStorageTextureBindingLayout){
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA8Unorm,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };
  state.tonemapper.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Tonemapper - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.tonemapper.bind_group_layout != NULL);

  /* Create pipeline layout */
  WGPUBindGroupLayout bg_layouts[1] = {
    state.tonemapper.bind_group_layout,
  };
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Tonemapper - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bg_layouts),
      .bindGroupLayouts     = bg_layouts,
    });
  ASSERT(pipeline_layout != NULL);

  /* Build shader with fixed rgba8unorm format */
  static const char* shader_code
    = "// The linear-light input framebuffer\n"
      "@group(0) @binding(0) var input  : texture_2d<f32>;\n"
      "// The tonemapped, gamma-corrected output framebuffer\n"
      "@group(0) @binding(1) var output : texture_storage_2d<rgba8unorm, "
      "write>;\n"
      "const TonemapExposure = 0.5;\n"
      "const Gamma = 2.2;\n"
      "override WorkgroupSizeX : u32;\n"
      "override WorkgroupSizeY : u32;\n"
      "@compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)\n"
      "fn main(@builtin(global_invocation_id) invocation_id : vec3u) {\n"
      "  let color = textureLoad(input, invocation_id.xy, 0).rgb;\n"
      "  let tonemapped = reinhard_tonemap(color);\n"
      "  textureStore(output, invocation_id.xy, vec4f(tonemapped, 1));\n"
      "}\n"
      "fn reinhard_tonemap(linearColor: vec3f) -> vec3f {\n"
      "  let color = linearColor * TonemapExposure;\n"
      "  let mapped = color / (1+color);\n"
      "  return pow(mapped, vec3f(1 / Gamma));\n"
      "}\n";

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, shader_code);
  ASSERT(shader_module != NULL);

  /* Create compute pipeline */
  WGPUConstantEntry constants[2] = {
    {.key = STRVIEW("WorkgroupSizeX"), .value = TONEMAPPER_WORKGROUP_SIZE_X},
    {.key = STRVIEW("WorkgroupSizeY"), .value = TONEMAPPER_WORKGROUP_SIZE_Y},
  };
  state.tonemapper.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Tonemapper - Pipeline"),
      .layout  = pipeline_layout,
      .compute = (WGPUComputeState){
        .module        = shader_module,
        .entryPoint    = STRVIEW("main"),
        .constantCount = (uint32_t)ARRAY_SIZE(constants),
        .constants     = constants,
      },
    });
  ASSERT(state.tonemapper.pipeline != NULL);

  /* Cleanup */
  wgpuShaderModuleRelease(shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

static void init_tonemapper_bind_group(wgpu_context_t* wgpu_context)
{
  /* Destroy existing bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.tonemapper.bind_group)

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.framebuffer.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding     = 1,
      .textureView = state.tonemapper.output_view,
    },
  };
  state.tonemapper.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Tonemapper - Bind group"),
                            .layout     = state.tonemapper.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.tonemapper.bind_group != NULL);
}

/* -------------------------------------------------------------------------- *
 * Blit (fullscreen copy) initialization
 * -------------------------------------------------------------------------- */

static void init_blit(wgpu_context_t* wgpu_context)
{
  /* Create sampler */
  state.blit.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Blit - Sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.blit.sampler != NULL);

  /* Create bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };
  state.blit.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Blit - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.blit.bind_group_layout != NULL);

  /* Create pipeline layout */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Blit - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts = &state.blit.bind_group_layout,
                          });
  ASSERT(pipeline_layout != NULL);

  /* Create shaders */
  static const char* vertex_shader_code
    = "struct VertexOutput {\n"
      "  @builtin(position) position : vec4<f32>,\n"
      "  @location(0) tex_coord : vec2<f32>,\n"
      "};\n"
      "@vertex fn main(@builtin(vertex_index) vertex_index : u32) -> "
      "VertexOutput {\n"
      "  var output : VertexOutput;\n"
      "  let x = f32((vertex_index << 1u) & 2u);\n"
      "  let y = f32(vertex_index & 2u);\n"
      "  output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);\n"
      "  output.tex_coord = vec2<f32>(x, y);\n"
      "  return output;\n"
      "}\n";

  static const char* fragment_shader_code
    = "@group(0) @binding(0) var render_texture : texture_2d<f32>;\n"
      "@group(0) @binding(1) var tex_sampler : sampler;\n"
      "@fragment fn main(@location(0) tex_coord : vec2<f32>) -> @location(0) "
      "vec4<f32> {\n"
      "  return textureSample(render_texture, tex_sampler, tex_coord);\n"
      "}\n";

  WGPUShaderModule vertex_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_code);
  ASSERT(vertex_shader_module != NULL);

  WGPUShaderModule fragment_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_code);
  ASSERT(fragment_shader_module != NULL);

  /* Create render pipeline */
  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .writeMask = WGPUColorWriteMask_All,
  };

  state.blit.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Blit - Render pipeline"),
      .layout = pipeline_layout,
      .vertex = (WGPUVertexState){
        .module     = vertex_shader_module,
        .entryPoint = STRVIEW("main"),
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
      .fragment = &(WGPUFragmentState){
        .module      = fragment_shader_module,
        .entryPoint  = STRVIEW("main"),
        .targetCount = 1,
        .targets     = &color_target,
      },
    });
  ASSERT(state.blit.pipeline != NULL);

  /* Setup render pass descriptor */
  state.blit.color_attachment = (WGPURenderPassColorAttachment){
    .view       = NULL, /* Set per frame */
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = (WGPUColor){0.0, 0.0, 0.0, 1.0},
  };
  state.blit.render_pass_descriptor = (WGPURenderPassDescriptor){
    .label                = STRVIEW("Blit - Render pass"),
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.blit.color_attachment,
  };

  /* Cleanup */
  wgpuShaderModuleRelease(vertex_shader_module);
  wgpuShaderModuleRelease(fragment_shader_module);
  wgpuPipelineLayoutRelease(pipeline_layout);
}

static void init_blit_bind_group(wgpu_context_t* wgpu_context)
{
  /* Destroy existing bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.blit.bind_group)

  /* Create bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.tonemapper.output_view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.blit.sampler,
    },
  };
  state.blit.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Blit - Bind group"),
                            .layout     = state.blit.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.blit.bind_group != NULL);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer updates
 * -------------------------------------------------------------------------- */

static void update_common_uniforms(wgpu_context_t* wgpu_context)
{
  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Projection matrix */
  glm_perspective(PI2 / 8.0f, aspect, 0.5f, 100.0f, state.matrices.projection);

  /* Camera rotation */
  float view_rotation = 0.0f;
  if (state.settings.rotate_camera) {
    view_rotation = (float)state.common.frame / 1000.0f;
  }

  /* View matrix */
  vec3 eye = {sinf(view_rotation) * 15.0f, 5.0f, cosf(view_rotation) * 15.0f};
  vec3 center = {0.0f, 5.0f, 0.0f};
  vec3 up     = {0.0f, 1.0f, 0.0f};
  glm_lookat(eye, center, up, state.matrices.view);

  /* MVP matrix */
  glm_mat4_mul(state.matrices.projection, state.matrices.view,
               state.matrices.mvp);

  /* Inverse MVP */
  glm_mat4_inv(state.matrices.mvp, state.matrices.inv_mvp);

  /* Prepare uniform data */
  float uniform_data[36]; /* 16 + 16 + 4 = 36 floats */
  memcpy(&uniform_data[0], state.matrices.mvp, 16 * sizeof(float));
  memcpy(&uniform_data[16], state.matrices.inv_mvp, 16 * sizeof(float));

  /* Random seed */
  uint32_t* seed_ptr = (uint32_t*)&uniform_data[32];
  seed_ptr[0]        = (uint32_t)(random_float() * (float)0xFFFFFFFF);
  seed_ptr[1]        = (uint32_t)(random_float() * (float)0xFFFFFFFF);
  seed_ptr[2]        = (uint32_t)(random_float() * (float)0xFFFFFFFF);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.common.uniform_buffer, 0,
                       uniform_data, sizeof(uniform_data));

  state.common.frame++;
}

static void update_radiosity_uniforms(wgpu_context_t* wgpu_context)
{
  const uint32_t total_lightmap_texels
    = LIGHTMAP_WIDTH * LIGHTMAP_HEIGHT * state.scene.quad_count;
  const uint32_t photons_per_frame
    = PHOTONS_PER_WORKGROUP * WORKGROUPS_PER_FRAME;

  /* Calculate the new mean value for the accumulation buffer */
  state.radiosity.accumulation_mean
    += (float)(photons_per_frame * PHOTON_ENERGY)
       / (float)total_lightmap_texels;

  /* Calculate scales */
  float accumulation_to_lightmap_scale
    = 1.0f / state.radiosity.accumulation_mean;
  float accumulation_buffer_scale = 1.0f;
  if (state.radiosity.accumulation_mean > 2.0f * ACCUMULATION_MEAN_MAX) {
    accumulation_buffer_scale = 0.5f;
  }
  state.radiosity.accumulation_mean *= accumulation_buffer_scale;

  /* Prepare uniform data */
  float uniform_data[8] = {
    accumulation_to_lightmap_scale, accumulation_buffer_scale,
    state.scene.light_width,        state.scene.light_height,
    state.scene.light_center[0],    state.scene.light_center[1],
    state.scene.light_center[2],    0.0f, /* padding */
  };

  wgpuQueueWriteBuffer(wgpu_context->queue, state.radiosity.uniform_buffer, 0,
                       uniform_data, sizeof(uniform_data));
}

/* -------------------------------------------------------------------------- *
 * Render passes
 * -------------------------------------------------------------------------- */

static void run_radiosity(WGPUCommandEncoder command_encoder)
{
  WGPUComputePassEncoder pass_encoder
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);

  wgpuComputePassEncoderSetBindGroup(pass_encoder, 0, state.common.bind_group,
                                     0, NULL);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 1,
                                     state.radiosity.bind_group, 0, NULL);

  /* Dispatch radiosity workgroups */
  wgpuComputePassEncoderSetPipeline(pass_encoder,
                                    state.radiosity.radiosity_pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(pass_encoder, WORKGROUPS_PER_FRAME,
                                           1, 1);

  /* Copy accumulation to lightmap */
  wgpuComputePassEncoderSetPipeline(
    pass_encoder, state.radiosity.accumulation_to_lightmap_pipeline);
  uint32_t wg_x
    = (LIGHTMAP_WIDTH + ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_X - 1)
      / ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_X;
  uint32_t wg_y
    = (LIGHTMAP_HEIGHT + ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_Y - 1)
      / ACCUMULATION_TO_LIGHTMAP_WORKGROUP_SIZE_Y;
  wgpuComputePassEncoderDispatchWorkgroups(pass_encoder, wg_x, wg_y,
                                           state.scene.quad_count);

  wgpuComputePassEncoderEnd(pass_encoder);
  wgpuComputePassEncoderRelease(pass_encoder);
}

static void run_rasterizer(WGPUCommandEncoder command_encoder)
{
  WGPURenderPassColorAttachment color_attachment = {
    .view       = state.framebuffer.view,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1f, 0.2f, 0.3f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  WGPURenderPassDepthStencilAttachment depth_attachment = {
    .view            = state.rasterizer.depth_view,
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
  };
  WGPURenderPassDescriptor render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &color_attachment,
    .depthStencilAttachment = &depth_attachment,
  };

  WGPURenderPassEncoder pass_encoder
    = wgpuCommandEncoderBeginRenderPass(command_encoder, &render_pass_desc);

  wgpuRenderPassEncoderSetPipeline(pass_encoder, state.rasterizer.pipeline);
  wgpuRenderPassEncoderSetVertexBuffer(
    pass_encoder, 0, state.rasterizer.vertex_buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    pass_encoder, state.rasterizer.index_buffer, WGPUIndexFormat_Uint16, 0,
    WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0, state.common.bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderSetBindGroup(pass_encoder, 1,
                                    state.rasterizer.bind_group, 0, NULL);
  wgpuRenderPassEncoderDrawIndexed(pass_encoder, state.rasterizer.index_count,
                                   1, 0, 0, 0);

  wgpuRenderPassEncoderEnd(pass_encoder);
  wgpuRenderPassEncoderRelease(pass_encoder);
}

static void run_raytracer(WGPUCommandEncoder command_encoder)
{
  WGPUComputePassEncoder pass_encoder
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);

  wgpuComputePassEncoderSetPipeline(pass_encoder, state.raytracer.pipeline);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 0, state.common.bind_group,
                                     0, NULL);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 1,
                                     state.raytracer.bind_group, 0, NULL);

  uint32_t wg_x = (state.framebuffer.width + RAYTRACER_WORKGROUP_SIZE_X - 1)
                  / RAYTRACER_WORKGROUP_SIZE_X;
  uint32_t wg_y = (state.framebuffer.height + RAYTRACER_WORKGROUP_SIZE_Y - 1)
                  / RAYTRACER_WORKGROUP_SIZE_Y;
  wgpuComputePassEncoderDispatchWorkgroups(pass_encoder, wg_x, wg_y, 1);

  wgpuComputePassEncoderEnd(pass_encoder);
  wgpuComputePassEncoderRelease(pass_encoder);
}

static void run_tonemapper(WGPUCommandEncoder command_encoder)
{
  WGPUComputePassEncoder pass_encoder
    = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);

  wgpuComputePassEncoderSetPipeline(pass_encoder, state.tonemapper.pipeline);
  wgpuComputePassEncoderSetBindGroup(pass_encoder, 0,
                                     state.tonemapper.bind_group, 0, NULL);

  uint32_t wg_x = (state.framebuffer.width + TONEMAPPER_WORKGROUP_SIZE_X - 1)
                  / TONEMAPPER_WORKGROUP_SIZE_X;
  uint32_t wg_y = (state.framebuffer.height + TONEMAPPER_WORKGROUP_SIZE_Y - 1)
                  / TONEMAPPER_WORKGROUP_SIZE_Y;
  wgpuComputePassEncoderDispatchWorkgroups(pass_encoder, wg_x, wg_y, 1);

  wgpuComputePassEncoderEnd(pass_encoder);
  wgpuComputePassEncoderRelease(pass_encoder);
}

static void run_blit(WGPUCommandEncoder command_encoder,
                     WGPUTextureView swapchain_view)
{
  state.blit.color_attachment.view = swapchain_view;

  WGPURenderPassEncoder pass_encoder = wgpuCommandEncoderBeginRenderPass(
    command_encoder, &state.blit.render_pass_descriptor);

  wgpuRenderPassEncoderSetPipeline(pass_encoder, state.blit.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass_encoder, 0, state.blit.bind_group, 0,
                                    NULL);
  wgpuRenderPassEncoderDraw(pass_encoder, 3, 1, 0, 0);

  wgpuRenderPassEncoderEnd(pass_encoder);
  wgpuRenderPassEncoderRelease(pass_encoder);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){200.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Cornell Box", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Renderer selector */
  if (imgui_overlay_combo_box("Renderer", &state.settings.renderer,
                              state.renderer_names, RENDERER_COUNT)) {
    /* Renderer changed */
  }

  /* Rotate camera checkbox */
  igCheckbox("Rotate Camera", &state.settings.rotate_camera);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Resize handling
 * -------------------------------------------------------------------------- */

static void handle_resize(wgpu_context_t* wgpu_context)
{
  init_framebuffer(wgpu_context);
  init_rasterizer_depth_texture(wgpu_context);
  init_raytracer_bind_group(wgpu_context);
  init_tonemapper_output_texture(wgpu_context);
  init_tonemapper_bind_group(wgpu_context);
  init_blit_bind_group(wgpu_context);
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    handle_resize(wgpu_context);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR) {
    if (input_event->char_code == (uint32_t)'r') {
      /* Toggle renderer */
      state.settings.renderer = (state.settings.renderer + 1) % RENDERER_COUNT;
    }
    else if (input_event->char_code == (uint32_t)'c') {
      /* Toggle camera rotation */
      state.settings.rotate_camera = !state.settings.rotate_camera;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Main functions
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();

    init_scene();
    init_scene_buffers(wgpu_context);
    init_common_resources(wgpu_context);
    init_framebuffer(wgpu_context);
    init_radiosity(wgpu_context);
    init_rasterizer(wgpu_context);
    init_rasterizer_depth_texture(wgpu_context);
    init_raytracer(wgpu_context);
    init_raytracer_bind_group(wgpu_context);
    init_tonemapper(wgpu_context);
    init_tonemapper_bind_group(wgpu_context);
    init_blit(wgpu_context);
    init_blit_bind_group(wgpu_context);

    imgui_overlay_init(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);

  /* Render GUI */
  render_gui(wgpu_context);

  /* Update uniforms */
  update_common_uniforms(wgpu_context);
  update_radiosity_uniforms(wgpu_context);

  WGPUCommandEncoder command_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Run radiosity pass */
  run_radiosity(command_encoder);

  /* Run renderer (rasterizer or raytracer) */
  if (state.settings.renderer == RENDERER_RASTERIZER) {
    run_rasterizer(command_encoder);
  }
  else {
    run_raytracer(command_encoder);
  }

  /* Run tonemapper (writes to intermediate RGBA8Unorm texture) */
  run_tonemapper(command_encoder);

  /* Blit tonemapper output to swapchain */
  run_blit(command_encoder, wgpu_context->swapchain_view);

  /* Submit commands */
  WGPUCommandBuffer command_buffer
    = wgpuCommandEncoderFinish(command_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &command_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(command_buffer);
  wgpuCommandEncoderRelease(command_encoder);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Common resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.common.uniform_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.common.quad_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.common.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.common.bind_group)

  /* Radiosity resources */
  WGPU_RELEASE_RESOURCE(TextureView, state.radiosity.lightmap_view)
  WGPU_RELEASE_RESOURCE(Texture, state.radiosity.lightmap)
  WGPU_RELEASE_RESOURCE(Buffer, state.radiosity.accumulation_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.radiosity.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.radiosity.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.radiosity.bind_group)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.radiosity.radiosity_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline,
                        state.radiosity.accumulation_to_lightmap_pipeline)

  /* Rasterizer resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.rasterizer.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.rasterizer.index_buffer)
  WGPU_RELEASE_RESOURCE(TextureView, state.rasterizer.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.rasterizer.depth_texture)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.rasterizer.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.rasterizer.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.rasterizer.pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, state.rasterizer.sampler)

  /* Raytracer resources */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.raytracer.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.raytracer.bind_group)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.raytracer.pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, state.raytracer.sampler)

  /* Tonemapper resources */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.tonemapper.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.tonemapper.bind_group)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.tonemapper.pipeline)
  WGPU_RELEASE_RESOURCE(TextureView, state.tonemapper.output_view)
  WGPU_RELEASE_RESOURCE(Texture, state.tonemapper.output_texture)

  /* Blit resources */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.blit.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blit.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blit.pipeline)
  WGPU_RELEASE_RESOURCE(Sampler, state.blit.sampler)

  /* Framebuffer resources */
  WGPU_RELEASE_RESOURCE(TextureView, state.framebuffer.view)
  WGPU_RELEASE_RESOURCE(Texture, state.framebuffer.texture)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Cornell Box",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* common_shader_wgsl = CODE(
  const pi = 3.14159265359;

  // Quad describes 2D rectangle on a plane
  struct Quad {
    // The surface plane
    plane    : vec4f,
    // A plane with a normal in the 'u' direction, intersecting the origin, at
    // right-angles to the surface plane.
    right    : vec4f,
    // A plane with a normal in the 'v' direction, intersecting the origin, at
    // right-angles to the surface plane.
    up       : vec4f,
    // The diffuse color of the quad
    color    : vec3f,
    // Emissive value. 0=no emissive, 1=full emissive.
    emissive : f32,
  };

  // Ray is a start point and direction.
  struct Ray {
    start : vec3f,
    dir   : vec3f,
  }

  // Value for HitInfo.quad if no intersection occured.
  const kNoHit = 0xffffffff;

  // HitInfo describes the hit location of a ray-quad intersection
  struct HitInfo {
    // Distance along the ray to the intersection
    dist : f32,
    // The quad index that was hit
    quad : u32,
    // The position of the intersection
    pos : vec3f,
    // The UVs of the quad at the point of intersection
    uv : vec2f,
  }

  // CommonUniforms uniform buffer data
  struct CommonUniforms {
    // Model View Projection matrix
    mvp : mat4x4f,
    // Inverse of mvp
    inv_mvp : mat4x4f,
    // Random seed for the workgroup
    seed : vec3u,
  }

  // The common uniform buffer binding.
  @group(0) @binding(0) var<uniform> common_uniforms : CommonUniforms;

  // The quad buffer binding.
  @group(0) @binding(1) var<storage> quads : array<Quad>;

  // intersect_ray_quad will check to see if the ray 'r' intersects the quad 'q'.
  fn intersect_ray_quad(r : Ray, quad : u32, closest : HitInfo) -> HitInfo {
    let q = quads[quad];
    let plane_dist = dot(q.plane, vec4(r.start, 1));
    let ray_dist = plane_dist / -dot(q.plane.xyz, r.dir);
    let pos = r.start + r.dir * ray_dist;
    let uv = vec2(dot(vec4f(pos, 1), q.right),
                  dot(vec4f(pos, 1), q.up)) * 0.5 + 0.5;
    let hit = plane_dist > 0 &&
              ray_dist > 0 &&
              ray_dist < closest.dist &&
              all((uv > vec2f()) & (uv < vec2f(1)));
    return HitInfo(
      select(closest.dist, ray_dist, hit),
      select(closest.quad, quad,     hit),
      select(closest.pos,  pos,      hit),
      select(closest.uv,   uv,       hit),
    );
  }

  // raytrace finds the closest intersecting quad for the given ray
  fn raytrace(ray : Ray) -> HitInfo {
    var hit = HitInfo();
    hit.dist = 1e20;
    hit.quad = kNoHit;
    for (var quad = 0u; quad < arrayLength(&quads); quad++) {
      hit = intersect_ray_quad(ray, quad, hit);
    }
    return hit;
  }

  // A pseudo random number. Initialized with init_rand(), updated with rand().
  var<private> rnd : vec3u;

  // Initializes the random number generator.
  fn init_rand(invocation_id : vec3u) {
    const A = vec3(1741651 * 1009,
                   140893  * 1609 * 13,
                   6521    * 983  * 7 * 2);
    rnd = (invocation_id * A) ^ common_uniforms.seed;
  }

  // Returns a random number between 0 and 1.
  fn rand() -> f32 {
    const C = vec3(60493  * 9377,
                   11279  * 2539 * 23,
                   7919   * 631  * 5 * 3);

    rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
    return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
  }

  // Returns a random point within a unit sphere centered at (0,0,0).
  fn rand_unit_sphere() -> vec3f {
      var u = rand();
      var v = rand();
      var theta = u * 2.0 * pi;
      var phi = acos(2.0 * v - 1.0);
      var r = pow(rand(), 1.0/3.0);
      var sin_theta = sin(theta);
      var cos_theta = cos(theta);
      var sin_phi = sin(phi);
      var cos_phi = cos(phi);
      var x = r * sin_phi * sin_theta;
      var y = r * sin_phi * cos_theta;
      var z = r * cos_phi;
      return vec3f(x, y, z);
  }

  fn rand_concentric_disk() -> vec2f {
      let u = vec2f(rand(), rand());
      let uOffset = 2.f * u - vec2f(1, 1);

      if (uOffset.x == 0 && uOffset.y == 0){
          return vec2f(0, 0);
      }

      var theta = 0.0;
      var r = 0.0;
      if (abs(uOffset.x) > abs(uOffset.y)) {
          r = uOffset.x;
          theta = (pi / 4) * (uOffset.y / uOffset.x);
      } else {
          r = uOffset.y;
          theta = (pi / 2) - (pi / 4) * (uOffset.x / uOffset.y);
      }
      return r * vec2f(cos(theta), sin(theta));
  }

  fn rand_cosine_weighted_hemisphere() -> vec3f {
      let d = rand_concentric_disk();
      let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
      return vec3f(d.x, d.y, z);
  }
);

static const char* radiosity_shader_wgsl = CODE(
  // A storage buffer holding an array of atomic<u32>.
  @group(1) @binding(0)
  var<storage, read_write> accumulation : array<atomic<u32>>;

  // The output lightmap texture.
  @group(1) @binding(1)
  var lightmap : texture_storage_2d_array<rgba16float, write>;

  // Uniform data used by the accumulation_to_lightmap entry point
  struct Uniforms {
    accumulation_to_lightmap_scale : f32,
    accumulation_buffer_scale : f32,
    light_width : f32,
    light_height : f32,
    light_center : vec3f,
  }

  @group(1) @binding(2) var<uniform> uniforms : Uniforms;

  override PhotonsPerWorkgroup : u32;
  override PhotonEnergy : f32;

  const PhotonBounces = 4;
  const LightAbsorbtion = 0.5;

  @compute @workgroup_size(PhotonsPerWorkgroup)
  fn radiosity(@builtin(global_invocation_id) invocation_id : vec3u) {
    init_rand(invocation_id);
    photon();
  }

  fn photon() {
    var ray = new_light_ray();
    var color = PhotonEnergy * vec3f(1, 0.8, 0.6);

    for (var i = 0; i < (PhotonBounces+1); i++) {
      let hit = raytrace(ray);
      let quad = quads[hit.quad];

      ray.start = hit.pos + quad.plane.xyz * 1e-5;
      ray.dir = normalize(reflect(ray.dir, quad.plane.xyz) + rand_unit_sphere() * 0.75);

      color *= quad.color;
      accumulate(hit.uv, hit.quad, color * LightAbsorbtion);
      color *= 1 - LightAbsorbtion;
    }
  }

  fn accumulate(uv : vec2f, quad : u32, color : vec3f) {
    let dims = textureDimensions(lightmap);
    let base_idx = accumulation_base_index(vec2u(uv * vec2f(dims)), quad);
    atomicAdd(&accumulation[base_idx + 0], u32(color.r + 0.5));
    atomicAdd(&accumulation[base_idx + 1], u32(color.g + 0.5));
    atomicAdd(&accumulation[base_idx + 2], u32(color.b + 0.5));
  }

  fn accumulation_base_index(coord : vec2u, quad : u32) -> u32 {
    let dims = textureDimensions(lightmap);
    let c = min(vec2u(dims) - 1, coord);
    return 3 * (c.x + dims.x * c.y + dims.x * dims.y * quad);
  }

  fn new_light_ray() -> Ray {
    let center = uniforms.light_center;
    let pos = center + vec3f(uniforms.light_width * (rand() - 0.5),
                             0,
                             uniforms.light_height * (rand() - 0.5));
    var dir = rand_cosine_weighted_hemisphere().xzy;
    dir.y = -dir.y;
    return Ray(pos, dir);
  }

  override AccumulationToLightmapWorkgroupSizeX : u32;
  override AccumulationToLightmapWorkgroupSizeY : u32;

  @compute @workgroup_size(AccumulationToLightmapWorkgroupSizeX, AccumulationToLightmapWorkgroupSizeY)
  fn accumulation_to_lightmap(@builtin(global_invocation_id) invocation_id : vec3u,
                              @builtin(workgroup_id)         workgroup_id  : vec3u) {
    let dims = textureDimensions(lightmap);
    let quad = workgroup_id.z;
    let coord = invocation_id.xy;
    if (all(coord < dims)) {
      let base_idx = accumulation_base_index(coord, quad);
      let color = vec3(f32(atomicLoad(&accumulation[base_idx + 0])),
                       f32(atomicLoad(&accumulation[base_idx + 1])),
                       f32(atomicLoad(&accumulation[base_idx + 2])));

      textureStore(lightmap, coord, quad, vec4(color * uniforms.accumulation_to_lightmap_scale, 1));

      if (uniforms.accumulation_buffer_scale != 1.0) {
        let scaled = color * uniforms.accumulation_buffer_scale + 0.5;
        atomicStore(&accumulation[base_idx + 0], u32(scaled.r));
        atomicStore(&accumulation[base_idx + 1], u32(scaled.g));
        atomicStore(&accumulation[base_idx + 2], u32(scaled.b));
      }
    }
  }
);

static const char* rasterizer_shader_wgsl = CODE(
  // The lightmap data
  @group(1) @binding(0) var lightmap : texture_2d_array<f32>;

  // The sampler used to sample the lightmap
  @group(1) @binding(1) var smpl : sampler;

  // Vertex shader input data
  struct VertexIn {
    @location(0) position : vec4f,
    @location(1) uv : vec3f,
    @location(2) emissive : vec3f,
  }

  // Vertex shader output data
  struct VertexOut {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
    @location(1) emissive : vec3f,
    @interpolate(flat)
    @location(2) quad : u32,
  }

  @vertex
  fn vs_main(input : VertexIn) -> VertexOut {
    var output : VertexOut;
    output.position = common_uniforms.mvp * input.position;
    output.uv = input.uv.xy;
    output.quad = u32(input.uv.z + 0.5);
    output.emissive = input.emissive;
    return output;
  }

  @fragment
  fn fs_main(vertex_out : VertexOut) -> @location(0) vec4f {
    return textureSample(lightmap, smpl, vertex_out.uv, vertex_out.quad) + vec4f(vertex_out.emissive, 1);
  }
);

static const char* raytracer_shader_wgsl = CODE(
  // The lightmap data
  @group(1) @binding(0) var lightmap : texture_2d_array<f32>;

  // The sampler used to sample the lightmap
  @group(1) @binding(1) var smpl : sampler;

  // The output framebuffer
  @group(1) @binding(2) var framebuffer : texture_storage_2d<rgba16float, write>;

  override WorkgroupSizeX : u32;
  override WorkgroupSizeY : u32;

  const NumReflectionRays = 5;

  @compute @workgroup_size(WorkgroupSizeX, WorkgroupSizeY)
  fn main(@builtin(global_invocation_id) invocation_id : vec3u) {
    if (all(invocation_id.xy < textureDimensions(framebuffer))) {
      init_rand(invocation_id);

      let uv = vec2f(invocation_id.xy) / vec2f(textureDimensions(framebuffer).xy);
      let ndcXY = (uv - 0.5) * vec2(2, -2);

      var near = common_uniforms.inv_mvp * vec4f(ndcXY, 0.0, 1);
      var far = common_uniforms.inv_mvp * vec4f(ndcXY, 1, 1);
      near /= near.w;
      far /= far.w;

      let ray = Ray(near.xyz, normalize(far.xyz - near.xyz));
      let hit = raytrace(ray);

      let hit_color = sample_hit(hit);
      var normal = quads[hit.quad].plane.xyz;

      let bounce = reflect(ray.dir, normal);
      var reflection : vec3f;
      for (var i = 0; i < NumReflectionRays; i++) {
        let reflection_dir = normalize(bounce + rand_unit_sphere()*0.1);
        let reflection_ray = Ray(hit.pos + bounce * 1e-5, reflection_dir);
        let reflection_hit = raytrace(reflection_ray);
        reflection += sample_hit(reflection_hit);
      }
      let color = mix(reflection / NumReflectionRays, hit_color, 0.95);

      textureStore(framebuffer, invocation_id.xy, vec4(color, 1));
    }
  }

  fn sample_hit(hit : HitInfo) -> vec3f {
    let quad = quads[hit.quad];
    return textureSampleLevel(lightmap, smpl, hit.uv, hit.quad, 0).rgb +
           quad.emissive * quad.color;
  }
);
// clang-format on
