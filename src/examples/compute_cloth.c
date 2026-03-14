/* -------------------------------------------------------------------------- *
 * Compute Cloth Simulation
 *
 * Demonstrates a GPU-based cloth simulation using compute shaders. A grid of
 * particles is connected by springs (structural + shear) and simulated with
 * Verlet integration. The cloth drapes over a sphere rendered beneath it.
 *
 * Based on the Vulkan example by Sascha Willems:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/computecloth
 *
 * Key features:
 *  - Mass-spring-damper system with 8-neighbor connectivity
 *  - Sphere collision response
 *  - Per-frame normal recalculation on the GPU
 *  - Double-buffered storage buffers (ping-pong) for compute
 *  - Textured Phong-lit cloth with two-sided rendering
 *  - Optional wind simulation
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>
#define SOKOL_LOG_IMPL
#include <sokol_log.h>
#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
#pragma GCC diagnostic pop

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define CLOTH_GRID_X 60u
#define CLOTH_GRID_Y 60u
#define CLOTH_SIZE_X 5.0f
#define CLOTH_SIZE_Y 5.0f
#define PARTICLE_COUNT (CLOTH_GRID_X * CLOTH_GRID_Y) /* 3600 */

/* Compute workgroup & dispatch sizes */
#define WORKGROUP_SIZE_X 10u
#define WORKGROUP_SIZE_Y 10u
#define DISPATCH_X (CLOTH_GRID_X / WORKGROUP_SIZE_X) /* 6 */
#define DISPATCH_Y (CLOTH_GRID_Y / WORKGROUP_SIZE_Y) /* 6 */

/* Number of compute iterations per frame */
#define COMPUTE_ITERATIONS 64u

/* Texture file buffer (1024x1024 RGBA) */
#define TEX_FILE_BUFFER_SIZE (1024 * 1024 * 4 + 4096)

/* -------------------------------------------------------------------------- *
 * Particle struct — matches the WGSL storage buffer layout
 *
 * Each field is a vec4<f32> for natural 16-byte alignment.
 * -------------------------------------------------------------------------- */

typedef struct {
  float pos[4];    /* position xyz, w=1.0  */
  float vel[4];    /* velocity xyz, w=0.0  */
  float uv[4];     /* texcoord xy, zw=0.0  */
  float normal[4]; /* normal xyz, w=0.0    */
} particle_t;

/* -------------------------------------------------------------------------- *
 * UBO structures
 * -------------------------------------------------------------------------- */

/* Graphics UBO — shared by cloth and sphere vertex shaders */
typedef struct {
  mat4 projection;
  mat4 view;
  float light_pos[4]; /* vec4 */
} graphics_ubo_t;

/* Compute UBO — cloth simulation parameters */
typedef struct {
  float delta_t;
  float particle_mass;
  float spring_stiffness;
  float damping;
  float rest_dist_h;
  float rest_dist_v;
  float rest_dist_d;
  float sphere_radius;
  float sphere_pos[4];       /* vec4 */
  float gravity[4];          /* vec4 */
  int32_t particle_count[2]; /* ivec2 */
  int32_t _pad[2];           /* pad to 16-byte alignment */
} compute_ubo_t;

/* -------------------------------------------------------------------------- *
 * Shader forward declarations
 * -------------------------------------------------------------------------- */

static const char* cloth_shader_wgsl;
static const char* sphere_shader_wgsl;
static const char* compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Timing */
  uint64_t last_frame_time;
  float frame_timer;

  /* Models */
  gltf_model_t sphere_model;
  WGPUBuffer sphere_vertex_buffer;
  WGPUBuffer sphere_index_buffer;
  bool models_loaded;

  /* Cloth texture */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUSampler sampler;
    bool loaded;
    bool is_dirty;
    int width;
    int height;
  } cloth_texture;
  uint8_t tex_file_buffer[TEX_FILE_BUFFER_SIZE];

  /* Cloth geometry (CPU-side for initial upload) */
  WGPUBuffer cloth_index_buffer;
  uint32_t cloth_index_count;

  /* Compute storage buffers (ping-pong) */
  WGPUBuffer storage_buffers[2]; /* [0]=input, [1]=output */
  uint32_t read_set;             /* current input index for ping-pong */

  /* Graphics uniform buffer */
  WGPUBuffer graphics_ubo_buffer;
  graphics_ubo_t graphics_ubo_data;

  /* Compute uniform buffer */
  WGPUBuffer compute_ubo_buffer;
  compute_ubo_t compute_ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout graphics_bgl;
  WGPUBindGroupLayout compute_bgl;

  /* Bind groups */
  WGPUBindGroup graphics_bind_group;
  WGPUBindGroup compute_bind_groups[2]; /* [0] and [1] for ping-pong */

  /* Pipeline layouts */
  WGPUPipelineLayout graphics_pipeline_layout;
  WGPUPipelineLayout compute_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline cloth_pipeline;
  WGPURenderPipeline sphere_pipeline;

  /* Compute pipeline */
  WGPUComputePipeline compute_pipeline;

  /* GUI settings */
  struct {
    bool simulate_wind;
  } settings;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  WGPUBool initialized;
} state = {
  /* Vulkan original: {-2, 4, -2, 1} — Y negated for WebGPU Y-up */
  .graphics_ubo_data.light_pos = VKY_TO_WGPU_VEC4(-2.0f, 4.0f, -2.0f, 1.0f),
  .compute_ubo_data = {
    .particle_mass    = 0.1f,
    .spring_stiffness = 2000.0f,
    .damping          = 0.25f,
    .sphere_radius    = 1.0f,
    .sphere_pos       = {0.0f, 0.0f, 0.0f, 0.0f},
    /* Vulkan original: {0, 9.8, 0, 0} — Y negated for WebGPU Y-up */
    .gravity          = VKY_TO_WGPU_VEC4(0.0f, 9.8f, 0.0f, 0.0f),
    .particle_count   = {CLOTH_GRID_X, CLOTH_GRID_Y},
  },
  .settings.simulate_wind = false,
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
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
};

/* -------------------------------------------------------------------------- *
 * Model loading descriptor — pre-transform vertices & colors
 * -------------------------------------------------------------------------- */

static const gltf_model_desc_t sphere_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

/* -------------------------------------------------------------------------- *
 * Cloth grid initialization
 * -------------------------------------------------------------------------- */

static void init_cloth_particles(particle_t* particles)
{
  const float dx = CLOTH_SIZE_X / (float)(CLOTH_GRID_X - 1);
  const float dy = CLOTH_SIZE_Y / (float)(CLOTH_GRID_Y - 1);
  const float du = 1.0f / (float)(CLOTH_GRID_X - 1);
  const float dv = 1.0f / (float)(CLOTH_GRID_Y - 1);

  /* Vulkan original: Y = -2.0 (above sphere in Y-down convention).
   * Y negated for WebGPU Y-up: cloth starts above the sphere. */
  mat4 trans;
  glm_mat4_identity(trans);
  glm_translate(trans, (vec3)VKY_TO_WGPU_VEC3(-CLOTH_SIZE_X / 2.0f, -2.0f,
                                              -CLOTH_SIZE_Y / 2.0f));

  for (uint32_t i = 0; i < CLOTH_GRID_Y; i++) {
    for (uint32_t j = 0; j < CLOTH_GRID_X; j++) {
      /* Column-major indexing (matches Vulkan reference) */
      uint32_t idx = i + j * CLOTH_GRID_Y;

      /* Local position before transform */
      vec4 local_pos = {dx * (float)j, 0.0f, dy * (float)i, 1.0f};
      vec4 world_pos;
      glm_mat4_mulv(trans, local_pos, world_pos);

      particles[idx].pos[0] = world_pos[0];
      particles[idx].pos[1] = world_pos[1];
      particles[idx].pos[2] = world_pos[2];
      particles[idx].pos[3] = 1.0f;

      particles[idx].vel[0] = 0.0f;
      particles[idx].vel[1] = 0.0f;
      particles[idx].vel[2] = 0.0f;
      particles[idx].vel[3] = 0.0f;

      /* UVs (flipped to match Vulkan reference) */
      particles[idx].uv[0] = 1.0f - du * (float)i;
      particles[idx].uv[1] = dv * (float)j;
      particles[idx].uv[2] = 0.0f;
      particles[idx].uv[3] = 0.0f;

      particles[idx].normal[0] = 0.0f;
      particles[idx].normal[1] = 1.0f;
      particles[idx].normal[2] = 0.0f;
      particles[idx].normal[3] = 0.0f;
    }
  }
}

/* Build index buffer for the cloth mesh (triangle list, no primitive restart)
 *
 * WebGPU does not support primitive restart, so we convert the Vulkan
 * triangle-strip-with-restart to explicit triangle indices.
 */
static uint32_t build_cloth_indices(uint32_t* indices)
{
  uint32_t count = 0;
  for (uint32_t y = 0; y < CLOTH_GRID_Y - 1; y++) {
    for (uint32_t x = 0; x < CLOTH_GRID_X - 1; x++) {
      uint32_t top_left     = y * CLOTH_GRID_X + x;
      uint32_t top_right    = y * CLOTH_GRID_X + (x + 1);
      uint32_t bottom_left  = (y + 1) * CLOTH_GRID_X + x;
      uint32_t bottom_right = (y + 1) * CLOTH_GRID_X + (x + 1);

      /* Triangle 1 */
      indices[count++] = bottom_left;
      indices[count++] = top_left;
      indices[count++] = top_right;

      /* Triangle 2 */
      indices[count++] = bottom_left;
      indices[count++] = top_right;
      indices[count++] = bottom_right;
    }
  }
  return count;
}

/* -------------------------------------------------------------------------- *
 * Storage buffer creation (compute ping-pong)
 * -------------------------------------------------------------------------- */

static void create_storage_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Initialize particle data on CPU */
  particle_t* particles
    = (particle_t*)calloc(PARTICLE_COUNT, sizeof(particle_t));
  init_cloth_particles(particles);

  const size_t buf_size = PARTICLE_COUNT * sizeof(particle_t);

  /* Create both storage buffers with VERTEX | STORAGE | COPY_DST usage */
  for (int i = 0; i < 2; i++) {
    state.storage_buffers[i] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW(i == 0 ? "Particle SSBO Input" :
                                          "Particle SSBO Output"),
                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex
                         | WGPUBufferUsage_CopyDst,
                .size = buf_size,
              });

    /* Upload initial data to both buffers */
    wgpuQueueWriteBuffer(queue, state.storage_buffers[i], 0, particles,
                         buf_size);
  }

  free(particles);
}

/* -------------------------------------------------------------------------- *
 * Cloth index buffer
 * -------------------------------------------------------------------------- */

static void create_cloth_index_buffer(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Max triangles: (CLOTH_GRID_X-1) * (CLOTH_GRID_Y-1) * 2 triangles * 3
   * indices */
  const uint32_t max_indices = (CLOTH_GRID_X - 1) * (CLOTH_GRID_Y - 1) * 6;
  uint32_t* indices       = (uint32_t*)malloc(max_indices * sizeof(uint32_t));
  state.cloth_index_count = build_cloth_indices(indices);

  const size_t ib_size     = state.cloth_index_count * sizeof(uint32_t);
  state.cloth_index_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Cloth Index Buffer"),
              .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
              .size  = ib_size,
              .mappedAtCreation = true,
            });
  void* mapped = wgpuBufferGetMappedRange(state.cloth_index_buffer, 0, ib_size);
  memcpy(mapped, indices, ib_size);
  wgpuBufferUnmap(state.cloth_index_buffer);

  free(indices);
}

/* -------------------------------------------------------------------------- *
 * Sphere model loading
 * -------------------------------------------------------------------------- */

static void load_sphere_model(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  bool ok = gltf_model_load_from_file_ext(
    &state.sphere_model, "assets/models/sphere.gltf", 1.0f, &sphere_load_desc);
  if (!ok) {
    printf("[compute_cloth] Failed to load sphere.gltf\n");
    return;
  }

  gltf_model_t* m = &state.sphere_model;

  /* Create vertex buffer */
  size_t vb_size             = m->vertex_count * sizeof(gltf_vertex_t);
  state.sphere_vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Sphere - Vertex Buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata
    = wgpuBufferGetMappedRange(state.sphere_vertex_buffer, 0, vb_size);
  memcpy(vdata, m->vertices, vb_size);
  wgpuBufferUnmap(state.sphere_vertex_buffer);

  /* Create index buffer */
  if (m->index_count > 0) {
    size_t ib_size            = m->index_count * sizeof(uint32_t);
    state.sphere_index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Sphere Index Buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = true,
              });
    void* idata
      = wgpuBufferGetMappedRange(state.sphere_index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.sphere_index_buffer);
  }

  state.models_loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Cloth texture loading (async via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void cloth_texture_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("[compute_cloth] Cloth texture fetch failed, error: %d\n",
           response->error_code);
    return;
  }

  int w, h, num_ch;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &w, &h, &num_ch, 4);
  if (pixels) {
    state.cloth_texture.width  = w;
    state.cloth_texture.height = h;
    memcpy(state.tex_file_buffer, pixels, (size_t)(w * h * 4));
    image_free(pixels);
    state.cloth_texture.is_dirty = true;
    printf("[compute_cloth] Cloth texture loaded: %dx%d\n", w, h);
  }
}

static void fetch_cloth_texture(void)
{
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/vulkan_cloth_rgba.png",
    .callback = cloth_texture_fetch_callback,
    .buffer   = SFETCH_RANGE(state.tex_file_buffer),
    .channel  = 0,
  });
}

static void create_cloth_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  int w = state.cloth_texture.width;
  int h = state.cloth_texture.height;
  if (w == 0 || h == 0) {
    return;
  }

  /* Destroy old texture if resizing */
  if (state.cloth_texture.texture) {
    wgpuTextureDestroy(state.cloth_texture.texture);
    wgpuTextureRelease(state.cloth_texture.texture);
  }
  if (state.cloth_texture.view) {
    wgpuTextureViewRelease(state.cloth_texture.view);
  }

  state.cloth_texture.texture
    = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
                                        .label = STRVIEW("Cloth Texture"),
                                        .usage = WGPUTextureUsage_TextureBinding
                                                 | WGPUTextureUsage_CopyDst,
                                        .dimension = WGPUTextureDimension_2D,
                                        .size   = {(uint32_t)w, (uint32_t)h, 1},
                                        .format = WGPUTextureFormat_RGBA8Unorm,
                                        .mipLevelCount = 1,
                                        .sampleCount   = 1,
                                      });

  state.cloth_texture.view = wgpuTextureCreateView(
    state.cloth_texture.texture, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("Cloth Texture View"),
                                   .format    = WGPUTextureFormat_RGBA8Unorm,
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .mipLevelCount   = 1,
                                   .arrayLayerCount = 1,
                                 });

  /* Upload pixel data */
  wgpuQueueWriteTexture(queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.cloth_texture.texture,
                          .mipLevel = 0,
                          .origin   = {0, 0, 0},
                        },
                        state.tex_file_buffer, (size_t)(w * h * 4),
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = (uint32_t)(w * 4),
                          .rowsPerImage = (uint32_t)h,
                        },
                        &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});

  state.cloth_texture.loaded   = true;
  state.cloth_texture.is_dirty = false;
}

static void create_cloth_sampler(struct wgpu_context_t* wgpu_context)
{
  state.cloth_texture.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Cloth Sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void create_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Graphics UBO */
  state.graphics_ubo_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Graphics UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(graphics_ubo_t),
            });

  /* Compute UBO */
  const float dx                     = CLOTH_SIZE_X / (float)(CLOTH_GRID_X - 1);
  const float dy                     = CLOTH_SIZE_Y / (float)(CLOTH_GRID_Y - 1);
  state.compute_ubo_data.rest_dist_h = dx;
  state.compute_ubo_data.rest_dist_v = dy;
  state.compute_ubo_data.rest_dist_d = sqrtf(dx * dx + dy * dy);

  state.compute_ubo_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Compute UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(compute_ubo_t),
            });
}

static void update_graphics_ubo(struct wgpu_context_t* wgpu_context)
{
  glm_mat4_copy(state.camera.matrices.perspective,
                state.graphics_ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.graphics_ubo_data.view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.graphics_ubo_buffer, 0,
                       &state.graphics_ubo_data, sizeof(graphics_ubo_t));
}

static void update_compute_ubo(struct wgpu_context_t* wgpu_context)
{
  /* Clamp delta time to prevent simulation explosion */
  state.compute_ubo_data.delta_t = fminf(state.frame_timer, 0.02f) * 0.0025f;

  /* Wind simulation */
  if (state.settings.simulate_wind) {
    float time_val
      = (float)stm_sec(stm_now()) * 0.5f; /* slow time progression */
    float rd1 = 1.0f + ((float)(rand() % 1000) / 1000.0f) * 11.0f;
    float rd2 = 1.0f + ((float)(rand() % 1000) / 1000.0f) * 11.0f;
    state.compute_ubo_data.gravity[0]
      = cosf(glm_rad(-time_val * 360.0f)) * (rd1 - rd2);
    state.compute_ubo_data.gravity[2]
      = sinf(glm_rad(time_val * 360.0f)) * (rd1 - rd2);
  }
  else {
    state.compute_ubo_data.gravity[0] = 0.0f;
    state.compute_ubo_data.gravity[2] = 0.0f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.compute_ubo_buffer, 0,
                       &state.compute_ubo_data, sizeof(compute_ubo_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts & bind groups
 * -------------------------------------------------------------------------- */

static void create_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Graphics bind group layout: UBO (vert) + texture (frag) + sampler (frag) */
  WGPUBindGroupLayoutEntry gfx_entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = {.type = WGPUBufferBindingType_Uniform},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
  };
  state.graphics_bgl = wgpuDeviceCreateBindGroupLayout(
    device, &(WGPUBindGroupLayoutDescriptor){
              .label      = STRVIEW("Graphics BGL"),
              .entryCount = 3,
              .entries    = gfx_entries,
            });

  /* Compute bind group layout: SSBO in + SSBO out + UBO */
  WGPUBindGroupLayoutEntry comp_entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_ReadOnlyStorage},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Storage},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {.type = WGPUBufferBindingType_Uniform},
    },
  };
  state.compute_bgl
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .label = STRVIEW("Compute BGL"),
                                                .entryCount = 3,
                                                .entries    = comp_entries,
                                              });
}

/* Create a 1x1 white fallback texture for rendering before the real texture
 * loads */
static void create_fallback_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.cloth_texture.texture
    = wgpuDeviceCreateTexture(device, &(WGPUTextureDescriptor){
                                        .label = STRVIEW("Fallback Texture"),
                                        .usage = WGPUTextureUsage_TextureBinding
                                                 | WGPUTextureUsage_CopyDst,
                                        .dimension = WGPUTextureDimension_2D,
                                        .size      = {1, 1, 1},
                                        .format = WGPUTextureFormat_RGBA8Unorm,
                                        .mipLevelCount = 1,
                                        .sampleCount   = 1,
                                      });

  state.cloth_texture.view = wgpuTextureCreateView(
    state.cloth_texture.texture, &(WGPUTextureViewDescriptor){
                                   .label     = STRVIEW("Fallback TV"),
                                   .format    = WGPUTextureFormat_RGBA8Unorm,
                                   .dimension = WGPUTextureViewDimension_2D,
                                   .mipLevelCount   = 1,
                                   .arrayLayerCount = 1,
                                 });

  const uint8_t white[4] = {255, 255, 255, 255};
  wgpuQueueWriteTexture(
    queue, &(WGPUTexelCopyTextureInfo){.texture = state.cloth_texture.texture},
    white, 4, &(WGPUTexelCopyBufferLayout){.bytesPerRow = 4, .rowsPerImage = 1},
    &(WGPUExtent3D){1, 1, 1});

  state.cloth_texture.width  = 1;
  state.cloth_texture.height = 1;
}

static void create_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  const size_t ssbo_size = PARTICLE_COUNT * sizeof(particle_t);

  /* Graphics bind group */
  WGPUBindGroupEntry gfx_entries[3] = {
    [0] = {.binding = 0,
           .buffer  = state.graphics_ubo_buffer,
           .size    = sizeof(graphics_ubo_t)},
    [1] = {.binding = 1, .textureView = state.cloth_texture.view},
    [2] = {.binding = 2, .sampler = state.cloth_texture.sampler},
  };
  state.graphics_bind_group
    = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                          .label      = STRVIEW("Graphics BG"),
                                          .layout     = state.graphics_bgl,
                                          .entryCount = 3,
                                          .entries    = gfx_entries,
                                        });

  /* Compute bind groups — ping-pong:
   * BG[0]: read from storage[0], write to storage[1]
   * BG[1]: read from storage[1], write to storage[0] */
  for (int i = 0; i < 2; i++) {
    int in_idx                         = i;
    int out_idx                        = 1 - i;
    WGPUBindGroupEntry comp_entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.storage_buffers[in_idx],
             .size    = ssbo_size},
      [1] = {.binding = 1,
             .buffer  = state.storage_buffers[out_idx],
             .size    = ssbo_size},
      [2] = {.binding = 2,
             .buffer  = state.compute_ubo_buffer,
             .size    = sizeof(compute_ubo_t)},
    };
    char label[32];
    snprintf(label, sizeof(label), "Compute BG %d", i);
    state.compute_bind_groups[i]
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW(label),
                                            .layout     = state.compute_bgl,
                                            .entryCount = 3,
                                            .entries    = comp_entries,
                                          });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline creation
 * -------------------------------------------------------------------------- */

static void create_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Graphics pipeline layout */
  state.graphics_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Graphics PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.graphics_bgl,
            });

  /* Compute pipeline layout */
  state.compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Compute PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.compute_bgl,
            });
}

static void create_render_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* --- Cloth shader module --- */
  WGPUShaderModule cloth_sm
    = wgpu_create_shader_module(device, cloth_shader_wgsl);

  /* --- Sphere shader module --- */
  WGPUShaderModule sphere_sm
    = wgpu_create_shader_module(device, sphere_shader_wgsl);

  /* Depth stencil state — shared between cloth and sphere */
  WGPUDepthStencilState depth_stencil = {
    .format               = wgpu_context->depth_stencil_format,
    .depthWriteEnabled    = WGPUOptionalBool_True,
    .depthCompare         = WGPUCompareFunction_LessEqual,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  /* Color target state */
  WGPUBlendState blend = {
    .color = {.srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
              .operation = WGPUBlendOperation_Add},
    .alpha = {.srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
              .operation = WGPUBlendOperation_Add},
  };
  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend,
    .writeMask = WGPUColorWriteMask_All,
  };

  /* --- Cloth pipeline (vertex input from particle_t SSBO) --- */
  WGPUVertexAttribute cloth_attrs[3] = {
    /* position: offset 0, float32x3 (from vec4, only xyz used) */
    {.format = WGPUVertexFormat_Float32x3, .offset = 0, .shaderLocation = 0},
    /* uv: offset 32 (2 * vec4 = 32), float32x2 */
    {.format = WGPUVertexFormat_Float32x2, .offset = 32, .shaderLocation = 1},
    /* normal: offset 48 (3 * vec4 = 48), float32x3 */
    {.format = WGPUVertexFormat_Float32x3, .offset = 48, .shaderLocation = 2},
  };
  WGPUVertexBufferLayout cloth_vbl = {
    .arrayStride    = sizeof(particle_t), /* 64 bytes */
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 3,
    .attributes     = cloth_attrs,
  };

  state.cloth_pipeline = wgpuDeviceCreateRenderPipeline(
    device, &(WGPURenderPipelineDescriptor){
              .label  = STRVIEW("Cloth - Render pipeline"),
              .layout = state.graphics_pipeline_layout,
              .vertex = {
                .module      = cloth_sm,
                .entryPoint  = STRVIEW("vs_main"),
                .bufferCount = 1,
                .buffers     = &cloth_vbl,
              },
              .primitive = {
                .topology = WGPUPrimitiveTopology_TriangleList,
                .cullMode = WGPUCullMode_None, /* Both faces visible */
              },
              .depthStencil = &depth_stencil,
              .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
              .fragment     = &(WGPUFragmentState){
                .module      = cloth_sm,
                .entryPoint  = STRVIEW("fs_main"),
                .targetCount = 1,
                .targets     = &color_target,
              },
            });

  /* --- Sphere pipeline (vertex input from gltf_vertex_t) --- */
  WGPUVertexAttribute sphere_attrs[2] = {
    /* position: offset 0, float32x3 */
    {.format         = WGPUVertexFormat_Float32x3,
     .offset         = offsetof(gltf_vertex_t, position),
     .shaderLocation = 0},
    /* normal: offset 12 (vec3), float32x3 */
    {.format         = WGPUVertexFormat_Float32x3,
     .offset         = offsetof(gltf_vertex_t, normal),
     .shaderLocation = 1},
  };
  WGPUVertexBufferLayout sphere_vbl = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = sphere_attrs,
  };

  state.sphere_pipeline = wgpuDeviceCreateRenderPipeline(
    device, &(WGPURenderPipelineDescriptor){
              .label  = STRVIEW("Sphere Pipeline"),
              .layout = state.graphics_pipeline_layout,
              .vertex = {
                .module      = sphere_sm,
                .entryPoint  = STRVIEW("vs_main"),
                .bufferCount = 1,
                .buffers     = &sphere_vbl,
              },
              .primitive = {
                .topology = WGPUPrimitiveTopology_TriangleList,
                .cullMode = WGPUCullMode_Back,
                .frontFace = WGPUFrontFace_CCW,
              },
              .depthStencil = &depth_stencil,
              .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
              .fragment     = &(WGPUFragmentState){
                .module      = sphere_sm,
                .entryPoint  = STRVIEW("fs_main"),
                .targetCount = 1,
                .targets     = &color_target,
              },
            });

  wgpuShaderModuleRelease(cloth_sm);
  wgpuShaderModuleRelease(sphere_sm);
}

static void create_compute_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  WGPUShaderModule compute_sm
    = wgpu_create_shader_module(device, compute_shader_wgsl);

  state.compute_pipeline = wgpuDeviceCreateComputePipeline(
    device, &(WGPUComputePipelineDescriptor){
              .label   = STRVIEW("Cloth Compute Pipeline"),
              .layout  = state.compute_pipeline_layout,
              .compute = {
                .module     = compute_sm,
                .entryPoint = STRVIEW("cs_main"),
              },
            });

  wgpuShaderModuleRelease(compute_sm);
}

/* -------------------------------------------------------------------------- *
 * Draw helpers
 * -------------------------------------------------------------------------- */

static void draw_sphere(WGPURenderPassEncoder pass)
{
  if (!state.models_loaded) {
    return;
  }

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.sphere_vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.sphere_index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, state.sphere_index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  gltf_model_t* m = &state.sphere_model;
  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                         prim->first_index, 0, 0);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){250.0f, 80.0f}, ImGuiCond_FirstUseEver);
  igBegin("Settings", NULL, ImGuiWindowFlags_None);

  igCheckbox("Simulate wind", &state.settings.simulate_wind);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  camera_on_input_event(&state.camera, input_event);
  imgui_overlay_handle_input(wgpu_context, input_event);
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 1,
    .num_lanes    = 4,
    .logger.func  = slog_func,
  });

  /* Camera setup — Vulkan rotation (-30, -45, 0) adapted for WebGPU Y-up.
   * See doc/vulkan_to_webgpu_porting_guide.md for details. */
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -5.0f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-30.0f, -45.0f, 0.0f));
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 512.0f);

  /* Create resources */
  create_storage_buffers(wgpu_context);
  create_cloth_index_buffer(wgpu_context);
  load_sphere_model(wgpu_context);
  create_cloth_sampler(wgpu_context);
  create_fallback_texture(wgpu_context);
  create_uniform_buffers(wgpu_context);
  create_bind_group_layouts(wgpu_context);
  create_bind_groups(wgpu_context);
  create_pipeline_layouts(wgpu_context);
  create_render_pipelines(wgpu_context);
  create_compute_pipeline(wgpu_context);

  /* Start async texture loading */
  fetch_cloth_texture();

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Handle texture hot-loading */
  if (state.cloth_texture.is_dirty) {
    create_cloth_texture(wgpu_context);
    /* Recreate graphics bind group with new texture view */
    if (state.graphics_bind_group) {
      wgpuBindGroupRelease(state.graphics_bind_group);
    }
    WGPUBindGroupEntry gfx_entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.graphics_ubo_buffer,
             .size    = sizeof(graphics_ubo_t)},
      [1] = {.binding = 1, .textureView = state.cloth_texture.view},
      [2] = {.binding = 2, .sampler = state.cloth_texture.sampler},
    };
    state.graphics_bind_group = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Graphics BG"),
                              .layout     = state.graphics_bgl,
                              .entryCount = 3,
                              .entries    = gfx_entries,
                            });
  }

  /* Timing */
  uint64_t current_time = stm_now();
  state.frame_timer
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Camera update */
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 512.0f);
  camera_update(&state.camera, state.frame_timer);

  /* Update uniforms */
  update_graphics_ubo(wgpu_context);
  update_compute_ubo(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, state.frame_timer);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ---- Compute pass: 64 iterations of cloth simulation ---- */
  {
    WGPUComputePassEncoder cpass
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass, state.compute_pipeline);

    for (uint32_t iter = 0; iter < COMPUTE_ITERATIONS; iter++) {
      /* Ping-pong: alternate read/write buffers each iteration */
      state.read_set = 1 - state.read_set;
      wgpuComputePassEncoderSetBindGroup(
        cpass, 0, state.compute_bind_groups[state.read_set], 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(cpass, DISPATCH_X, DISPATCH_Y,
                                               1);
    }

    wgpuComputePassEncoderEnd(cpass);
    wgpuComputePassEncoderRelease(cpass);
  }

  /* ---- Render pass ---- */
  {
    state.color_attachment.view         = wgpu_context->swapchain_view;
    state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    /* Set viewport */
    wgpuRenderPassEncoderSetViewport(rpass, 0.0f, 0.0f,
                                     (float)wgpu_context->width,
                                     (float)wgpu_context->height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(rpass, 0, 0, wgpu_context->width,
                                        wgpu_context->height);

    /* Draw sphere */
    wgpuRenderPassEncoderSetPipeline(rpass, state.sphere_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.graphics_bind_group, 0,
                                      NULL);
    draw_sphere(rpass);

    /* Draw cloth — use the output SSBO as vertex buffer */
    wgpuRenderPassEncoderSetPipeline(rpass, state.cloth_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.graphics_bind_group, 0,
                                      NULL);

    /* After compute, the output buffer is storage_buffers[1 - read_set]
     * because we just finished writing into it */
    uint32_t output_idx = 1 - state.read_set;
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass, 0, state.storage_buffers[output_idx], 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass, state.cloth_index_buffer,
                                        WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rpass, state.cloth_index_count, 1, 0, 0,
                                     0);

    wgpuRenderPassEncoderEnd(rpass);
    wgpuRenderPassEncoderRelease(rpass);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay render */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown_func(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Destroy models */
  gltf_model_destroy(&state.sphere_model);

  /* Release WebGPU resources */
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.sphere_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.cloth_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.storage_buffers[0])
  WGPU_RELEASE_RESOURCE(Buffer, state.storage_buffers[1])
  WGPU_RELEASE_RESOURCE(Buffer, state.graphics_ubo_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute_ubo_buffer)

  if (state.cloth_texture.texture) {
    wgpuTextureDestroy(state.cloth_texture.texture);
    wgpuTextureRelease(state.cloth_texture.texture);
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.cloth_texture.view)
  WGPU_RELEASE_RESOURCE(Sampler, state.cloth_texture.sampler)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute_bgl)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[0])
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute_bind_groups[1])
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.cloth_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.sphere_pipeline)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Cloth",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown_func,
    .input_event_cb = input_event_cb,
  });
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* cloth_shader_wgsl = CODE(
  /* Graphics UBO */
  struct Uniforms {
    projection : mat4x4f,
    view       : mat4x4f,
    lightPos   : vec4f,
  };
  @group(0) @binding(0) var<uniform> ubo : Uniforms;
  @group(0) @binding(1) var clothTexture : texture_2d<f32>;
  @group(0) @binding(2) var clothSampler : sampler;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) uv       : vec2f,
    @location(1) normal   : vec3f,
    @location(2) viewVec  : vec3f,
    @location(3) lightVec : vec3f,
  };

  @vertex
  fn vs_main(
    @location(0) inPos    : vec3f,
    @location(1) inUV     : vec2f,
    @location(2) inNormal : vec3f
  ) -> VSOutput {
    var out : VSOutput;
    out.uv = inUV;
    /* Negate normal: the compute shader's cross products produce normals
     * oriented for Vulkan's Y-down convention. In WebGPU (Y-up) we negate
     * them so the lit face matches the Vulkan visual appearance. */
    out.normal = -inNormal;
    let eyePos = ubo.view * vec4f(inPos, 1.0);
    out.position = ubo.projection * eyePos;
    let pos = vec4f(inPos, 1.0);
    let lPos = ubo.lightPos.xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec = -pos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    let color = textureSample(clothTexture, clothSampler, in.uv).rgb;
    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(in.viewVec);
    let R = reflect(-L, N);
    let diffuse = max(dot(N, L), 0.15) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 8.0) * vec3f(0.2);
    return vec4f(diffuse * color + specular, 1.0);
  }
);

static const char* sphere_shader_wgsl = CODE(
  /* Shared UBO (same layout as cloth) */
  struct Uniforms {
    projection : mat4x4f,
    view       : mat4x4f,
    lightPos   : vec4f,
  };
  @group(0) @binding(0) var<uniform> ubo : Uniforms;

  struct VSOutput {
    @builtin(position) position : vec4f,
    @location(0) normal   : vec3f,
    @location(1) viewVec  : vec3f,
    @location(2) lightVec : vec3f,
  };

  @vertex
  fn vs_main(
    @location(0) inPos    : vec3f,
    @location(1) inNormal : vec3f
  ) -> VSOutput {
    var out : VSOutput;
    let eyePos = ubo.view * vec4f(inPos, 1.0);
    out.position = ubo.projection * eyePos;
    let pos = vec4f(inPos, 1.0);
    let lPos = ubo.lightPos.xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec = -pos.xyz;
    out.normal = inNormal;
    return out;
  }

  @fragment
  fn fs_main(in : VSOutput) -> @location(0) vec4f {
    let color = vec3f(0.5);
    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(in.viewVec);
    let R = reflect(-L, N);
    let diffuse = max(dot(N, L), 0.15) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 32.0) * vec3f(1.0);
    return vec4f(diffuse * color + specular, 1.0);
  }
);

static const char* compute_shader_wgsl = CODE(
  /* Particle structure — matches CPU-side particle_t (4 × vec4f = 64 bytes) */
  struct Particle {
    pos    : vec4f,
    vel    : vec4f,
    uv     : vec4f,
    normal : vec4f,
  };

  /* Compute UBO */
  struct Params {
    deltaT          : f32,
    particleMass    : f32,
    springStiffness : f32,
    damping         : f32,
    restDistH       : f32,
    restDistV       : f32,
    restDistD       : f32,
    sphereRadius    : f32,
    spherePos       : vec4f,
    gravity         : vec4f,
    particleCount   : vec2i,
    _pad            : vec2i,
  };

  @group(0) @binding(0) var<storage, read>       particleIn  : array<Particle>;
  @group(0) @binding(1) var<storage, read_write>  particleOut : array<Particle>;
  @group(0) @binding(2) var<uniform>              params      : Params;

  fn springForce(p0 : vec3f, p1 : vec3f, restDist : f32) -> vec3f {
    let dist = p0 - p1;
    let len = length(dist);
    if (len < 0.0001) { return vec3f(0.0); }
    return normalize(dist) * params.springStiffness * (len - restDist);
  }

  @compute @workgroup_size(10, 10, 1)
  fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    let pcx = u32(params.particleCount.x);
    let pcy = u32(params.particleCount.y);
    let index = id.y * pcx + id.x;

    if (index >= pcx * pcy) { return; }

    /* Gravity force */
    var force = params.gravity.xyz * params.particleMass;
    let pos = particleIn[index].pos.xyz;
    let vel = particleIn[index].vel.xyz;

    /* Spring forces — 8 neighbors (structural + shear) */
    /* Left */
    if (id.x > 0u) {
      force += springForce(particleIn[index - 1u].pos.xyz, pos, params.restDistH);
    }
    /* Right */
    if (id.x < pcx - 1u) {
      force += springForce(particleIn[index + 1u].pos.xyz, pos, params.restDistH);
    }
    /* Upper */
    if (id.y < pcy - 1u) {
      force += springForce(particleIn[index + pcx].pos.xyz, pos, params.restDistV);
    }
    /* Lower */
    if (id.y > 0u) {
      force += springForce(particleIn[index - pcx].pos.xyz, pos, params.restDistV);
    }
    /* Upper-left */
    if (id.x > 0u && id.y < pcy - 1u) {
      force += springForce(particleIn[index + pcx - 1u].pos.xyz, pos, params.restDistD);
    }
    /* Lower-left */
    if (id.x > 0u && id.y > 0u) {
      force += springForce(particleIn[index - pcx - 1u].pos.xyz, pos, params.restDistD);
    }
    /* Upper-right */
    if (id.x < pcx - 1u && id.y < pcy - 1u) {
      force += springForce(particleIn[index + pcx + 1u].pos.xyz, pos, params.restDistD);
    }
    /* Lower-right */
    if (id.x < pcx - 1u && id.y > 0u) {
      force += springForce(particleIn[index - pcx + 1u].pos.xyz, pos, params.restDistD);
    }

    /* Damping */
    force += -params.damping * vel;

    /* Verlet integration */
    let f = force * (1.0 / params.particleMass);
    let dt = params.deltaT;
    let newPos = pos + vel * dt + 0.5 * f * dt * dt;
    var newVel = vel + f * dt;

    /* Sphere collision */
    let sphereDist = newPos - params.spherePos.xyz;
    let sphereLen = length(sphereDist);
    if (sphereLen < params.sphereRadius + 0.01) {
      particleOut[index].pos = vec4f(
        params.spherePos.xyz + normalize(sphereDist) * (params.sphereRadius + 0.01),
        1.0
      );
      particleOut[index].vel = vec4f(0.0, 0.0, 0.0, 0.0);
    } else {
      particleOut[index].pos = vec4f(newPos, 1.0);
      particleOut[index].vel = vec4f(newVel, 0.0);
    }

    /* Normal calculation — average of surrounding triangle face normals */
    var normal = vec3f(0.0);
    if (id.y > 0u) {
      if (id.x > 0u) {
        let a = particleIn[index - 1u].pos.xyz - pos;
        let b = particleIn[index - pcx - 1u].pos.xyz - pos;
        let c = particleIn[index - pcx].pos.xyz - pos;
        normal += cross(a, b) + cross(b, c);
      }
      if (id.x < pcx - 1u) {
        let a = particleIn[index - pcx].pos.xyz - pos;
        let b = particleIn[index - pcx + 1u].pos.xyz - pos;
        let c = particleIn[index + 1u].pos.xyz - pos;
        normal += cross(a, b) + cross(b, c);
      }
    }
    if (id.y < pcy - 1u) {
      if (id.x > 0u) {
        let a = particleIn[index + pcx].pos.xyz - pos;
        let b = particleIn[index + pcx - 1u].pos.xyz - pos;
        let c = particleIn[index - 1u].pos.xyz - pos;
        normal += cross(a, b) + cross(b, c);
      }
      if (id.x < pcx - 1u) {
        let a = particleIn[index + 1u].pos.xyz - pos;
        let b = particleIn[index + pcx + 1u].pos.xyz - pos;
        let c = particleIn[index + pcx].pos.xyz - pos;
        normal += cross(a, b) + cross(b, c);
      }
    }
    particleOut[index].normal = vec4f(normalize(normal), 0.0);

    /* Copy UV through */
    particleOut[index].uv = particleIn[index].uv;
  }
);
// clang-format on
