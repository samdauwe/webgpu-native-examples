/* -------------------------------------------------------------------------- *
 * WebGPU Example - CPU Based Particle System
 *
 * This example renders a CPU-based particle system simulating a campfire with
 * two particle types (flame and smoke). The particles are updated on the CPU
 * each frame and uploaded to a GPU vertex buffer. A normal-mapped fireplace
 * environment is rendered as the backdrop.
 *
 * Ported from the Vulkan example:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/particlesystem
 * -------------------------------------------------------------------------- */

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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
 * WGSL Shaders (forward declarations)
 * -------------------------------------------------------------------------- */

static const char* particle_shader_wgsl;
static const char* env_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define PARTICLE_COUNT (512u)
#define FLAME_RADIUS (8.0f)
#define PARTICLE_TYPE_FLAME (0)
#define PARTICLE_TYPE_SMOKE (1)

/* File buffer size for sokol_fetch (largest texture is 1024x1024 RGBA) */
#define TEXTURE_FILE_BUFFER_SIZE (1024 * 1024 * 4 + 4096)

#define DEPTH_FORMAT (WGPUTextureFormat_Depth24PlusStencil8)

/* Number of textures loaded asynchronously */
#define TEX_FIRE 0
#define TEX_SMOKE 1
#define TEX_COLORMAP 2
#define TEX_NORMALMAP 3
#define TEXTURE_COUNT 4

/* -------------------------------------------------------------------------- *
 * Particle struct (must match Vulkan vertex attribute layout exactly)
 * -------------------------------------------------------------------------- */

typedef struct particle_t {
  float pos[4];         /* offset  0, size 16 — @location(0) */
  float color[4];       /* offset 16, size 16 — @location(1) */
  float alpha;          /* offset 32, size  4 — @location(2) */
  float size;           /* offset 36, size  4 — @location(3) */
  float rotation;       /* offset 40, size  4 — @location(4) */
  int32_t type;         /* offset 44, size  4 — @location(5) */
  float vel[4];         /* offset 48, size 16 — CPU only */
  float rotation_speed; /* offset 64, size  4 — CPU only */
  float _pad[3];        /* offset 68, size 12 — alignment padding */
} particle_t;           /* Total: 80 bytes */

/* Uniform buffer for particle pipeline */
typedef struct particles_ubo_t {
  mat4 projection;       /* 64 bytes */
  mat4 model_view;       /* 64 bytes */
  float viewport_dim[2]; /*  8 bytes */
  float point_size;      /*  4 bytes */
  float _pad;            /*  4 bytes */
} particles_ubo_t;       /* Total: 144 bytes */

/* Uniform buffer for environment pipeline */
typedef struct env_ubo_t {
  mat4 projection;    /* 64 bytes */
  mat4 model;         /* 64 bytes — actually modelview */
  mat4 normal;        /* 64 bytes — inverse-transpose of modelview */
  float light_pos[4]; /* 16 bytes */
} env_ubo_t;          /* Total: 208 bytes */

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Particles (CPU-side array) */
  particle_t particles[PARTICLE_COUNT];

  /* Particle vertex buffer (dynamic, uploaded every frame) */
  WGPUBuffer particle_vb;

  /* Environment GLTF model */
  gltf_model_t env_model;
  WGPUBuffer env_vb;
  WGPUBuffer env_ib;
  uint32_t env_index_count;
  bool env_model_loaded;

  /* Textures */
  struct {
    wgpu_texture_t fire;
    wgpu_texture_t smoke;
    wgpu_texture_t colormap;
    wgpu_texture_t normalmap;

    /* Async load file buffers */
    uint8_t file_buf[TEXTURE_COUNT][TEXTURE_FILE_BUFFER_SIZE];

    /* Load state tracking */
    bool loaded[TEXTURE_COUNT];
    bool all_loaded;
  } textures;

  /* Samplers */
  WGPUSampler particle_sampler; /* custom: border clamp, anisotropy     */
  WGPUSampler env_sampler;      /* standard: linear repeat               */

  /* Uniform buffers */
  wgpu_buffer_t particles_ub;
  wgpu_buffer_t env_ub;

  /* Shared bind group layout and pipeline layout */
  WGPUBindGroupLayout bind_group_layout;
  WGPUPipelineLayout pipeline_layout;

  /* Bind groups */
  WGPUBindGroup particle_bg;
  WGPUBindGroup env_bg;

  /* Pipelines */
  WGPURenderPipeline particle_pipeline;
  WGPURenderPipeline env_pipeline;

  /* Depth texture */
  wgpu_texture_t depth;

  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_desc;

  /* Uniform data (CPU-side) */
  particles_ubo_t particles_ubo;
  env_ubo_t env_ubo;

  /* Animation */
  float timer;
  float frame_timer;
  uint64_t last_frame_time;

  /* Settings */
  struct {
    bool paused;
    float point_size;
  } settings;

  /* Random state */
  uint32_t rng_state;

  WGPUBool initialized;
} state = {
  /* clang-format off */
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
  .render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  /* clang-format on */
  .settings = {
    .paused     = false,
    .point_size = 10.0f,
  },
  .rng_state = 12345678u,
};

/* -------------------------------------------------------------------------- *
 * Random number generator (xorshift32, deterministic)
 * -------------------------------------------------------------------------- */

static uint32_t rng_next(void)
{
  state.rng_state ^= state.rng_state << 13;
  state.rng_state ^= state.rng_state >> 17;
  state.rng_state ^= state.rng_state << 5;
  return state.rng_state;
}

/* Returns a random float in [0, range) */
static float rnd(float range)
{
  return ((float)(rng_next() & 0x7FFFFFFF) / (float)0x7FFFFFFF) * range;
}

/* -------------------------------------------------------------------------- *
 * Particle system — initialization and update
 * -------------------------------------------------------------------------- */

/* WebGPU Y-up: emitter at Y = +(FLAME_RADIUS - 2) = +6
 * (Vulkan had emitter at Y = -(FLAME_RADIUS - 2) = -6 in Y-down space) */
static const float EMITTER_POS[3] = {0.0f, FLAME_RADIUS - 2.0f, 0.0f};
static const float MIN_VEL[3]     = {-3.0f, 0.5f, -3.0f};
static const float MAX_VEL[3]     = {3.0f, 7.0f, 3.0f};

static void init_particle(particle_t* p)
{
  /* Initial velocity: upward (Y) with random X/Z */
  p->vel[0] = 0.0f;
  p->vel[1] = MIN_VEL[1] + rnd(MAX_VEL[1] - MIN_VEL[1]);
  p->vel[2] = 0.0f;
  p->vel[3] = 0.0f;

  p->alpha          = rnd(0.75f);
  p->size           = 1.0f + rnd(0.5f);
  p->color[0]       = 1.0f;
  p->color[1]       = 1.0f;
  p->color[2]       = 1.0f;
  p->color[3]       = 1.0f;
  p->type           = PARTICLE_TYPE_FLAME;
  p->rotation       = rnd(2.0f * GLM_PIf);
  p->rotation_speed = rnd(2.0f) - rnd(2.0f);

  /* Random point on sphere */
  float theta = rnd(2.0f * GLM_PIf);
  float phi   = rnd(GLM_PIf) - GLM_PIf / 2.0f;
  float r     = rnd(FLAME_RADIUS);

  p->pos[0] = r * cosf(theta) * cosf(phi) + EMITTER_POS[0];
  p->pos[1] = r * sinf(phi) + EMITTER_POS[1];
  p->pos[2] = r * sinf(theta) * cosf(phi) + EMITTER_POS[2];
  p->pos[3] = 0.0f;
}

static void transition_particle(particle_t* p)
{
  switch (p->type) {
    case PARTICLE_TYPE_FLAME:
      if (rnd(1.0f) < 0.05f) {
        /* Flame → smoke */
        p->alpha    = 0.0f;
        p->color[0] = 0.25f + rnd(0.25f);
        p->color[1] = p->color[0];
        p->color[2] = p->color[0];
        p->color[3] = 1.0f;
        p->pos[0] *= 0.5f;
        p->pos[2] *= 0.5f;
        /* Random smoke velocity (rising upward in Y-up WebGPU) */
        p->vel[0]         = rnd(1.0f) - rnd(1.0f);
        p->vel[1]         = (MIN_VEL[1] * 2.0f) + rnd(MAX_VEL[1] - MIN_VEL[1]);
        p->vel[2]         = rnd(1.0f) - rnd(1.0f);
        p->vel[3]         = 0.0f;
        p->size           = 1.0f + rnd(0.5f);
        p->rotation_speed = rnd(1.0f) - rnd(1.0f);
        p->type           = PARTICLE_TYPE_SMOKE;
      }
      else {
        init_particle(p);
      }
      break;

    case PARTICLE_TYPE_SMOKE:
      /* Smoke respawns as flame */
      init_particle(p);
      break;
  }
}

static void prepare_particles(void)
{
  for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
    init_particle(&state.particles[i]);
    /* Initial alpha based on distance from Y=0 (floor), matching Vulkan */
    float abs_y              = fabsf(state.particles[i].pos[1]);
    state.particles[i].alpha = 1.0f - (abs_y / (FLAME_RADIUS * 2.0f));
  }
}

/* Update all particles on the CPU and upload to GPU */
static void update_particles(wgpu_context_t* wgpu_context)
{
  float particle_timer = state.frame_timer * 0.45f;

  for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
    particle_t* p = &state.particles[i];

    switch (p->type) {
      case PARTICLE_TYPE_FLAME:
        /* Flames rise: Y increases in WebGPU Y-up
         * (Vulkan had pos.y -= vel.y * ... since Vulkan Y-down) */
        p->pos[1] += p->vel[1] * particle_timer * 3.5f;
        p->alpha += particle_timer * 2.5f;
        p->size -= particle_timer * 0.5f;
        break;

      case PARTICLE_TYPE_SMOKE:
        /* Smoke drifts: flip Y sign vs Vulkan (same logic as flames) */
        p->pos[0] -= p->vel[0] * state.frame_timer;
        p->pos[1] += p->vel[1] * state.frame_timer; /* Y rises in WebGPU */
        p->pos[2] -= p->vel[2] * state.frame_timer;
        p->alpha += particle_timer * 1.25f;
        p->size += particle_timer * 0.125f;
        p->color[0] -= particle_timer * 0.05f;
        p->color[1] -= particle_timer * 0.05f;
        p->color[2] -= particle_timer * 0.05f;
        break;
    }

    p->rotation += particle_timer * p->rotation_speed;

    /* Transition when particle has faded out (alpha > 2.0) */
    if (p->alpha > 2.0f) {
      transition_particle(p);
    }
  }

  /* Upload updated particle data to GPU vertex buffer */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.particle_vb, 0,
                       state.particles, PARTICLE_COUNT * sizeof(particle_t));
}

/* -------------------------------------------------------------------------- *
 * Async texture loading
 * -------------------------------------------------------------------------- */

typedef struct tex_fetch_user_data_t {
  wgpu_texture_t* texture;
  int index;
} tex_fetch_user_data_t;

static tex_fetch_user_data_t tex_user_data[TEXTURE_COUNT];

static void texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    fprintf(stderr, "Failed to fetch texture (error %d)\n",
            response->error_code);
    return;
  }

  tex_fetch_user_data_t* ud = (tex_fetch_user_data_t*)response->user_data;
  if (!ud) {
    return;
  }

  image_t img = {0};
  if (image_load_from_memory(response->data.ptr, (int)response->data.size, 4,
                             &img)) {
    ud->texture->desc = (wgpu_texture_desc_t){
      .extent = {
        .width              = (uint32_t)img.width,
        .height             = (uint32_t)img.height,
        .depthOrArrayLayers = 1,
      },
      .format       = WGPUTextureFormat_RGBA8Unorm,
      .address_mode = WGPUAddressMode_ClampToEdge,
      .pixels       = {
        .ptr  = img.pixels.u8,
        .size = (size_t)(img.width * img.height * 4),
      },
    };
    ud->texture->desc.is_dirty       = true;
    state.textures.loaded[ud->index] = true;
  }
}

static void load_textures(wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures */
  state.textures.fire      = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.smoke     = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.colormap  = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.normalmap = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Set up user data for callbacks */
  tex_user_data[TEX_FIRE]
    = (tex_fetch_user_data_t){&state.textures.fire, TEX_FIRE};
  tex_user_data[TEX_SMOKE]
    = (tex_fetch_user_data_t){&state.textures.smoke, TEX_SMOKE};
  tex_user_data[TEX_COLORMAP]
    = (tex_fetch_user_data_t){&state.textures.colormap, TEX_COLORMAP};
  tex_user_data[TEX_NORMALMAP]
    = (tex_fetch_user_data_t){&state.textures.normalmap, TEX_NORMALMAP};

  const char* paths[TEXTURE_COUNT] = {
    "assets/textures/particle_fire.png",
    "assets/textures/particle_smoke.png",
    "assets/textures/fireplace_colormap_rgba.png",
    "assets/textures/fireplace_normalmap_rgba.png",
  };

  for (int i = 0; i < TEXTURE_COUNT; ++i) {
    sfetch_send(&(sfetch_request_t){
      .path     = paths[i],
      .callback = texture_fetch_cb,
      .buffer   = SFETCH_RANGE(state.textures.file_buf[i]),
      .user_data = {
        .ptr  = &tex_user_data[i],
        .size = sizeof(tex_fetch_user_data_t),
      },
    });
  }
}

/* -------------------------------------------------------------------------- *
 * GLTF Environment Model
 * -------------------------------------------------------------------------- */

static void load_env_model(wgpu_context_t* wgpu_context)
{
  /* Load fireplace model (without FlipY — WebGPU is Y-up) */
  bool ok = gltf_model_load_from_file(&state.env_model,
                                      "assets/models/fireplace.gltf", 1.0f);
  if (!ok) {
    fprintf(stderr, "Failed to load fireplace.gltf\n");
    return;
  }

  /* Bake node transforms: PreTransformVertices | PreMultiplyVertexColors
   * (same as Vulkan but without FlipY) */
  static const gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  const size_t vb_size   = state.env_model.vertex_count * sizeof(gltf_vertex_t);
  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  if (!xformed) {
    fprintf(stderr, "Failed to allocate env vertex buffer\n");
    return;
  }
  memcpy(xformed, state.env_model.vertices, vb_size);
  gltf_model_bake_node_transforms(&state.env_model, xformed, &desc);

  WGPUDevice device = wgpu_context->device;

  /* Upload transformed vertex data */
  state.env_vb = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Environment vertex buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  memcpy(wgpuBufferGetMappedRange(state.env_vb, 0, vb_size), xformed, vb_size);
  wgpuBufferUnmap(state.env_vb);
  free(xformed);

  /* Upload index data */
  if (state.env_model.index_count > 0) {
    const size_t ib_size = state.env_model.index_count * sizeof(uint32_t);
    state.env_ib         = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                        .label = STRVIEW("Environment index buffer"),
                        .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                        .size  = ib_size,
                        .mappedAtCreation = true,
              });
    memcpy(wgpuBufferGetMappedRange(state.env_ib, 0, ib_size),
           state.env_model.indices, ib_size);
    wgpuBufferUnmap(state.env_ib);
    state.env_index_count = state.env_model.index_count;
  }

  state.env_model_loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth);

  state.depth.handle = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth texture"),
                            .size          = {.width              = wgpu_context->width,
                                              .height             = wgpu_context->height,
                                              .depthOrArrayLayers = 1},
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = DEPTH_FORMAT,
                            .usage         = WGPUTextureUsage_RenderAttachment,
                          });
  ASSERT(state.depth.handle != NULL);

  state.depth.view = wgpuTextureCreateView(
    state.depth.handle, &(WGPUTextureViewDescriptor){
                          .label           = STRVIEW("Depth texture view"),
                          .dimension       = WGPUTextureViewDimension_2D,
                          .format          = DEPTH_FORMAT,
                          .baseMipLevel    = 0,
                          .mipLevelCount   = 1,
                          .baseArrayLayer  = 0,
                          .arrayLayerCount = 1,
                        });
  ASSERT(state.depth.view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Samplers
 * -------------------------------------------------------------------------- */

static void init_samplers(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Particle sampler: clamp-to-border, anisotropic filtering
   * Matches Vulkan's custom particle sampler */
  state.particle_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Particle sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 8.0f,
              .maxAnisotropy = 8,
            });
  ASSERT(state.particle_sampler != NULL);

  /* Environment sampler: linear repeat */
  state.env_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Environment sampler"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 8.0f,
              .maxAnisotropy = 1,
            });
  ASSERT(state.env_sampler != NULL);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.particles_ub = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particles UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(particles_ubo_t),
                  });

  state.env_ub = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Environment UBO",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(env_ubo_t),
                  });
}

/* -------------------------------------------------------------------------- *
 * Particle vertex buffer
 * -------------------------------------------------------------------------- */

static void init_particle_vertex_buffer(wgpu_context_t* wgpu_context)
{
  /* Dynamic vertex buffer — updated every frame via wgpuQueueWriteBuffer */
  state.particle_vb = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Particle vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = PARTICLE_COUNT * sizeof(particle_t),
      .mappedAtCreation = false,
    });
  ASSERT(state.particle_vb != NULL);

  /* Upload initial particle positions */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.particle_vb, 0,
                       state.particles, PARTICLE_COUNT * sizeof(particle_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout and pipeline layout (shared by both pipelines)
 * -------------------------------------------------------------------------- */

/* Layout:
 *   binding 0 — uniform buffer (particles UBO or env UBO)
 *   binding 1 — sampler
 *   binding 2 — texture_2d (smoke or colormap)
 *   binding 3 — texture_2d (fire or normalmap)
 */
static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[4] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = 0, /* accept both UBO sizes */
      },
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Shared bind group layout"),
                            .entryCount = 4,
                            .entries    = entries,
                          });
  ASSERT(state.bind_group_layout != NULL);

  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Shared pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void rebuild_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Release old bind groups if any */
  WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.env_bg)

  /* Particle bind group: particles_ubo + particle_sampler + smoke + fire */
  {
    WGPUBindGroupEntry entries[4] = {
      [0] = {.binding = 0,
             .buffer  = state.particles_ub.buffer,
             .size    = state.particles_ub.size},
      [1] = {.binding = 1, .sampler = state.particle_sampler},
      [2] = {.binding = 2, .textureView = state.textures.smoke.view},
      [3] = {.binding = 3, .textureView = state.textures.fire.view},
    };
    state.particle_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Particle bind group"),
                              .layout     = state.bind_group_layout,
                              .entryCount = 4,
                              .entries    = entries,
                            });
    ASSERT(state.particle_bg != NULL);
  }

  /* Environment bind group: env_ubo + env_sampler + colormap + normalmap */
  {
    WGPUBindGroupEntry entries[4] = {
      [0] = {.binding = 0,
             .buffer  = state.env_ub.buffer,
             .size    = state.env_ub.size},
      [1] = {.binding = 1, .sampler = state.env_sampler},
      [2] = {.binding = 2, .textureView = state.textures.colormap.view},
      [3] = {.binding = 3, .textureView = state.textures.normalmap.view},
    };
    state.env_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Environment bind group"),
                              .layout     = state.bind_group_layout,
                              .entryCount = 4,
                              .entries    = entries,
                            });
    ASSERT(state.env_bg != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ------------------------------------------------------------------ *
   * Particle pipeline
   * ------------------------------------------------------------------ */
  {
    WGPUShaderModule vert
      = wgpu_create_shader_module(device, particle_shader_wgsl);
    WGPUShaderModule frag
      = wgpu_create_shader_module(device, particle_shader_wgsl);

    /* Premultiplied alpha blend (src=ONE, dst=ONE_MINUS_SRC_ALPHA):
     * Matches Vulkan: srcColorBlendFactor=ONE,
     * dstColorBlendFactor=ONE_MINUS_SRC_ALPHA */
    WGPUBlendState premul_blend = {
      .color = {
        .operation = WGPUBlendOperation_Add,
        .srcFactor = WGPUBlendFactor_One,
        .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      },
      .alpha = {
        .operation = WGPUBlendOperation_Add,
        .srcFactor = WGPUBlendFactor_One,
        .dstFactor = WGPUBlendFactor_Zero,
      },
    };

    /* Per-instance particle vertex attributes */
    WGPUVertexAttribute particle_attrs[] = {
      /* location 0: pos (vec4) */
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(particle_t, pos)},
      /* location 1: color (vec4) */
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(particle_t, color)},
      /* location 2: alpha (float) */
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32,
       .offset         = offsetof(particle_t, alpha)},
      /* location 3: size (float) */
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32,
       .offset         = offsetof(particle_t, size)},
      /* location 4: rotation (float) */
      {.shaderLocation = 4,
       .format         = WGPUVertexFormat_Float32,
       .offset         = offsetof(particle_t, rotation)},
      /* location 5: type (int32) */
      {.shaderLocation = 5,
       .format         = WGPUVertexFormat_Sint32,
       .offset         = offsetof(particle_t, type)},
    };
    WGPUVertexBufferLayout particle_vb_layout = {
      .arrayStride    = sizeof(particle_t),
      .stepMode       = WGPUVertexStepMode_Instance, /* per-instance data */
      .attributeCount = ARRAY_SIZE(particle_attrs),
      .attributes     = particle_attrs,
    };

    /* Depth stencil: test but do NOT write (particles are transparent) */
    WGPUDepthStencilState depth_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = false,
      });
    depth_state.depthCompare = WGPUCompareFunction_LessEqual;

    state.particle_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Particle render pipeline"),
        .layout = state.pipeline_layout,
        .vertex = {
          .module      = vert,
          .entryPoint  = STRVIEW("vs_particle"),
          .bufferCount = 1,
          .buffers     = &particle_vb_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = frag,
          .entryPoint  = STRVIEW("fs_particle"),
          .targetCount = 1,
          .targets = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .blend     = &premul_blend,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_state,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
      });
    ASSERT(state.particle_pipeline != NULL);

    wgpuShaderModuleRelease(vert);
    wgpuShaderModuleRelease(frag);
  }

  /* ------------------------------------------------------------------ *
   * Environment (normalmap) pipeline
   * ------------------------------------------------------------------ */
  {
    WGPUShaderModule vert = wgpu_create_shader_module(device, env_shader_wgsl);
    WGPUShaderModule frag = wgpu_create_shader_module(device, env_shader_wgsl);

    /* Vertex layout matching gltf_vertex_t
     * Shader inputs: position(0), uv0(1), normal(2), tangent(3) */
    WGPUVertexAttribute env_attrs[] = {
      /* location 0: position (vec3) */
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      /* location 1: uv0 (vec2) */
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      /* location 2: normal (vec3) */
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      /* location 3: tangent (vec4) */
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, tangent)},
    };
    WGPUVertexBufferLayout env_vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(env_attrs),
      .attributes     = env_attrs,
    };

    /* Opaque depth: test and write */
    WGPUDepthStencilState depth_state
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    depth_state.depthCompare = WGPUCompareFunction_LessEqual;

    WGPUBlendState no_blend = wgpu_create_blend_state(false);

    state.env_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Environment render pipeline"),
        .layout = state.pipeline_layout,
        .vertex = {
          .module      = vert,
          .entryPoint  = STRVIEW("vs_env"),
          .bufferCount = 1,
          .buffers     = &env_vb_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = frag,
          .entryPoint  = STRVIEW("fs_env"),
          .targetCount = 1,
          .targets = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .blend     = &no_blend,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_state,
        .multisample  = {.count = 1, .mask = 0xFFFFFFFF},
      });
    ASSERT(state.env_pipeline != NULL);

    wgpuShaderModuleRelease(vert);
    wgpuShaderModuleRelease(frag);
  }
}

/* -------------------------------------------------------------------------- *
 * Camera setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.movement_speed = 5.0f;
  state.camera.rotation_speed = 0.25f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;

  /* Vulkan: position(0, 0, -75), rotation(-15, 45, 0)
   * WebGPU: Y=0 no change; negate pitch: rotation(15, 45, 0) */
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -75.0f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-15.0f, 45.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer update
 * -------------------------------------------------------------------------- */

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  camera_update_view_matrix(&state.camera);

  /* Particle UBO */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.particles_ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.particles_ubo.model_view);
  state.particles_ubo.viewport_dim[0] = (float)wgpu_context->width;
  state.particles_ubo.viewport_dim[1] = (float)wgpu_context->height;
  state.particles_ubo.point_size      = state.settings.point_size;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.particles_ub.buffer, 0,
                       &state.particles_ubo, sizeof(particles_ubo_t));

  /* Environment UBO */
  glm_mat4_copy(state.camera.matrices.perspective, state.env_ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.env_ubo.model);

  /* Normal matrix = inverse-transpose of modelview */
  mat4 inv_mv;
  glm_mat4_inv(state.env_ubo.model, inv_mv);
  glm_mat4_transpose_to(inv_mv, state.env_ubo.normal);

  /* Animated light position (circles in XZ plane, Y=0) */
  if (!state.settings.paused) {
    state.env_ubo.light_pos[0] = sinf(state.timer * 2.0f * GLM_PIf) * 1.5f;
    state.env_ubo.light_pos[1] = 0.0f;
    state.env_ubo.light_pos[2] = cosf(state.timer * 2.0f * GLM_PIf) * 1.5f;
    state.env_ubo.light_pos[3] = 0.0f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.env_ub.buffer, 0,
                       &state.env_ubo, sizeof(env_ubo_t));
}

/* -------------------------------------------------------------------------- *
 * Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = TEXTURE_COUNT + 2,
    .num_channels = 2,
    .num_lanes    = 2,
    .logger.func  = slog_func,
  });

  /* Seed RNG with time for variety each run */
  state.rng_state = (uint32_t)stm_now() ^ 0xDEADBEEFu;
  if (state.rng_state == 0) {
    state.rng_state = 1;
  }

  /* Initialize camera */
  init_camera(wgpu_context);

  /* Load GLTF environment (synchronous) */
  load_env_model(wgpu_context);

  /* Initialize particle system */
  prepare_particles();

  /* GPU resources */
  init_depth_texture(wgpu_context);
  init_samplers(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_particle_vertex_buffer(wgpu_context);
  init_bind_group_layout(wgpu_context);

  /* Start async texture loading — placeholder textures must exist
   * before bind groups are built */
  load_textures(wgpu_context);

  rebuild_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  /* GUI */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);
  igBegin("CPU Particle System", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  igText("Particles: %u", PARTICLE_COUNT);

  igCheckbox("Paused", &state.settings.paused);
  imgui_overlay_slider_float("Point Size", &state.settings.point_size, 1.0f,
                             32.0f, "%.1f");

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate depth texture and update projection aspect ratio */
    init_depth_texture(wgpu_context);
    camera_set_perspective(
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);
    return;
  }

  /* Skip camera input when ImGui captures the mouse */
  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Draw the environment model
 * -------------------------------------------------------------------------- */

static void draw_env(WGPURenderPassEncoder pass)
{
  if (!state.env_model_loaded || !state.env_vb) {
    return;
  }

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.env_vb, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.env_ib) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.env_ib, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }

  /* Draw all primitives */
  for (uint32_t n = 0; n < state.env_model.linear_node_count; ++n) {
    gltf_node_t* node = state.env_model.linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; ++p) {
      gltf_primitive_t* prim = &mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                         prim->first_index, 0, 0);
      }
      else if (prim->vertex_count > 0) {
        wgpuRenderPassEncoderDraw(pass, prim->vertex_count, 1, 0, 0);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async texture loads */
  sfetch_dowork();

  /* Upload newly loaded textures to GPU */
  bool rebind                             = false;
  wgpu_texture_t* tex_ptrs[TEXTURE_COUNT] = {
    &state.textures.fire,
    &state.textures.smoke,
    &state.textures.colormap,
    &state.textures.normalmap,
  };
  for (int i = 0; i < TEXTURE_COUNT; ++i) {
    if (tex_ptrs[i]->desc.is_dirty) {
      wgpu_recreate_texture(wgpu_context, tex_ptrs[i]);
      FREE_TEXTURE_PIXELS(*tex_ptrs[i]);
      rebind = true;
    }
  }
  if (rebind) {
    rebuild_bind_groups(wgpu_context);
  }

  /* Delta time */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  state.frame_timer     = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Animation timer: Vulkan base default timerSpeed = 0.25, example scales it
   * by 8 → effective timerSpeed = 0.25 * 8 = 2.0 (drives light orbit speed) */
  if (!state.settings.paused) {
    state.timer += state.frame_timer * 2.0f;
    if (state.timer > 1.0f) {
      state.timer -= 1.0f;
    }
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* Update + upload particles */
  if (!state.settings.paused) {
    update_particles(wgpu_context);
  }

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, state.frame_timer);
  render_gui(wgpu_context);

  /* Render */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth.view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_desc);

  /* 1. Draw environment (opaque, writes depth) */
  wgpuRenderPassEncoderSetPipeline(rpass, state.env_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.env_bg, 0, NULL);
  draw_env(rpass);

  /* 2. Draw particles (transparent, reads but does not write depth)
   *    Draw 6 vertices (2 triangles) per particle instance */
  wgpuRenderPassEncoderSetPipeline(rpass, state.particle_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.particle_bg, 0, NULL);
  wgpuRenderPassEncoderSetVertexBuffer(rpass, 0, state.particle_vb, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(rpass, 6, PARTICLE_COUNT, 0, 0);

  wgpuRenderPassEncoderEnd(rpass);

  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buf);

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass);
  wgpuCommandBufferRelease(cmd_buf);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Particle vertex buffer */
  WGPU_RELEASE_RESOURCE(Buffer, state.particle_vb)

  /* Environment model buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.env_vb)
  WGPU_RELEASE_RESOURCE(Buffer, state.env_ib)
  gltf_model_destroy(&state.env_model);

  /* Textures */
  wgpu_destroy_texture(&state.textures.fire);
  wgpu_destroy_texture(&state.textures.smoke);
  wgpu_destroy_texture(&state.textures.colormap);
  wgpu_destroy_texture(&state.textures.normalmap);
  wgpu_destroy_texture(&state.depth);

  /* Samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.particle_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.env_sampler)

  /* Uniform buffers */
  wgpu_destroy_buffer(&state.particles_ub);
  wgpu_destroy_buffer(&state.env_ub);

  /* Pipelines + layouts */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.particle_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.env_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.particle_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.env_bg)
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "CPU Based Particle System",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* ========================================================================== *
 * WGSL Shaders
 * ========================================================================== */

/* -------------------------------------------------------------------------- *
 * Particle shaders (vertex + fragment in one module)
 *
 * WebGPU has no gl_PointSize / gl_PointCoord, so each particle is rendered
 * as an instanced billboard quad (6 vertices = 2 triangles per particle).
 * The quad size is computed to match the Vulkan gl_PointSize calculation
 * exactly.
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* particle_shader_wgsl = CODE(

/* ---------- Uniforms ----------------------------------------------------- */
struct ParticlesUbo {
  projection  : mat4x4f,
  model_view  : mat4x4f,
  viewport_dim : vec2f,
  point_size  : f32,
  _pad        : f32,
}

@group(0) @binding(0) var<uniform> ubo : ParticlesUbo;
@group(0) @binding(1) var psampler    : sampler;
@group(0) @binding(2) var smoke_tex   : texture_2d<f32>;
@group(0) @binding(3) var fire_tex    : texture_2d<f32>;

/* ---------- Vertex input -------------------------------------------------- */
struct VertIn {
  @builtin(vertex_index)   vi       : u32,      /* 0..5 quad corner   */
  /* Per-instance particle data */
  @location(0)             pos      : vec4f,
  @location(1)             color    : vec4f,
  @location(2)             alpha    : f32,
  @location(3)             size     : f32,
  @location(4)             rotation : f32,
  @location(5)             ptype    : i32,
}

/* ---------- Vertex output ------------------------------------------------- */
struct VertOut {
  @builtin(position)                 clip_pos  : vec4f,
  @location(0)                       color     : vec4f,
  @location(1)                       alpha     : f32,
  @location(2) @interpolate(flat)    ptype     : i32,
  @location(3)                       rotation  : f32,
  @location(4)                       uv        : vec2f,
}

/* ---------- Vertex shader ------------------------------------------------- */
@vertex
fn vs_particle(in: VertIn) -> VertOut {
  /* Billboard quad corners (2 triangles, CCW winding, Y-up clip space)
   * Index: 0=bottom-left, 1=bottom-right, 2=top-right,
   *        3=bottom-left, 4=top-right,    5=top-left  */
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0,  1.0),
  );

  /* UV coordinates match gl_PointCoord convention:
   * (0,0) = top-left of sprite, (1,1) = bottom-right */
  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 0.0),
    vec2f(0.0, 0.0),
  );

  let corner = corners[in.vi];
  let uv     = uvs[in.vi];

  /* Project particle center to clip space */
  let eye_pos  = ubo.model_view * vec4f(in.pos.xyz, 1.0);
  let clip_pos = ubo.projection * eye_pos;

  /* Compute point size in pixels (matches Vulkan gl_PointSize formula):
   *   spriteSize = 8.0 * inSize
   *   projectedCorner = projection * vec4(0.5 * spriteSize, 0.5 * spriteSize,
   *                                       eyePos.z, eyePos.w)
   *   gl_PointSize = viewportDim.x * projectedCorner.x / projectedCorner.w  */
  let sprite_size = 8.0 * in.size;
  let half_size   = 0.5 * sprite_size;
  let proj_corner = ubo.projection * vec4f(half_size, half_size, eye_pos.z, eye_pos.w);
  let point_px    = ubo.viewport_dim.x * proj_corner.x / proj_corner.w;

  /* Convert pixel radius to NDC, then to clip space offset:
   *   NDC offset  = (point_px / 2) / (viewport / 2) = point_px / viewport
   *   Clip offset = NDC offset * clip_pos.w           (undo perspective div)  */
  let ndc_half_x = point_px / ubo.viewport_dim.x;
  let ndc_half_y = point_px / ubo.viewport_dim.y;
  let clip_ox    = corner.x * ndc_half_x * clip_pos.w;
  let clip_oy    = corner.y * ndc_half_y * clip_pos.w;

  var out : VertOut;
  out.clip_pos = vec4f(clip_pos.x + clip_ox, clip_pos.y + clip_oy,
                       clip_pos.z, clip_pos.w);
  out.color    = in.color;
  out.alpha    = in.alpha;
  out.ptype    = in.ptype;
  out.rotation = in.rotation;
  out.uv       = uv;
  return out;
}

/* ---------- Fragment shader ----------------------------------------------- */
@fragment
fn fs_particle(in: VertOut) -> @location(0) vec4f {
  /* Triangle-wave alpha: 0→transparent, 1→opaque, 2→transparent */
  let a = select(2.0 - in.alpha, in.alpha, in.alpha <= 1.0);

  /* Rotate UV around centre (0.5, 0.5) to spin the particle sprite */
  let rot_cos  = cos(in.rotation);
  let rot_sin  = sin(in.rotation);
  let c        = in.uv - vec2f(0.5);
  let rot_uv   = vec2f(
    rot_cos * c.x + rot_sin * c.y + 0.5,
    rot_cos * c.y - rot_sin * c.x + 0.5,
  );

  /* Sample both textures outside the if to satisfy WGSL uniform control flow */
  let fire_color  = textureSample(fire_tex,  psampler, rot_uv);
  let smoke_color = textureSample(smoke_tex, psampler, rot_uv);

  var out_color : vec4f;
  if (in.ptype == 0) {
    /* Flame: fire texture, no alpha write (additive blending) */
    out_color = vec4f(fire_color.rgb * in.color.rgb * a, 0.0);
  } else {
    /* Smoke: smoke texture with pre-multiplied alpha */
    let smoke_alpha = smoke_color.a * a;
    out_color = vec4f(smoke_color.rgb * in.color.rgb * a, smoke_alpha);
  }

  return out_color;
}

); // end particle_shader_wgsl
// clang-format on

/* -------------------------------------------------------------------------- *
 * Environment (normal-mapped fireplace) shaders
 *
 * Faithful WGSL translation of the Vulkan GLSL normalmap.vert / normalmap.frag
 * shaders. Lighting math is in tangent space.
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* env_shader_wgsl = CODE(

const LIGHT_RADIUS : f32 = 45.0;

/* ---------- Uniforms ----------------------------------------------------- */
struct EnvUbo {
  projection : mat4x4f,
  model      : mat4x4f,   /* modelview (fireplace at world origin)          */
  normal     : mat4x4f,   /* inverse-transpose of modelview                */
  light_pos  : vec4f,     /* world-space animated light position            */
}

@group(0) @binding(0) var<uniform> ubo        : EnvUbo;
@group(0) @binding(1) var          esampler   : sampler;
@group(0) @binding(2) var          color_map  : texture_2d<f32>;
@group(0) @binding(3) var          normal_map : texture_2d<f32>;

/* ---------- Vertex I/O --------------------------------------------------- */
struct EnvVertIn {
  @location(0) in_pos     : vec3f,
  @location(1) in_uv      : vec2f,
  @location(2) in_normal  : vec3f,
  @location(3) in_tangent : vec4f,
}

struct EnvVertOut {
  @builtin(position) clip_pos      : vec4f,
  @location(0)       uv            : vec2f,
  @location(1)       light_vec     : vec3f,   /* light vec in tangent space  */
  @location(2)       light_vec_b   : vec3f,   /* light dist in tangent space */
  @location(3)       light_dir     : vec3f,   /* light dir in world space    */
  @location(4)       view_vec      : vec3f,   /* view vec in tangent space   */
}

/* ---------- Vertex shader ------------------------------------------------- */
@vertex
fn vs_env(in: EnvVertIn) -> EnvVertOut {
  /* Extract 3x3 normal matrix from the 4x4 */
  let nm = mat3x3f(ubo.normal[0].xyz, ubo.normal[1].xyz, ubo.normal[2].xyz);

  /* World-space (actually view-space since model = view) vertex position */
  let vertex_pos = (ubo.model * vec4f(in.in_pos, 1.0)).xyz;

  /* Light direction in world space */
  let light_dir = normalize(ubo.light_pos.xyz - vertex_pos);

  /* Build TBN matrix: tangent, bitangent, normal in view space */
  let bi_tangent = cross(in.in_normal, in.in_tangent.xyz);

  var tbn : mat3x3f;
  tbn[0] = nm * in.in_tangent.xyz;
  tbn[1] = nm * bi_tangent;
  tbn[2] = nm * in.in_normal;

  /* Light vector in tangent space */
  let light_vec = (ubo.light_pos.xyz - vertex_pos) * tbn;

  /* Light distance in tangent space (object space minus object space) */
  let light_dist = ubo.light_pos.xyz - in.in_pos;
  let light_vec_b = vec3f(
    dot(in.in_tangent.xyz, light_dist),
    dot(bi_tangent,        light_dist),
    dot(in.in_normal,      light_dist),
  );

  /* View vector in tangent space */
  let view_vec = vec3f(
    dot(in.in_tangent.xyz, in.in_pos),
    dot(bi_tangent,        in.in_pos),
    dot(in.in_normal,      in.in_pos),
  );

  var out : EnvVertOut;
  out.clip_pos    = ubo.projection * ubo.model * vec4f(in.in_pos, 1.0);
  out.uv          = in.in_uv;
  out.light_vec   = light_vec;
  out.light_vec_b = light_vec_b;
  out.light_dir   = light_dir;
  out.view_vec    = view_vec;
  return out;
}

/* ---------- Fragment shader ----------------------------------------------- */
@fragment
fn fs_env(in: EnvVertOut) -> @location(0) vec4f {
  let specular_color = vec3f(0.85, 0.5, 0.0);
  let inv_radius     = 1.0 / LIGHT_RADIUS;
  let ambient        = 0.25;

  /* Sample color and normal maps */
  let rgb    = textureSample(color_map,  esampler, in.uv).rgb;
  let normal = normalize((textureSample(normal_map, esampler, in.uv).rgb - 0.5) * 2.0);

  /* Attenuation based on light distance */
  let dist_sqr = dot(in.light_vec_b, in.light_vec_b);
  let l_vec    = in.light_vec_b * inverseSqrt(dist_sqr);
  let atten    = max(clamp(1.0 - inv_radius * sqrt(dist_sqr), 0.0, 1.0), ambient);

  /* Diffuse term */
  let diffuse = clamp(dot(l_vec, normal), 0.0, 1.0);

  /* Specular term (Phong) */
  let light       = normalize(-in.light_vec);
  let view        = normalize(in.view_vec);
  let reflect_dir = reflect(-light, normal);
  let specular    = pow(max(dot(view, reflect_dir), 0.0), 4.0);

  let out_rgb = (rgb * atten + (diffuse * rgb + 0.5 * specular * specular_color)) * atten;
  return vec4f(out_rgb, 1.0);
}

); // end env_shader_wgsl
// clang-format on
