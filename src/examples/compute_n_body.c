/* -------------------------------------------------------------------------- *
 * WebGPU Example - Compute Shader N-Body Simulation
 *
 * GPU-driven N-body gravity simulation using compute shaders with shared
 * workgroup memory optimization. Particles are organized into galaxies around
 * attractors and interact via gravitational forces computed on the GPU.
 *
 * Two compute passes per frame:
 *   1. Force calculation: Tile-based N-body with shared memory
 *   2. Position integration: Euler integration of velocities
 *
 * The particles are rendered as additive-blended point sprites with animated
 * color gradient, creating a visually rich galaxy simulation.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/computenbody
 * -------------------------------------------------------------------------- */

/* WebGPU uses [0,1] depth range, not OpenGL's [-1,1] */
#define CGLM_FORCE_DEPTH_ZERO_TO_ONE

#include "core/camera.h"
#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static const char* particle_vertex_shader_wgsl;
static const char* particle_fragment_shader_wgsl;
static const char* particle_calculate_shader_wgsl;
static const char* particle_integrate_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define PARTICLES_PER_ATTRACTOR (4096u)
#define NUM_ATTRACTORS (6u)
#define NUM_PARTICLES (NUM_ATTRACTORS * PARTICLES_PER_ATTRACTOR)
#define WORKGROUP_SIZE (256u)

/* File buffer for async texture loading (largest texture: 64x64 RGBA PNG) */
#define TEXTURE_FILE_BUFFER_SIZE (64 * 1024)

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* Per-particle data stored in SSBO
 * pos: xyz = position, w = mass
 * vel: xyz = velocity, w = gradient texture coordinate */
typedef struct particle_t {
  float pos[4]; /* position (xyz) + mass (w)                */
  float vel[4]; /* velocity (xyz) + gradient position (w)   */
} particle_t;

/* Graphics uniform buffer (std140 compatible) */
typedef struct graphics_ubo_t {
  mat4 projection;    /* 64 bytes */
  mat4 view;          /* 64 bytes */
  float screen_dim_x; /* 4 bytes */
  float screen_dim_y; /* 4 bytes */
  float _pad[2];      /* 8 bytes padding to 16-byte alignment */
} graphics_ubo_t;

/* Compute uniform buffer */
typedef struct compute_ubo_t {
  float delta_t;
  int32_t particle_count;
  float gravity;
  float power;
  float soften;
  float _pad[3]; /* pad to 32 bytes for alignment */
} compute_ubo_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Textures */
  struct {
    wgpu_texture_t particle; /* Point sprite texture */
    wgpu_texture_t gradient; /* Color gradient ramp  */
    uint8_t particle_buf[TEXTURE_FILE_BUFFER_SIZE];
    uint8_t gradient_buf[TEXTURE_FILE_BUFFER_SIZE];
    bool particle_loaded;
    bool gradient_loaded;
  } textures;

  /* Graphics pipeline */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
    wgpu_buffer_t uniform_buffer;
    graphics_ubo_t ubo;
  } graphics;

  /* Compute pipeline */
  struct {
    wgpu_buffer_t storage_buffer;
    wgpu_buffer_t uniform_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline_calculate; /* Pass 1: force calculation    */
    WGPUComputePipeline pipeline_integrate; /* Pass 2: position integration */
    compute_ubo_t ubo;
  } compute;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Timing */
  uint64_t last_frame_time;
  float frame_timer;

  /* GUI settings */
  struct {
    bool paused;
    float gravity;
    float power;
    float soften;
  } settings;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .settings = {
    .paused  = false,
    .gravity = 0.002f,
    .power   = 0.75f,
    .soften  = 0.05f,
  },
};

/* -------------------------------------------------------------------------- *
 * Random number generation (Normal distribution via Box-Muller)
 * -------------------------------------------------------------------------- */

static float rand_uniform(void)
{
  return (float)rand() / (float)RAND_MAX;
}

static float rand_normal(void)
{
  /* Box-Muller transform */
  float u1 = rand_uniform();
  float u2 = rand_uniform();
  while (u1 < 1e-7f) {
    u1 = rand_uniform();
  }
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * GLM_PIf * u2);
}

/* -------------------------------------------------------------------------- *
 * Particle initialization
 * -------------------------------------------------------------------------- */

static void init_particles(particle_t* particles)
{
  /* 6 galaxy attractors in 3D space */
  static const float attractors[NUM_ATTRACTORS][3] = {
    {5.0f, 0.0f, 0.0f},  {-5.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 5.0f},
    {0.0f, 0.0f, -5.0f}, {0.0f, 4.0f, 0.0f},  {0.0f, -8.0f, 0.0f},
  };

  srand((unsigned)time(NULL));

  for (uint32_t i = 0; i < NUM_ATTRACTORS; i++) {
    for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++) {
      uint32_t idx  = i * PARTICLES_PER_ATTRACTOR + j;
      particle_t* p = &particles[idx];

      if (j == 0) {
        /* Center particle: heavy attractor */
        p->pos[0] = attractors[i][0] * 1.5f;
        p->pos[1] = attractors[i][1] * 1.5f;
        p->pos[2] = attractors[i][2] * 1.5f;
        p->pos[3] = 90000.0f; /* Very high mass */
        p->vel[0] = 0.0f;
        p->vel[1] = 0.0f;
        p->vel[2] = 0.0f;
      }
      else {
        /* Scatter particles around attractor (normal distribution) */
        float px = attractors[i][0] + rand_normal() * 0.75f;
        float py = attractors[i][1] + rand_normal() * 0.75f;
        float pz = attractors[i][2] + rand_normal() * 0.75f;

        /* Flatten into disc shape */
        float dx  = px - attractors[i][0];
        float dy  = py - attractors[i][1];
        float dz  = pz - attractors[i][2];
        float len = sqrtf(dx * dx + dy * dy + dz * dz);
        if (len > 1e-6f) {
          float inv_len = 1.0f / len;
          float nx      = dx * inv_len;
          float ny      = dy * inv_len;
          float nz      = dz * inv_len;
          float norm_len
            = sqrtf(nx * nx + ny * ny + nz * nz); /* should be ~1 */
          py *= 2.0f - (norm_len * norm_len);
        }

        /* Orbital velocity via cross product */
        float ax = 0.5f, ay = 1.5f, az = 0.5f;
        float sign = ((i % 2) == 0) ? 1.0f : -1.0f;
        ax *= sign;
        ay *= sign;
        az *= sign;

        /* cross((pos - attractor), angular) + uniform jitter */
        float jitter = rand_normal() * 0.025f;
        float vx     = (dy * az - dz * ay) + jitter;
        float vy     = (dz * ax - dx * az) + jitter;
        float vz     = (dx * ay - dy * ax) + jitter;

        float mass = (rand_normal() * 0.5f + 0.5f) * 75.0f;

        p->pos[0] = px;
        p->pos[1] = py;
        p->pos[2] = pz;
        p->pos[3] = mass;
        p->vel[0] = vx;
        p->vel[1] = vy;
        p->vel[2] = vz;
      }

      /* Color gradient offset: cycles through attractors */
      p->vel[3] = (float)i / (float)NUM_ATTRACTORS;
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Storage buffer (particle SSBO)
 * -------------------------------------------------------------------------- */

static void init_storage_buffer(wgpu_context_t* wgpu_context)
{
  particle_t particles[NUM_PARTICLES];
  init_particles(particles);

  state.compute.storage_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "N-body particle - Storage buffer",
                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_CopyDst,
                    .size    = sizeof(particles),
                    .initial = {.data = particles, .size = sizeof(particles)},
                  });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Graphics UBO */
  state.graphics.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Graphics - Uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(graphics_ubo_t),
                  });

  /* Compute UBO */
  state.compute.ubo.particle_count = (int32_t)NUM_PARTICLES;
  state.compute.ubo.gravity        = state.settings.gravity;
  state.compute.ubo.power          = state.settings.power;
  state.compute.ubo.soften         = state.settings.soften;

  state.compute.uniform_buffer = wgpu_create_buffer(
    wgpu_context,
    &(wgpu_buffer_desc_t){
      .label   = "Compute uniform buffer",
      .usage   = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size    = sizeof(compute_ubo_t),
      .initial = {.data = &state.compute.ubo, .size = sizeof(compute_ubo_t)},
    });
}

static void update_graphics_uniform_buffer(wgpu_context_t* wgpu_context)
{
  glm_mat4_copy(state.camera.matrices.perspective,
                state.graphics.ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.graphics.ubo.view);
  state.graphics.ubo.screen_dim_x = (float)wgpu_context->width;
  state.graphics.ubo.screen_dim_y = (float)wgpu_context->height;

  wgpuQueueWriteBuffer(wgpu_context->queue,
                       state.graphics.uniform_buffer.buffer, 0,
                       &state.graphics.ubo, sizeof(graphics_ubo_t));
}

static void update_compute_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.compute.ubo.delta_t
    = state.settings.paused ? 0.0f : state.frame_timer * 0.05f;
  state.compute.ubo.gravity = state.settings.gravity;
  state.compute.ubo.power   = state.settings.power;
  state.compute.ubo.soften  = state.settings.soften;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.compute.uniform_buffer.buffer,
                       0, &state.compute.ubo, sizeof(compute_ubo_t));
}

/* -------------------------------------------------------------------------- *
 * Texture loading (async via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void rebuild_graphics_bind_group(wgpu_context_t* wgpu_context);

static void particle_texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("ERROR: Failed to fetch particle texture\n");
    return;
  }

  image_t img = {0};
  if (image_load_from_memory(response->data.ptr, (int)response->data.size, 4,
                             &img)) {
    state.textures.particle.desc = (wgpu_texture_desc_t){
      .extent = {.width              = (uint32_t)img.width,
                 .height             = (uint32_t)img.height,
                 .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels
      = {.ptr = img.pixels.u8, .size = (size_t)(img.width * img.height * 4)},
    };
    state.textures.particle.desc.is_dirty = true;
    state.textures.particle_loaded        = true;
  }
}

static void gradient_texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("ERROR: Failed to fetch gradient texture\n");
    return;
  }

  image_t img = {0};
  if (image_load_from_memory(response->data.ptr, (int)response->data.size, 4,
                             &img)) {
    state.textures.gradient.desc = (wgpu_texture_desc_t){
      .extent       = {.width              = (uint32_t)img.width,
                       .height             = (uint32_t)img.height,
                       .depthOrArrayLayers = 1},
      .format       = WGPUTextureFormat_RGBA8Unorm,
      .address_mode = WGPUAddressMode_ClampToEdge,
      .pixels
      = {.ptr = img.pixels.u8, .size = (size_t)(img.width * img.height * 4)},
    };
    state.textures.gradient.desc.is_dirty = true;
    state.textures.gradient_loaded        = true;
  }
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures */
  state.textures.particle = wgpu_create_color_bars_texture(wgpu_context, NULL);
  state.textures.gradient = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Request async loads */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/particle01_rgba.png",
    .callback = particle_texture_fetch_cb,
    .buffer   = SFETCH_RANGE(state.textures.particle_buf),
  });

  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/particle_gradient_rgba.png",
    .callback = gradient_texture_fetch_cb,
    .buffer   = SFETCH_RANGE(state.textures.gradient_buf),
  });
}

/* -------------------------------------------------------------------------- *
 * Graphics pipeline
 * -------------------------------------------------------------------------- */

static void init_graphics_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[4] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = (WGPUBindGroupLayoutEntry){
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    [3] = (WGPUBindGroupLayoutEntry){
      .binding    = 3,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(graphics_ubo_t),
      },
    },
  };

  state.graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Graphics - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.graphics.bind_group_layout != NULL);
}

static void rebuild_graphics_bind_group(wgpu_context_t* wgpu_context)
{
  /* Release previous bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)

  WGPUBindGroupEntry bg_entries[4] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.textures.particle.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.textures.particle.sampler,
    },
    [2] = (WGPUBindGroupEntry){
      .binding     = 2,
      .textureView = state.textures.gradient.view,
    },
    [3] = (WGPUBindGroupEntry){
      .binding = 3,
      .buffer  = state.graphics.uniform_buffer.buffer,
      .size    = state.graphics.uniform_buffer.size,
    },
  };

  state.graphics.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Graphics - Bind group"),
                            .layout     = state.graphics.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.graphics.bind_group != NULL);
}

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule vert_module = wgpu_create_shader_module(
    wgpu_context->device, particle_vertex_shader_wgsl);
  WGPUShaderModule frag_module = wgpu_create_shader_module(
    wgpu_context->device, particle_fragment_shader_wgsl);

  /* Additive blending */
  WGPUBlendState additive_blend = {
    .color = {
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_One,
      .dstFactor = WGPUBlendFactor_One,
    },
    .alpha = {
      .operation = WGPUBlendOperation_Add,
      .srcFactor = WGPUBlendFactor_SrcAlpha,
      .dstFactor = WGPUBlendFactor_DstAlpha,
    },
  };

  /* Vertex buffer layout: particle_t stride, 2 attributes (pos, vel) */
  /* Vertex buffer layout: per-instance particle data */
  WGPU_VERTEX_BUFFER_LAYOUT(particle_vb_layout, sizeof(particle_t),
                            WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4,
                                               offsetof(particle_t, pos)),
                            WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                                               offsetof(particle_t, vel)))

  /* Override to per-instance stepping */
  particle_vb_layout_vertex_buffer_layout.stepMode
    = WGPUVertexStepMode_Instance;

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Particle render pipeline"),
    .layout = state.graphics.pipeline_layout,
    .vertex = {
      .module      = vert_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &particle_vb_layout_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = frag_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &additive_blend,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.graphics.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_module);
  wgpuShaderModuleRelease(frag_module);
}

/* -------------------------------------------------------------------------- *
 * Compute pipelines
 * -------------------------------------------------------------------------- */

static void init_compute_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = NUM_PARTICLES * sizeof(particle_t),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(compute_ubo_t),
      },
    },
  };

  state.compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Compute bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.compute.bind_group_layout != NULL);
}

static void init_compute_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.compute.storage_buffer.buffer,
      .size    = state.compute.storage_buffer.size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.compute.uniform_buffer.buffer,
      .size    = state.compute.uniform_buffer.size,
    },
  };

  state.compute.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Compute bind group"),
                            .layout     = state.compute.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.compute.bind_group != NULL);
}

static void init_compute_pipelines(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Compute pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    });
  ASSERT(state.compute.pipeline_layout != NULL);

  /* Pass 1: Force calculation shader */
  WGPUShaderModule calc_module = wgpu_create_shader_module(
    wgpu_context->device, particle_calculate_shader_wgsl);

  state.compute.pipeline_calculate = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("N-body calculate pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = calc_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline_calculate != NULL);
  wgpuShaderModuleRelease(calc_module);

  /* Pass 2: Integration shader */
  WGPUShaderModule integ_module = wgpu_create_shader_module(
    wgpu_context->device, particle_integrate_shader_wgsl);

  state.compute.pipeline_integrate = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("N-body integrate pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = integ_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline_integrate != NULL);
  wgpuShaderModuleRelease(integ_module);
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

  igBegin("N-Body Simulation", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  igText("Particles: %u", NUM_PARTICLES);
  igText("FPS: %.1f", 1.0f / fmaxf(state.frame_timer, 0.001f));
  igSeparator();

  igCheckbox("Paused", &state.settings.paused);
  igSliderFloat("Gravity", &state.settings.gravity, 0.0001f, 0.01f, "%.4f", 0);
  igSliderFloat("Power", &state.settings.power, 0.1f, 2.0f, "%.2f", 0);
  igSliderFloat("Soften", &state.settings.soften, 0.001f, 0.5f, "%.3f", 0);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Camera setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.movement_speed = 2.5f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;

  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 512.0f);
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-26.0f, 75.0f, 0.0f));
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -14.0f});
}

/* -------------------------------------------------------------------------- *
 * Main callbacks
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 2,
    .num_lanes    = 2,
    .logger.func  = slog_func,
  });

  init_camera(wgpu_context);
  init_storage_buffer(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_textures(wgpu_context);

  /* Graphics pipeline */
  init_graphics_bind_group_layout(wgpu_context);
  rebuild_graphics_bind_group(wgpu_context);
  init_graphics_pipeline(wgpu_context);

  /* Compute pipelines */
  init_compute_bind_group_layout(wgpu_context);
  init_compute_bind_group(wgpu_context);
  init_compute_pipelines(wgpu_context);

  /* GUI */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async loads */
  sfetch_dowork();

  /* Recreate textures if loaded from disk */
  bool rebind = false;
  if (state.textures.particle.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.particle);
    FREE_TEXTURE_PIXELS(state.textures.particle);
    rebind = true;
  }
  if (state.textures.gradient.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.textures.gradient);
    FREE_TEXTURE_PIXELS(state.textures.gradient);
    rebind = true;
  }
  if (rebind) {
    rebuild_graphics_bind_group(wgpu_context);
  }

  /* Timing */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  state.frame_timer     = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Update camera */
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 512.0f);
  camera_update(&state.camera, state.frame_timer);
  camera_update_view_matrix(&state.camera);

  /* Update uniform buffers */
  update_graphics_uniform_buffer(wgpu_context);
  update_compute_uniform_buffer(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, state.frame_timer);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Update render pass views */
  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass: update particle positions */
  {
    WGPUComputePassEncoder cpass
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);

    uint32_t workgroup_count = NUM_PARTICLES / WORKGROUP_SIZE;

    /* Pass 1: Force calculation */
    wgpuComputePassEncoderSetPipeline(cpass, state.compute.pipeline_calculate);
    wgpuComputePassEncoderSetBindGroup(cpass, 0, state.compute.bind_group, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroup_count, 1, 1);

    /* Pass 2: Position integration */
    wgpuComputePassEncoderSetPipeline(cpass, state.compute.pipeline_integrate);
    wgpuComputePassEncoderSetBindGroup(cpass, 0, state.compute.bind_group, 0,
                                       NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroup_count, 1, 1);

    wgpuComputePassEncoderEnd(cpass);
    wgpuComputePassEncoderRelease(cpass);
  }

  /* Render pass: draw particles */
  {
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    wgpuRenderPassEncoderSetPipeline(rpass, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.graphics.bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass, 0, state.compute.storage_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    // 6 vertices per quad (2 triangles), NUM_PARTICLES instances
    wgpuRenderPassEncoderDraw(rpass, 6, NUM_PARTICLES, 0, 0);

    wgpuRenderPassEncoderEnd(rpass);
    wgpuRenderPassEncoderRelease(rpass);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render GUI overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Textures */
  wgpu_destroy_texture(&state.textures.particle);
  wgpu_destroy_texture(&state.textures.gradient);

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)
  WGPU_RELEASE_RESOURCE(Buffer, state.graphics.uniform_buffer.buffer)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.storage_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline_calculate)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline_integrate)
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  ImGuiIO* io = igGetIO_Nil();
  if (io->WantCaptureMouse) {
    return;
  }

  camera_on_input_event(&state.camera, input_event);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader N-Body Simulation",
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
static const char* particle_vertex_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4<f32>,
    view       : mat4x4<f32>,
    screenDim  : vec2<f32>,
  };

  @group(0) @binding(3) var<uniform> ubo : UBO;

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) gradientPos    : f32,
    @location(1) uv             : vec2<f32>,
  };

  // Billboard quad: 2 triangles, 6 vertices
  // vertex_index 0..5 maps to quad corners
  @vertex
  fn vs_main(
    @builtin(vertex_index) vertexIndex : u32,
    @location(0) inPos : vec4<f32>,
    @location(1) inVel : vec4<f32>,
  ) -> VertexOutput {
    var output : VertexOutput;

    // Quad corner offsets (2 triangles forming a quad)
    var corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0,  1.0),
    );

    // UV coordinates for the quad corners
    var uvs = array<vec2<f32>, 6>(
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 0.0),
    );

    let corner = corners[vertexIndex];

    // Transform particle center to eye space
    let eyePos = ubo.view * vec4<f32>(inPos.xyz, 1.0);

    // Compute screen-space point size (matching Vulkan: clamped to [1, 128] px)
    let spriteSize = 0.005 * inPos.w;
    let projCorner = ubo.projection * vec4<f32>(
      0.5 * spriteSize, 0.5 * spriteSize, eyePos.z, eyePos.w
    );
    let pointSize = clamp(
      ubo.screenDim.x * projCorner.x / projCorner.w, 1.0, 128.0
    );

    // Project particle center to clip space
    let clipPos = ubo.projection * eyePos;

    // Convert pixel-space billboard offset to clip space
    // corner is +/-1, half pointSize gives radius in pixels
    // pixels to NDC: multiply by 2/screenDim, then to clip: multiply by w
    let pixelOffset = corner * pointSize * 0.5;
    let ndcOffset = pixelOffset * vec2<f32>(
      2.0 / ubo.screenDim.x, 2.0 / ubo.screenDim.y
    );
    output.position = clipPos + vec4<f32>(ndcOffset * clipPos.w, 0.0, 0.0);

    output.gradientPos = inVel.w;
    output.uv = uvs[vertexIndex];
    return output;
  }
);

static const char* particle_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var colorMap : texture_2d<f32>;
  @group(0) @binding(1) var colorSampler : sampler;
  @group(0) @binding(2) var gradientRamp : texture_2d<f32>;

  @fragment
  fn fs_main(
    @location(0) gradientPos : f32,
    @location(1) uv : vec2<f32>,
  ) -> @location(0) vec4<f32> {
    let spriteColor = textureSample(colorMap, colorSampler, uv);
    let gradientColor = textureSample(
      gradientRamp, colorSampler, vec2<f32>(gradientPos, 0.0)
    );
    return vec4<f32>(spriteColor.rgb * gradientColor.rgb, spriteColor.a);
  }
);

static const char* particle_calculate_shader_wgsl = CODE(
  struct Particle {
    pos : vec4<f32>,
    vel : vec4<f32>,
  };

  struct Params {
    deltaT         : f32,
    particleCount  : i32,
    gravity        : f32,
    power          : f32,
    soften         : f32,
  };

  @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
  @group(0) @binding(1) var<uniform> params : Params;

  var<workgroup> sharedData : array<vec4<f32>, 256>;

  @compute @workgroup_size(256)
  fn main(
    @builtin(global_invocation_id) globalId : vec3<u32>,
    @builtin(local_invocation_id) localId : vec3<u32>,
  ) {
    let index = globalId.x;
    let pCount = u32(params.particleCount);
    let isValid = index < pCount;

    var position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (isValid) {
      position = particles[index].pos;
    }

    var acceleration = vec3<f32>(0.0, 0.0, 0.0);

    for (var tile : u32 = 0u; tile < pCount; tile += 256u) {
      let tileIdx = tile + localId.x;
      if (tileIdx < pCount) {
        sharedData[localId.x] = particles[tileIdx].pos;
      } else {
        sharedData[localId.x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      }

      workgroupBarrier();

      if (isValid) {
        for (var j : u32 = 0u; j < 256u; j = j + 1u) {
          let other = sharedData[j];
          let diff = other.xyz - position.xyz;
          let distSq = dot(diff, diff) + params.soften;
          acceleration += params.gravity * diff * other.w
            / pow(distSq, params.power);
        }
      }

      workgroupBarrier();
    }

    if (isValid) {
      particles[index].vel = vec4<f32>(
        particles[index].vel.xyz + params.deltaT * acceleration,
        particles[index].vel.w
      );

      // Animate gradient texture coordinate
      var gradPos = particles[index].vel.w + 0.1 * params.deltaT;
      if (gradPos > 1.0) {
        gradPos = gradPos - 1.0;
      }
      particles[index].vel.w = gradPos;
    }
  }
);

static const char* particle_integrate_shader_wgsl = CODE(
  struct Particle {
    pos : vec4<f32>,
    vel : vec4<f32>,
  };

  struct Params {
    deltaT         : f32,
    particleCount  : i32,
  };

  @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
  @group(0) @binding(1) var<uniform> params : Params;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
    let index = globalId.x;
    if (index >= u32(params.particleCount)) {
      return;
    }

    var position = particles[index].pos;
    let velocity = particles[index].vel;
    position = position + params.deltaT * velocity;
    particles[index].pos = position;
  }
);
// clang-format on
