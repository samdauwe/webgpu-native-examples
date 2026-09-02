/* WebGPU uses [0,1] depth range, not OpenGL's [-1,1] */
#define CGLM_FORCE_DEPTH_ZERO_TO_ONE

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __WAJIC__
#define WAJIC_TIME_IMPL
#include <wajic_time.h>
#else
#define SOKOL_LOG_IMPL
#include <sokol_log.h>
#define SOKOL_TIME_IMPL
#include <sokol_time.h>
#endif

/* WAjic WebGPU handles are uint32_t — redefine NULL to 0 for handle assignments
 */
#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

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
 * WebGPU Example - Momentous
 *
 * GPU-driven particle system reimplementation of the particle effect from
 * "fr-059: momentum" by Farbrausch/ryg. Uses verlet integration on the GPU
 * via render-to-texture passes. 48 K small cubes are driven by a procedural
 * divergence-free 3-D force field and rendered as instanced elongated cubes
 * with faceted normals and a tri-light shading model.
 *
 * Ref:
 * https://github.com/rygorous/momentous
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* update_pos_shader_wgsl;
static const char* update_vel_shader_wgsl;
static const char* cube_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define CHUNK_SIZE 1024u        /* particles per chunk (texture width)  */
#define NUM_CUBES (48u * 1024u) /* total particles                       */
#define TEX_HEIGHT ((NUM_CUBES + CHUNK_SIZE - 1u) / CHUNK_SIZE) /* = 48  */
#define SPAWN_COUNT 256u /* particles spawned per frame           */
#define FORCE_SIZE 32u   /* 3-D force field side length            */
#define PART_SIZE 0.001f /* cube half-length along velocity axis  */

/* -------------------------------------------------------------------------- *
 * Uniform buffer structures (must match WGSL struct alignment rules)
 * -------------------------------------------------------------------------- */

/* mat4x4f(64) + vec3f+f32(16) + 5*(vec3f+f32)(80) = 160 bytes */
typedef struct {
  float clip_from_world[16];
  float world_down[3];
  float time_offs;
  float ambient[3];
  float _p0;
  float key_col[3];
  float _p1;
  float fill_col[3];
  float _p2;
  float back_col[3];
  float _p3;
  float light_dir[3];
  float _p4;
} cube_consts_t;

/* 3*(vec3f+f32) = 48 bytes */
typedef struct {
  float field_scale[3];
  float damping;
  float field_offs[3];
  float accel;
  float field_sample_scale[3];
  float vel_scale;
} update_consts_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Particle textures: [0..2] = position triple-buffer, [3] = velocity */
  WGPUTexture part_tex[4];
  WGPUTextureView part_view[4];
  /* 3-D force field texture (RGBA32F, FORCE_SIZE^3) */
  WGPUTexture force_tex;
  WGPUTextureView force_view;
  WGPUSampler force_sampler; /* unused placeholder kept for struct layout */
  /* Depth texture for cube rendering */
  WGPUTexture depth_tex;
  WGPUTextureView depth_view;
  /* Uniform buffers */
  WGPUBuffer cube_consts_buf;
  WGPUBuffer update_consts_buf;
  /* Index buffer: CHUNK_SIZE cubes × 15 uint16 per cube */
  WGPUBuffer cube_index_buf;
  /* Pipelines */
  WGPURenderPipeline update_pos_pipeline;
  WGPURenderPipeline update_vel_pipeline;
  WGPURenderPipeline cube_pipeline;
  /* Bind group layouts */
  WGPUBindGroupLayout update_pos_bgl;
  WGPUBindGroupLayout update_vel_bgl;
  WGPUBindGroupLayout cube_bgl;
  /* Pre-built bind groups indexed by cur_part (0..2) */
  WGPUBindGroup update_pos_bg[3];
  WGPUBindGroup update_vel_bg[3];
  WGPUBindGroup cube_bg[3];
  /* Simulation state */
  uint32_t cur_part;      /* index of the "current" position buffer (0-2) */
  uint32_t spawn_counter; /* cycles 0 .. NUM_CUBES-1                       */
  int frame;
  /* GUI settings */
  struct {
    float damping;
    float accel;
    bool animate;
  } settings;
  /* Timing */
  float time;
  uint64_t last_time;
  bool initialized;
  wgpu_context_t* wgpu_ctx;
} state;

/* -------------------------------------------------------------------------- *
 * Helpers
 * -------------------------------------------------------------------------- */

static float srgb2lin(float x)
{
  return x < 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

static void srgb_color(int hex, float out[3])
{
  out[0] = srgb2lin(((hex >> 16) & 0xff) / 255.0f);
  out[1] = srgb2lin(((hex >> 8) & 0xff) / 255.0f);
  out[2] = srgb2lin(((hex >> 0) & 0xff) / 255.0f);
}

static float randf(void)
{
  return (float)rand() / (float)RAND_MAX;
}

static void rand_sphere(float out[3])
{
  float l;
  do {
    out[0] = 2.f * randf() - 1.f;
    out[1] = 2.f * randf() - 1.f;
    out[2] = 2.f * randf() - 1.f;
    l      = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
  } while (l > 1.f);
}

/* Sample a uniformly-distributed unit vector (on the sphere surface). */
static void rand_unit_vec3(float out[3])
{
  float l;
  do {
    rand_sphere(out);
    l = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
  } while (l == 0.f);
  float inv = 1.f / sqrtf(l);
  out[0] *= inv;
  out[1] *= inv;
  out[2] *= inv;
}

/* -------------------------------------------------------------------------- *
 * Force field generation
 *
 * Produces a divergence-free random 3-D vector field via Gauss-Seidel
 * gradient removal (Helmholtz decomposition), as in the original D3D version.
 * -------------------------------------------------------------------------- */

static void create_force_texture(wgpu_context_t* wgpu_context)
{
  const int S  = (int)FORCE_SIZE;
  const int N  = S * S * S;
  const int sx = 1, mx = S - 1;
  const int sy = S, my = (S - 1) * S;
  const int sz = S * S, mz = (S - 1) * S * S;

  /* Build a vec4 force field (w=0) */
  float (*forces)[4] = (float (*)[4])malloc(N * sizeof(float[4]));
  float* div         = (float*)malloc(N * sizeof(float));
  float* high        = (float*)calloc(N, sizeof(float));

  /* Random velocity field — unit vectors, matching rand_unit_vec3() in
   * reference */
  const float strength = 1.0f;
  for (int i = 0; i < N; ++i) {
    rand_unit_vec3(forces[i]);
    forces[i][0] *= strength;
    forces[i][1] *= strength;
    forces[i][2] *= strength;
    forces[i][3] = 0.f;
  }

  /* Compute divergences */
  float ds = -0.5f / S;
#define IDX(o, step, mask) (((o) & ~(mask)) | (((o) + (step)) & (mask)))
  for (int zo = 0; zo < N; zo += sz) {
    for (int yo = 0; yo < N - sz; yo += sy) {
      if (zo + yo >= N)
        break;
      for (int xo = 0; xo < sy; xo += sx) {
        int o = xo + yo + zo;
        if (o >= N)
          break;
        div[o] = ds
                 * (forces[IDX(o, sx, mx)][0] - forces[IDX(o, -sx, mx)][0]
                    + forces[IDX(o, sy, my)][1] - forces[IDX(o, -sy, my)][1]
                    + forces[IDX(o, sz, mz)][2] - forces[IDX(o, -sz, mz)][2]);
      }
    }
  }

  /* Gauss-Seidel to compute pressure field */
  for (int iter = 0; iter < 40; ++iter) {
    for (int zo = 0; zo < N; zo += sz) {
      for (int yo = 0; yo < N - sz; yo += sy) {
        for (int xo = 0; xo < sy; xo += sx) {
          int o = xo + yo + zo;
          if (o >= N)
            break;
          high[o] = (high[IDX(o, -sx, mx)] + high[IDX(o, sx, mx)]
                     + high[IDX(o, -sy, my)] + high[IDX(o, sy, my)]
                     + high[IDX(o, -sz, mz)] + high[IDX(o, sz, mz)])
                      * (1.f / 6.f)
                    - div[o];
        }
      }
    }
  }

  /* Remove gradient from vector field */
  float gs               = 0.5f * S;
  const float post_scale = 0.001f;
  for (int zo = 0; zo < N; zo += sz) {
    for (int yo = 0; yo < N - sz; yo += sy) {
      for (int xo = 0; xo < sy; xo += sx) {
        int o = xo + yo + zo;
        if (o >= N)
          break;
        forces[o][0]
          = (forces[o][0] - gs * (high[IDX(o, sx, mx)] - high[IDX(o, -sx, mx)]))
            * post_scale;
        forces[o][1]
          = (forces[o][1] - gs * (high[IDX(o, sy, my)] - high[IDX(o, -sy, my)]))
            * post_scale;
        forces[o][2]
          = (forces[o][2] - gs * (high[IDX(o, sz, mz)] - high[IDX(o, -sz, mz)]))
            * post_scale;
      }
    }
  }
#undef IDX

  free(div);
  free(high);

  /* Create 3-D texture */
  state.force_tex = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label         = STRVIEW("Force - 3D texture"),
      .dimension     = WGPUTextureDimension_3D,
      .format        = WGPUTextureFormat_RGBA32Float,
      .size          = {FORCE_SIZE, FORCE_SIZE, FORCE_SIZE},
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    });

  WGPUExtent3D extent = {FORCE_SIZE, FORCE_SIZE, FORCE_SIZE};
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.force_tex,
                          .mipLevel = 0,
                          .origin   = {0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        forces, N * sizeof(float[4]),
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = FORCE_SIZE * sizeof(float[4]),
                          .rowsPerImage = FORCE_SIZE,
                        },
                        &extent);
  free(forces);

  state.force_view = wgpuTextureCreateView(
    state.force_tex, &(WGPUTextureViewDescriptor){
                       .label           = STRVIEW("Force - 3D texture view"),
                       .format          = WGPUTextureFormat_RGBA32Float,
                       .dimension       = WGPUTextureViewDimension_3D,
                       .mipLevelCount   = 1,
                       .arrayLayerCount = 1,
                     });

  /* No sampler needed: force field is sampled via textureLoad (manual
   * trilinear) */
  state.force_sampler = NULL;
}

/* -------------------------------------------------------------------------- *
 * Particle textures
 * -------------------------------------------------------------------------- */

static void create_particle_textures(wgpu_context_t* wgpu_context)
{
  for (int i = 0; i < 4; ++i) {
    state.part_tex[i] = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label         = STRVIEW("Particle - RGBA32F texture"),
        .dimension     = WGPUTextureDimension_2D,
        .format        = WGPUTextureFormat_RGBA32Float,
        .size          = {CHUNK_SIZE, TEX_HEIGHT, 1},
        .mipLevelCount = 1,
        .sampleCount   = 1,
        .usage         = WGPUTextureUsage_RenderAttachment
                 | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      });

    state.part_view[i] = wgpuTextureCreateView(
      state.part_tex[i], &(WGPUTextureViewDescriptor){
                           .label  = STRVIEW("Particle - RGBA32F texture view"),
                           .format = WGPUTextureFormat_RGBA32Float,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });
  }

  /* Clear all particle textures to zero (w=0 → all dead initially) */
  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  for (int i = 0; i < 4; ++i) {
    WGPURenderPassColorAttachment ca = {
      .view       = state.part_view[i],
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
}

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void create_depth_texture(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_tex)

  state.depth_tex = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth - Texture"),
                            .dimension     = WGPUTextureDimension_2D,
                            .format        = WGPUTextureFormat_Depth24Plus,
                            .size          = {(uint32_t)wgpu_context->width,
                                              (uint32_t)wgpu_context->height, 1},
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                            .usage         = WGPUTextureUsage_RenderAttachment,
                          });

  state.depth_view = wgpuTextureCreateView(
    state.depth_tex, &(WGPUTextureViewDescriptor){
                       .label           = STRVIEW("Depth - Texture view"),
                       .format          = WGPUTextureFormat_Depth24Plus,
                       .dimension       = WGPUTextureViewDimension_2D,
                       .mipLevelCount   = 1,
                       .arrayLayerCount = 1,
                     });
}

/* -------------------------------------------------------------------------- *
 * Index buffer (CHUNK_SIZE cube triangle strips + primitive restart)
 * -------------------------------------------------------------------------- */

static void create_index_buffer(wgpu_context_t* wgpu_context)
{
  static const uint16_t cube_inds[] = {
    0, 2, 1, 3, 7, 2, 6, 0, 4, 1, 5, 7, 4, 6, /* 14 verts, then restart */
  };

  uint16_t* ind_data = (uint16_t*)malloc(CHUNK_SIZE * 15 * sizeof(uint16_t));
  for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
    uint16_t* out = ind_data + i * 15;
    for (int j = 0; j < 14; ++j)
      out[j] = cube_inds[j] + (uint16_t)(i * 8);
    out[14] = 0xFFFFu; /* primitive restart */
  }

  state.cube_index_buf = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Cube - Index buffer"),
      .size             = CHUNK_SIZE * 15 * sizeof(uint16_t),
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, state.cube_index_buf, 0, ind_data,
                       CHUNK_SIZE * 15 * sizeof(uint16_t));
  free(ind_data);
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void create_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* UpdatePos: uniform, 2×texture_2d, sampler, texture_3d */
  {
    WGPUBindGroupLayoutEntry e[5] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Fragment,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(update_consts_t)}},
      /* particle textures — accessed via textureLoad, no filtering needed */
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
      /* force 3D texture — sampled via manual textureLoad, no
         float32-filterable needed */
      [3] = {.binding    = 3,
             .visibility = WGPUShaderStage_Fragment,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_3D}},
    };
    state.update_pos_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("UpdatePos - Bind group layout"),
                              .entryCount = 4,
                              .entries    = e,
                            });
  }

  /* UpdateVel: 2×texture_2d */
  {
    /* particle textures — accessed via textureLoad, no filtering needed */
    WGPUBindGroupLayoutEntry e[2] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Fragment,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
    };
    state.update_vel_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("UpdateVel - Bind group layout"),
                              .entryCount = 2,
                              .entries    = e,
                            });
  }

  /* Cube: uniform, 2×texture_2d (vertex-visible for vertex pulling) */
  {
    WGPUBindGroupLayoutEntry e[3] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(cube_consts_t)}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Vertex,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Vertex,
             .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                         .viewDimension = WGPUTextureViewDimension_2D}},
    };
    state.cube_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label      = STRVIEW("Cube - Bind group layout"),
                              .entryCount = 3,
                              .entries    = e,
                            });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 * -------------------------------------------------------------------------- */

static void create_pipelines(wgpu_context_t* wgpu_context)
{
  /* --- UpdatePos pipeline --- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(wgpu_context->device, update_pos_shader_wgsl);
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("UpdatePos - Pipeline layout"),
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &state.update_pos_bgl,
                            });
    state.update_pos_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("UpdatePos - Render pipeline"),
        .layout = pl,
        .vertex = {.module = sm, .entryPoint = STRVIEW("vs")},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = WGPUTextureFormat_RGBA32Float,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive   = {.topology = WGPUPrimitiveTopology_TriangleList},
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
      });
    wgpuShaderModuleRelease(sm);
    wgpuPipelineLayoutRelease(pl);
  }

  /* --- UpdateVel pipeline --- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(wgpu_context->device, update_vel_shader_wgsl);
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("UpdateVel - Pipeline layout"),
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &state.update_vel_bgl,
                            });
    state.update_vel_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("UpdateVel - Render pipeline"),
        .layout = pl,
        .vertex = {.module = sm, .entryPoint = STRVIEW("vs")},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = WGPUTextureFormat_RGBA32Float,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive   = {.topology = WGPUPrimitiveTopology_TriangleList},
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
      });
    wgpuShaderModuleRelease(sm);
    wgpuPipelineLayoutRelease(pl);
  }

  /* --- Cube render pipeline --- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(wgpu_context->device, cube_shader_wgsl);
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("Cube - Pipeline layout"),
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &state.cube_bgl,
                            });
    state.cube_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Cube - Render pipeline"),
        .layout = pl,
        .vertex = {.module = sm, .entryPoint = STRVIEW("vs")},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology        = WGPUPrimitiveTopology_TriangleStrip,
          .stripIndexFormat = WGPUIndexFormat_Uint16,
          .cullMode         = WGPUCullMode_None,
        },
        .depthStencil = &(WGPUDepthStencilState){
          .format            = WGPUTextureFormat_Depth24Plus,
          .depthWriteEnabled = true,
          .depthCompare      = WGPUCompareFunction_Less,
          .stencilFront      = {.compare = WGPUCompareFunction_Always},
          .stencilBack       = {.compare = WGPUCompareFunction_Always},
        },
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
      });
    wgpuShaderModuleRelease(sm);
    wgpuPipelineLayoutRelease(pl);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * Pre-built for each possible cur_part value (0, 1, 2).
 * UpdatePos at cur_part=k reads tex[(k+1)%3] (older) and tex[(k+2)%3] (newer).
 * UpdateVel at cur_part=k reads tex[(k+2)%3] (older) and tex[k] (newer).
 * -------------------------------------------------------------------------- */

static void create_bind_groups(wgpu_context_t* wgpu_context)
{
  for (int k = 0; k < 3; ++k) {
    /* UpdatePos */
    {
      WGPUBindGroupEntry e[4] = {
        [0] = {.binding = 0,
               .buffer  = state.update_consts_buf,
               .size    = sizeof(update_consts_t)},
        [1] = {.binding = 1, .textureView = state.part_view[(k + 1) % 3]},
        [2] = {.binding = 2, .textureView = state.part_view[(k + 2) % 3]},
        [3] = {.binding = 3, .textureView = state.force_view},
      };
      state.update_pos_bg[k] = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .label      = STRVIEW("UpdatePos - Bind group"),
                                .layout     = state.update_pos_bgl,
                                .entryCount = 4,
                                .entries    = e,
                              });
    }

    /* UpdateVel */
    {
      WGPUBindGroupEntry e[2] = {
        [0] = {.binding = 0, .textureView = state.part_view[(k + 2) % 3]},
        [1] = {.binding = 1, .textureView = state.part_view[k]},
      };
      state.update_vel_bg[k] = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .label      = STRVIEW("UpdateVel - Bind group"),
                                .layout     = state.update_vel_bgl,
                                .entryCount = 2,
                                .entries    = e,
                              });
    }

    /* Cube render */
    {
      WGPUBindGroupEntry e[3] = {
        [0] = {.binding = 0,
               .buffer  = state.cube_consts_buf,
               .size    = sizeof(cube_consts_t)},
        [1] = {.binding = 1, .textureView = state.part_view[k]},
        [2] = {.binding = 2, .textureView = state.part_view[3]},
      };
      state.cube_bg[k] = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .label      = STRVIEW("Cube - Bind group"),
                                .layout     = state.cube_bgl,
                                .entryCount = 3,
                                .entries    = e,
                              });
    }
  }
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.f, 10.f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.f, 0.f});
  igSetNextWindowSize((ImVec2){220.f, 0.f}, ImGuiCond_FirstUseEver);

  igBegin("Momentous", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  igCheckbox("Animate", &state.settings.animate);
  imgui_overlay_slider_float("Damping", &state.settings.damping, 0.9f, 1.0f,
                             "%.4f");
  imgui_overlay_slider_float("Accel", &state.settings.accel, 0.0f, 2.0f,
                             "%.2f");

  igSeparator();
  igText("Particles: %u", NUM_CUBES);
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Spawn helper
 * -------------------------------------------------------------------------- */

static void spawn_particles(wgpu_context_t* wgpu_context, float emit_x)
{
  /* The spawn writes to the two "older" buffers (before cur_part is advanced).
   * [cur_part]         → pos_new  (will be read as "newer" by UpdatePos)
   * [(cur_part+2)%3]   → pos_old  (will be read as "older" by UpdatePos)    */
  float pos_old[SPAWN_COUNT * 4];
  float pos_new[SPAWN_COUNT * 4];

  for (uint32_t i = 0; i < SPAWN_COUNT; ++i) {
    float rv[3];
    rand_sphere(rv);
    float px = emit_x + rv[0] * 0.002f;
    float py = rv[1] * 0.002f;
    float pz = rv[2] * 0.002f;

    float vv[3];
    rand_sphere(vv);
    float vx = vv[0] * 0.003f;
    float vy = vv[1] * 0.003f;
    float vz = vv[2] * 0.003f;

    pos_old[i * 4 + 0] = px - vx;
    pos_old[i * 4 + 1] = py - vy;
    pos_old[i * 4 + 2] = pz - vz;
    pos_old[i * 4 + 3] = PART_SIZE;

    pos_new[i * 4 + 0] = px;
    pos_new[i * 4 + 1] = py;
    pos_new[i * 4 + 2] = pz;
    pos_new[i * 4 + 3] = PART_SIZE;
  }

  uint32_t col = state.spawn_counter % CHUNK_SIZE;
  uint32_t row = state.spawn_counter / CHUNK_SIZE;

  WGPUExtent3D sz                  = {SPAWN_COUNT, 1, 1};
  WGPUTexelCopyBufferLayout layout = {
    .offset       = 0,
    .bytesPerRow  = SPAWN_COUNT * sizeof(float[4]),
    .rowsPerImage = 1,
  };

  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.part_tex[(state.cur_part + 2) % 3],
                          .mipLevel = 0,
                          .origin   = {col, row, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        pos_old, sizeof(pos_old), &layout, &sz);

  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.part_tex[state.cur_part],
                          .mipLevel = 0,
                          .origin   = {col, row, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        pos_new, sizeof(pos_new), &layout, &sz);

  state.spawn_counter = (state.spawn_counter + SPAWN_COUNT) % NUM_CUBES;
}

/* -------------------------------------------------------------------------- *
 * Initialization
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context)
    return EXIT_FAILURE;

  state.wgpu_ctx = wgpu_context;
  stm_setup();
  srand((unsigned int)(stm_now() & 0xFFFFFFFFu));

  /* Default simulation settings */
  state.settings.damping = 0.99f;
  state.settings.accel   = 0.75f;
  state.settings.animate = true;

  /* Uniform buffers */
  state.update_consts_buf = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("UpdateConsts - Uniform buffer"),
      .size  = sizeof(update_consts_t),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    });

  state.cube_consts_buf = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("CubeConsts - Uniform buffer"),
      .size  = sizeof(cube_consts_t),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    });

  create_force_texture(wgpu_context);
  create_particle_textures(wgpu_context);
  create_depth_texture(wgpu_context);
  create_index_buffer(wgpu_context);
  create_bind_group_layouts(wgpu_context);
  create_pipelines(wgpu_context);
  create_bind_groups(wgpu_context);
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized)
    return EXIT_FAILURE;

  /* Frame timing */
  uint64_t now = stm_now();
  if (state.last_time == 0)
    state.last_time = now;
  float dt        = (float)stm_sec(stm_diff(now, state.last_time));
  state.last_time = now;

  /* Emitter x-position drives both spawn and camera */
  float emit_x = 0.7f * sinf((float)state.frame * 0.001f);

  if (state.settings.animate) {
    state.time = (float)stm_sec(now);

    /* 1. Spawn new particles (before advancing cur_part) */
    spawn_particles(wgpu_context, emit_x);

    /* 2. Advance cur_part */
    state.cur_part = (state.cur_part + 1) % 3;
  }
  uint32_t k = state.cur_part;

  /* 3. Upload update consts */
  {
    update_consts_t uc = {
      .field_scale        = {32.f, 32.f, 32.f},
      .damping            = state.settings.damping,
      .field_offs         = {0.f, 0.f, 0.f},
      .accel              = state.settings.accel,
      .field_sample_scale = {1.f / 32.f, 1.f / 32.f, 1.f / 32.f},
      .vel_scale          = PART_SIZE * 6.f,
    };
    wgpuQueueWriteBuffer(wgpu_context->queue, state.update_consts_buf, 0, &uc,
                         sizeof(uc));
  }

  /* 4. Upload cube consts */
  {
    /* Camera: fixed position looking at emitter */
    vec3 eye    = {0.f, 0.f, -0.9f};
    vec3 center = {emit_x, 0.f, 0.f};
    vec3 up     = {0.f, 1.f, 0.f};

    mat4 view, proj, vp;
    glm_lookat(eye, center, up, view);
    float fov = 2.f * atanf(0.5f);
    float asp = (float)wgpu_context->width / (float)wgpu_context->height;
    glm_perspective(fov, asp, 0.01f, 50.f, proj);
    glm_mat4_mul(proj, view, vp);

    cube_consts_t cc;
    memcpy(cc.clip_from_world, vp, sizeof(cc.clip_from_world));
    cc.world_down[0] = 0.f;
    cc.world_down[1] = 1.f;
    cc.world_down[2] = 0.f;
    cc.time_offs     = (float)state.frame * 0.0001f;
    srgb_color(0x202020, cc.ambient);
    cc._p0 = 0.f;
    srgb_color(0xc0c0c0, cc.key_col);
    cc._p1 = 0.f;
    srgb_color(0x602020, cc.fill_col);
    cc._p2 = 0.f;
    srgb_color(0x101040, cc.back_col);
    cc._p3 = 0.f;
    /* Normalised light direction */
    vec3 ld = {0.f, -0.7f, -0.3f};
    glm_normalize(ld);
    cc.light_dir[0] = ld[0];
    cc.light_dir[1] = ld[1];
    cc.light_dir[2] = ld[2];
    cc._p4          = 0.f;

    wgpuQueueWriteBuffer(wgpu_context->queue, state.cube_consts_buf, 0, &cc,
                         sizeof(cc));
  }

  /* 5. ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* 6. Build command buffer */
  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* --- UpdatePos pass: write to part_tex[k] --- */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = state.part_view[k],
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .label                = STRVIEW("UpdatePos - Render pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetPipeline(rp, state.update_pos_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.update_pos_bg[k], 0, NULL);
    wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* --- UpdateVel pass: write to part_tex[3] --- */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = state.part_view[3],
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .label                = STRVIEW("UpdateVel - Pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetPipeline(rp, state.update_vel_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.update_vel_bg[k], 0, NULL);
    wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* --- Cube render pass: write to swapchain --- */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = wgpu_context->swapchain_view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.2, 0.4, 0.6, 1.0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDepthStencilAttachment dsa = {
      .view            = state.depth_view,
      .depthLoadOp     = WGPULoadOp_Clear,
      .depthStoreOp    = WGPUStoreOp_Store,
      .depthClearValue = 1.0f,
    };
    WGPURenderPassDescriptor rpd = {
      .label                  = STRVIEW("Cube - Pass"),
      .colorAttachmentCount   = 1,
      .colorAttachments       = &ca,
      .depthStencilAttachment = &dsa,
    };
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetPipeline(rp, state.cube_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.cube_bg[k], 0, NULL);
    wgpuRenderPassEncoderSetIndexBuffer(
      rp, state.cube_index_buf, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
    /* Draw CHUNK_SIZE*15 indices × TEX_HEIGHT instances */
    wgpuRenderPassEncoderDrawIndexed(rp, CHUNK_SIZE * 15, TEX_HEIGHT, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  /* ImGui overlay (separate pass managed internally) */
  imgui_overlay_render(wgpu_context);

  if (state.settings.animate)
    ++state.frame;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  for (int i = 0; i < 4; ++i) {
    WGPU_RELEASE_RESOURCE(TextureView, state.part_view[i])
    WGPU_RELEASE_RESOURCE(Texture, state.part_tex[i])
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.force_view)
  WGPU_RELEASE_RESOURCE(Texture, state.force_tex)
  WGPU_RELEASE_RESOURCE(Sampler, state.force_sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_tex)

  WGPU_RELEASE_RESOURCE(Buffer, state.update_consts_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_consts_buf)
  WGPU_RELEASE_RESOURCE(Buffer, state.cube_index_buf)

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.update_pos_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.update_vel_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.cube_pipeline)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.update_pos_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.update_vel_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.cube_bgl)

  for (int i = 0; i < 3; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.update_pos_bg[i])
    WGPU_RELEASE_RESOURCE(BindGroup, state.update_vel_bg[i])
    WGPU_RELEASE_RESOURCE(BindGroup, state.cube_bg[i])
  }
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* ev)
{
  imgui_overlay_handle_input(wgpu_context, ev);
  if (ev->type == INPUT_EVENT_TYPE_RESIZED) {
    create_depth_texture(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Momentous",
    .width          = 1280,
    .height         = 720,
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

/* ---------- UpdatePos ---------- */
static const char* update_pos_shader_wgsl = CODE(
  struct UpdateConsts {
    field_scale:        vec3f,
    damping:            f32,
    field_offs:         vec3f,
    accel:              f32,
    field_sample_scale: vec3f,
    vel_scale:          f32,
  }

  @group(0) @binding(0) var<uniform> u:          UpdateConsts;
  @group(0) @binding(1) var          tex_older:  texture_2d<f32>;
  @group(0) @binding(2) var          tex_newer:  texture_2d<f32>;
  @group(0) @binding(3) var          force_tex:  texture_3d<f32>;

  @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let x = f32(vid >> 1u) * 4.0 - 1.0;
    let y = 1.0 - f32(vid & 1u) * 4.0;
    return vec4f(x, y, 0.5, 1.0);
  }

  @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let c      = vec2i(i32(pos.x), i32(pos.y));
    let older  = textureLoad(tex_older, c, 0);
    let newer  = textureLoad(tex_newer, c, 0);

    // Compute force field sample coordinate with smoothstep weights
    let fp_raw = newer.xyz * u.field_scale + u.field_offs;  // pos * 32
    let fr     = fract(fp_raw);
    let fs     = fr * fr * (3.0 - 2.0 * fr);  // smoothstep weights
    // Manual trilinear interpolation via textureLoad (avoids float32-filterable requirement)
    let b  = vec3i(i32(floor(fp_raw.x)) & 31, i32(floor(fp_raw.y)) & 31,
                   i32(floor(fp_raw.z)) & 31);
    let nx = (b.x + 1) & 31;
    let ny = (b.y + 1) & 31;
    let nz = (b.z + 1) & 31;
    let v000 = textureLoad(force_tex, vec3i(b.x,  b.y,  b.z),  0).xyz;
    let v100 = textureLoad(force_tex, vec3i(nx,   b.y,  b.z),  0).xyz;
    let v010 = textureLoad(force_tex, vec3i(b.x,  ny,   b.z),  0).xyz;
    let v110 = textureLoad(force_tex, vec3i(nx,   ny,   b.z),  0).xyz;
    let v001 = textureLoad(force_tex, vec3i(b.x,  b.y,  nz),   0).xyz;
    let v101 = textureLoad(force_tex, vec3i(nx,   b.y,  nz),   0).xyz;
    let v011 = textureLoad(force_tex, vec3i(b.x,  ny,   nz),   0).xyz;
    let v111 = textureLoad(force_tex, vec3i(nx,   ny,   nz),   0).xyz;
    let force = mix(mix(mix(v000, v100, fs.x), mix(v010, v110, fs.x), fs.y),
                    mix(mix(v001, v101, fs.x), mix(v011, v111, fs.x), fs.y), fs.z);

    // Verlet integration
    let new_xyz = newer.xyz + u.damping * (newer.xyz - older.xyz) + u.accel * force;
    let w       = select(newer.w, 0.0, dot(new_xyz, new_xyz) > 16.0);

    return vec4f(new_xyz, w);
  }
);

/* ---------- UpdateVel ---------- */
static const char* update_vel_shader_wgsl = CODE(
  @group(0) @binding(0) var tex_older: texture_2d<f32>;
  @group(0) @binding(1) var tex_newer: texture_2d<f32>;

  @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let x = f32(vid >> 1u) * 4.0 - 1.0;
    let y = 1.0 - f32(vid & 1u) * 4.0;
    return vec4f(x, y, 0.5, 1.0);
  }

  @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let c = vec2i(i32(pos.x), i32(pos.y));
    return textureLoad(tex_newer, c, 0) - textureLoad(tex_older, c, 0);
  }
);

/* ---------- Cube render ---------- */
static const char* cube_shader_wgsl = CODE(
  struct CubeConsts {
    clip_from_world: mat4x4f,
    world_down:      vec3f,   time_offs:  f32,
    ambient:         vec3f,   _p0:        f32,
    key_col:         vec3f,   _p1:        f32,
    fill_col:        vec3f,   _p2:        f32,
    back_col:        vec3f,   _p3:        f32,
    light_dir:       vec3f,   _p4:        f32,
  }

  @group(0) @binding(0) var<uniform> c:       CubeConsts;
  @group(0) @binding(1) var          tex_pos: texture_2d<f32>;
  @group(0) @binding(2) var          tex_vel: texture_2d<f32>;

  struct VOut {
    @builtin(position)              clip_pos:  vec4f,
    // Flat-interpolated: avoids subpixel-width derivative collapse to zero.
    // Provoking vertex (first of each triangle in strip) carries the outward
    // face normal for that triangle.  Assignment per corner index below.
    @location(0) @interpolate(flat) face_norm: vec3f,
  }

  @vertex fn vs(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32
  ) -> VOut {
    var v: VOut;

    // Fetch per-cube position and velocity from particle textures.
    // vertex_id encodes cube-within-chunk (bits 3+) and corner (bits 0-2).
    let tc       = vec2i(i32(vid >> 3u), i32(iid));
    let cube_pos = textureLoad(tex_pos, tc, 0);
    let cube_vel = textureLoad(tex_vel, tc, 0);

    if cube_pos.w == 0.0 {
      v.clip_pos  = vec4f(0.0);
      v.face_norm = vec3f(0.0);
      return v;
    }

    // Build local frame from velocity direction
    let x_axis = cube_vel.xyz;
    let z_axis = normalize(cross(x_axis, c.world_down));
    let y_axis = normalize(cross(z_axis, x_axis));

    // Cube corners: x elongated along velocity, y/z by across_size
    let s  = cube_pos.w;   // across-size (= PART_SIZE)
    var wp = cube_pos.xyz;
    wp += select(-1.0, 1.0, (vid & 1u) != 0u) * x_axis;
    wp += select( -s,   s,  (vid & 2u) != 0u) * y_axis;
    wp += select( -s,   s,  (vid & 4u) != 0u) * z_axis;

    v.clip_pos = c.clip_from_world * vec4f(wp, 1.0);

    // Assign outward face normal for this vertex's role as provoking vertex.
    // Strip [0,2,1,3,7,2,6,0,4,1,5,7,4,6] → provoking = first of each triple:
    //  corner 0 → T0(-z front), T7(-y bottom)  → -z_axis (front wins)
    //  corner 1 → T2/T9(+x end caps)            → +x_hat
    //  corner 2 → T1(-z front), T5(-x end cap)  → -z_axis (front wins)
    //  corner 3 → T3(+y top)                    → +y_axis
    //  corner 4 → T8(-y bottom)                 → -y_axis
    //  corner 5 → T10(+z back, camera-hidden)   → +z_axis
    //  corner 6 → T6(-x end cap)                → -x_hat
    //  corner 7 → T4(+y top), T11(+z back)      → +y_axis (top wins)
    let vlen  = length(x_axis);
    let x_hat = select(vec3f(1,0,0), x_axis / vlen, vlen > 1e-6);
    switch vid & 7u {
      case 0u: { v.face_norm = -z_axis; }
      case 1u: { v.face_norm =  x_hat;  }
      case 2u: { v.face_norm = -z_axis; }
      case 3u: { v.face_norm =  y_axis; }
      case 4u: { v.face_norm = -y_axis; }
      case 5u: { v.face_norm =  z_axis; }
      case 6u: { v.face_norm = -x_hat;  }
      case 7u: { v.face_norm =  y_axis; }
      default: { v.face_norm = vec3f(0,0,1); }
    }
    return v;
  }

  @fragment fn fs(v: VOut) -> @location(0) vec4f {
    let NdotL = dot(v.face_norm, c.light_dir);

    // Tri-light model: key + fill + back + ambient
    let lit = c.ambient
            + max( NdotL, 0.0) * c.key_col
            + (1.0 - abs(NdotL)) * c.fill_col
            + max(-NdotL, 0.0)  * c.back_col;

    return vec4f(lit, 1.0);
  }
);

// clang-format on
