/* clang-format off */
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __WAJIC__
#define WAJIC_SFETCH_IMPL
#include <wajic_sfetch.h>
#include <wajic_time.h>
#else
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>
#define SOKOL_LOG_IMPL
#include <sokol_log.h>
#include <sokol_time.h>
#endif

#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#ifndef CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/image_loader.h"
/* clang-format on */

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Clear Water
 *
 * Real-time photoreal shallow-water rendering ported to C99 WebGPU.
 * Features FFT ocean spectrum, refracted caustics with chromatic dispersion,
 * physically-based Fresnel + GGX sun glints, interactive ripples, procedural
 * pebble seabed, and ACES tone-mapping with bloom.
 *
 * Ref:
 * https://github.com/Aureliengmz/clearwater
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader Forward Declarations
 * -------------------------------------------------------------------------- */

static const char* cw_spectrum_shader_wgsl;
static const char* cw_fft_shader_wgsl;
static const char* cw_resolve_shader_wgsl;
static const char* cw_ripple_shader_wgsl;
static const char* cw_ripn_shader_wgsl;
static const char* cw_caus_shader_wgsl;
static const char* cw_bright_shader_wgsl;
static const char* cw_blur_shader_wgsl;
static const char* cw_copy_shader_wgsl;
static const char* cw_final_shader_wgsl;
/* Main water shader is split due to C99 string literal length limit */
static const char* cw_water_shader_part1;
static const char* cw_water_shader_part2;
static const char* cw_water_shader_part3;
static char cw_water_shader_buf[32 * 1024];
static const char* cw_get_water_shader(void);

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Ocean FFT */
#define CW_N 256
#define CW_LOGN 8
static const float CW_L            = 4.6f;
static const float CW_DEPTH        = 1.6f;
static const float CW_TARGET_SLOPE = 0.078f;

/* Ripple simulation */
#define CW_RN 256
static const float CW_RSIZE = 7.0f;

/* Caustics */
#define CW_G 256
#define CW_C 1024

/* Chromatic dispersion IORs (R, G, B) */
static const float CW_IORS[3] = {1.3315f, 1.3335f, 1.3365f};

/* Sun direction (elevation 31°, azimuth 6°) */
#define CW_SUN_EL (31.0f * 3.14159265f / 180.0f)
#define CW_SUN_AZ (6.0f * 3.14159265f / 180.0f)

/* Camera */
#define CW_VFOV (64.0f * 3.14159265f / 180.0f)
#define CW_CAM_H 1.55f
#define CW_MAX_DROPS 16

/* File buffer */
#define CW_PEB_FILE_BUF_SIZE (300u * 1024u)

/* -------------------------------------------------------------------------- *
 * Aligned Uniform Structs
 * -------------------------------------------------------------------------- */

/* Spectrum pass: time and patch length */
typedef struct {
  float t, L, pad[2];
} cw_spec_ub_t; /* 16 bytes */

/* FFT butterfly pass */
typedef struct {
  int32_t P, horiz, half_n, pad;
} cw_fft_ub_t; /* 16 bytes */

/* Ripple simulation step */
typedef struct {
  float shift[2], pad0[2]; /* 16 bytes */
  float drop[4];           /* 16 bytes */
} cw_ripple_ub_t;          /* 32 bytes */

/* Ripple normals */
typedef struct {
  float texel, pad[3];
} cw_ripn_ub_t; /* 16 bytes */

/* Caustics (one per colour channel) */
typedef struct {
  float L, depth, ior, norm; /* 16 bytes */
  float sun[3], pad0;        /* 16 bytes */
  float shift[2], pad1[2];   /* 16 bytes */
} cw_caus_ub_t;              /* 48 bytes */

/* Main water shader */
typedef struct {
  float cam[3], pad0;                  /* 16 */
  float R[3], pad1;                    /* 16 */
  float U[3], pad2;                    /* 16 */
  float F[3], pad3;                    /* 16 */
  float sun[3], pad4;                  /* 16 */
  float tanF, aspect, L, depth;        /* 16 */
  float time, rip_size, rip_center[2]; /* 16 */
  float caus_shift[2], pad5[2];        /* 16 */
} cw_water_ub_t;                       /* 96 bytes */

/* Bright pass */
typedef struct {
  float threshold, pad[3];
} cw_bright_ub_t; /* 16 bytes */

/* Blur pass */
typedef struct {
  float dir[2], pad[2];
} cw_blur_ub_t; /* 16 bytes */

/* Copy pass */
typedef struct {
  float k, pad[3];
} cw_copy_ub_t; /* 16 bytes */

/* Final tonemapping pass */
typedef struct {
  float exposure, time, no_post, pad0; /* 16 */
  float res[2], pad1[2];               /* 16 */
} cw_final_ub_t;                       /* 32 bytes */

/* -------------------------------------------------------------------------- *
 * Render Target helper
 * -------------------------------------------------------------------------- */

typedef struct {
  WGPUTexture tex;
  WGPUTextureView view;        /* full mip range (for sampling) */
  WGPUTextureView render_view; /* mip 0 only (for render attachment) */
  uint32_t w, h;
  WGPUTextureFormat fmt;
} cw_rt_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* ---- Simulation textures (fixed size) ---- */
  WGPUTexture h0_tex; /* initial spectrum RGBA32Float NxN  */
  WGPUTextureView h0_view;
  cw_rt_t fft[2];           /* FFT ping-pong RGBA16Float NxN     */
  cw_rt_t surf;             /* resolved surface RGBA16Float NxN  */
  WGPUSampler surf_sampler; /* linear+repeat+mip for surf         */
  WGPUSampler surf_aniso;   /* aniso 8 for surf                   */
  cw_rt_t rip[2];           /* ripple ping-pong RGBA16Float RNxRN */
  WGPUSampler rip_sampler;  /* linear+clamp                       */
  cw_rt_t rip_n;            /* ripple normals RGBA16Float RNxRN  */
  cw_rt_t caus;             /* caustics RGBA16Float CxC+mip       */
  WGPUSampler caus_sampler; /* linear+repeat+mip+aniso 8          */

  /* ---- Screen-sized render targets ---- */
  cw_rt_t hdr;                /* RGBA16Float WxH                   */
  cw_rt_t qa, qb;             /* half-res RGBA16Float               */
  cw_rt_t b1;                 /* half-res bloom RGBA16Float         */
  cw_rt_t b2, b2t;            /* quarter-res bloom                  */
  cw_rt_t streak;             /* 4x4 black (no lens glare)          */
  WGPUSampler screen_sampler; /* linear+clamp for screen passes     */

  /* ---- Pebbles texture ---- */
  wgpu_texture_t peb_tex;
  WGPUSampler peb_sampler;
  uint8_t* peb_file_buf;
  bool peb_loaded;

  /* ---- Pipelines ---- */
  WGPURenderPipeline spec_pipe;
  WGPURenderPipeline fft_pipe;
  WGPURenderPipeline resolve_pipe;
  WGPURenderPipeline ripple_pipe;
  WGPURenderPipeline ripn_pipe;
  WGPURenderPipeline caus_pipe[3]; /* per colour-channel write mask    */
  WGPURenderPipeline water_pipe;
  WGPURenderPipeline bright_pipe;
  WGPURenderPipeline blur_pipe;
  WGPURenderPipeline copy_pipe;
  WGPURenderPipeline final_pipe;

  /* ---- Caustics geometry ---- */
  WGPUBuffer caus_vb;
  WGPUBuffer caus_ib;
  uint32_t caus_idx_count;

  /* ---- Uniform buffers ---- */
  WGPUBuffer spec_ub;
  WGPUBuffer fft_ub[16]; /* one per butterfly stage             */
  WGPUBuffer ripple_ub;
  WGPUBuffer ripn_ub;
  WGPUBuffer caus_ub[3]; /* one per colour channel              */
  WGPUBuffer water_ub;
  WGPUBuffer bright_ub;
  WGPUBuffer blur_h_ub;
  WGPUBuffer blur_v_ub;
  WGPUBuffer blur_h2_ub;
  WGPUBuffer blur_v2_ub;
  WGPUBuffer copy_ub;
  WGPUBuffer final_ub;

  /* ---- Bind groups ---- */
  WGPUBindGroup spec_bg;
  WGPUBindGroup fft_bg[16]; /* alternating source texture          */
  WGPUBindGroup resolve_bg;
  WGPUBindGroup ripple_bg[2];
  WGPUBindGroup ripn_bg[2];
  WGPUBindGroup caus_bg[3];
  WGPUBindGroup water_bg;  /* recreated on resize                 */
  WGPUBindGroup bright_bg; /* recreated on resize                 */
  WGPUBindGroup blur_qa_bg;
  WGPUBindGroup blur_qb_bg;
  WGPUBindGroup blur_b1_bg;
  WGPUBindGroup blur_b2_bg;
  WGPUBindGroup blur_b2t_bg;
  WGPUBindGroup copy_b1_bg;
  WGPUBindGroup final_bg; /* recreated on resize                 */

  /* ---- Simulation state ---- */
  int rip_idx;
  float rip_center[2];
  int rip_active;
  /* circular drop queue */
  float drops[CW_MAX_DROPS][4];
  int drop_head, drop_tail;

  /* ---- Camera ---- */
  float cam_yaw, cam_pitch;
  float cam_vy, cam_vp; /* inertia */

  /* ---- Input ---- */
  struct {
    bool drag;
    float dx, dy;
    float x0, y0;
    double t0;
    bool tap;
    float tap_x, tap_y;
  } inp;

  /* ---- Timing ---- */
  float t_sim;
  uint64_t last_ns;

  /* ---- Render pass descriptors ---- */
  WGPURenderPassColorAttachment hdr_ca;
  WGPURenderPassDescriptor hdr_rpd;
  WGPURenderPassColorAttachment fft_ca;
  WGPURenderPassDescriptor fft_rpd;
  WGPURenderPassColorAttachment caus_ca;
  WGPURenderPassDescriptor caus_rpd;
  WGPURenderPassColorAttachment rip_ca;
  WGPURenderPassDescriptor rip_rpd;
  WGPURenderPassColorAttachment surf_ca;
  WGPURenderPassDescriptor surf_rpd;
  WGPURenderPassColorAttachment ripn_ca;
  WGPURenderPassDescriptor ripn_rpd;

  /* ---- Settings (GUI) ---- */
  struct {
    float exposure;
    bool no_post;
    bool paused;
  } settings;

  int screen_w, screen_h;
  WGPUBool initialized;
} state = {
  .rip_center = {0.0f, 0.0f},
  .rip_idx    = 0,
  .rip_active = 0,
  .cam_yaw    = 0.0f,
  .cam_pitch  = -0.72f,
  .cam_vy     = 0.0f,
  .cam_vp     = 0.0f,
  .t_sim      = 0.0f,
  .settings   = {.exposure = 0.63f, .no_post = false, .paused = false},
};

/* -------------------------------------------------------------------------- *
 * CPU-side H0 spectrum (Mulberry32 PRNG + Gaussian noise)
 * -------------------------------------------------------------------------- */

static uint32_t cw_prng_seed = 7;
static float cw_prng(void)
{
  uint32_t a = cw_prng_seed;
  a |= 0;
  a            = a + 0x6D2B79F5u;
  a            = a ^ (a >> 15);
  uint32_t t   = a * (1u | a);
  t            = t + ((t ^ (t >> 7)) * (61u | t));
  t            = t ^ (t >> 14);
  cw_prng_seed = a;
  return (float)(t >> 0) / 4294967296.0f;
}

static float cw_gauss(void)
{
  float u = 0.0f, v = 0.0f;
  while (u == 0.0f)
    u = cw_prng();
  v = cw_prng();
  return sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.14159265f * v);
}

/* Build the initial Gaussian spectrum H0 and upload to h0_tex */
static void cw_build_h0(wgpu_context_t* ctx)
{
  const int N       = CW_N;
  const float L     = CW_L;
  const float kp    = 2.0f * 3.14159265f / 0.62f;
  const float kcut  = 2.0f * 3.14159265f / 0.045f;
  const float wd[2] = {0.8f, 0.6f};

  float* re = (float*)calloc((size_t)(N * N), sizeof(float));
  float* im = (float*)calloc((size_t)(N * N), sizeof(float));

  double s2 = 0.0;
  for (int m = 0; m < N; m++) {
    for (int n = 0; n < N; n++) {
      int nx   = (n < N / 2) ? n : n - N;
      int nz   = (m < N / 2) ? m : m - N;
      float kx = 2.0f * 3.14159265f * (float)nx / L;
      float kz = 2.0f * 3.14159265f * (float)nz / L;
      float k  = sqrtf(kx * kx + kz * kz);
      float P  = 0.0f;
      if (k > 1e-6f) {
        float lk   = logf(k / kp);
        float bump = expf(-0.5f * (lk / 0.36f) * (lk / 0.36f));
        float tail = 0.035f * expf(-((kp / k) * (kp / k)))
                     * expf(-((k / kcut) * (k / kcut)));
        float swell_k = 2.0f * 3.14159265f / 1.6f;
        float sw_l    = logf(k / swell_k) / 0.3f;
        float swell   = 0.35f * expf(-0.5f * sw_l * sw_l);
        float c       = (kx * wd[0] + kz * wd[1]) / k;
        float spread  = (0.3f + 0.7f * c * c) * ((c < 0.0f) ? 0.35f : 1.0f);
        P             = (bump + tail + swell) * spread / (k * k * k * k);
      }
      float a = sqrtf(P * 0.5f);
      int i   = m * N + n;
      re[i]   = cw_gauss() * a;
      im[i]   = cw_gauss() * a;
      s2 += 2.0 * (double)k * (double)k
            * ((double)re[i] * (double)re[i] + (double)im[i] * (double)im[i]);
    }
  }

  float sc = (s2 > 0.0) ? (float)((double)CW_TARGET_SLOPE / sqrt(s2)) : 1.0f;

  /* Pack into RGBA32Float: (re, im, re_conj, -im_conj) */
  float* data = (float*)malloc((size_t)(N * N * 4) * sizeof(float));
  for (int m = 0; m < N; m++) {
    for (int n = 0; n < N; n++) {
      int i           = m * N + n;
      int j           = ((N - m) % N) * N + ((N - n) % N);
      data[i * 4 + 0] = re[i] * sc;
      data[i * 4 + 1] = im[i] * sc;
      data[i * 4 + 2] = re[j] * sc;
      data[i * 4 + 3] = -im[j] * sc;
    }
  }

  /* Create GPU texture */
  WGPUTextureDescriptor td = {
    .label         = STRVIEW("CW H0 Spectrum"),
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {(uint32_t)N, (uint32_t)N, 1},
    .format        = WGPUTextureFormat_RGBA32Float,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.h0_tex  = wgpuDeviceCreateTexture(ctx->device, &td);
  state.h0_view = wgpuTextureCreateView(
    state.h0_tex, &(WGPUTextureViewDescriptor){
                    .label           = STRVIEW("CW H0 View"),
                    .format          = WGPUTextureFormat_RGBA32Float,
                    .dimension       = WGPUTextureViewDimension_2D,
                    .mipLevelCount   = 1,
                    .arrayLayerCount = 1,
                  });

  WGPUTexelCopyTextureInfo dst = {
    .texture  = state.h0_tex,
    .mipLevel = 0,
    .aspect   = WGPUTextureAspect_All,
  };
  WGPUTexelCopyBufferLayout layout = {
    .offset       = 0,
    .bytesPerRow  = (uint32_t)(N * 4 * sizeof(float)),
    .rowsPerImage = (uint32_t)N,
  };
  WGPUExtent3D ext = {(uint32_t)N, (uint32_t)N, 1};
  wgpuQueueWriteTexture(ctx->queue, &dst, data,
                        (size_t)(N * N * 4) * sizeof(float), &layout, &ext);

  free(data);
  free(re);
  free(im);
}

/* -------------------------------------------------------------------------- *
 * Render-target helpers
 * -------------------------------------------------------------------------- */

static cw_rt_t cw_rt_create(wgpu_context_t* ctx, uint32_t w, uint32_t h,
                            WGPUTextureFormat fmt, const char* label,
                            bool mipmaps)
{
  uint32_t mip_count = mipmaps ? wgpu_texture_mip_level_count(w, h) : 1;
  WGPUTextureUsage usage
    = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment
      | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopyDst;
  if (mipmaps)
    usage |= WGPUTextureUsage_TextureBinding;

  WGPUTextureDescriptor td = {
    .label         = {.data = label, .length = label ? strlen(label) : 0},
    .usage         = usage,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {w, h, 1},
    .format        = fmt,
    .mipLevelCount = mip_count,
    .sampleCount   = 1,
  };
  WGPUTexture tex = wgpuDeviceCreateTexture(ctx->device, &td);
  WGPUTextureView view
    = wgpuTextureCreateView(tex, &(WGPUTextureViewDescriptor){
                                   .format        = fmt,
                                   .dimension     = WGPUTextureViewDimension_2D,
                                   .mipLevelCount = mip_count,
                                   .arrayLayerCount = 1,
                                 });
  /* Separate mip-0 view for render attachment */
  WGPUTextureView render_view
    = wgpuTextureCreateView(tex, &(WGPUTextureViewDescriptor){
                                   .format        = fmt,
                                   .dimension     = WGPUTextureViewDimension_2D,
                                   .baseMipLevel  = 0,
                                   .mipLevelCount = 1,
                                   .arrayLayerCount = 1,
                                 });
  return (cw_rt_t){.tex         = tex,
                   .view        = view,
                   .render_view = render_view,
                   .w           = w,
                   .h           = h,
                   .fmt         = fmt};
}

static void cw_rt_destroy(cw_rt_t* rt)
{
  WGPU_RELEASE_RESOURCE(TextureView, rt->render_view)
  WGPU_RELEASE_RESOURCE(TextureView, rt->view)
  WGPU_RELEASE_RESOURCE(Texture, rt->tex)
  rt->w = rt->h = 0;
}

/* -------------------------------------------------------------------------- *
 * Uniform-buffer helpers
 * -------------------------------------------------------------------------- */

static WGPUBuffer cw_ub_create(wgpu_context_t* ctx, uint32_t size,
                               const char* label)
{
  return wgpuDeviceCreateBuffer(
    ctx->device,
    &(WGPUBufferDescriptor){
      .label            = {.data = label, .length = label ? strlen(label) : 0},
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = size,
      .mappedAtCreation = false,
    });
}

static void cw_ub_write(wgpu_context_t* ctx, WGPUBuffer buf, const void* data,
                        uint32_t size)
{
  wgpuQueueWriteBuffer(ctx->queue, buf, 0, data, size);
}

/* -------------------------------------------------------------------------- *
 * Sampler helpers
 * -------------------------------------------------------------------------- */

static WGPUSampler cw_sampler(wgpu_context_t* ctx, WGPUAddressMode wrap,
                              WGPUFilterMode filter, uint16_t aniso,
                              WGPUMipmapFilterMode mip)
{
  return wgpuDeviceCreateSampler(ctx->device, &(WGPUSamplerDescriptor){
                                                .addressModeU  = wrap,
                                                .addressModeV  = wrap,
                                                .addressModeW  = wrap,
                                                .magFilter     = filter,
                                                .minFilter     = filter,
                                                .mipmapFilter  = mip,
                                                .lodMinClamp   = 0.0f,
                                                .lodMaxClamp   = 1024.0f,
                                                .maxAnisotropy = aniso,
                                              });
}

/* -------------------------------------------------------------------------- *
 * Fullscreen render-pass helpers
 * -------------------------------------------------------------------------- */

/* Begin a render pass rendering into rt - placeholder for potential future use
 */
static WGPURenderPassEncoder cw_begin_rt_pass_impl(WGPUCommandEncoder enc,
                                                   WGPUTextureView view)
{
  (void)enc;
  (void)view;
  return NULL;
}

static inline void cw_begin_rt_pass_unused(void)
{
  (void)cw_begin_rt_pass_impl;
}

/* Draw fullscreen triangle (vertex_index based, no vertex buffer) */
static void cw_fullscreen(WGPURenderPassEncoder pass)
{
  wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
}

/* -------------------------------------------------------------------------- *
 * Pipeline creation helpers
 * -------------------------------------------------------------------------- */

/* Simple fullscreen pipeline (vertex: vertex_index, fragment: wgsl_fs) */
static WGPURenderPipeline cw_fs_pipeline(wgpu_context_t* ctx,
                                         WGPUBindGroupLayout bgl,
                                         const char* fs_wgsl,
                                         WGPUTextureFormat rt_fmt,
                                         WGPUColorWriteMask write_mask,
                                         bool additive_blend, const char* label)
{
  /* Single module containing both vertex and fragment shaders */
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(ctx->device,
    &(WGPUShaderModuleDescriptor){
      .label    = {.data = label, .length = label ? strlen(label) : 0},
      .nextInChain = (WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code  = {.data = fs_wgsl, .length = strlen(fs_wgsl)},
      },
    });

  WGPUBlendState blend_add = {
    .color = {WGPUBlendOperation_Add, WGPUBlendFactor_One, WGPUBlendFactor_One},
    .alpha = {WGPUBlendOperation_Add, WGPUBlendFactor_One, WGPUBlendFactor_One},
  };

  WGPUColorTargetState ct = {
    .format    = rt_fmt,
    .blend     = additive_blend ? &blend_add : NULL,
    .writeMask = write_mask,
  };
  WGPUFragmentState fs = {
    .module      = mod,
    .entryPoint  = STRVIEW("fs_main"),
    .targetCount = 1,
    .targets     = &ct,
  };
  WGPUPipelineLayoutDescriptor pld = {
    .bindGroupLayoutCount = (bgl ? 1u : 0u),
    .bindGroupLayouts     = bgl ? &bgl : NULL,
  };
  WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(ctx->device, &pld);

  WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(ctx->device,
    &(WGPURenderPipelineDescriptor){
      .label  = {.data = label, .length = label ? strlen(label) : 0},
      .layout = layout,
      .vertex = {
        .module     = mod,
        .entryPoint = STRVIEW("vs_main"),
      },
      .fragment    = &fs,
      .primitive   = {.topology = WGPUPrimitiveTopology_TriangleList},
      .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    });

  WGPU_RELEASE_RESOURCE(PipelineLayout, layout)
  WGPU_RELEASE_RESOURCE(ShaderModule, mod)
  return pipe;
}

/* -------------------------------------------------------------------------- *
 * Bind-group creation helpers
 * -------------------------------------------------------------------------- */

/* Create BGL for: texture + uniform-buffer */
static WGPUBindGroupLayout cw_bgl_tex_ub(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry entries[2] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .texture    = {.sampleType = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = 0},
    },
  };
  return wgpuDeviceCreateBindGroupLayout(
    ctx->device,
    &(WGPUBindGroupLayoutDescriptor){.entryCount = 2, .entries = entries});
}

/* Create BGL for: texture + sampler + uniform-buffer */
static WGPUBindGroupLayout cw_bgl_texsamp_ub(wgpu_context_t* ctx)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .buffer     = {.type = WGPUBufferBindingType_Uniform,
                     .minBindingSize = 0},
    },
  };
  return wgpuDeviceCreateBindGroupLayout(
    ctx->device,
    &(WGPUBindGroupLayoutDescriptor){.entryCount = 3, .entries = entries});
}

/* -------------------------------------------------------------------------- *
 * Pebbles texture loading
 * -------------------------------------------------------------------------- */

static void cw_peb_fetch_callback(const sfetch_response_t* resp)
{
  if (!resp->fetched) {
    printf("[CW] Pebble texture fetch failed (error %d)\n", resp->error_code);
    free(state.peb_file_buf);
    state.peb_file_buf = NULL;
    /* Create a fallback grey texture */
    wgpu_context_t* ctx = *(wgpu_context_t**)resp->user_data;
    uint8_t grey[4 * 4] = {0};
    for (int i = 0; i < 16; i += 4) {
      grey[i]     = 128;
      grey[i + 1] = 120;
      grey[i + 2] = 100;
      grey[i + 3] = 255;
    }
    state.peb_tex = wgpu_create_texture(
      ctx,
      &(wgpu_texture_desc_t){
        .extent = {4, 4, 1},
        .format = WGPUTextureFormat_RGBA8UnormSrgb,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .pixels = {.ptr = grey, .size = sizeof(grey)},
      });
    state.peb_loaded = true;
    return;
  }
  int w, h, ch;
  uint8_t* px = image_pixels_from_memory(resp->data.ptr, (int)resp->data.size,
                                         &w, &h, &ch, 4);
  free(state.peb_file_buf);
  state.peb_file_buf = NULL;
  if (!px) {
    printf("[CW] Pebble decode failed\n");
    return;
  }
  wgpu_context_t* ctx = *(wgpu_context_t**)resp->user_data;
  state.peb_tex       = wgpu_create_texture(
    ctx, &(wgpu_texture_desc_t){
                 .extent = {(uint32_t)w, (uint32_t)h, 1},
                 .format = WGPUTextureFormat_RGBA8UnormSrgb,
                 .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
                 .pixels = {.ptr = px, .size = (size_t)(w * h * 4)},
                 .generate_mipmaps      = 1,
                 .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D,
         });
  image_free(px);
  state.peb_loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Caustics geometry (G × G grid)
 * -------------------------------------------------------------------------- */

static void cw_init_caus_geom(wgpu_context_t* ctx)
{
  const int G      = CW_G;
  const int vcount = (G + 1) * (G + 1);
  const int icount = G * G * 6;

  float* verts  = (float*)malloc((size_t)(vcount * 2) * sizeof(float));
  uint32_t* idx = (uint32_t*)malloc((size_t)icount * sizeof(uint32_t));

  int o = 0;
  for (int j = 0; j <= G; j++)
    for (int i = 0; i <= G; i++) {
      verts[o++] = (float)i / (float)G;
      verts[o++] = (float)j / (float)G;
    }
  o = 0;
  for (int j = 0; j < G; j++)
    for (int i = 0; i < G; i++) {
      uint32_t a = (uint32_t)(j * (G + 1) + i);
      uint32_t b = a + 1;
      uint32_t c = a + (uint32_t)(G + 1);
      uint32_t d = c + 1;
      idx[o++]   = a;
      idx[o++]   = b;
      idx[o++]   = c;
      idx[o++]   = b;
      idx[o++]   = d;
      idx[o++]   = c;
    }

  state.caus_vb = wgpu_create_buffer_from_data(
    ctx, verts, (size_t)(vcount * 2) * sizeof(float), WGPUBufferUsage_Vertex);
  state.caus_ib = wgpu_create_buffer_from_data(
    ctx, idx, (size_t)icount * sizeof(uint32_t), WGPUBufferUsage_Index);
  state.caus_idx_count = (uint32_t)icount;

  free(verts);
  free(idx);
}

/* -------------------------------------------------------------------------- *
 * Screen-sized render-target (re)allocation
 * -------------------------------------------------------------------------- */

static void cw_alloc_screen_rts(wgpu_context_t* ctx)
{
  int w = ctx->width, h = ctx->height;
  if (state.hdr.w == (uint32_t)w && state.hdr.h == (uint32_t)h)
    return;

  /* Destroy old */
  cw_rt_destroy(&state.hdr);
  cw_rt_destroy(&state.qa);
  cw_rt_destroy(&state.qb);
  cw_rt_destroy(&state.b1);
  cw_rt_destroy(&state.b2);
  cw_rt_destroy(&state.b2t);

  uint32_t qw = (uint32_t)MAX(1, w >> 1);
  uint32_t qh = (uint32_t)MAX(1, h >> 1);
  uint32_t bw = (uint32_t)MAX(1, w >> 2);
  uint32_t bh = (uint32_t)MAX(1, h >> 2);

  state.hdr = cw_rt_create(ctx, (uint32_t)w, (uint32_t)h,
                           WGPUTextureFormat_RGBA16Float, "CW HDR", false);
  state.qa
    = cw_rt_create(ctx, qw, qh, WGPUTextureFormat_RGBA16Float, "CW QA", false);
  state.qb
    = cw_rt_create(ctx, qw, qh, WGPUTextureFormat_RGBA16Float, "CW QB", false);
  state.b1
    = cw_rt_create(ctx, qw, qh, WGPUTextureFormat_RGBA16Float, "CW B1", false);
  state.b2
    = cw_rt_create(ctx, bw, bh, WGPUTextureFormat_RGBA16Float, "CW B2", false);
  state.b2t
    = cw_rt_create(ctx, bw, bh, WGPUTextureFormat_RGBA16Float, "CW B2T", false);

  state.screen_w = w;
  state.screen_h = h;
}

/* -------------------------------------------------------------------------- *
 * Bind-group (re)creation for screen-sized passes
 * -------------------------------------------------------------------------- */

static void cw_init_screen_bgs(wgpu_context_t* ctx,
                               WGPUBindGroupLayout bgl_tex_samp)
{
  (void)bgl_tex_samp;
  /* Release old */
  WGPU_RELEASE_RESOURCE(BindGroup, state.water_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bright_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_qa_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_qb_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b1_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b2_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b2t_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.copy_b1_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.final_bg)

  WGPUSampler ss = state.screen_sampler;

  /* water_bg: surf, caus, peb, rip_n (4 tex/samp + 1 ub) */
  {
    WGPUBindGroupLayoutEntry e[9] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [3] = {.binding    = 3,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [4] = {.binding    = 4,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [5] = {.binding    = 5,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [6] = {.binding    = 6,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [7] = {.binding    = 7,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [8] = {.binding    = 8,
             .visibility = WGPUShaderStage_Fragment,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(cw_water_ub_t)}},
    };
    WGPUBindGroupLayout wbgl = wgpuDeviceCreateBindGroupLayout(
      ctx->device,
      &(WGPUBindGroupLayoutDescriptor){.entryCount = 9, .entries = e});

    WGPUBindGroupEntry be[9] = {
      {.binding = 0, .textureView = state.surf.view},
      {.binding = 1, .sampler = state.surf_aniso},
      {.binding = 2, .textureView = state.caus.view},
      {.binding = 3, .sampler = state.caus_sampler},
      {.binding = 4, .textureView = state.peb_tex.view},
      {.binding = 5, .sampler = state.peb_sampler},
      {.binding = 6, .textureView = state.rip_n.view},
      {.binding = 7, .sampler = state.rip_sampler},
      {.binding = 8, .buffer = state.water_ub, .size = sizeof(cw_water_ub_t)},
    };
    state.water_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = wbgl, .entryCount = 9, .entries = be});

    /* Re-create the water pipeline with this layout */
    if (state.water_pipe) {
      wgpuRenderPipelineRelease(state.water_pipe);
      state.water_pipe = 0;
    }
    state.water_pipe = cw_fs_pipeline(
      ctx, wbgl, cw_get_water_shader(), WGPUTextureFormat_RGBA16Float,
      WGPUColorWriteMask_All, false, "CW Water");
    WGPU_RELEASE_RESOURCE(BindGroupLayout, wbgl)
  }

  /* bright_bg */
  {
    WGPUBindGroupLayout bgl  = cw_bgl_texsamp_ub(ctx);
    WGPUBindGroupEntry be[3] = {
      {.binding = 0, .textureView = state.hdr.view},
      {.binding = 1, .sampler = ss},
      {.binding = 2, .buffer = state.bright_ub, .size = sizeof(cw_bright_ub_t)},
    };
    state.bright_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = bgl, .entryCount = 3, .entries = be});
    if (!state.bright_pipe) {
      state.bright_pipe = cw_fs_pipeline(
        ctx, bgl, cw_bright_shader_wgsl, WGPUTextureFormat_RGBA16Float,
        WGPUColorWriteMask_All, false, "CW Bright");
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* blur bind groups */
  {
    WGPUBindGroupLayout bgl  = cw_bgl_texsamp_ub(ctx);
    WGPUTextureView views[5] = {state.qa.view, state.qb.view, state.b2.view,
                                state.b2.view, state.b2t.view};
    WGPUBindGroup* bgs[5]
      = {&state.blur_qa_bg, &state.blur_qb_bg, &state.blur_b1_bg,
         &state.blur_b2_bg, &state.blur_b2t_bg};
    WGPUBuffer ubs[5] = {state.blur_h_ub, state.blur_v_ub, state.blur_v_ub,
                         state.blur_h2_ub, state.blur_v2_ub};
    for (int i = 0; i < 5; i++) {
      WGPUBindGroupEntry be[3] = {
        {.binding = 0, .textureView = views[i]},
        {.binding = 1, .sampler = ss},
        {.binding = 2, .buffer = ubs[i], .size = sizeof(cw_blur_ub_t)},
      };
      *bgs[i] = wgpuDeviceCreateBindGroup(
        ctx->device, &(WGPUBindGroupDescriptor){
                       .layout = bgl, .entryCount = 3, .entries = be});
    }
    if (!state.blur_pipe) {
      state.blur_pipe = cw_fs_pipeline(
        ctx, bgl, cw_blur_shader_wgsl, WGPUTextureFormat_RGBA16Float,
        WGPUColorWriteMask_All, false, "CW Blur");
    }
    /* copy */
    WGPUBindGroupEntry cbe[3] = {
      {.binding = 0, .textureView = state.b1.view},
      {.binding = 1, .sampler = ss},
      {.binding = 2, .buffer = state.copy_ub, .size = sizeof(cw_copy_ub_t)},
    };
    state.copy_b1_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = bgl, .entryCount = 3, .entries = cbe});
    if (!state.copy_pipe) {
      state.copy_pipe = cw_fs_pipeline(
        ctx, bgl, cw_copy_shader_wgsl, WGPUTextureFormat_RGBA16Float,
        WGPUColorWriteMask_All, false, "CW Copy");
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* final_bg: hdr, streak, b1, b2 + ub */
  {
    WGPUBindGroupLayoutEntry fe[9] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [3] = {.binding    = 3,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [4] = {.binding    = 4,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [5] = {.binding    = 5,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [6] = {.binding    = 6,
             .visibility = WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [7] = {.binding    = 7,
             .visibility = WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [8] = {.binding    = 8,
             .visibility = WGPUShaderStage_Fragment,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(cw_final_ub_t)}},
    };
    WGPUBindGroupLayout fbgl = wgpuDeviceCreateBindGroupLayout(
      ctx->device,
      &(WGPUBindGroupLayoutDescriptor){.entryCount = 9, .entries = fe});

    WGPUBindGroupEntry fbe[9] = {
      {.binding = 0, .textureView = state.hdr.view},
      {.binding = 1, .sampler = ss},
      {.binding = 2, .textureView = state.streak.view},
      {.binding = 3, .sampler = ss},
      {.binding = 4, .textureView = state.b1.view},
      {.binding = 5, .sampler = ss},
      {.binding = 6, .textureView = state.b2.view},
      {.binding = 7, .sampler = ss},
      {.binding = 8, .buffer = state.final_ub, .size = sizeof(cw_final_ub_t)},
    };
    state.final_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = fbgl, .entryCount = 9, .entries = fbe});

    if (state.final_pipe) {
      wgpuRenderPipelineRelease(state.final_pipe);
      state.final_pipe = 0;
    }
    state.final_pipe
      = cw_fs_pipeline(ctx, fbgl, cw_final_shader_wgsl, ctx->render_format,
                       WGPUColorWriteMask_All, false, "CW Final");
    WGPU_RELEASE_RESOURCE(BindGroupLayout, fbgl)
  }
}

/* -------------------------------------------------------------------------- *
 * Initialisation of fixed (non-screen-sized) GPU resources
 * -------------------------------------------------------------------------- */

static void cw_init_fixed(wgpu_context_t* ctx)
{
  /* Simulation render targets */
  state.fft[0] = cw_rt_create(ctx, CW_N, CW_N, WGPUTextureFormat_RGBA16Float,
                              "CW FFT0", false);
  state.fft[1] = cw_rt_create(ctx, CW_N, CW_N, WGPUTextureFormat_RGBA16Float,
                              "CW FFT1", false);
  state.surf   = cw_rt_create(ctx, CW_N, CW_N, WGPUTextureFormat_RGBA16Float,
                              "CW Surf", true);
  state.rip[0] = cw_rt_create(ctx, CW_RN, CW_RN, WGPUTextureFormat_RGBA16Float,
                              "CW Rip0", false);
  state.rip[1] = cw_rt_create(ctx, CW_RN, CW_RN, WGPUTextureFormat_RGBA16Float,
                              "CW Rip1", false);
  state.rip_n  = cw_rt_create(ctx, CW_RN, CW_RN, WGPUTextureFormat_RGBA16Float,
                              "CW RipN", false);
  state.caus   = cw_rt_create(ctx, CW_C, CW_C, WGPUTextureFormat_RGBA16Float,
                              "CW Caus", true);

  /* Samplers */
  state.surf_sampler
    = cw_sampler(ctx, WGPUAddressMode_Repeat, WGPUFilterMode_Linear, 1,
                 WGPUMipmapFilterMode_Linear);
  state.surf_aniso
    = cw_sampler(ctx, WGPUAddressMode_Repeat, WGPUFilterMode_Linear, 8,
                 WGPUMipmapFilterMode_Linear);
  state.rip_sampler
    = cw_sampler(ctx, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, 1,
                 WGPUMipmapFilterMode_Nearest);
  state.caus_sampler
    = cw_sampler(ctx, WGPUAddressMode_Repeat, WGPUFilterMode_Linear, 8,
                 WGPUMipmapFilterMode_Linear);
  state.peb_sampler
    = cw_sampler(ctx, WGPUAddressMode_Repeat, WGPUFilterMode_Linear, 16,
                 WGPUMipmapFilterMode_Linear);
  state.screen_sampler
    = cw_sampler(ctx, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, 1,
                 WGPUMipmapFilterMode_Nearest);

  /* Uniform buffers */
  state.spec_ub   = cw_ub_create(ctx, sizeof(cw_spec_ub_t), "CW Spec UB");
  state.ripple_ub = cw_ub_create(ctx, sizeof(cw_ripple_ub_t), "CW Ripple UB");
  state.ripn_ub   = cw_ub_create(ctx, sizeof(cw_ripn_ub_t), "CW RipN UB");
  for (int c = 0; c < 3; c++)
    state.caus_ub[c] = cw_ub_create(ctx, sizeof(cw_caus_ub_t), "CW Caus UB");
  state.water_ub   = cw_ub_create(ctx, sizeof(cw_water_ub_t), "CW Water UB");
  state.bright_ub  = cw_ub_create(ctx, sizeof(cw_bright_ub_t), "CW Bright UB");
  state.blur_h_ub  = cw_ub_create(ctx, sizeof(cw_blur_ub_t), "CW BlurH UB");
  state.blur_v_ub  = cw_ub_create(ctx, sizeof(cw_blur_ub_t), "CW BlurV UB");
  state.blur_h2_ub = cw_ub_create(ctx, sizeof(cw_blur_ub_t), "CW BlurH2 UB");
  state.blur_v2_ub = cw_ub_create(ctx, sizeof(cw_blur_ub_t), "CW BlurV2 UB");
  state.copy_ub    = cw_ub_create(ctx, sizeof(cw_copy_ub_t), "CW Copy UB");
  state.final_ub   = cw_ub_create(ctx, sizeof(cw_final_ub_t), "CW Final UB");
  for (int i = 0; i < 16; i++)
    state.fft_ub[i] = cw_ub_create(ctx, sizeof(cw_fft_ub_t), "CW FFT UB");

  /* Write static FFT params */
  int src = 0; /* tracks which fft[] is currently source */
  for (int horiz = 1; horiz >= 0; horiz--) {
    for (int s = 0; s < CW_LOGN; s++) {
      int step = (1 - horiz) * CW_LOGN + s;
      cw_fft_ub_t p
        = {.P = 1 << s, .horiz = horiz, .half_n = CW_N / 2, .pad = 0};
      cw_ub_write(ctx, state.fft_ub[step], &p, sizeof(p));
      (void)src;
    }
  }

  /* Write static post-processing params */
  cw_bright_ub_t bp = {.threshold = 2.5f};
  cw_ub_write(ctx, state.bright_ub, &bp, sizeof(bp));
  cw_blur_ub_t blh  = {.dir = {1.0f, 0.0f}};
  cw_blur_ub_t blv  = {.dir = {0.0f, 1.0f}};
  cw_blur_ub_t blh2 = {.dir = {1.5f, 0.0f}};
  cw_blur_ub_t blv2 = {.dir = {0.0f, 1.5f}};
  cw_ub_write(ctx, state.blur_h_ub, &blh, sizeof(blh));
  cw_ub_write(ctx, state.blur_v_ub, &blv, sizeof(blv));
  cw_ub_write(ctx, state.blur_h2_ub, &blh2, sizeof(blh2));
  cw_ub_write(ctx, state.blur_v2_ub, &blv2, sizeof(blv2));
  cw_copy_ub_t cub = {.k = 1.0f};
  cw_ub_write(ctx, state.copy_ub, &cub, sizeof(cub));
  cw_ripn_ub_t rnub = {.texel = CW_RSIZE / (float)CW_RN};
  cw_ub_write(ctx, state.ripn_ub, &rnub, sizeof(rnub));

  /* Build H0 spectrum */
  cw_build_h0(ctx);

  /* Build caustics grid geometry */
  cw_init_caus_geom(ctx);

  /* 4×4 black streak texture (no lens glare) */
  {
    uint8_t blk[4 * 4 * 4] = {0};
    state.streak = cw_rt_create(ctx, 4, 4, WGPUTextureFormat_RGBA16Float,
                                "CW Streak", false);
    /* write zeros to clear it */
    (void)blk;
    /* Use a command encoder to clear it */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
    WGPURenderPassColorAttachment ca = {
      .view       = state.streak.view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(ctx->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
  }

  /* Clear ripple textures */
  {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
    for (int i = 0; i < 2; i++) {
      WGPURenderPassColorAttachment ca = {
        .view       = state.rip[i].view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      };
      WGPURenderPassDescriptor rpd
        = {.colorAttachmentCount = 1, .colorAttachments = &ca};
      WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
      wgpuRenderPassEncoderEnd(rp);
      wgpuRenderPassEncoderRelease(rp);
    }
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(ctx->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
  }

  /* --- Pipelines for fixed resources --- */

  /* Spectrum pipeline - H0 is RGBA32Float (UnfilterableFloat) */
  {
    WGPUBindGroupLayoutEntry spec_entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform, .minBindingSize = 0},
      },
    };
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(
      ctx->device, &(WGPUBindGroupLayoutDescriptor){.entryCount = 2,
                                                    .entries = spec_entries});
    state.spec_pipe = cw_fs_pipeline(ctx, bgl, cw_spectrum_shader_wgsl,
                                     WGPUTextureFormat_RGBA16Float,
                                     WGPUColorWriteMask_All, false, "CW Spec");
    /* spec bind group */
    WGPUBindGroupEntry be[2] = {
      {.binding = 0, .textureView = state.h0_view},
      {.binding = 1, .buffer = state.spec_ub, .size = sizeof(cw_spec_ub_t)},
    };
    state.spec_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = bgl, .entryCount = 2, .entries = be});
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* FFT pipeline + bind groups */
  {
    WGPUBindGroupLayout bgl = cw_bgl_tex_ub(ctx);
    state.fft_pipe          = cw_fs_pipeline(ctx, bgl, cw_fft_shader_wgsl,
                                             WGPUTextureFormat_RGBA16Float,
                                             WGPUColorWriteMask_All, false, "CW FFT");
    /* 16 bind groups, alternating source */
    int cur_src = 0;
    for (int horiz = 1; horiz >= 0; horiz--) {
      for (int s = 0; s < CW_LOGN; s++) {
        int step                 = (1 - horiz) * CW_LOGN + s;
        WGPUBindGroupEntry be[2] = {
          {.binding = 0, .textureView = state.fft[cur_src].view},
          {.binding = 1,
           .buffer  = state.fft_ub[step],
           .size    = sizeof(cw_fft_ub_t)},
        };
        state.fft_bg[step] = wgpuDeviceCreateBindGroup(
          ctx->device, &(WGPUBindGroupDescriptor){
                         .layout = bgl, .entryCount = 2, .entries = be});
        cur_src = 1 - cur_src;
      }
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* Resolve pipeline */
  {
    WGPUBindGroupLayoutEntry e[1] = {{
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                     .viewDimension = WGPUTextureViewDimension_2D},
    }};
    WGPUBindGroupLayout bgl       = wgpuDeviceCreateBindGroupLayout(
      ctx->device,
      &(WGPUBindGroupLayoutDescriptor){.entryCount = 1, .entries = e});
    state.resolve_pipe = cw_fs_pipeline(
      ctx, bgl, cw_resolve_shader_wgsl, WGPUTextureFormat_RGBA16Float,
      WGPUColorWriteMask_All, false, "CW Resolve");
    /* After 16 FFT steps, source is fft[0] (src=0 after 16 swaps from 0) */
    WGPUBindGroupEntry be[1]
      = {{.binding = 0, .textureView = state.fft[0].view}};
    state.resolve_bg = wgpuDeviceCreateBindGroup(
      ctx->device, &(WGPUBindGroupDescriptor){
                     .layout = bgl, .entryCount = 1, .entries = be});
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* Ripple pipeline */
  {
    WGPUBindGroupLayout bgl = cw_bgl_texsamp_ub(ctx);
    state.ripple_pipe       = cw_fs_pipeline(
      ctx, bgl, cw_ripple_shader_wgsl, WGPUTextureFormat_RGBA16Float,
      WGPUColorWriteMask_All, false, "CW Ripple");
    for (int i = 0; i < 2; i++) {
      WGPUBindGroupEntry be[3] = {
        {.binding = 0, .textureView = state.rip[i].view},
        {.binding = 1, .sampler = state.rip_sampler},
        {.binding = 2,
         .buffer  = state.ripple_ub,
         .size    = sizeof(cw_ripple_ub_t)},
      };
      state.ripple_bg[i] = wgpuDeviceCreateBindGroup(
        ctx->device, &(WGPUBindGroupDescriptor){
                       .layout = bgl, .entryCount = 3, .entries = be});
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* Ripple-normals pipeline */
  {
    WGPUBindGroupLayout bgl = cw_bgl_texsamp_ub(ctx);
    state.ripn_pipe         = cw_fs_pipeline(ctx, bgl, cw_ripn_shader_wgsl,
                                             WGPUTextureFormat_RGBA16Float,
                                             WGPUColorWriteMask_All, false, "CW RipN");
    for (int i = 0; i < 2; i++) {
      WGPUBindGroupEntry be[3] = {
        {.binding = 0, .textureView = state.rip[i].view},
        {.binding = 1, .sampler = state.rip_sampler},
        {.binding = 2, .buffer = state.ripn_ub, .size = sizeof(cw_ripn_ub_t)},
      };
      state.ripn_bg[i] = wgpuDeviceCreateBindGroup(
        ctx->device, &(WGPUBindGroupDescriptor){
                       .layout = bgl, .entryCount = 3, .entries = be});
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl)
  }

  /* Caustics pipelines (3 colour channels, one per write mask) */
  {
    WGPUColorWriteMask masks[3] = {
      WGPUColorWriteMask_Red,
      WGPUColorWriteMask_Green,
      WGPUColorWriteMask_Blue,
    };
    /* Caustics BGL: surf tex/samp (vertex shader needs tex), ub */
    WGPUBindGroupLayoutEntry ce[3] = {
      [0] = {.binding    = 0,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                            .viewDimension = WGPUTextureViewDimension_2D}},
      [1] = {.binding    = 1,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .sampler    = {.type = WGPUSamplerBindingType_Filtering}},
      [2] = {.binding    = 2,
             .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
             .buffer     = {.type           = WGPUBufferBindingType_Uniform,
                            .minBindingSize = sizeof(cw_caus_ub_t)}},
    };
    WGPUBindGroupLayout cbgl = wgpuDeviceCreateBindGroupLayout(
      ctx->device,
      &(WGPUBindGroupLayoutDescriptor){.entryCount = 3, .entries = ce});

    for (int c = 0; c < 3; c++) {
      /* Caustics pipeline with grid vertex buffer */
      WGPUShaderModule mod = wgpuDeviceCreateShaderModule(ctx->device,
        &(WGPUShaderModuleDescriptor){
          .label = STRVIEW("CW Caus Shader"),
          .nextInChain = (WGPUChainedStruct*)&(WGPUShaderSourceWGSL){
            .chain = {.sType = WGPUSType_ShaderSourceWGSL},
            .code  = {.data = cw_caus_shader_wgsl,
                      .length = strlen(cw_caus_shader_wgsl)},
          },
        });
      WGPUBlendState badd = {
        .color
        = {WGPUBlendOperation_Add, WGPUBlendFactor_One, WGPUBlendFactor_One},
        .alpha
        = {WGPUBlendOperation_Add, WGPUBlendFactor_One, WGPUBlendFactor_One},
      };
      WGPUColorTargetState ct = {
        .format    = WGPUTextureFormat_RGBA16Float,
        .blend     = &badd,
        .writeMask = masks[c],
      };
      WGPUFragmentState fs     = {.module      = mod,
                                  .entryPoint  = STRVIEW("fs_main"),
                                  .targetCount = 1,
                                  .targets     = &ct};
      WGPUVertexAttribute attr = {
        .shaderLocation = 0, .format = WGPUVertexFormat_Float32x2, .offset = 0};
      WGPUVertexBufferLayout vbl = {.arrayStride    = 8,
                                    .stepMode       = WGPUVertexStepMode_Vertex,
                                    .attributeCount = 1,
                                    .attributes     = &attr};
      WGPUPipelineLayoutDescriptor pld
        = {.bindGroupLayoutCount = 1, .bindGroupLayouts = &cbgl};
      WGPUPipelineLayout layout
        = wgpuDeviceCreatePipelineLayout(ctx->device, &pld);
      state.caus_pipe[c] = wgpuDeviceCreateRenderPipeline(
        ctx->device,
        &(WGPURenderPipelineDescriptor){
          .label       = STRVIEW("CW Caus Pipe"),
          .layout      = layout,
          .vertex      = {.module      = mod,
                          .entryPoint  = STRVIEW("vs_main"),
                          .bufferCount = 1,
                          .buffers     = &vbl},
          .fragment    = &fs,
          .primitive   = {.topology = WGPUPrimitiveTopology_TriangleList},
          .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        });
      WGPU_RELEASE_RESOURCE(PipelineLayout, layout)
      WGPU_RELEASE_RESOURCE(ShaderModule, mod)

      WGPUBindGroupEntry cbe[3] = {
        {.binding = 0, .textureView = state.surf.view},
        {.binding = 1, .sampler = state.surf_sampler},
        {.binding = 2,
         .buffer  = state.caus_ub[c],
         .size    = sizeof(cw_caus_ub_t)},
      };
      state.caus_bg[c] = wgpuDeviceCreateBindGroup(
        ctx->device, &(WGPUBindGroupDescriptor){
                       .layout = cbgl, .entryCount = 3, .entries = cbe});
    }
    WGPU_RELEASE_RESOURCE(BindGroupLayout, cbgl)
  }
}

/* -------------------------------------------------------------------------- *
 * Per-frame GPU pass functions
 * -------------------------------------------------------------------------- */

static void cw_compute_cam_basis(float t, float out_f[3], float out_r[3],
                                 float out_u[3], float out_pos[3])
{
  /* subtle idle sway when not dragging */
  float yaw = state.cam_yaw + 0.010f * sinf(t * 0.31f)
              + 0.005f * sinf(t * 0.83f + 1.3f);
  float pitch = state.cam_pitch + 0.007f * sinf(t * 0.47f + 2.0f)
                + 0.003f * sinf(t * 1.13f);
  float roll = 0.006f * sinf(t * 0.39f + 0.4f);

  float f[3] = {sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch)};
  float r[3] = {cosf(yaw), 0.0f, sinf(yaw)};
  /* up = cross(r, f) */
  float u[3] = {
    r[1] * f[2] - r[2] * f[1],
    r[2] * f[0] - r[0] * f[2],
    r[0] * f[1] - r[1] * f[0],
  };
  float cr = cosf(roll), sr = sinf(roll);
  float r2[3]
    = {r[0] * cr + u[0] * sr, r[1] * cr + u[1] * sr, r[2] * cr + u[2] * sr};
  float u2[3]
    = {u[0] * cr - r[0] * sr, u[1] * cr - r[1] * sr, u[2] * cr - r[2] * sr};

  for (int i = 0; i < 3; i++) {
    out_f[i] = f[i];
    out_r[i] = r2[i];
    out_u[i] = u2[i];
  }
  out_pos[0] = 0.03f * sinf(t * 0.21f);
  out_pos[1] = CW_CAM_H + 0.015f * sinf(t * 0.57f);
  out_pos[2] = 0.03f * cosf(t * 0.17f);
}

/* Map screen tap to ripple UV */
static void cw_tap_to_drop(float sx, float sy, int sw, int sh, const float F[3],
                           const float R[3], const float U[3],
                           const float pos[3])
{
  float nx   = (sx / (float)sw) * 2.0f - 1.0f;
  float ny   = 1.0f - (sy / (float)sh) * 2.0f;
  float tf   = tanf(CW_VFOV * 0.5f);
  float asp  = (float)sw / (float)sh;
  float d[3] = {
    F[0] + nx * asp * tf * R[0] + ny * tf * U[0],
    F[1] + nx * asp * tf * R[1] + ny * tf * U[1],
    F[2] + nx * asp * tf * R[2] + ny * tf * U[2],
  };
  if (d[1] >= -0.01f)
    return;
  float t  = -pos[1] / d[1];
  float px = pos[0] + d[0] * t;
  float pz = pos[2] + d[2] * t;
  float u  = (px - state.rip_center[0]) / CW_RSIZE + 0.5f;
  float v  = (pz - state.rip_center[1]) / CW_RSIZE + 0.5f;
  if (u < 0.05f || u > 0.95f || v < 0.05f || v > 0.95f)
    return;
  int next = (state.drop_head + 1) % CW_MAX_DROPS;
  if (next != state.drop_tail) {
    state.drops[state.drop_head][0] = u;
    state.drops[state.drop_head][1] = v;
    state.drops[state.drop_head][2] = 0.022f;
    state.drops[state.drop_head][3] = 0.07f;
    state.drop_head                 = next;
  }
}

/* Run the FFT ocean simulation for time t */
static void cw_run_fft(wgpu_context_t* ctx, WGPUCommandEncoder enc, float t)
{
  /* Spectrum pass → fft[0] (= fftA in the reference) */
  {
    cw_spec_ub_t p = {.t = t * 0.9f, .L = CW_L};
    cw_ub_write(ctx, state.spec_ub, &p, sizeof(p));

    WGPURenderPassColorAttachment ca = {
      .view       = state.fft[0].view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_N, (float)CW_N, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.spec_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.spec_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* 16 butterfly FFT passes, ping-ponging between fft[0] and fft[1] */
  int dst_idx = 1; /* first pass writes to fft[1] */
  for (int horiz = 1; horiz >= 0; horiz--) {
    for (int s = 0; s < CW_LOGN; s++) {
      int step                         = (1 - horiz) * CW_LOGN + s;
      WGPURenderPassColorAttachment ca = {
        .view       = state.fft[dst_idx].view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      };
      WGPURenderPassDescriptor rpd
        = {.colorAttachmentCount = 1, .colorAttachments = &ca};
      WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
      wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_N, (float)CW_N, 0,
                                       1);
      wgpuRenderPassEncoderSetPipeline(rp, state.fft_pipe);
      wgpuRenderPassEncoderSetBindGroup(rp, 0, state.fft_bg[step], 0, NULL);
      cw_fullscreen(rp);
      wgpuRenderPassEncoderEnd(rp);
      wgpuRenderPassEncoderRelease(rp);
      dst_idx = 1 - dst_idx;
    }
  }

  /* After 16 passes, the last rendered result is in fft[0]
     (dst_idx ends at 1 after 16 alternations starting at 1, but it was
     written to 1-dst before last swap, so result is in fft[0]).
     resolve_bg is pre-bound to fft[0]. */

  /* Resolve pass → surf */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = state.surf.render_view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_N, (float)CW_N, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.resolve_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.resolve_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }
  /* Generate mipmaps for surf */
  wgpu_generate_mipmaps(ctx, state.surf.tex, WGPU_MIPMAP_VIEW_2D);
}

/* Run the ripple simulation */
static void cw_step_ripples(wgpu_context_t* ctx, WGPUCommandEncoder enc,
                            float shift_u, float shift_v)
{
  /* Consume one drop from the queue */
  float dr[4] = {0, 0, 0, 0};
  if (state.drop_head != state.drop_tail) {
    dr[0]           = state.drops[state.drop_tail][0];
    dr[1]           = state.drops[state.drop_tail][1];
    dr[2]           = state.drops[state.drop_tail][2];
    dr[3]           = state.drops[state.drop_tail][3];
    state.drop_tail = (state.drop_tail + 1) % CW_MAX_DROPS;
  }
  cw_ripple_ub_t p = {
    .shift = {shift_u, shift_v},
    .drop  = {dr[0], dr[1], dr[2], dr[3]},
  };
  cw_ub_write(ctx, state.ripple_ub, &p, sizeof(p));

  int src = state.rip_idx;
  int dst = 1 - src;

  /* Ripple step */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = state.rip[dst].view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_RN, (float)CW_RN, 0,
                                     1);
    wgpuRenderPassEncoderSetPipeline(rp, state.ripple_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.ripple_bg[src], 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }
  state.rip_idx = dst;

  /* Ripple normals */
  {
    WGPURenderPassColorAttachment ca = {
      .view       = state.rip_n.view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_RN, (float)CW_RN, 0,
                                     1);
    wgpuRenderPassEncoderSetPipeline(rp, state.ripn_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.ripn_bg[dst], 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }
  (void)ctx;
}

/* Render caustics */
static void cw_render_caustics(wgpu_context_t* ctx, WGPUCommandEncoder enc,
                               const float sun[3])
{
  /* Compute refracted sun shift for green channel (flat surface) */
  float sy    = sun[1];
  float ior_g = CW_IORS[1];
  float sinI  = sqrtf(1.0f - sy * sy);
  float sinT  = sinI / ior_g;
  float cosT  = sqrtf(1.0f - sinT * sinT);
  float hd    = sqrtf(sun[0] * sun[0] + sun[2] * sun[2]);
  if (hd < 1e-6f)
    hd = 1e-6f;
  float tanT = sinT / cosT;
  float cs[2]
    = {-sun[0] / hd * CW_DEPTH * tanT, -sun[2] / hd * CW_DEPTH * tanT};

  /* Update 3 caustic UBOs (one per channel, different IOR) */
  for (int c = 0; c < 3; c++) {
    /* per-channel refraction shift */
    float ior_c  = CW_IORS[c];
    float sinT_c = sinI / ior_c;
    float cosT_c = sqrtf(1.0f - sinT_c * sinT_c);
    float tanT_c = sinT_c / cosT_c;
    float shift_c[2]
      = {-sun[0] / hd * CW_DEPTH * tanT_c, -sun[2] / hd * CW_DEPTH * tanT_c};
    cw_caus_ub_t p = {
      .L     = CW_L,
      .depth = CW_DEPTH,
      .ior   = CW_IORS[c],
      .norm  = ((float)CW_C / CW_L) * ((float)CW_C / CW_L),
      .sun   = {sun[0], sun[1], sun[2]},
      .shift = {shift_c[0], shift_c[1]},
    };
    cw_ub_write(ctx, state.caus_ub[c], &p, sizeof(p));
  }
  /* Update causShift for the main water shader (green = representative) */
  state.rip_center[0] = state.rip_center[0]; /* unchanged */
  (void)cs; /* causShift set from main water pass */

  /* Render to causRT with clear */
  WGPURenderPassColorAttachment ca = {
    .view       = state.caus.render_view,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0, 0, 0, 0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  WGPURenderPassDescriptor rpd
    = {.colorAttachmentCount = 1, .colorAttachments = &ca};
  WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
  wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)CW_C, (float)CW_C, 0, 1);

  for (int c = 0; c < 3; c++) {
    wgpuRenderPassEncoderSetPipeline(rp, state.caus_pipe[c]);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.caus_bg[c], 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(
      rp, 0, state.caus_vb, 0,
      (uint64_t)((CW_G + 1) * (CW_G + 1) * 2 * sizeof(float)));
    wgpuRenderPassEncoderSetIndexBuffer(
      rp, state.caus_ib, WGPUIndexFormat_Uint32, 0,
      (uint64_t)(state.caus_idx_count * sizeof(uint32_t)));
    wgpuRenderPassEncoderDrawIndexed(rp, state.caus_idx_count, 9, 0, 0, 0);
  }
  wgpuRenderPassEncoderEnd(rp);
  wgpuRenderPassEncoderRelease(rp);

  /* Generate mips for caustics */
  wgpu_generate_mipmaps(ctx, state.caus.tex, WGPU_MIPMAP_VIEW_2D);
}

/* Main water render */
static void cw_render_water(wgpu_context_t* ctx, WGPUCommandEncoder enc,
                            float t, const float sun[3])
{
  float F[3], R[3], U[3], pos[3];
  cw_compute_cam_basis(t, F, R, U, pos);

  float asp = (float)state.screen_w / (float)state.screen_h;

  /* Green-channel caustic shift for the main water shader */
  float sy   = sun[1];
  float sinI = sqrtf(1.0f - sy * sy);
  float sinT = sinI / CW_IORS[1];
  float cosT = sqrtf(1.0f - sinT * sinT);
  float hd   = sqrtf(sun[0] * sun[0] + sun[2] * sun[2]);
  if (hd < 1e-6f)
    hd = 1e-6f;
  float tanT = sinT / cosT;
  float cs[2]
    = {-sun[0] / hd * CW_DEPTH * tanT, -sun[2] / hd * CW_DEPTH * tanT};

  cw_water_ub_t wu = {
    .cam        = {pos[0], pos[1], pos[2]},
    .R          = {R[0], R[1], R[2]},
    .U          = {U[0], U[1], U[2]},
    .F          = {F[0], F[1], F[2]},
    .sun        = {sun[0], sun[1], sun[2]},
    .tanF       = tanf(CW_VFOV * 0.5f),
    .aspect     = asp,
    .L          = CW_L,
    .depth      = CW_DEPTH,
    .time       = t,
    .rip_size   = CW_RSIZE,
    .rip_center = {state.rip_center[0], state.rip_center[1]},
    .caus_shift = {cs[0], cs[1]},
  };
  cw_ub_write(ctx, state.water_ub, &wu, sizeof(wu));

  WGPURenderPassColorAttachment ca = {
    .view       = state.hdr.view,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.05, 0.12, 0.18, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  WGPURenderPassDescriptor rpd
    = {.colorAttachmentCount = 1, .colorAttachments = &ca};
  WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
  wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.screen_w,
                                   (float)state.screen_h, 0, 1);
  wgpuRenderPassEncoderSetPipeline(rp, state.water_pipe);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.water_bg, 0, NULL);
  cw_fullscreen(rp);
  wgpuRenderPassEncoderEnd(rp);
  wgpuRenderPassEncoderRelease(rp);
}

/* Post-processing: bloom + tonemapping */
static void cw_post(wgpu_context_t* ctx, WGPUCommandEncoder enc, float t)
{
  /* Bright pass: hdrRT → qa */
  {
    WGPURenderPassColorAttachment ca
      = {.view       = state.qa.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.qa.w,
                                     (float)state.qa.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.bright_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.bright_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Blur H: qa → qb */
  {
    WGPURenderPassColorAttachment ca
      = {.view       = state.qb.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.qb.w,
                                     (float)state.qb.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.blur_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.blur_qa_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Blur V: qb → b1 */
  {
    WGPURenderPassColorAttachment ca
      = {.view       = state.b1.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.b1.w,
                                     (float)state.b1.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.blur_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.blur_qb_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Copy b1 → b2 */
  {
    WGPURenderPassColorAttachment ca
      = {.view       = state.b2.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.b2.w,
                                     (float)state.b2.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.copy_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.copy_b1_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
  }

  /* Wide blur ×2 iterations: H b2→b2t, V b2t→b2 */
  for (int i = 0; i < 2; i++) {
    WGPURenderPassColorAttachment ca
      = {.view       = state.b2t.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd
      = {.colorAttachmentCount = 1, .colorAttachments = &ca};
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.b2t.w,
                                     (float)state.b2t.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp, state.blur_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, state.blur_b2_bg, 0, NULL);
    cw_fullscreen(rp);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);

    WGPURenderPassColorAttachment ca2
      = {.view       = state.b2.view,
         .loadOp     = WGPULoadOp_Clear,
         .storeOp    = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 0},
         .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED};
    WGPURenderPassDescriptor rpd2
      = {.colorAttachmentCount = 1, .colorAttachments = &ca2};
    WGPURenderPassEncoder rp2 = wgpuCommandEncoderBeginRenderPass(enc, &rpd2);
    wgpuRenderPassEncoderSetViewport(rp2, 0, 0, (float)state.b2.w,
                                     (float)state.b2.h, 0, 1);
    wgpuRenderPassEncoderSetPipeline(rp2, state.blur_pipe);
    wgpuRenderPassEncoderSetBindGroup(rp2, 0, state.blur_b2t_bg, 0, NULL);
    cw_fullscreen(rp2);
    wgpuRenderPassEncoderEnd(rp2);
    wgpuRenderPassEncoderRelease(rp2);
  }

  /* Final pass → swapchain */
  cw_final_ub_t fu = {
    .exposure = state.settings.exposure,
    .time     = t,
    .no_post  = state.settings.no_post ? 1.0f : 0.0f,
    .res      = {(float)state.screen_w, (float)state.screen_h},
  };
  cw_ub_write(ctx, state.final_ub, &fu, sizeof(fu));

  WGPURenderPassColorAttachment sca = {
    .view       = ctx->swapchain_view,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0, 0, 0, 1},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  };
  WGPURenderPassDescriptor srpd
    = {.colorAttachmentCount = 1, .colorAttachments = &sca};
  WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &srpd);
  wgpuRenderPassEncoderSetViewport(rp, 0, 0, (float)state.screen_w,
                                   (float)state.screen_h, 0, 1);
  wgpuRenderPassEncoderSetPipeline(rp, state.final_pipe);
  wgpuRenderPassEncoderSetBindGroup(rp, 0, state.final_bg, 0, NULL);
  cw_fullscreen(rp);

  wgpuRenderPassEncoderEnd(rp);
  wgpuRenderPassEncoderRelease(rp);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void cw_update_gui(wgpu_context_t* ctx, float dt)
{
  imgui_overlay_new_frame(ctx, dt);
  igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_Once, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){240, 120}, ImGuiCond_Once);
  igBegin("Clear Water", NULL, 0);
  igSliderFloat("Exposure", &state.settings.exposure, 0.1f, 3.0f, "%.2f", 0);
  igCheckbox("No Post", &state.settings.no_post);
  igCheckbox("Pause", &state.settings.paused);
  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* ctx, const input_event_t* ev)
{
  imgui_overlay_handle_input(ctx, ev);
  /* Don't process scene input when ImGui is capturing the mouse */
  if (imgui_overlay_want_capture_mouse())
    return;

  switch (ev->type) {
    case INPUT_EVENT_TYPE_MOUSE_DOWN:
      if (ev->mouse_button == BUTTON_LEFT) {
        state.inp.drag = true;
        state.inp.x0   = ev->mouse_x;
        state.inp.y0   = ev->mouse_y;
      }
      break;
    case INPUT_EVENT_TYPE_MOUSE_UP:
      if (ev->mouse_button == BUTTON_LEFT && state.inp.drag) {
        float dx = ev->mouse_x - state.inp.x0;
        float dy = ev->mouse_y - state.inp.y0;
        if (sqrtf(dx * dx + dy * dy) < 8.0f) {
          state.inp.tap   = true;
          state.inp.tap_x = ev->mouse_x;
          state.inp.tap_y = ev->mouse_y;
        }
      }
      state.inp.drag = false;
      break;
    case INPUT_EVENT_TYPE_MOUSE_MOVE:
      if (state.inp.drag) {
        float k  = 1.2f / (float)MIN(ctx->width, ctx->height);
        float dx = ev->mouse_dx * k;
        float dy = ev->mouse_dy * k;
        state.cam_yaw -= dx;
        state.cam_pitch += dy;
        state.cam_pitch = CLAMP(state.cam_pitch, -1.45f, 0.35f);
        state.cam_vy    = -dx;
        state.cam_vp    = dy;
      }
      break;
    case INPUT_EVENT_TYPE_RESIZED:
      /* Screen-sized resources are recreated in frame() */
      break;
    default:
      break;
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* ctx)
{
  sfetch_desc_t fd = {0};
#ifndef __WAJIC__
  fd.logger.func = slog_func;
#endif
  sfetch_setup(&fd);
  stm_setup();

  cw_init_fixed(ctx);
  cw_alloc_screen_rts(ctx);
  imgui_overlay_init(ctx);

  /* Start loading the pebbles texture */
  state.peb_file_buf = (uint8_t*)malloc(CW_PEB_FILE_BUF_SIZE);
  static wgpu_context_t* ctx_ptr;
  ctx_ptr = ctx;
  sfetch_send(&(sfetch_request_t){
    .path      = "assets/textures/clearwater_pebbles.jpg",
    .callback  = cw_peb_fetch_callback,
    .buffer    = {.ptr = state.peb_file_buf, .size = CW_PEB_FILE_BUF_SIZE},
    .user_data = {.ptr = &ctx_ptr, .size = sizeof(ctx_ptr)},
  });

  state.last_ns     = stm_now();
  state.initialized = (WGPUBool) true;
  return 0;
}

static int frame(wgpu_context_t* ctx)
{
  sfetch_dowork();

  /* Wait for pebbles texture before rendering */
  if (!state.peb_loaded)
    return 0;

  /* Static BGL for water-shader re-build (only needed for screen resize) */
  static WGPUBindGroupLayout water_bgl_cache = NULL;
  (void)water_bgl_cache; /* managed inside cw_init_screen_bgs */

  /* Handle resize */
  if (ctx->width != state.screen_w || ctx->height != state.screen_h) {
    cw_alloc_screen_rts(ctx);
    cw_init_screen_bgs(ctx, NULL);
  }
  /* First-time screen BG setup (after pebbles loaded) */
  if (!state.water_bg) {
    cw_init_screen_bgs(ctx, NULL);
  }
  if (!state.water_bg)
    return 0;

  /* Update time */
  if (!state.settings.paused) {
    uint64_t now = stm_now();
    float dt     = (float)stm_sec(stm_laptime(&state.last_ns));
    dt           = CLAMP(dt, 0.0f, 0.05f);
    state.t_sim += dt;
    (void)now;
  }
  else {
    state.last_ns = stm_now();
  }
  float t = state.t_sim;

  /* Camera inertia */
  if (!state.inp.drag) {
    state.cam_yaw += state.cam_vy * 0.9f;
    state.cam_pitch += state.cam_vp * 0.9f;
    state.cam_pitch = CLAMP(state.cam_pitch, -1.45f, 0.35f);
    state.cam_vy *= 0.9f;
    state.cam_vp *= 0.9f;
  }

  /* Camera basis for ripple centering and tap */
  float F[3], R[3], U[3], pos[3];
  cw_compute_cam_basis(t, F, R, U, pos);

  /* Handle tap → drop */
  if (state.inp.tap) {
    cw_tap_to_drop(state.inp.tap_x, state.inp.tap_y, ctx->width, ctx->height, F,
                   R, U, pos);
    state.inp.tap = false;
  }

  /* Advance ripple centre to follow look-point */
  float look    = -pos[1] / CLAMP(-F[1], 0.2f, 1.0f);
  float want[2] = {pos[0] + F[0] * look * 0.9f, pos[2] + F[2] * look * 0.9f};
  float tx      = CW_RSIZE / (float)CW_RN;
  int dxT       = (int)roundf((want[0] - state.rip_center[0]) / tx);
  int dzT       = (int)roundf((want[1] - state.rip_center[1]) / tx);
  float shift_u = (float)dxT / (float)CW_RN;
  float shift_v = (float)dzT / (float)CW_RN;

  /* Sun direction (constant) */
  float sun[3] = {
    sinf(CW_SUN_AZ) * cosf(CW_SUN_EL),
    sinf(CW_SUN_EL),
    -cosf(CW_SUN_AZ) * cosf(CW_SUN_EL),
  };

  /* Update GUI */
  float dt_gui = (!state.settings.paused) ?
                   (float)stm_sec(stm_diff(stm_now(), state.last_ns)) :
                   0.016f;
  cw_update_gui(ctx, dt_gui);

  /* Build command buffer */
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);

  /* 1. FFT ocean simulation */
  cw_run_fft(ctx, enc, t);

  /* 2. Ripple simulation */
  if (state.rip_active < 900) {
    cw_step_ripples(ctx, enc, shift_u, shift_v);
    state.rip_center[0] += (float)dxT * tx;
    state.rip_center[1] += (float)dzT * tx;
    state.rip_active++;
  }
  else {
    state.rip_center[0] += (float)dxT * tx;
    state.rip_center[1] += (float)dzT * tx;
  }

  /* 3. Caustics */
  cw_render_caustics(ctx, enc, sun);

  /* 4. Main water render */
  cw_render_water(ctx, enc, t, sun);

  /* 5. Post-processing */
  cw_post(ctx, enc, t);

  /* Submit */
  WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(ctx->queue, 1, &cb);
  wgpuCommandBufferRelease(cb);
  wgpuCommandEncoderRelease(enc);

  /* Render ImGui on top of the final frame */
  imgui_overlay_render(ctx);

  return 0;
}

static void shutdown(wgpu_context_t* ctx)
{
  (void)ctx;
  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Fixed resources */
  WGPU_RELEASE_RESOURCE(TextureView, state.h0_view)
  WGPU_RELEASE_RESOURCE(Texture, state.h0_tex)
  for (int i = 0; i < 2; i++) {
    cw_rt_destroy(&state.fft[i]);
  }
  cw_rt_destroy(&state.surf);
  for (int i = 0; i < 2; i++) {
    cw_rt_destroy(&state.rip[i]);
  }
  cw_rt_destroy(&state.rip_n);
  cw_rt_destroy(&state.caus);
  cw_rt_destroy(&state.streak);

  /* Screen-sized */
  cw_rt_destroy(&state.hdr);
  cw_rt_destroy(&state.qa);
  cw_rt_destroy(&state.qb);
  cw_rt_destroy(&state.b1);
  cw_rt_destroy(&state.b2);
  cw_rt_destroy(&state.b2t);

  /* Samplers */
  WGPU_RELEASE_RESOURCE(Sampler, state.surf_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.surf_aniso)
  WGPU_RELEASE_RESOURCE(Sampler, state.rip_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.caus_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.peb_sampler)
  WGPU_RELEASE_RESOURCE(Sampler, state.screen_sampler)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.spec_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.fft_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.resolve_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.ripple_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.ripn_pipe)
  for (int c = 0; c < 3; c++)
    WGPU_RELEASE_RESOURCE(RenderPipeline, state.caus_pipe[c])
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.water_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.bright_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blur_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.copy_pipe)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.final_pipe)

  /* Caustics geometry */
  WGPU_RELEASE_RESOURCE(Buffer, state.caus_vb)
  WGPU_RELEASE_RESOURCE(Buffer, state.caus_ib)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.spec_ub)
  for (int i = 0; i < 16; i++)
    WGPU_RELEASE_RESOURCE(Buffer, state.fft_ub[i])
  WGPU_RELEASE_RESOURCE(Buffer, state.ripple_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.ripn_ub)
  for (int c = 0; c < 3; c++)
    WGPU_RELEASE_RESOURCE(Buffer, state.caus_ub[c])
  WGPU_RELEASE_RESOURCE(Buffer, state.water_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.bright_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_h_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_v_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_h2_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_v2_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.copy_ub)
  WGPU_RELEASE_RESOURCE(Buffer, state.final_ub)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.spec_bg)
  for (int i = 0; i < 16; i++)
    WGPU_RELEASE_RESOURCE(BindGroup, state.fft_bg[i])
  WGPU_RELEASE_RESOURCE(BindGroup, state.resolve_bg)
  for (int i = 0; i < 2; i++)
    WGPU_RELEASE_RESOURCE(BindGroup, state.ripple_bg[i])
  for (int i = 0; i < 2; i++)
    WGPU_RELEASE_RESOURCE(BindGroup, state.ripn_bg[i])
  for (int c = 0; c < 3; c++)
    WGPU_RELEASE_RESOURCE(BindGroup, state.caus_bg[c])
  WGPU_RELEASE_RESOURCE(BindGroup, state.water_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bright_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_qa_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_qb_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b1_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b2_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_b2t_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.copy_b1_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.final_bg)

  /* Pebbles texture */
  wgpu_destroy_texture(&state.peb_tex);
  free(state.peb_file_buf);
}

/* -------------------------------------------------------------------------- *
 * main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title           = "Clear Water — Real-time Shallow Water (WebGPU)",
    .width           = 1280,
    .height          = 720,
    .no_depth_buffer = (WGPUBool) true,
    .init_cb         = init,
    .frame_cb        = frame,
    .shutdown_cb     = shutdown,
    .input_event_cb  = input_event_cb,
  });
  return 0;
}

/* ========================================================================== *
 * WGSL Shaders
 * ========================================================================== */

/* Common vertex shader (fullscreen triangle via vertex_index) */
#define CW_VS                                                                  \
  "struct VsOut {\n"                                                           \
  "  @builtin(position) pos: vec4f,\n"                                         \
  "  @location(0) uv: vec2f,\n"                                                \
  "}\n"                                                                        \
  "@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {\n"            \
  "  let x = f32((vi << 1u) & 2u);\n"                                          \
  "  let y = f32(vi & 2u);\n"                                                  \
  "  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));\n"      \
  "}\n"

/* Combined water shader */
static const char* cw_get_water_shader(void)
{
  snprintf(cw_water_shader_buf, sizeof(cw_water_shader_buf), "%s%s%s",
           cw_water_shader_part1, cw_water_shader_part2, cw_water_shader_part3);
  return cw_water_shader_buf;
}

// clang-format off

/* -------------------------------------------------------------------------- *
 * Spectrum shader
 * -------------------------------------------------------------------------- */
static const char* cw_spectrum_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uH0 : texture_2d<f32>;
struct SpecParams { t: f32, L: f32, pad: vec2f }
@group(0) @binding(1) var<uniform> p: SpecParams;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let id = vec2i(in.pos.xy);
  let s  = textureLoad(uH0, id, 0);
  var n  = vec2f(f32(id.x), f32(id.y));
  n -= step(vec2f(128.0), n) * 256.0;
  let k  = 6.28318530718 * n / p.L;
  let kl = length(k);
  var w  = sqrt(9.81*kl + 7.4e-5*kl*kl*kl);
  let w0 = 6.28318530718 / 60.0;
  w = floor(w/w0)*w0;
  let c  = cos(w*p.t); let sn = sin(w*p.t);
  let H  = cmul(s.xy, vec2f(c, sn)) + cmul(s.zw, vec2f(c, -sn));
  let C1 = H - k.x*H;
  let C2 = vec2f(-k.y*H.y, k.y*H.x);
  return vec4f(C1, C2);
}

);

/* -------------------------------------------------------------------------- *
 * FFT butterfly shader
 * -------------------------------------------------------------------------- */
static const char* cw_fft_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc : texture_2d<f32>;
struct FftParams { P: i32, horiz: i32, half_n: i32, pad: i32 }
@group(0) @binding(1) var<uniform> p: FftParams;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let id = vec2i(in.pos.xy);
  let j  = select(id.y, id.x, p.horiz == 1);
  let k  = j & (p.P - 1);
  let i  = ((j - (j & (2*p.P - 1))) >> 1) + k;
  let y1 = (j & p.P) != 0;
  let a  = select(vec2i(id.x, i), vec2i(i, id.y), p.horiz == 1);
  let b  = select(vec2i(id.x, i + p.half_n), vec2i(i + p.half_n, id.y), p.horiz == 1);
  let x0 = textureLoad(uSrc, a, 0);
  let x1 = textureLoad(uSrc, b, 0);
  let ang = 3.14159265359 * f32(k) / f32(p.P);
  let w  = vec2f(cos(ang), sin(ang));
  let wx = vec4f(cmul(w, x1.xy), cmul(w, x1.zw));
  return select(x0 + wx, x0 - wx, y1);
}

);

/* -------------------------------------------------------------------------- *
 * Resolve shader  (height, slope.x, slope.z, slope_variance)
 * -------------------------------------------------------------------------- */
static const char* cw_resolve_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc : texture_2d<f32>;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let s  = textureLoad(uSrc, vec2i(in.pos.xy), 0);
  let sl = vec2f(s.y, s.z);
  return vec4f(s.x, sl, dot(sl, sl));
}

);

/* -------------------------------------------------------------------------- *
 * Ripple wave-equation step
 * -------------------------------------------------------------------------- */
static const char* cw_ripple_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc  : texture_2d<f32>;
@group(0) @binding(1) var uSamp : sampler;
struct RipParams { shift: vec2f, pad: vec2f, drop: vec4f }
@group(0) @binding(2) var<uniform> p: RipParams;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let px  = 1.0 / vec2f(textureDimensions(uSrc));
  let uv  = in.uv + p.shift;
  let c   = textureSample(uSrc, uSamp, uv);
  let avg = 0.25*(
    textureSample(uSrc, uSamp, uv + vec2f( px.x, 0.0)).x +
    textureSample(uSrc, uSamp, uv + vec2f(-px.x, 0.0)).x +
    textureSample(uSrc, uSamp, uv + vec2f(0.0,  px.y)).x +
    textureSample(uSrc, uSamp, uv + vec2f(0.0, -px.y)).x);
  var v = c.y + (avg - c.x)*0.9;
  v *= 0.9955;
  var h = c.x + v;
  h *= 0.9985;
  if (p.drop.w != 0.0) {
    let d = length(in.uv - p.drop.xy);
    let r = p.drop.z;
    if (d < r) { let f = 0.5 + 0.5*cos(3.14159265*d/r); h -= p.drop.w*f; }
  }
  let e = min(in.uv, 1.0 - in.uv);
  let edge = smoothstep(0.0, 0.06, min(e.x, e.y));
  h *= mix(0.9, 1.0, edge); v *= mix(0.9, 1.0, edge);
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { h = 0.0; v = 0.0; }
  return vec4f(h, v, 0.0, 1.0);
}

);

/* -------------------------------------------------------------------------- *
 * Ripple normals
 * -------------------------------------------------------------------------- */
static const char* cw_ripn_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc  : texture_2d<f32>;
@group(0) @binding(1) var uSamp : sampler;
struct RipNParams { texel: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(2) var<uniform> p: RipNParams;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let px = 1.0 / vec2f(textureDimensions(uSrc));
  let hx = textureSample(uSrc, uSamp, in.uv + vec2f( px.x, 0.0)).x
         - textureSample(uSrc, uSamp, in.uv + vec2f(-px.x, 0.0)).x;
  let hz = textureSample(uSrc, uSamp, in.uv + vec2f(0.0,  px.y)).x
         - textureSample(uSrc, uSamp, in.uv + vec2f(0.0, -px.y)).x;
  let h   = textureSample(uSrc, uSamp, in.uv).x;
  let lap = (textureSample(uSrc, uSamp, in.uv + vec2f( px.x, 0.0)).x +
             textureSample(uSrc, uSamp, in.uv + vec2f(-px.x, 0.0)).x +
             textureSample(uSrc, uSamp, in.uv + vec2f(0.0,  px.y)).x +
             textureSample(uSrc, uSamp, in.uv + vec2f(0.0, -px.y)).x - 4.0*h)
           / (p.texel * p.texel);
  return vec4f(h, hx/(2.0*p.texel), hz/(2.0*p.texel), lap);
}

);

/* -------------------------------------------------------------------------- *
 * Caustics (grid mesh, instanced 3x3, additive blend, per-channel mask)
 * -------------------------------------------------------------------------- */
static const char* cw_caus_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) src: vec2f, }

@group(0) @binding(0) var uSurf  : texture_2d<f32>;
@group(0) @binding(1) var uSamp  : sampler;
struct CausParams { L: f32, depth: f32, ior: f32, norm: f32,
                   sun: vec3f, pad0: f32, shift: vec2f, pad1: vec2f }
@group(0) @binding(2) var<uniform> p: CausParams;

@vertex fn vs_main(
  @location(0) aUV: vec2f,
  @builtin(instance_index) inst_idx: u32
) -> VsOut {
  let off = vec2i(i32(inst_idx) % 3 - 1, i32(inst_idx) / 3 - 1);
  let s   = textureSampleLevel(uSurf, uSamp, aUV, 0.0);
  let n   = normalize(vec3f(-s.y, 1.0, -s.z));
  let r   = refract(-p.sun, n, 1.0/p.ior);
  let P   = vec3f(aUV.x*p.L, s.x, aUV.y*p.L);
  let F   = P + r*((-p.depth - s.x)/r.y);
  let c   = (F.xz - p.shift)/p.L + vec2f(f32(off.x), f32(off.y));
  return VsOut(vec4f(c*2.0-1.0, 0.0, 1.0), aUV*p.L);
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let a    = dpdx(in.src); let b = dpdy(in.src);
  let area = abs(a.x*b.y - a.y*b.x);
  let I    = min(area*p.norm, 40.0);
  return vec4f(I, I, I, I);
}

);

/* -------------------------------------------------------------------------- *
 * Bright pass
 * -------------------------------------------------------------------------- */
static const char* cw_bright_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc  : texture_2d<f32>;
@group(0) @binding(1) var uSamp : sampler;
struct BrightParams { threshold: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(2) var<uniform> p: BrightParams;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let px = 1.0 / vec2f(textureDimensions(uSrc));
  var c  = vec3f(0.0);
  for (var dy: i32 = -1; dy <= 2; dy++) {
    for (var dx: i32 = -1; dx <= 2; dx++) {
      c += textureSample(uSrc, uSamp, in.uv + (vec2f(f32(dx), f32(dy))-0.5)*px).rgb;
    }
  }
  c /= 16.0;
  let l = max(max(c.r, c.g), c.b);
  let k = max(l - p.threshold, 0.0) / max(l, 1e-4);
  return vec4f(min(c*k, vec3f(160.0)), 1.0);
}

);

/* -------------------------------------------------------------------------- *
 * Gaussian blur (separable 5-tap)
 * -------------------------------------------------------------------------- */
static const char* cw_blur_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc  : texture_2d<f32>;
@group(0) @binding(1) var uSamp : sampler;
struct BlurParams { dir: vec2f, pad: vec2f }
@group(0) @binding(2) var<uniform> p: BlurParams;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  let px = p.dir / vec2f(textureDimensions(uSrc));
  var c  = textureSample(uSrc, uSamp, in.uv).rgb * 0.2270270270;
  c += (textureSample(uSrc, uSamp, in.uv + px*1.3846153846).rgb +
        textureSample(uSrc, uSamp, in.uv - px*1.3846153846).rgb) * 0.3162162162;
  c += (textureSample(uSrc, uSamp, in.uv + px*3.2307692308).rgb +
        textureSample(uSrc, uSamp, in.uv - px*3.2307692308).rgb) * 0.0702702703;
  return vec4f(c, 1.0);
}

);

/* -------------------------------------------------------------------------- *
 * Copy / scale pass
 * -------------------------------------------------------------------------- */
static const char* cw_copy_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSrc  : texture_2d<f32>;
@group(0) @binding(1) var uSamp : sampler;
struct CopyParams { k: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(2) var<uniform> p: CopyParams;

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  return vec4f(textureSample(uSrc, uSamp, in.uv).rgb * p.k, 1.0);
}

);

/* -------------------------------------------------------------------------- *
 * Final tonemapping + film grain
 * -------------------------------------------------------------------------- */
static const char* cw_final_shader_wgsl = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uHdr    : texture_2d<f32>;
@group(0) @binding(1) var uHdrS   : sampler;
@group(0) @binding(2) var uStreak : texture_2d<f32>;
@group(0) @binding(3) var uStrkS  : sampler;
@group(0) @binding(4) var uB1     : texture_2d<f32>;
@group(0) @binding(5) var uB1S    : sampler;
@group(0) @binding(6) var uB2     : texture_2d<f32>;
@group(0) @binding(7) var uB2S    : sampler;
struct FinalParams { exposure: f32, time: f32, no_post: f32, pad0: f32, res: vec2f, pad1: vec2f }
@group(0) @binding(8) var<uniform> fp: FinalParams;

fn aces(x: vec3f) -> vec3f {
  const a = 2.51; const b = 0.03; const c = 2.43; const d = 0.59; const e = 0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3f(0.0), vec3f(1.0));
}
fn hash_g(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x)*0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y)*p3.z);
}
fn bicubic(uv: vec2f, t: texture_2d<f32>, s: sampler) -> vec3f {
  let ts = vec2f(textureDimensions(t));
  let p  = uv*ts - 0.5; let f = fract(p); let pp = floor(p);
  let w0 = f*(-0.5+f*(1.0-0.5*f)); let w1 = 1.0+f*f*(-2.5+1.5*f);
  let w2 = f*(0.5+f*(2.0-1.5*f));  let w3 = f*f*(-0.5+0.5*f);
  let g0 = w0+w1; let g1 = w2+w3;
  let h0 = (w1/g0 - 0.5 + pp)/ts; let h1 = (w3/g1 + 1.5 + pp)/ts;
  return (textureSample(t,s,vec2f(h0.x,h0.y)).rgb*g0.x + textureSample(t,s,vec2f(h1.x,h0.y)).rgb*g1.x)*g0.y
       + (textureSample(t,s,vec2f(h0.x,h1.y)).rgb*g0.x + textureSample(t,s,vec2f(h1.x,h1.y)).rgb*g1.x)*g1.y;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  var uv = in.uv;
  let cc = uv - 0.5;
  let ca = 0.0012*dot(cc,cc)*4.0;
  var c: vec3f;
  c.r = textureSample(uHdr, uHdrS, uv + cc*ca).r;
  c.g = textureSample(uHdr, uHdrS, uv).g;
  c.b = textureSample(uHdr, uHdrS, uv - cc*ca).b;
  if (fp.no_post < 0.5) { c += textureSample(uStreak, uStrkS, uv).rgb * 0.9; }
  if (fp.no_post < 0.5) {
    c += textureSample(uB1, uB1S, uv).rgb * 0.035 + bicubic(uv, uB2, uB2S) * 0.035;
  }
  c *= fp.exposure;
  let vig = 1.0 - 0.22*dot(cc*vec2f(1.0,0.8), cc*vec2f(1.0,0.8))*2.2;
  c *= vig;
  c = aces(c);
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = mix(vec3f(lum), c, 0.90);
  c = mix(c, c*vec3f(0.96, 1.0, 1.05), 1.0 - smoothstep(0.0, 0.35, lum));
  c = pow(c, vec3f(1.0/2.2));
  let g = hash_g(in.pos.xy + fract(fp.time*7.13)*917.0) - 0.5;
  c += g * 0.018 * (1.0 - c*0.6);
  return vec4f(c, 1.0);
}

);

/* -------------------------------------------------------------------------- *
 * Main water shader — part 1: structs, vertex, utility functions
 * -------------------------------------------------------------------------- */
static const char* cw_water_shader_part1 = CODE(

struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let x = f32((vi << 1u) & 2u); let y = f32(vi & 2u);
  return VsOut(vec4f(x*2.0-1.0, 1.0-y*2.0, 0.0, 1.0), vec2f(x, y));
}

@group(0) @binding(0) var uSurf   : texture_2d<f32>;
@group(0) @binding(1) var uSurfS  : sampler;
@group(0) @binding(2) var uCaus   : texture_2d<f32>;
@group(0) @binding(3) var uCausS  : sampler;
@group(0) @binding(4) var uPeb    : texture_2d<f32>;
@group(0) @binding(5) var uPebS   : sampler;
@group(0) @binding(6) var uRip    : texture_2d<f32>;
@group(0) @binding(7) var uRipS   : sampler;

struct WaterParams {
  cam: vec3f, pad0: f32,
  R:   vec3f, pad1: f32,
  U:   vec3f, pad2: f32,
  F:   vec3f, pad3: f32,
  sun: vec3f, pad4: f32,
  tanF: f32, aspect: f32, L: f32, depth: f32,
  time: f32, rip_size: f32, rip_center: vec2f,
  caus_shift: vec2f, pad5: vec2f,
}
@group(0) @binding(8) var<uniform> wp: WaterParams;

const PI = 3.14159265359;
const IOR = 1.3335;
const SIG_A = vec3f(0.40, 0.074, 0.088);
const SIG_S = vec3f(0.028, 0.052, 0.068);
const SIG_T = SIG_A + SIG_S;
const SUN_C = vec3f(1.0, 0.90, 0.74) * 6.0;

fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x)*0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y)*p3.z);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f*f*(3.0 - 2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2f(1,0)), u.x),
             mix(hash12(i+vec2f(0,1)), hash12(i+vec2f(1,1)), u.x), u.y);
}
fn ridge(a: f32) -> f32 {
  return 0.040 + 0.016*sin(a*2.0+0.7) + 0.011*sin(a*5.0+2.1)
               + 0.006*sin(a*11.0+0.3) + 0.003*sin(a*23.0+1.7);
}
fn fbm2(pin: vec2f) -> f32 {
  var p = pin; var v = 0.0; var am = 0.5;
  for (var i = 0; i < 4; i++) { v += am*vnoise(p); p = p*2.03+17.1; am *= 0.5; }
  return v;
}
fn sky(d: vec3f) -> vec3f {
  let e  = d.y;
  let mu = dot(d, wp.sun);
  let zen = vec3f(0.11, 0.27, 0.62); let hor = vec3f(0.66, 0.78, 0.90);
  var c  = mix(hor, zen, pow(clamp(e, 0.0, 1.0), 0.42));
  c += vec3f(1.0,0.86,0.66)*(0.22*pow(max(mu,0.0),6.0)+0.30*pow(max(mu,0.0),64.0)+1.6*pow(max(mu,0.0),2400.0));
  let a  = atan2(d.z, d.x);
  let r  = ridge(a) + 0.0045*(vnoise(vec2f(a*260.0, 0.0))-0.5) + 0.002*(vnoise(vec2f(a*900.0, 3.0))-0.5);
  let back = smoothstep(-0.3, 0.95, dot(normalize(vec2f(d.x,d.z)+1e-5), normalize(vec2f(wp.sun.x,wp.sun.z))));
  let u  = clamp(e / max(r, 1e-3), 0.0, 1.0);
  let q  = vec2f(a*420.0, e*420.0);
  let tex2 = fbm2(q);
  let pine = vec3f(0.045,0.070,0.042)*(0.6+0.8*tex2);
  let rock = vec3f(0.30,0.28,0.23)*(0.55+0.7*fbm2(q*1.7+5.0));
  let cliff = smoothstep(0.42,0.18, u + 0.25*(tex2-0.5)) * smoothstep(0.35,0.75, vnoise(vec2f(a*18.0, 1.0)));
  var land = mix(pine, rock, cliff);
  land *= mix(1.0, 0.45, back);
  land = mix(land, hor*0.92, 0.38 + 0.25*back);
  let w2 = fwidth(e)*1.2 + 2e-4;
  c = mix(c, land, smoothstep(r+w2, r-w2, e) * step(-0.3, e));
  return c;
}

);

/* -------------------------------------------------------------------------- *
 * Main water shader — part 2: more helpers + fragment start
 * -------------------------------------------------------------------------- */
static const char* cw_water_shader_part2 = CODE(

fn texBS(uv: vec2f) -> vec4f {
  let ts = vec2f(textureDimensions(uSurf));
  let p  = uv*ts - 0.5; let f = fract(p); let pp = floor(p);
  let f2 = f*f; let f3 = f2*f;
  let w0 = (-f3+3.0*f2-3.0*f+1.0)/6.0; let w1 = (3.0*f3-6.0*f2+4.0)/6.0;
  let w2 = (-3.0*f3+3.0*f2+3.0*f+1.0)/6.0; let w3 = f3/6.0;
  let g0 = w0+w1; let g1 = w2+w3;
  let h0 = (w1/g0 - 0.5 + pp)/ts; let h1 = (w3/g1 + 1.5 + pp)/ts;
  return (textureSample(uSurf,uSurfS,vec2f(h0.x,h0.y))*g0.x + textureSample(uSurf,uSurfS,vec2f(h1.x,h0.y))*g1.x)*g0.y
       + (textureSample(uSurf,uSurfS,vec2f(h0.x,h1.y))*g0.x + textureSample(uSurf,uSurfS,vec2f(h1.x,h1.y))*g1.x)*g1.y;
}
fn floorDepth(xz: vec2f) -> f32 {
  let shelf = 0.95 + 0.17*clamp(-xz.y + 1.5, 0.0, 14.0);
  return shelf + 0.30*(vnoise(xz*0.22)-0.5) + 0.10*(vnoise(xz*0.9+7.0)-0.5);
}

struct PebOut { col: vec3f, hgt: f32 }
fn pebbles(x: vec2f, sc: f32) -> PebOut {
  let uv  = x / (vec2f(0.78)*sc);
  let dx  = dpdx(uv); let dy = dpdy(uv);
  let k   = vnoise(x*0.85);
  let l   = k*8.0; let ia = floor(l); let f = fract(l);
  let oa  = sin(vec2f(3.0,7.0)*ia); let ob = sin(vec2f(3.0,7.0)*(ia+1.0));
  let a   = textureSampleGrad(uPeb, uPebS, uv+oa, dx, dy).rgb;
  let b   = textureSampleGrad(uPeb, uPebS, uv+ob, dx, dy).rgb;
  let s   = dot(a-b, vec3f(1.0));
  let m   = smoothstep(0.2, 0.8, f - 0.1*s);
  let ca  = textureSampleGrad(uPeb, uPebS, uv+oa, dx*6.0, dy*6.0).rgb;
  let cb  = textureSampleGrad(uPeb, uPebS, uv+ob, dx*6.0, dy*6.0).rgb;
  let hgt = dot(mix(ca,cb,m), vec3f(0.3,0.55,0.15));
  return PebOut(mix(a,b,m), hgt);
}
fn fresnel(ci: f32, n: f32) -> f32 {
  let ci2 = clamp(ci, 0.0, 1.0);
  let st2 = (1.0-ci2*ci2)/(n*n);
  if (st2 >= 1.0) { return 1.0; }
  let ct  = sqrt(1.0-st2);
  let rs  = (ci2-n*ct)/(ci2+n*ct); let rp = (n*ci2-ct)/(n*ci2+ct);
  return 0.5*(rs*rs + rp*rp);
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4f {
  /* WebGPU UV convention: V=0 is at the top of the render target (row 0),
   * so ndc.y = 1 - 2*uv.y  (not uv.y*2-1, which would flip Y). */
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  var rd  = normalize(wp.F + ndc.x*wp.aspect*wp.tanF*wp.R + ndc.y*wp.tanF*wp.U);
  var wd  = rd; wd.y = min(wd.y, -0.0015); wd = normalize(wd);

  var t = -wp.cam.y / wd.y;
  let M  = mat2x2f(0.8,-0.6, 0.6,0.8);
  const SC = 0.41; const WB = 0.10;
  var xz = vec2f(0.0); var A = vec4f(0.0); var B = vec4f(0.0); var R = vec4f(0.0);
  var hsum = 0.0;
  for (var i = 0; i < 3; i++) {
    xz = wp.cam.xz + wd.xz*t;
    A  = textureSample(uSurf, uSurfS, xz/wp.L);
    B  = textureSample(uSurf, uSurfS, (M*xz)/(wp.L*SC) + 0.37);
    let ruv = (xz - wp.rip_center)/wp.rip_size + 0.5;
    R  = textureSample(uRip, uRipS, ruv);
    hsum = A.x + WB*SC*B.x + R.x;
    t = (hsum - wp.cam.y) / wd.y;
  }
  let P  = wp.cam + wd*t;
  A = texBS(P.xz/wp.L);
  B = texBS((M*P.xz)/(wp.L*SC) + 0.37);
  var slope = A.yz + WB*(transpose(M)*B.yz) + R.yz;
  let M2 = mat2x2f(0.28,0.96, -0.96,0.28);
  let Cm = textureSample(uSurf, uSurfS, (M2*P.xz)/(wp.L*0.13) + 0.71);
  slope += 0.13*exp(-t*0.18)*(transpose(M2)*Cm.yz);
  let vr = max(A.w - dot(A.yz,A.yz), 0.0) + WB*WB*max(B.w - dot(B.yz,B.yz), 0.0);
  let n  = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let dist = t;

  let v  = -wd;
  let nv0 = dot(n, v);
  var nn = n;
  if (nv0 < 0.02) { nn = normalize(n + v*(0.02-nv0)); }
  let nv = dot(nn, v);
  let F  = fresnel(nv, IOR);

  let rr0  = reflect(wd, nn); let rr = vec3f(rr0.x, abs(rr0.y), rr0.z);
  let refl = sky(rr) * 1.25;

  let a2 = 0.00012 + 1.2*vr;
  let h  = normalize(v + wp.sun);
  let nh = max(dot(nn,h), 0.0); let nl = max(dot(nn,wp.sun), 0.0);
  let c2 = max(nh*nh, 1e-4); let tan2 = (1.0-c2)/c2;
  let D  = exp(-tan2/a2)/(PI*a2*c2*c2);
  let Vis = 0.5/(nl*sqrt(nv*nv*(1.0-a2)+a2) + nv*sqrt(nl*nl*(1.0-a2)+a2) + 1e-5);
  let Fh = fresnel(max(dot(h,v), 0.0), IOR);
  let spec = SUN_C * min(D*Vis*Fh*nl, 12000.0);

);

/* -------------------------------------------------------------------------- *
 * Main water shader — part 3: underwater, in-scattering, final
 * -------------------------------------------------------------------------- */
static const char* cw_water_shader_part3 = CODE(

  let tr  = refract(wd, nn, 1.0/IOR);
  var fy  = -floorDepth(P.xz);
  var s   = (fy - P.y)/tr.y; var FP = P + tr*s;
  for (var i = 0; i < 2; i++) { fy = -floorDepth(FP.xz); s = (fy - P.y)/tr.y; FP = P + tr*s; }
  s = max(s, 0.0);

  let pf  = pebbles(FP.xz, 1.0);
  let pc  = pebbles(FP.xz.yx*vec2f(-1.0,1.0) + 5.3, 1.7);
  let coarse = smoothstep(0.45, 0.62, fbm2(FP.xz*0.21 + 40.0));
  var alb = mix(pf.col, pc.col, coarse); var hgt = mix(pf.hgt, pc.hgt, coarse);
  let zone  = fbm2(FP.xz*0.16 + 3.0) + 0.10*(vnoise(FP.xz*2.5)-0.5);
  let sandM = smoothstep(hgt+0.02, hgt+0.16, (zone-0.46)*1.6);
  let marks = 0.5 + 0.5*sin(dot(FP.xz, vec2f(0.93,0.37))*16.0 + 3.0*vnoise(FP.xz*0.8));
  let sand  = pow(vec3f(0.60,0.55,0.44)*(0.82+0.22*vnoise(FP.xz*40.0)+0.10*marks), vec3f(2.0)) * 1.4;
  alb = mix(alb, sand, sandM); hgt = mix(hgt, 0.42+0.05*marks, sandM);
  alb = mix(vec3f(dot(alb, vec3f(0.3,0.55,0.15))), alb, 0.8) * vec3f(1.10,1.0,0.86);
  let big  = vnoise(FP.xz*0.45)*0.65 + vnoise(FP.xz*1.3+3.1)*0.35;
  let weed = smoothstep(0.55, 0.85, vnoise(FP.xz*0.32+11.0));
  alb *= mix(0.62, 1.22, big);
  alb = mix(alb, alb*vec3f(0.55,0.62,0.40), weed*0.7);
  alb = mix(vec3f(0.30,0.29,0.27), pow(alb, vec3f(1.2)), 0.72) * 0.6;

  let sunT = refract(-wp.sun, vec3f(0,1,0), 1.0/IOR);
  let Ts   = 1.0 - fresnel(wp.sun.y, IOR);
  let depthHere = max(P.y - FP.y, 0.0);
  let cuv  = (FP.xz - wp.caus_shift + sunT.xz/(-sunT.y)*(hgt-0.35)*0.05) / wp.L;
  let caus = textureSampleBias(uCaus, uCausS, cuv, 1.0).rgb;
  let S2   = FP.xz - sunT.xz*depthHere/(-sunT.y);
  let lap  = textureSample(uRip, uRipS, (S2-wp.rip_center)/wp.rip_size+0.5).a;
  let caus2 = caus * clamp(1.0/(1.0 + 0.12*depthHere*lap), 0.45, 3.0);
  let ao   = mix(0.55, 1.0, smoothstep(0.08, 0.42, hgt));
  let Esun = SUN_C * Ts * exp(-SIG_T*depthHere/(-sunT.y)) * caus2 * (-sunT.y) * mix(0.75, 1.0, ao);
  let skyIrr = vec3f(0.62,0.70,0.78) * PI * 0.22;
  let Esky = skyIrr * exp(-(SIG_A+0.4*SIG_S)*depthHere*1.25) * ao;
  let Lfloor = alb/PI * (Esun + Esky);

  let Tv   = exp(-SIG_T*s);
  let cosS = dot(sunT, -tr);
  let g    = 0.8; let ph = (1.0-g*g)/(4.0*PI*pow(1.0+g*g-2.0*g*cosS, 1.5));
  let Lmid = SUN_C*Ts*exp(-SIG_T*depthHere*0.5/(-sunT.y))*(ph+0.02) + skyIrr*exp(-SIG_A*depthHere*0.6)/(4.0*PI);
  let Lin  = SIG_S/SIG_T * Lmid * (1.0 - Tv) * 3.2;
  var under = Lfloor*Tv + Lin;

  for (var k = 0; k < 3; k++) {
    let dz  = 0.22 + 0.38*f32(k);
    let tt  = dz / max(-tr.y, 0.05);
    let q2  = (P.xz + tr.xz*tt)*48.0 + vec2f(wp.time*(0.05+0.03*f32(k)), wp.time*0.02) + f32(k)*17.0;
    let id2 = floor(q2); let ff = fract(q2) - 0.5;
    let rr2 = hash12(id2 + f32(k)*13.1);
    let of2 = vec2f(hash12(id2+3.1), hash12(id2+7.7)) - 0.5;
    let fw  = fwidth(q2.x) + fwidth(q2.y);
    let dot_ = smoothstep(0.10+fw, 0.0, length(ff-of2*0.6)) * step(0.988, rr2) * step(tt, s);
    let fade = exp(-SIG_T.g*tt*2.0) * smoothstep(1.2, 0.3, fw);
    under += dot_ * fade * SUN_C * Ts * 0.022 * mix(vec3f(0.9,1.0,0.95), vec3f(0.4,0.35,0.3), step(0.992, rr2));
  }

  var col = F*refl + (1.0-F)*under + spec;

  let haze = 1.0 - exp(-dist*0.004);
  let muh  = max(dot(normalize(vec3f(wd.x, 0.0, wd.z)), wp.sun), 0.0);
  let hazeC = vec3f(0.60,0.71,0.82) + vec3f(1.0,0.86,0.66)*(0.22*pow(muh,6.0)+0.3*pow(muh,64.0));
  col = mix(col, hazeC*0.95, haze*0.8);

  let skyc = sky(rd);
  let mu2  = dot(rd, wp.sun);
  var skyc2 = skyc + SUN_C*18.0*smoothstep(0.99996, 0.999985, mu2);
  let hz   = smoothstep(-0.0005, 0.0015, rd.y);
  col = mix(col, skyc2, hz);

  return vec4f(max(col, vec3f(0.0)), 1.0);
}

);
// clang-format on
