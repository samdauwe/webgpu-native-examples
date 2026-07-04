#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __WAJIC__
#define WAJIC_IMAGE_IMPL
#include <wajic_image.h>
#define WAJIC_SFETCH_IMPL
#include <wajic_sfetch.h>
#define WAJIC_TIME_IMPL
#include <wajic_time.h>
#else
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>
#define SOKOL_LOG_IMPL
#include <sokol_log.h>
#define SOKOL_TIME_IMPL
#include <sokol_time.h>
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

#include "core/image_loader.h"

/* In WAjic, WGPU handles are uint32_t */
#ifdef __WAJIC__
#ifdef NULL
#undef NULL
#define NULL 0
#endif
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Fur Rendering
 *
 * Demonstrates GPU-accelerated fur/grass rendering using the "shell fur"
 * technique. The model geometry is rendered in multiple instanced passes,
 * each pass displacing vertices outward along normals by an increasing amount.
 * A sinusoidal wave animation simulates wind-driven fur movement. Five fur
 * presets are provided (Leopard, Cow, Chick, Timber Wolf, Moss) with
 * adjustable layer count and layer thickness.
 *
 * Ported from the WebGL 2 Fur demo:
 * https://github.com/keaukraine/webgl-fur
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* fur_vignette_shader_wgsl;
static const char* fur_diffuse_shader_wgsl;
static const char* fur_shell_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Depth texture format */
#define FUR_DEPTH_FORMAT (WGPUTextureFormat_Depth24PlusStencil8)

/* Animation speeds (from WebGL demo) */
#define FUR_YAW_SPEED (12000.0f) /* ms per degree */
#define FUR_ANIM_SPEED (1500.0f) /* ms per 0..1 cycle */
#define FUR_WIND_SPEED (8310.0f) /* ms per 0..1 cycle */
#define FUR_STIFFNESS (2.75f)

/* Camera / projection */
#define FUR_FOV_DEG (15.0f) /* 25 * 0.6 multiplier */
#define FUR_CAM_NEAR (20.0f)
#define FUR_CAM_FAR (1000.0f)

/* File buffer sizes */
#define FUR_IDX_BUF_SIZE (8u * 1024u)  /* 7200 bytes needed  */
#define FUR_STR_BUF_SIZE (24u * 1024u) /* 23040 bytes needed */
#define FUR_TEX_BUF_SIZE (64u * 1024u) /* 46 KB max texture  */

/* Maximum layers */
#define FUR_MAX_LAYERS (30)

/* Number of preset textures (bg + 5 diffuse + 3 alpha) */
#define FUR_TEX_COUNT (9u)

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* Texture slot indices */
typedef enum {
  FUR_TEX_BG            = 0,
  FUR_TEX_DIFFUSE_LEO   = 1,
  FUR_TEX_DIFFUSE_COW   = 2,
  FUR_TEX_DIFFUSE_CHICK = 3,
  FUR_TEX_DIFFUSE_WOLF  = 4,
  FUR_TEX_DIFFUSE_MOSS  = 5,
  FUR_TEX_ALPHA_UNEVEN  = 6,
  FUR_TEX_ALPHA_EVEN    = 7,
  FUR_TEX_ALPHA_MOSS    = 8,
} fur_tex_id_t;

/* Fur preset */
typedef struct {
  const char* name;
  int layers;
  float thickness;
  float wave_scale;
  fur_tex_id_t diffuse;
  fur_tex_id_t alpha;
  vec4 start_color;
  vec4 end_color;
} fur_preset_t;

/* Vignette uniform buffer (vertex stage) */
typedef struct {
  mat4 ortho_matrix; /* 64 bytes */
} fur_vignette_uniforms_t;

/* Diffuse colored uniform buffer (vertex stage) */
typedef struct {
  mat4 view_proj_matrix; /* 64 bytes */
  vec4 color;            /* 16 bytes */
} fur_diffuse_uniforms_t;

/* Fur shell uniform buffer (vertex + fragment stages) */
typedef struct {
  mat4 view_proj_matrix; /* offset   0, 64 bytes */
  vec4 color_start;      /* offset  64, 16 bytes */
  vec4 color_end;        /* offset  80, 16 bytes */
  float layer_thickness; /* offset  96,  4 bytes */
  float layers_count;    /* offset 100,  4 bytes */
  float time;            /* offset 104,  4 bytes */
  float wave_scale;      /* offset 108,  4 bytes */
  float stiffness;       /* offset 112,  4 bytes */
  float _pad[3];         /* offset 116, 12 bytes (align to 128) */
} fur_shell_uniforms_t;

/* -------------------------------------------------------------------------- *
 * Preset definitions
 * -------------------------------------------------------------------------- */

// clang-format off
static const fur_preset_t fur_presets[] = {
  { "Leopard",     20, 0.15f, 0.50f, FUR_TEX_DIFFUSE_LEO,  FUR_TEX_ALPHA_UNEVEN, {0.60f, 0.60f, 0.60f, 1.0f}, {1.00f, 1.00f, 1.00f, 0.0f} },
  { "Cow",         10, 0.15f, 0.20f, FUR_TEX_DIFFUSE_COW,  FUR_TEX_ALPHA_UNEVEN, {0.70f, 0.70f, 0.70f, 1.0f}, {1.00f, 1.00f, 1.00f, 0.0f} },
  { "Chick",       13, 0.13f, 0.12f, FUR_TEX_DIFFUSE_CHICK, FUR_TEX_ALPHA_EVEN,  {1.15f, 1.15f, 1.15f, 1.0f}, {0.95f, 0.95f, 0.95f, 0.2f} },
  { "Timber Wolf", 20, 0.15f, 0.30f, FUR_TEX_DIFFUSE_WOLF, FUR_TEX_ALPHA_UNEVEN, {0.00f, 0.00f, 0.00f, 1.0f}, {1.00f, 1.00f, 1.00f, 0.0f} },
  { "Moss",         7, 0.13f, 0.00f, FUR_TEX_DIFFUSE_MOSS, FUR_TEX_ALPHA_MOSS,   {0.20f, 0.20f, 0.20f, 1.0f}, {1.00f, 1.00f, 1.00f, 0.8f} },
};
// clang-format on

#define FUR_PRESET_COUNT ((int)(ARRAY_SIZE(fur_presets)))

/* -------------------------------------------------------------------------- *
 * Fur Example State
 * -------------------------------------------------------------------------- */

static struct {
  /* WebGPU context reference (set during init, used in callbacks) */
  wgpu_context_t* wgpu_context;

  /* Box model loaded from binary files (pos3 + uv2 + normal3, uint16 indices)
   */
  struct {
    wgpu_buffer_t vertex_buffer;
    wgpu_buffer_t index_buffer;
    uint32_t index_count; /* = 3600 */
    bool is_ready;
    /* Pending binary data before GPU upload */
    uint8_t* pending_index_data;
    size_t pending_index_size;
    bool index_pending;
    uint8_t* pending_stride_data;
    size_t pending_stride_size;
    bool stride_pending;
  } model;

  /* Fullscreen quad for background vignette (pos3 + uv2) */
  wgpu_buffer_t vignette_vb;

  /* Textures */
  wgpu_texture_t textures[FUR_TEX_COUNT];
  WGPUSampler sampler;

  /* Uniform buffers */
  wgpu_buffer_t vignette_ubo;
  wgpu_buffer_t diffuse_ubo;
  wgpu_buffer_t fur_ubo;

  /* Bind group layouts */
  WGPUBindGroupLayout
    common_bgl; /* UBO + sampler + 1 texture (vignette/diffuse) */
  WGPUBindGroupLayout fur_bgl; /* UBO + sampler + 2 textures (shell fur) */

  /* Bind groups (rebuilt on preset change) */
  WGPUBindGroup vignette_bg;
  WGPUBindGroup diffuse_bg;
  WGPUBindGroup fur_bg;

  /* Pipeline layouts and pipelines */
  WGPUPipelineLayout vignette_pipeline_layout;
  WGPUPipelineLayout diffuse_pipeline_layout;
  WGPUPipelineLayout fur_pipeline_layout;
  WGPURenderPipeline vignette_pipeline;
  WGPURenderPipeline diffuse_pipeline;
  WGPURenderPipeline fur_pipeline;

  /* Depth texture */
  wgpu_texture_t depth_texture;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_desc;

  /* Animation */
  float angle_yaw;  /* degrees, wraps at 360 */
  float fur_timer;  /* [0..1) fur wave phase */
  float wind_timer; /* [0..1) wind phase     */
  uint64_t last_time;

  /* Current preset and GUI-editable overrides */
  int preset_idx;
  int preset_layers;
  float preset_thickness;
  bool bind_groups_dirty;

  /* Number of textures successfully loaded */
  uint32_t textures_loaded;

  /* Frame timing for ImGui */
  uint64_t last_frame_time;

  /* Ready to render */
  bool initialized;
} state = {
  /* Render pass setup */
  // clang-format off
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.3f, 0.3f, 0.3f, 1.0f},
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
  // clang-format on
  .preset_idx       = 0,
  .preset_layers    = 20,
  .preset_thickness = 0.15f,
};

/* -------------------------------------------------------------------------- *
 * Depth texture
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  wgpu_destroy_texture(&state.depth_texture);

  WGPUTextureDescriptor desc = {
    .label         = STRVIEW("Fur - Depth texture"),
    .dimension     = WGPUTextureDimension_2D,
    .format        = FUR_DEPTH_FORMAT,
    .mipLevelCount = 1,
    .sampleCount   = 1,
    .size          = {wgpu_context->width, wgpu_context->height, 1},
    .usage         = WGPUTextureUsage_RenderAttachment,
  };
  state.depth_texture.handle
    = wgpuDeviceCreateTexture(wgpu_context->device, &desc);
  ASSERT(state.depth_texture.handle != NULL);

  WGPUTextureViewDescriptor vdesc = {
    .label           = STRVIEW("Fur - Depth texture view"),
    .format          = FUR_DEPTH_FORMAT,
    .dimension       = WGPUTextureViewDimension_2D,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = 1,
  };
  state.depth_texture.view
    = wgpuTextureCreateView(state.depth_texture.handle, &vdesc);
  ASSERT(state.depth_texture.view != NULL);
}

/* -------------------------------------------------------------------------- *
 * Sampler
 * -------------------------------------------------------------------------- */

static void init_sampler(wgpu_context_t* wgpu_context)
{
  state.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Fur - Sampler"),
                            .addressModeU  = WGPUAddressMode_Repeat,
                            .addressModeV  = WGPUAddressMode_Repeat,
                            .addressModeW  = WGPUAddressMode_Repeat,
                            .minFilter     = WGPUFilterMode_Linear,
                            .magFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = 1.0f,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.sampler != NULL);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.vignette_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Vignette uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(fur_vignette_uniforms_t),
                  });

  state.diffuse_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Diffuse uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(fur_diffuse_uniforms_t),
                  });

  state.fur_ubo = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Shell uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(fur_shell_uniforms_t),
                  });
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  /* Common layout: UBO + sampler + 1 texture (used by vignette and diffuse) */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 0, /* flexible: vignette uses 64, diffuse uses 80 */
        },
      },
      [1] = (WGPUBindGroupLayoutEntry){
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry){
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    state.common_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device,
      &(WGPUBindGroupLayoutDescriptor){
        .label      = STRVIEW("Fur - Common bind group layout"),
        .entryCount = (uint32_t)ARRAY_SIZE(entries),
        .entries    = entries,
      });
    ASSERT(state.common_bgl != NULL);
  }

  /* Fur layout: UBO + sampler + diffuse texture + alpha texture */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      [0] = (WGPUBindGroupLayoutEntry){
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = (WGPUBufferBindingLayout){
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(fur_shell_uniforms_t),
        },
      },
      [1] = (WGPUBindGroupLayoutEntry){
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = (WGPUSamplerBindingLayout){
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = (WGPUBindGroupLayoutEntry){
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [3] = (WGPUBindGroupLayoutEntry){
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = (WGPUTextureBindingLayout){
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    state.fur_bgl = wgpuDeviceCreateBindGroupLayout(
      wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                              .label = STRVIEW("Fur - Shell bind group layout"),
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.fur_bgl != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups (rebuilt on preset change once textures are ready)
 * -------------------------------------------------------------------------- */

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Vignette bind group */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.vignette_bg)
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.vignette_ubo.buffer,
             .size    = state.vignette_ubo.size},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[FUR_TEX_BG].view},
    };
    state.vignette_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label  = STRVIEW("Fur - Vignette bind group"),
                              .layout = state.common_bgl,
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.vignette_bg != NULL);
  }

  /* Diffuse (base cube) bind group - uses current preset's diffuse texture */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.diffuse_bg)
    fur_tex_id_t diff_tex         = fur_presets[state.preset_idx].diffuse;
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.diffuse_ubo.buffer,
             .size    = state.diffuse_ubo.size},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[diff_tex].view},
    };
    state.diffuse_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Fur - Diffuse bind group"),
                              .layout     = state.common_bgl,
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.diffuse_bg != NULL);
  }

  /* Fur shell bind group */
  {
    WGPU_RELEASE_RESOURCE(BindGroup, state.fur_bg)
    fur_tex_id_t diff_tex         = fur_presets[state.preset_idx].diffuse;
    fur_tex_id_t alpha_tex        = fur_presets[state.preset_idx].alpha;
    WGPUBindGroupEntry entries[4] = {
      [0] = {.binding = 0,
             .buffer  = state.fur_ubo.buffer,
             .size    = state.fur_ubo.size},
      [1] = {.binding = 1, .sampler = state.sampler},
      [2] = {.binding = 2, .textureView = state.textures[diff_tex].view},
      [3] = {.binding = 3, .textureView = state.textures[alpha_tex].view},
    };
    state.fur_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Fur - Shell bind group"),
                              .layout     = state.fur_bgl,
                              .entryCount = (uint32_t)ARRAY_SIZE(entries),
                              .entries    = entries,
                            });
    ASSERT(state.fur_bg != NULL);
  }

  state.bind_groups_dirty = false;
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat swap_fmt = wgpu_context->render_format;

  /* --- Vignette pipeline --- */
  {
    state.vignette_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device,
      &(WGPUPipelineLayoutDescriptor){
        .label                = STRVIEW("Fur - Vignette pipeline layout"),
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts     = &state.common_bgl,
      });

    WGPUShaderModule vert = wgpu_create_shader_module(wgpu_context->device,
                                                      fur_vignette_shader_wgsl);

    /* Vertex layout: pos3 + uv2 (stride = 20 bytes) */
    WGPU_VERTEX_BUFFER_LAYOUT(
      vignette, 5 * sizeof(float),
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 3 * sizeof(float)))

    WGPUDepthStencilState ds
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = FUR_DEPTH_FORMAT,
        .depth_write_enabled = false, /* vignette must not write depth */
      });
    ds.depthCompare = WGPUCompareFunction_Always; /* always draw vignette */

    state.vignette_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Fur - Vignette pipeline"),
        .layout = state.vignette_pipeline_layout,
        .vertex = {
          .module      = vert,
          .entryPoint  = STRVIEW("vertexMain"),
          .bufferCount = 1,
          .buffers     = &vignette_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = vert,
          .entryPoint  = STRVIEW("fragmentMain"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = swap_fmt,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleStrip,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &ds,
        .multisample  = {.count = 1, .mask = 0xffffffff},
      });
    ASSERT(state.vignette_pipeline != NULL);
    wgpuShaderModuleRelease(vert);
  }

  /* --- Diffuse colored pipeline (base cube) --- */
  {
    state.diffuse_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("Fur - Diffuse pipeline layout"),
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &state.common_bgl,
                            });

    WGPUShaderModule vert = wgpu_create_shader_module(wgpu_context->device,
                                                      fur_diffuse_shader_wgsl);

    /* Vertex layout: pos3 + uv2 + normal3 (stride = 32 bytes) */
    WGPU_VERTEX_BUFFER_LAYOUT(
      model, 8 * sizeof(float),
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 3 * sizeof(float)),
      WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3, 5 * sizeof(float)))

    WGPUDepthStencilState ds
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = FUR_DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    ds.depthCompare = WGPUCompareFunction_Less;

    state.diffuse_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Fur - Diffuse pipeline"),
        .layout = state.diffuse_pipeline_layout,
        .vertex = {
          .module      = vert,
          .entryPoint  = STRVIEW("vertexMain"),
          .bufferCount = 1,
          .buffers     = &model_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = vert,
          .entryPoint  = STRVIEW("fragmentMain"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = swap_fmt,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &ds,
        .multisample  = {.count = 1, .mask = 0xffffffff},
      });
    ASSERT(state.diffuse_pipeline != NULL);
    wgpuShaderModuleRelease(vert);
  }

  /* --- Fur shell pipeline (instanced, alpha blend, no culling) --- */
  {
    state.fur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                              .label = STRVIEW("Fur - Shell pipeline layout"),
                              .bindGroupLayoutCount = 1,
                              .bindGroupLayouts     = &state.fur_bgl,
                            });

    WGPUShaderModule vert
      = wgpu_create_shader_module(wgpu_context->device, fur_shell_shader_wgsl);

    /* Vertex layout: pos3 + uv2 + normal3 (stride = 32 bytes) - same as model
     */
    WGPU_VERTEX_BUFFER_LAYOUT(
      fur_model, 8 * sizeof(float),
      WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, 0),
      WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, 3 * sizeof(float)),
      WGPU_VERTATTR_DESC(2, WGPUVertexFormat_Float32x3, 5 * sizeof(float)))

    /* Alpha blending: SRC_ALPHA / (1-SRC_ALPHA) for RGB; preserve alpha */
    WGPUBlendState blend = {
      .color = {
        .operation = WGPUBlendOperation_Add,
        .srcFactor = WGPUBlendFactor_SrcAlpha,
        .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
      },
      .alpha = {
        .operation = WGPUBlendOperation_Add,
        .srcFactor = WGPUBlendFactor_Zero,
        .dstFactor = WGPUBlendFactor_One,
      },
    };

    WGPUDepthStencilState ds
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = FUR_DEPTH_FORMAT,
        .depth_write_enabled = true,
      });
    ds.depthCompare = WGPUCompareFunction_LessEqual;

    state.fur_pipeline = wgpuDeviceCreateRenderPipeline(
      wgpu_context->device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Fur - Shell pipeline"),
        .layout = state.fur_pipeline_layout,
        .vertex = {
          .module      = vert,
          .entryPoint  = STRVIEW("vertexMain"),
          .bufferCount = 1,
          .buffers     = &fur_model_vertex_buffer_layout,
        },
        .fragment = &(WGPUFragmentState){
          .module      = vert,
          .entryPoint  = STRVIEW("fragmentMain"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = swap_fmt,
            .blend     = &blend,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None, /* no culling for fur layers */
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &ds,
        .multisample  = {.count = 1, .mask = 0xffffffff},
      });
    ASSERT(state.fur_pipeline != NULL);
    wgpuShaderModuleRelease(vert);
  }
}

/* -------------------------------------------------------------------------- *
 * Vignette quad geometry
 * -------------------------------------------------------------------------- */

static void init_vignette_buffer(wgpu_context_t* wgpu_context)
{
  /* Fullscreen quad: X, Y, Z, U, V (triangle strip, 4 vertices)
   * WebGL demo uses ortho(-1,1,-1,1,2,250) with z=-5 vertices */
  // clang-format off
  static const float verts[] = {
    -1.0f, -1.0f, -5.0f,  0.0f, 0.0f,  /* left-bottom  */
     1.0f, -1.0f, -5.0f,  1.0f, 0.0f,  /* right-bottom */
    -1.0f,  1.0f, -5.0f,  0.0f, 1.0f,  /* left-top     */
     1.0f,  1.0f, -5.0f,  1.0f, 1.0f,  /* right-top    */
  };
  // clang-format on

  state.vignette_vb = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Vignette vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(verts),
                    .initial.data = verts,
                  });
}

/* -------------------------------------------------------------------------- *
 * Model GPU buffers (uploaded after binary data is fetched)
 * -------------------------------------------------------------------------- */

static void init_model_buffers(wgpu_context_t* wgpu_context)
{
  /* Pre-allocate buffers; data filled by upload_pending_model_data() */
  state.model.index_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Model index buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = FUR_IDX_BUF_SIZE,
                  });

  state.model.vertex_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Fur - Model vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = FUR_STR_BUF_SIZE,
                  });
}

/* Upload pending binary model data (called from frame loop) */
static void upload_pending_model_data(wgpu_context_t* wgpu_context)
{
  if (state.model.index_pending && state.model.pending_index_data) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model.index_buffer.buffer,
                         0, state.model.pending_index_data,
                         state.model.pending_index_size);
    free(state.model.pending_index_data);
    state.model.pending_index_data = NULL;
    state.model.index_pending      = false;
    /* index_count is set when the callback fires */
  }

  if (state.model.stride_pending && state.model.pending_stride_data) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model.vertex_buffer.buffer,
                         0, state.model.pending_stride_data,
                         state.model.pending_stride_size);
    free(state.model.pending_stride_data);
    state.model.pending_stride_data = NULL;
    state.model.stride_pending      = false;
  }

  if (!state.model.is_ready && !state.model.index_pending
      && !state.model.stride_pending && state.model.index_count > 0) {
    state.model.is_ready = true;
  }
}

/* Upload pending textures (called from frame loop) */
static void upload_pending_textures(wgpu_context_t* wgpu_context)
{
  bool any_rebuilt = false;
  for (uint32_t i = 0; i < FUR_TEX_COUNT; ++i) {
    if (state.textures[i].desc.is_dirty) {
      wgpu_recreate_texture(wgpu_context, &state.textures[i]);
      FREE_TEXTURE_PIXELS(state.textures[i]);
      any_rebuilt = true;
    }
  }
  if (any_rebuilt) {
    state.bind_groups_dirty = true;
  }
}

/* -------------------------------------------------------------------------- *
 * Asset loading callbacks
 * -------------------------------------------------------------------------- */

/* Index buffer fetch callback */
static void fetch_indices_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("[FUR] Index file fetch failed (error %d)\n", response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  state.model.pending_index_data = (uint8_t*)malloc(response->data.size);
  if (state.model.pending_index_data) {
    memcpy(state.model.pending_index_data, response->data.ptr,
           response->data.size);
    state.model.pending_index_size = response->data.size;
    /* numIndices = byteLength / sizeof(uint16) / 3 triangles */
    state.model.index_count   = (uint32_t)(response->data.size / 2);
    state.model.index_pending = true;
  }
  free((void*)response->buffer.ptr);
}

/* Stride buffer fetch callback */
static void fetch_strides_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("[FUR] Stride file fetch failed (error %d)\n", response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  state.model.pending_stride_data = (uint8_t*)malloc(response->data.size);
  if (state.model.pending_stride_data) {
    memcpy(state.model.pending_stride_data, response->data.ptr,
           response->data.size);
    state.model.pending_stride_size = response->data.size;
    state.model.stride_pending      = true;
  }
  free((void*)response->buffer.ptr);
}

/* Texture fetch callback - user_data points to the target wgpu_texture_t* */
static void fetch_texture_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("[FUR] Texture fetch failed (error %d)\n", response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  int w, h, channels;
  uint8_t* pixels
    = image_pixels_from_memory((const uint8_t*)response->data.ptr,
                               (int)response->data.size, &w, &h, &channels, 4);

  if (pixels) {
    wgpu_texture_t* tex = *(wgpu_texture_t**)response->user_data;
    tex->desc           = (wgpu_texture_desc_t){
                .extent = {(uint32_t)w, (uint32_t)h, 1},
                .format = WGPUTextureFormat_RGBA8Unorm,
                .pixels = {.ptr = pixels, .size = (size_t)(w * h * 4)},
    };
    tex->desc.is_dirty = true;
    state.textures_loaded++;
  }
  free((void*)response->buffer.ptr);
}

/* -------------------------------------------------------------------------- *
 * Asset loading kick-off
 * -------------------------------------------------------------------------- */

/* Texture asset paths for each FUR_TEX_* slot */
// clang-format off
static const char* const fur_tex_paths[FUR_TEX_COUNT] = {
  [FUR_TEX_BG]           = "assets/models/Fur/textures/bg-gradient.png",
  [FUR_TEX_DIFFUSE_LEO]  = "assets/models/Fur/textures/fur-leo.png",
  [FUR_TEX_DIFFUSE_COW]  = "assets/models/Fur/textures/fur-cow.png",
  [FUR_TEX_DIFFUSE_CHICK]= "assets/models/Fur/textures/fur-chick.png",
  [FUR_TEX_DIFFUSE_WOLF] = "assets/models/Fur/textures/fur-wolf.png",
  [FUR_TEX_DIFFUSE_MOSS] = "assets/models/Fur/textures/moss.png",
  [FUR_TEX_ALPHA_UNEVEN] = "assets/models/Fur/textures/uneven-alpha.png",
  [FUR_TEX_ALPHA_EVEN]   = "assets/models/Fur/textures/even-alpha.png",
  [FUR_TEX_ALPHA_MOSS]   = "assets/models/Fur/textures/moss-alpha.png",
};
// clang-format on

static void load_assets(wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures for all slots */
  for (uint32_t i = 0; i < FUR_TEX_COUNT; ++i) {
    state.textures[i] = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });
    /* Fetch each texture.
     * Store the pointer to the texture slot so the callback can access it.
     * user_data.ptr must point to the pointer VALUE (ptr-to-ptr). */
    wgpu_texture_t* tex_ptr = &state.textures[i];
    uint8_t* buf            = (uint8_t*)malloc(FUR_TEX_BUF_SIZE);
    sfetch_send(&(sfetch_request_t){
      .path      = fur_tex_paths[i],
      .callback  = fetch_texture_cb,
      .buffer    = {.ptr = buf, .size = FUR_TEX_BUF_SIZE},
      .user_data = {.ptr = &tex_ptr, .size = sizeof(wgpu_texture_t*)},
    });
  }

  /* Fetch binary model files */
  {
    uint8_t* buf = (uint8_t*)malloc(FUR_IDX_BUF_SIZE);
    sfetch_send(&(sfetch_request_t){
      .path     = "assets/models/Fur/models/box10_rounded-indices.bin",
      .callback = fetch_indices_cb,
      .buffer   = {.ptr = buf, .size = FUR_IDX_BUF_SIZE},
    });
  }
  {
    uint8_t* buf = (uint8_t*)malloc(FUR_STR_BUF_SIZE);
    sfetch_send(&(sfetch_request_t){
      .path     = "assets/models/Fur/models/box10_rounded-strides.bin",
      .callback = fetch_strides_cb,
      .buffer   = {.ptr = buf, .size = FUR_STR_BUF_SIZE},
    });
  }
}

/* -------------------------------------------------------------------------- *
 * Matrix / uniform update helpers
 * -------------------------------------------------------------------------- */

static void compute_view_proj(wgpu_context_t* wgpu_context, mat4 out_vp)
{
  /* Camera: Z-up coordinate system, looking at origin from (190, 0, 270) */
  mat4 view, proj;
  glm_lookat((vec3){190.0f, 0.0f, 270.0f}, (vec3){0.0f, 0.0f, 0.0f},
             (vec3){0.0f, 0.0f, 1.0f}, view);

  const float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(TO_RADIANS(FUR_FOV_DEG), aspect, FUR_CAM_NEAR, FUR_CAM_FAR,
                  proj);

  /* Model matrix: rotate around Z by angleYaw */
  mat4 model;
  glm_mat4_identity(model);
  glm_rotate_z(model, state.angle_yaw, model);

  /* VP = proj * view, MVP = VP * model */
  mat4 vp;
  glm_mat4_mul(proj, view, vp);
  glm_mat4_mul(vp, model, out_vp);
}

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  mat4 mvp;
  compute_view_proj(wgpu_context, mvp);

  /* Vignette: orthographic projection */
  {
    fur_vignette_uniforms_t u;
    glm_ortho(-1.0f, 1.0f, -1.0f, 1.0f, 2.0f, 250.0f, u.ortho_matrix);
    wgpuQueueWriteBuffer(wgpu_context->queue, state.vignette_ubo.buffer, 0, &u,
                         sizeof(u));
  }

  /* Diffuse colored: MVP + preset start color */
  {
    fur_diffuse_uniforms_t u;
    glm_mat4_copy(mvp, u.view_proj_matrix);
    const fur_preset_t* p = &fur_presets[state.preset_idx];
    glm_vec4_copy((float*)p->start_color, u.color);
    wgpuQueueWriteBuffer(wgpu_context->queue, state.diffuse_ubo.buffer, 0, &u,
                         sizeof(u));
  }

  /* Fur shell: all fur animation parameters */
  {
    fur_shell_uniforms_t u = {0};
    glm_mat4_copy(mvp, u.view_proj_matrix);

    const fur_preset_t* p = &fur_presets[state.preset_idx];
    glm_vec4_copy((float*)p->start_color, u.color_start);
    glm_vec4_copy((float*)p->end_color, u.color_end);

    u.layer_thickness = state.preset_thickness;
    u.layers_count    = (float)state.preset_layers;
    u.time            = state.fur_timer;
    u.stiffness       = FUR_STIFFNESS;

    /* Wind-modulated wave scale: oscillates between 40% and 100% of preset */
    float a      = sinf(state.wind_timer * PI2) * 0.5f + 0.5f;
    u.wave_scale = p->wave_scale * 0.4f + (a * p->wave_scale * 0.6f);

    wgpuQueueWriteBuffer(wgpu_context->queue, state.fur_ubo.buffer, 0, &u,
                         sizeof(u));
  }
}

/* -------------------------------------------------------------------------- *
 * Animation tick
 * -------------------------------------------------------------------------- */

static void animate(void)
{
  uint64_t now = stm_now();
  if (state.last_time == 0) {
    state.last_time = now;
    return;
  }

  float elapsed_ms = (float)(stm_ms(stm_diff(now, state.last_time)));
  state.last_time  = now;

  state.angle_yaw += elapsed_ms / FUR_YAW_SPEED;
  if (state.angle_yaw >= 360.0f) {
    state.angle_yaw -= 360.0f;
  }

  state.fur_timer += elapsed_ms / FUR_ANIM_SPEED;
  if (state.fur_timer >= 1.0f) {
    state.fur_timer -= 1.0f;
  }

  state.wind_timer += elapsed_ms / FUR_WIND_SPEED;
  if (state.wind_timer >= 1.0f) {
    state.wind_timer -= 1.0f;
  }
}

/* -------------------------------------------------------------------------- *
 * ImGui / GUI
 * -------------------------------------------------------------------------- */

static void render_gui(void)
{
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_Once, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){220.0f, 0.0f}, ImGuiCond_Always);
  igBegin("Fur", NULL,
          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

  igText("Preset: %s", fur_presets[state.preset_idx].name);
  igSeparator();

  if (igButton("< Previous", (ImVec2){100.0f, 0.0f})) {
    state.preset_idx--;
    if (state.preset_idx < 0) {
      state.preset_idx = FUR_PRESET_COUNT - 1;
    }
    state.preset_layers     = fur_presets[state.preset_idx].layers;
    state.preset_thickness  = fur_presets[state.preset_idx].thickness;
    state.bind_groups_dirty = true;
  }
  igSameLine(0.0f, 5.0f);
  if (igButton("Next >", (ImVec2){100.0f, 0.0f})) {
    state.preset_idx        = (state.preset_idx + 1) % FUR_PRESET_COUNT;
    state.preset_layers     = fur_presets[state.preset_idx].layers;
    state.preset_thickness  = fur_presets[state.preset_idx].thickness;
    state.bind_groups_dirty = true;
  }

  igSeparator();
  igSliderInt("Layers##fur", &state.preset_layers, 1, FUR_MAX_LAYERS, "%d", 0);
  igSliderFloat("Thickness##fur", &state.preset_thickness, 0.01f, 0.30f, "%.3f",
                0);

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Init
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  state.wgpu_context = wgpu_context;

  /* sokol_fetch */
#ifndef __WAJIC__
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 16,
    .num_channels = 4,
    .num_lanes    = 4,
    .logger.func  = slog_func,
  });
#endif

  stm_setup();

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  /* GPU resources */
  init_depth_texture(wgpu_context);
  init_sampler(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_vignette_buffer(wgpu_context);
  init_model_buffers(wgpu_context);
  init_bind_group_layouts(wgpu_context);

  /* Build initial bind groups with placeholder textures (recreated on load) */
  /* We need at least one valid texture view before creating bind groups.
   * The placeholders will be created in load_assets(), so we create bind
   * groups AFTER that call. */
  load_assets(wgpu_context);
  init_bind_groups(wgpu_context);

  init_pipelines(wgpu_context);

  /* Sync preset */
  state.preset_layers    = fur_presets[0].layers;
  state.preset_thickness = fur_presets[0].thickness;

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Upload any pending binary model data */
  upload_pending_model_data(wgpu_context);

  /* Upload any newly decoded textures */
  upload_pending_textures(wgpu_context);

  /* Rebuild bind groups if preset changed or textures updated */
  if (state.bind_groups_dirty) {
    init_bind_groups(wgpu_context);
  }

  /* Animate */
  animate();

  /* Update GPU uniforms */
  update_uniforms(wgpu_context);

  /* ImGui frame */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  float dt              = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui();

  /* Render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture.view;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_desc);

  /* --- 1. Vignette (background gradient) --- */
  wgpuRenderPassEncoderSetPipeline(rpass, state.vignette_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.vignette_bg, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(rpass, 0, state.vignette_vb.buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(rpass, 4, 1, 0, 0);

  /* --- 2 & 3. Base cube + fur shells (only when model is ready) --- */
  if (state.model.is_ready) {
    /* Base cube (opaque, backface culled) */
    wgpuRenderPassEncoderSetPipeline(rpass, state.diffuse_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.diffuse_bg, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass, 0, state.model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass, state.model.index_buffer.buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rpass, state.model.index_count, 1, 0, 0,
                                     0);

    /* Fur shells (instanced, alpha blended) */
    wgpuRenderPassEncoderSetPipeline(rpass, state.fur_pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.fur_bg, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass, 0, state.model.vertex_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rpass, state.model.index_buffer.buffer,
                                        WGPUIndexFormat_Uint16, 0,
                                        WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rpass, state.model.index_count,
                                     (uint32_t)state.preset_layers, 0, 0, 0);
  }

  wgpuRenderPassEncoderEnd(rpass);

  WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buf);

  wgpuRenderPassEncoderRelease(rpass);
  wgpuCommandBufferRelease(cmd_buf);
  wgpuCommandEncoderRelease(cmd_enc);

  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Input events
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

#ifndef __WAJIC__
  sfetch_shutdown();
#endif

  /* Model buffers */
  wgpu_destroy_buffer(&state.model.vertex_buffer);
  wgpu_destroy_buffer(&state.model.index_buffer);
  free(state.model.pending_index_data);
  free(state.model.pending_stride_data);

  /* Vignette */
  wgpu_destroy_buffer(&state.vignette_vb);

  /* Textures */
  for (uint32_t i = 0; i < FUR_TEX_COUNT; ++i) {
    wgpu_destroy_texture(&state.textures[i]);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.sampler)

  /* Uniform buffers */
  wgpu_destroy_buffer(&state.vignette_ubo);
  wgpu_destroy_buffer(&state.diffuse_ubo);
  wgpu_destroy_buffer(&state.fur_ubo);

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.vignette_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.diffuse_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.fur_bg)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.common_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.fur_bgl)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.vignette_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.diffuse_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.fur_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.vignette_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.diffuse_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.fur_pipeline_layout)

  /* Depth texture */
  wgpu_destroy_texture(&state.depth_texture);
}

/* -------------------------------------------------------------------------- *
 * main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Fur Rendering",
    .width          = 1280,
    .height         = 720,
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
 * Vignette / background shader
 *
 * Renders the fullscreen gradient quad using an orthographic matrix.
 * Vertex layout: @location(0) pos:vec3, @location(1) uv:vec2
 * -------------------------------------------------------------------------- */
// clang-format off
static const char* fur_vignette_shader_wgsl = CODE(
  struct Uniforms {
    ortho_matrix: mat4x4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> uniforms:  Uniforms;
  @group(0) @binding(1) var          tex_samp:  sampler;
  @group(0) @binding(2) var          color_tex: texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.ortho_matrix * vec4f(in.position, 1.0);
    /* Flip V: model UVs are WebGL-convention (V=0 at bottom) */
    out.uv = vec2f(in.uv.x, 1.0 - in.uv.y);
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(color_tex, tex_samp, in.uv);
  }
);

/* -------------------------------------------------------------------------- *
 * Diffuse colored shader
 *
 * Renders the base (opaque) cube mesh tinted by the preset start color.
 * Vertex layout: @location(0) pos:vec3, @location(1) uv:vec2,
 *                @location(2) normal:vec3
 * -------------------------------------------------------------------------- */
static const char* fur_diffuse_shader_wgsl = CODE(
  struct Uniforms {
    view_proj_matrix: mat4x4f,
    color:            vec4f,
  }

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv:       vec2f,
    @location(2) normal:   vec3f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
  }

  @group(0) @binding(0) var<uniform> uniforms:  Uniforms;
  @group(0) @binding(1) var          tex_samp:  sampler;
  @group(0) @binding(2) var          color_tex: texture_2d<f32>;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.view_proj_matrix * vec4f(in.position, 1.0);
    out.uv       = vec2f(in.uv.x, 1.0 - in.uv.y);
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(color_tex, tex_samp, in.uv) * uniforms.color;
  }
);

/* -------------------------------------------------------------------------- *
 * Fur shell shader
 *
 * Instanced shell-fur: each instance displaces vertices outward along the
 * surface normal by layerThickness * (instanceIndex + 1). A sinusoidal wave
 * driven by time and vertex position simulates wind-blown fur movement.
 *
 * Vertex layout: @location(0) pos:vec3, @location(1) uv:vec2,
 *                @location(2) normal:vec3
 * -------------------------------------------------------------------------- */
static const char* fur_shell_shader_wgsl = CODE(
  struct FurUniforms {
    view_proj_matrix: mat4x4f,  /* offset   0 */
    color_start:      vec4f,    /* offset  64 */
    color_end:        vec4f,    /* offset  80 */
    layer_thickness:  f32,      /* offset  96 */
    layers_count:     f32,      /* offset 100 */
    time:             f32,      /* offset 104 */
    wave_scale:       f32,      /* offset 108 */
    stiffness:        f32,      /* offset 112 */
    _pad0:            f32,      /* offset 116 (3 × f32 padding to reach 128) */
    _pad1:            f32,      /* offset 120 */
    _pad2:            f32,      /* offset 124 */
  }

  struct VertexInput {
    @location(0)            position:    vec3f,
    @location(1)            uv:          vec2f,
    @location(2)            normal:      vec3f,
    @builtin(instance_index) instance_id: u32,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
    @location(1)       ao:       vec4f,
  }

  @group(0) @binding(0) var<uniform> u:         FurUniforms;
  @group(0) @binding(1) var          tex_samp:  sampler;
  @group(0) @binding(2) var          diffuse_tex: texture_2d<f32>;
  @group(0) @binding(3) var          alpha_tex:   texture_2d<f32>;

  const PI2:            f32 = 6.2831852;
  const RANDOM_COEFF_1: f32 = 0.1376;
  const RANDOM_COEFF_2: f32 = 0.3726;
  const RANDOM_COEFF_3: f32 = 0.2546;

  @vertex
  fn vertexMain(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    /* Layer offset along surface normal */
    let f: f32           = f32(in.instance_id + 1u) * u.layer_thickness;
    let layer_coeff: f32 = f32(in.instance_id) / u.layers_count;

    /* Displace vertex outward along normal */
    var v: vec4f = vec4f(in.position + in.normal * f, 1.0);

    /* Wind animation: sinusoidal wave proportional to layer depth */
    let time_pi2:      f32 = u.time * PI2;
    let wave_scale_f:  f32 = u.wave_scale * pow(layer_coeff, u.stiffness);
    let px: f32 = in.position.x;
    let py: f32 = in.position.y;
    let pz: f32 = in.position.z;

    v.x += sin(time_pi2 + (px + py + pz) * RANDOM_COEFF_1) * wave_scale_f;
    v.y += cos(time_pi2 + (px - py + pz) * RANDOM_COEFF_2) * wave_scale_f;
    v.z += sin(time_pi2 + (px + py - pz) * RANDOM_COEFF_3) * wave_scale_f;

    out.position = u.view_proj_matrix * v;
    out.uv       = vec2f(in.uv.x, 1.0 - in.uv.y);
    /* Ambient occlusion color blend from root (colorStart) to tip (colorEnd) */
    out.ao = mix(u.color_start, u.color_end, layer_coeff);
    return out;
  }

  @fragment
  fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
    let diffuse: vec4f = textureSample(diffuse_tex, tex_samp, in.uv);
    let alpha:   f32   = textureSample(alpha_tex,   tex_samp, in.uv).r;
    var color:   vec4f = diffuse * in.ao;
    color.a           *= alpha;
    return color;
  }
);
// clang-format on
