/* -------------------------------------------------------------------------- *
 * WebGPU Example - Offscreen Rendering
 *
 * Renders a scene (Chinese Dragon) into an offscreen framebuffer, then uses
 * that texture as a reflection map on a mirror plane in the main pass.
 * An optional debug mode displays the raw offscreen texture via a fullscreen
 * triangle.
 *
 * Render passes:
 *   1. Offscreen pass (512x512):
 *        Draws the dragon with Y-flipped model matrix → reflection image.
 *   2. Main pass (window-size):
 *        Debug mode  → fullscreen triangle sampling the offscreen texture.
 *        Normal mode → mirror plane (projective sampling) + lit dragon above.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/offscreen
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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

#include "core/camera.h"
#include "core/gltf_model.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations — defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* offscreen_phong_shader_wgsl;
static const char* offscreen_mirror_shader_wgsl;
static const char* offscreen_quad_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Fixed-size offscreen framebuffer — matches Vulkan reference (FB_DIM = 512) */
#define OFFSCREEN_WIDTH (512u)
#define OFFSCREEN_HEIGHT (512u)

/* Depth format shared by both offscreen and main passes */
#define DEPTH_FORMAT (WGPUTextureFormat_Depth24PlusStencil8)

/* -------------------------------------------------------------------------- *
 * Uniform data (matches Vulkan UBO layout: projection + view + model + light) *
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 projection; /* 64 bytes */
  mat4 view;       /* 64 bytes */
  mat4 model;      /* 64 bytes */
  vec4 light_pos;  /* 16 bytes */
} ubo_data_t;      /* 208 bytes total */

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Models */
  gltf_model_t dragon_model;
  gltf_model_t plane_model;
  bool models_loaded;

  /* GPU buffers (vertices + indices) */
  WGPUBuffer dragon_vb;
  WGPUBuffer dragon_ib;
  WGPUBuffer plane_vb;
  WGPUBuffer plane_ib;

  /* Offscreen framebuffer (fixed 512x512) */
  struct {
    WGPUTexture color_tex;
    WGPUTextureView color_view;
    WGPUTexture depth_tex;
    WGPUTextureView depth_view;
    WGPUSampler sampler;
  } offscreen;

  /* Main-pass depth texture (window-size, re-created on resize) */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
  } depth_tex;

  /* Uniform buffers (one per role) */
  WGPUBuffer ubo_model;     /* main dragon             */
  WGPUBuffer ubo_mirror;    /* mirror plane            */
  WGPUBuffer ubo_offscreen; /* offscreen (flipped) dragon */

  /* Uniform data CPU-side */
  ubo_data_t ubo_model_data;
  ubo_data_t ubo_mirror_data;
  ubo_data_t ubo_offscreen_data;

  /* Model animation */
  float model_rotation_y; /* degrees, updated each frame */

  /* Bind group layouts */
  WGPUBindGroupLayout shaded_bgl;   /* binding 0: UBO (vertex) */
  WGPUBindGroupLayout textured_bgl; /* binding 0: UBO (vertex)
                                       binding 1: sampler (fragment)
                                       binding 2: texture (fragment) */

  /* Pipeline layouts */
  WGPUPipelineLayout shaded_layout;
  WGPUPipelineLayout textured_layout;

  /* Bind groups */
  WGPUBindGroup offscreen_bg; /* shaded BGL — for offscreen pass      */
  WGPUBindGroup model_bg;     /* shaded BGL — for main dragon         */
  WGPUBindGroup mirror_bg;    /* textured BGL — mirror + debug pass   */

  /* Render pipelines */
  WGPURenderPipeline debug_pipeline;     /* fullscreen quad: offscreen tex */
  WGPURenderPipeline shaded_pipeline;    /* Phong-lit dragon (back cull)   */
  WGPURenderPipeline offscreen_pipeline; /* Phong offscreen (front cull)   */
  WGPURenderPipeline mirror_pipeline;    /* mirror plane with reflection   */

  /* Offscreen render pass descriptors */
  WGPURenderPassColorAttachment offscreen_color_att;
  WGPURenderPassDepthStencilAttachment offscreen_depth_att;
  WGPURenderPassDescriptor offscreen_pass;

  /* Main render pass descriptors */
  WGPURenderPassColorAttachment main_color_att;
  WGPURenderPassDepthStencilAttachment main_depth_att;
  WGPURenderPassDescriptor main_pass;

  /* GUI settings */
  bool debug_display;

  /* Timing */
  uint64_t last_frame_time;

  /* Window size (for resize detection) */
  int last_width;
  int last_height;

  WGPUBool initialized;
} state = {
  /* clang-format off */
  .offscreen_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .offscreen_depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Discard,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Discard,
    .stencilClearValue = 0,
  },
  .offscreen_pass = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.offscreen_color_att,
    .depthStencilAttachment = &state.offscreen_depth_att,
  },
  .main_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025f, 0.025f, 0.025f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .main_depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .main_pass = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.main_color_att,
    .depthStencilAttachment = &state.main_depth_att,
  },
  /* clang-format on */
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

/* No FlipY for WebGPU — gltf uses Y-up, same as WebGPU clip space.
 * No PreMultiplyVertexColors needed here (phong shader uses vertex color). */
static const gltf_model_desc_t model_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

static void load_models(void)
{
  bool ok = gltf_model_load_from_file_ext(&state.dragon_model,
                                          "assets/models/chinesedragon.gltf",
                                          1.0f, &model_load_desc);
  if (!ok) {
    printf("[offscreen] Failed to load chinesedragon.gltf\n");
    return;
  }

  ok = gltf_model_load_from_file_ext(
    &state.plane_model, "assets/models/plane.gltf", 1.0f, &model_load_desc);
  if (!ok) {
    printf("[offscreen] Failed to load plane.gltf\n");
    return;
  }

  state.models_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.models_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;

  /* Helper to upload a model's vertices and indices */
#define UPLOAD_MODEL(m, vb, ib)                                                       \
  do {                                                                                \
    size_t vb_size = (m)->vertex_count * sizeof(gltf_vertex_t);                       \
    (vb)           = wgpuDeviceCreateBuffer(                                          \
      device, &(WGPUBufferDescriptor){                                      \
                          .label = STRVIEW(#m " Vertex Buffer"),                      \
                          .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,  \
                          .size  = vb_size,                                           \
                          .mappedAtCreation = true,                                   \
              });                                                           \
    void* vdata = wgpuBufferGetMappedRange((vb), 0, vb_size);                         \
    memcpy(vdata, (m)->vertices, vb_size);                                            \
    wgpuBufferUnmap((vb));                                                            \
    if ((m)->index_count > 0) {                                                       \
      size_t ib_size = (m)->index_count * sizeof(uint32_t);                           \
      (ib)           = wgpuDeviceCreateBuffer(                                        \
        device, &(WGPUBufferDescriptor){                                    \
                            .label = STRVIEW(#m " Index Buffer"),                     \
                            .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst, \
                            .size  = ib_size,                                         \
                            .mappedAtCreation = true,                                 \
                });                                                         \
      void* idata = wgpuBufferGetMappedRange((ib), 0, ib_size);                       \
      memcpy(idata, (m)->indices, ib_size);                                           \
      wgpuBufferUnmap((ib));                                                          \
    }                                                                                 \
  } while (0)

  UPLOAD_MODEL(&state.dragon_model, state.dragon_vb, state.dragon_ib);
  UPLOAD_MODEL(&state.plane_model, state.plane_vb, state.plane_ib);

#undef UPLOAD_MODEL
}

/* -------------------------------------------------------------------------- *
 * Offscreen framebuffer (fixed 512x512, created once)
 * -------------------------------------------------------------------------- */

static void init_offscreen_framebuffer(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- Color attachment (sampled in mirror/debug pass) ---- */
  state.offscreen.color_tex = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Offscreen Color Texture"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 1},
              .format        = WGPUTextureFormat_RGBA8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  ASSERT(state.offscreen.color_tex != NULL);

  state.offscreen.color_view = wgpuTextureCreateView(
    state.offscreen.color_tex, &(WGPUTextureViewDescriptor){
                                 .label     = STRVIEW("Offscreen Color View"),
                                 .format    = WGPUTextureFormat_RGBA8Unorm,
                                 .dimension = WGPUTextureViewDimension_2D,
                                 .baseMipLevel    = 0,
                                 .mipLevelCount   = 1,
                                 .baseArrayLayer  = 0,
                                 .arrayLayerCount = 1,
                                 .aspect          = WGPUTextureAspect_All,
                               });
  ASSERT(state.offscreen.color_view != NULL);

  /* ---- Sampler for mirror/debug access ---- */
  state.offscreen.sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Offscreen Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 1.0f,
              .maxAnisotropy = 1,
            });
  ASSERT(state.offscreen.sampler != NULL);

  /* ---- Depth attachment (discard at end of pass, not sampled) ---- */
  state.offscreen.depth_tex = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Offscreen Depth Texture"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 1},
              .format        = DEPTH_FORMAT,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  ASSERT(state.offscreen.depth_tex != NULL);

  state.offscreen.depth_view = wgpuTextureCreateView(
    state.offscreen.depth_tex, &(WGPUTextureViewDescriptor){
                                 .label     = STRVIEW("Offscreen Depth View"),
                                 .format    = DEPTH_FORMAT,
                                 .dimension = WGPUTextureViewDimension_2D,
                                 .baseMipLevel    = 0,
                                 .mipLevelCount   = 1,
                                 .baseArrayLayer  = 0,
                                 .arrayLayerCount = 1,
                                 .aspect          = WGPUTextureAspect_All,
                               });
  ASSERT(state.offscreen.depth_view != NULL);

  /* Wire up pass descriptor views */
  state.offscreen_color_att.view = state.offscreen.color_view;
  state.offscreen_depth_att.view = state.offscreen.depth_view;
}

static void destroy_offscreen_framebuffer(void)
{
  WGPU_RELEASE_RESOURCE(Sampler, state.offscreen.sampler)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.color_tex)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen.depth_tex)
}

/* -------------------------------------------------------------------------- *
 * Main-pass depth texture (window-size, re-created on resize)
 * -------------------------------------------------------------------------- */

static void init_main_depth_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  state.depth_tex.handle = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Main Depth Texture"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = DEPTH_FORMAT,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  ASSERT(state.depth_tex.handle != NULL);

  state.depth_tex.view = wgpuTextureCreateView(
    state.depth_tex.handle, &(WGPUTextureViewDescriptor){
                              .label           = STRVIEW("Main Depth View"),
                              .format          = DEPTH_FORMAT,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });
  ASSERT(state.depth_tex.view != NULL);

  state.main_depth_att.view = state.depth_tex.view;
  state.last_width          = (int)w;
  state.last_height         = (int)h;
}

static void destroy_main_depth_texture(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_tex.view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_tex.handle)
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Light position at world origin (acts as headlight since view * (0,0,0,1)
   * = camera origin = (0,0,0) in view space) — matching Vulkan example. */
  vec4 light_pos = {0.0f, 0.0f, 0.0f, 1.0f};
  glm_vec4_copy(light_pos, state.ubo_model_data.light_pos);
  glm_vec4_copy(light_pos, state.ubo_mirror_data.light_pos);
  glm_vec4_copy(light_pos, state.ubo_offscreen_data.light_pos);

  WGPUBufferDescriptor ubo_desc = {
    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    .size  = sizeof(ubo_data_t),
  };

  ubo_desc.label  = STRVIEW("UBO Model");
  state.ubo_model = wgpuDeviceCreateBuffer(device, &ubo_desc);
  ASSERT(state.ubo_model != NULL);

  ubo_desc.label   = STRVIEW("UBO Mirror");
  state.ubo_mirror = wgpuDeviceCreateBuffer(device, &ubo_desc);
  ASSERT(state.ubo_mirror != NULL);

  ubo_desc.label      = STRVIEW("UBO Offscreen");
  state.ubo_offscreen = wgpuDeviceCreateBuffer(device, &ubo_desc);
  ASSERT(state.ubo_offscreen != NULL);
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context,
                                   float delta_time)
{
  /* Advance rotation */
  state.model_rotation_y += delta_time * 10.0f; /* °/s — matches Vulkan */

  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Shared projection + view */
  glm_perspective(glm_rad(60.0f), aspect, 0.1f, 256.0f,
                  state.ubo_model_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_model_data.view);

  /* -- Model UBO: rotate + translate dragon.
   * Vulkan: translate(0,-1,0) in Y-down convention = object sits on floor.
   * WebGPU: negate Y → translate(0,+1,0) so dragon is above floor (Y=0). */
  glm_mat4_identity(state.ubo_model_data.model);
  glm_rotate_y(state.ubo_model_data.model, glm_rad(state.model_rotation_y),
               state.ubo_model_data.model);
  glm_translate(state.ubo_model_data.model, (vec3){0.0f, 1.0f, 0.0f});

  wgpuQueueWriteBuffer(wgpu_context->queue, state.ubo_model, 0,
                       &state.ubo_model_data, sizeof(ubo_data_t));

  /* -- Mirror UBO: identity model (the plane sits at Y=0) -- */
  glm_mat4_copy(state.ubo_model_data.projection,
                state.ubo_mirror_data.projection);
  glm_mat4_copy(state.ubo_model_data.view, state.ubo_mirror_data.view);
  glm_mat4_identity(state.ubo_mirror_data.model);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.ubo_mirror, 0,
                       &state.ubo_mirror_data, sizeof(ubo_data_t));

  /* -- Offscreen UBO: rotate + Y-scale(-1) + translate dragon -- */
  /* Projection and view are the same as the main camera */
  glm_mat4_copy(state.ubo_model_data.projection,
                state.ubo_offscreen_data.projection);
  glm_mat4_copy(state.ubo_model_data.view, state.ubo_offscreen_data.view);
  glm_mat4_identity(state.ubo_offscreen_data.model);
  glm_rotate_y(state.ubo_offscreen_data.model, glm_rad(state.model_rotation_y),
               state.ubo_offscreen_data.model);
  /* Y-flip + same translate as main dragon (negated Vulkan -1 → +1). */
  glm_scale(state.ubo_offscreen_data.model, (vec3){1.0f, -1.0f, 1.0f});
  glm_translate(state.ubo_offscreen_data.model, (vec3){0.0f, 1.0f, 0.0f});

  wgpuQueueWriteBuffer(wgpu_context->queue, state.ubo_offscreen, 0,
                       &state.ubo_offscreen_data, sizeof(ubo_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Shaded BGL: single UBO at binding 0 (vertex stage) */
  {
    WGPUBindGroupLayoutEntry entry = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(ubo_data_t),
      },
    };
    state.shaded_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Shaded BGL"),
                .entryCount = 1,
                .entries    = &entry,
              });
    ASSERT(state.shaded_bgl != NULL);
  }

  /* Textured BGL: UBO (0), sampler (1), texture (2) */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(ubo_data_t),
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = { .type = WGPUSamplerBindingType_Filtering },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
    };
    state.textured_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Textured BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
    ASSERT(state.textured_bgl != NULL);
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline layouts
 * -------------------------------------------------------------------------- */

static void init_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.shaded_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Shaded Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.shaded_bgl,
            });
  ASSERT(state.shaded_layout != NULL);

  state.textured_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Textured Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.textured_bgl,
            });
  ASSERT(state.textured_layout != NULL);
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* offscreen_bg: shaded — draws the flipped dragon */
  {
    WGPUBindGroupEntry entry = {
      .binding = 0,
      .buffer  = state.ubo_offscreen,
      .offset  = 0,
      .size    = sizeof(ubo_data_t),
    };
    state.offscreen_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Offscreen BG"),
                                            .layout = state.shaded_bgl,
                                            .entryCount = 1,
                                            .entries    = &entry,
                                          });
    ASSERT(state.offscreen_bg != NULL);
  }

  /* model_bg: shaded — draws the main lit dragon */
  {
    WGPUBindGroupEntry entry = {
      .binding = 0,
      .buffer  = state.ubo_model,
      .offset  = 0,
      .size    = sizeof(ubo_data_t),
    };
    state.model_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW("Model BG"),
                                            .layout     = state.shaded_bgl,
                                            .entryCount = 1,
                                            .entries    = &entry,
                                          });
    ASSERT(state.model_bg != NULL);
  }

  /* mirror_bg: textured — mirror plane and debug quad */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.ubo_mirror,
        .offset  = 0,
        .size    = sizeof(ubo_data_t),
      },
      [1] = {
        .binding = 1,
        .sampler = state.offscreen.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.offscreen.color_view,
      },
    };
    state.mirror_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW("Mirror BG"),
                                            .layout     = state.textured_bgl,
                                            .entryCount = ARRAY_SIZE(entries),
                                            .entries    = entries,
                                          });
    ASSERT(state.mirror_bg != NULL);
  }
}

static void destroy_bind_groups(void)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.offscreen_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.model_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.mirror_bg)
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

/* gltf_vertex_t interleaved layout — locations match gltf_vertex_t offsets:
 *   location 0: position (vec3f) at offset 0
 *   location 1: normal   (vec3f) at offset 12
 *   location 2: uv0      (vec2f) at offset 24  (consumed but unused)
 *   location 3: color    (vec4f) at offset 56                               */
static WGPUVertexBufferLayout make_gltf_vbl(void)
{
  static WGPUVertexAttribute attrs[4];
  attrs[0] = (WGPUVertexAttribute){
    .shaderLocation = 0,
    .offset         = offsetof(gltf_vertex_t, position),
    .format         = WGPUVertexFormat_Float32x3,
  };
  attrs[1] = (WGPUVertexAttribute){
    .shaderLocation = 1,
    .offset         = offsetof(gltf_vertex_t, normal),
    .format         = WGPUVertexFormat_Float32x3,
  };
  attrs[2] = (WGPUVertexAttribute){
    .shaderLocation = 2,
    .offset         = offsetof(gltf_vertex_t, uv0),
    .format         = WGPUVertexFormat_Float32x2,
  };
  attrs[3] = (WGPUVertexAttribute){
    .shaderLocation = 3,
    .offset         = offsetof(gltf_vertex_t, color),
    .format         = WGPUVertexFormat_Float32x4,
  };
  return (WGPUVertexBufferLayout){
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 4,
    .attributes     = attrs,
  };
}

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device          = wgpu_context->device;
  WGPUVertexBufferLayout vbl = make_gltf_vbl();

  /* Depth stencil state shared by all mesh passes */
  WGPUDepthStencilState depth_state = {
    .format               = DEPTH_FORMAT,
    .depthWriteEnabled    = WGPUOptionalBool_True,
    .depthCompare         = WGPUCompareFunction_LessEqual,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  /* ---- (1) Debug pipeline: fullscreen triangle showing offscreen tex ---- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, offscreen_quad_shader_wgsl);

    WGPUColorTargetState ct = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.debug_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Debug Pipeline"),
        .layout = state.textured_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_quad"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = NULL, /* no depth for fullscreen pass */
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_quad"),
          .targetCount = 1,
          .targets     = &ct,
        },
      });
    ASSERT(state.debug_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- (2) Shaded pipeline: Phong-lit dragon (main pass, back cull) ---- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, offscreen_phong_shader_wgsl);

    WGPUColorTargetState ct = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.shaded_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Shaded Pipeline"),
        .layout = state.shaded_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_phong"),
          .bufferCount = 1,
          .buffers     = &vbl,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_state,
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_phong"),
          .targetCount = 1,
          .targets     = &ct,
        },
      });
    ASSERT(state.shaded_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- (3) Offscreen pipeline: Phong dragon (offscreen pass, front cull) --
   * The Y-flipped dragon is rendered back-to-front from the camera's
   * perspective, so we flip the cull mode to FRONT to match what Vulkan
   * does (VK_CULL_MODE_FRONT_BIT for the offscreen pass).                  */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, offscreen_phong_shader_wgsl);

    WGPUColorTargetState ct = {
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState ds = depth_state;

    state.offscreen_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Offscreen Phong Pipeline"),
        .layout = state.shaded_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_phong"),
          .bufferCount = 1,
          .buffers     = &vbl,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Front, /* flipped for mirror reflection */
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &ds,
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_phong"),
          .targetCount = 1,
          .targets     = &ct,
        },
      });
    ASSERT(state.offscreen_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }

  /* ---- (4) Mirror pipeline: plane with projective reflection texture ---- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, offscreen_mirror_shader_wgsl);

    WGPUColorTargetState ct = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.mirror_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Mirror Pipeline"),
        .layout = state.textured_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_mirror"),
          .bufferCount = 1,
          .buffers     = &vbl,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None, /* render both faces of the plane */
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_state,
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_mirror"),
          .targetCount = 1,
          .targets     = &ct,
        },
      });
    ASSERT(state.mirror_pipeline != NULL);
    wgpuShaderModuleRelease(sm);
  }
}

/* -------------------------------------------------------------------------- *
 * Draw helpers
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* m,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  if (!state.models_loaded || !vb) {
    return;
  }

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);

  if (ib) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    for (uint32_t p = 0; p < node->mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &node->mesh->primitives[p];
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
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Offscreen Rendering", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_checkbox("Display render target", &state.debug_display);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Skip camera input when ImGui captures the mouse */
  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Window resize
 * -------------------------------------------------------------------------- */

static void on_resize(struct wgpu_context_t* wgpu_context)
{
  destroy_main_depth_texture();
  init_main_depth_texture(wgpu_context);

  camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                              / (float)wgpu_context->height);
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

  /* Camera — Vulkan: position (0, 1, -6), rotation (-2.5, 0, 0).
   * Porting guide: negate camera Y (camera.c negates internally), negate X. */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
  camera_set_rotation(&state.camera, (vec3){2.5f, 0.0f, 0.0f});
  /* WebGPU +Y = top of screen, Vulkan +Y = bottom (flipY=false).
   * Pass the SAME value as the Vulkan setPosition({0,1,-6}) so that
   * camera_set_position's internal Y-negation compensates for the flip:
   * stored = (0, -1, -6) → view T(0,-1,-6) → camera above floor at Y=+1. */
  camera_set_position(&state.camera, (vec3){0.0f, 1.0f, -6.0f});
  state.camera.rotation_speed = 0.5f;

  /* Load gltf models */
  load_models();
  create_model_buffers(wgpu_context);

  /* Offscreen framebuffer (fixed 512x512) */
  init_offscreen_framebuffer(wgpu_context);

  /* Main depth texture (window-size) */
  init_main_depth_texture(wgpu_context);

  /* Uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* BGL → pipeline layouts → bind groups → pipelines */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  wgpuDeviceTick(wgpu_context->device);

  /* Resize detection */
  if (wgpu_context->width != state.last_width
      || wgpu_context->height != state.last_height) {
    on_resize(wgpu_context);
  }

  /* Timing */
  uint64_t now = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = now;
  }
  float dt              = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Camera update */
  camera_update(&state.camera, dt);

  /* Uniforms */
  update_uniform_buffers(wgpu_context, dt);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* ---- Encode render commands ---- */
  WGPUDevice device      = wgpu_context->device;
  WGPUQueue queue        = wgpu_context->queue;
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  uint32_t w             = (uint32_t)wgpu_context->width;
  uint32_t h             = (uint32_t)wgpu_context->height;

  /* ===== Pass 1: Offscreen — render Y-flipped dragon into 512x512 ===== */
  {
    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(enc, &state.offscreen_pass);

    wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, (float)OFFSCREEN_WIDTH,
                                     (float)OFFSCREEN_HEIGHT, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, OFFSCREEN_WIDTH,
                                        OFFSCREEN_HEIGHT);

    wgpuRenderPassEncoderSetPipeline(pass, state.offscreen_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.offscreen_bg, 0, NULL);
    draw_model(pass, &state.dragon_model, state.dragon_vb, state.dragon_ib);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ===== Pass 2: Main — scene or debug display ===== */
  {
    state.main_color_att.view = wgpu_context->swapchain_view;

    /* Debug pipeline has no depth stencil state, so omit the attachment when
     * running in debug mode to keep pipeline and pass compatible. */
    WGPURenderPassDescriptor main_pass_desc = state.main_pass;
    if (state.debug_display) {
      main_pass_desc.depthStencilAttachment = NULL;
    }

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(enc, &main_pass_desc);

    wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

    if (state.debug_display) {
      /* Show the raw offscreen texture via a fullscreen triangle */
      wgpuRenderPassEncoderSetPipeline(pass, state.debug_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.mirror_bg, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }
    else {
      /* Mirror plane (reflection) */
      wgpuRenderPassEncoderSetPipeline(pass, state.mirror_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.mirror_bg, 0, NULL);
      draw_model(pass, &state.plane_model, state.plane_vb, state.plane_ib);

      /* Main lit dragon */
      wgpuRenderPassEncoderSetPipeline(pass, state.shaded_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.model_bg, 0, NULL);
      draw_model(pass, &state.dragon_model, state.dragon_vb, state.dragon_ib);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  /* ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Models */
  gltf_model_destroy(&state.dragon_model);
  gltf_model_destroy(&state.plane_model);
  WGPU_RELEASE_RESOURCE(Buffer, state.dragon_vb)
  WGPU_RELEASE_RESOURCE(Buffer, state.dragon_ib)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_vb)
  WGPU_RELEASE_RESOURCE(Buffer, state.plane_ib)

  /* Offscreen framebuffer */
  destroy_offscreen_framebuffer();

  /* Main depth texture */
  destroy_main_depth_texture();

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.ubo_model)
  WGPU_RELEASE_RESOURCE(Buffer, state.ubo_mirror)
  WGPU_RELEASE_RESOURCE(Buffer, state.ubo_offscreen)

  /* Bind groups */
  destroy_bind_groups();

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.shaded_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.textured_bgl)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.shaded_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.textured_layout)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.debug_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.shaded_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.offscreen_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.mirror_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Offscreen Rendering",
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

/* ---- Phong shader (shared by main + offscreen pass) ----------------------
 *
 * Vertex layout (gltf_vertex_t interleaved):
 *   location 0: position vec3f
 *   location 1: normal   vec3f
 *   location 2: uv0      vec2f  (consumed, not used by lighting)
 *   location 3: color    vec4f  (pre-multiplied vertex color)
 *
 * Lighting: computed in view space.
 *   - eye_pos:   vertex position in view space
 *   - light_vec: light direction from vertex to light (headlight at camera)
 *   - normals are passed through in object space (matches Vulkan reference)
 *
 * The light is placed at (0,0,0) world space = camera origin in view space,
 * giving a headlight effect consistent with the Vulkan original.
 * -------------------------------------------------------------------------- */
// clang-format off
static const char* offscreen_phong_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    light_pos  : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv0      : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       color     : vec3f,
    @location(2)       eye_pos   : vec3f,
    @location(3)       light_vec : vec3f,
  };

  @vertex
  fn vs_phong(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    out.normal = in.normal;
    out.color  = in.color.rgb;

    let world_pos = ubo.model * vec4f(in.position, 1.0);
    out.position  = ubo.projection * ubo.view * world_pos;
    out.eye_pos   = (ubo.view * world_pos).xyz;

    // Light at (0,0,0) world space = headlight at camera in view space
    out.light_vec = normalize(ubo.light_pos.xyz - out.eye_pos);
    return out;
  }

  @fragment
  fn fs_phong(in : VertexOutput) -> @location(0) vec4f {
    let Eye       = normalize(-in.eye_pos);
    let Reflected = normalize(reflect(-in.light_vec, in.normal));

    let IAmbient  = vec4f(0.1, 0.1, 0.1, 1.0);
    let IDiffuse  = vec4f(max(dot(in.normal, in.light_vec), 0.0));
    var ISpecular = vec4f(0.0);
    // Only apply specular on the front face
    if (dot(in.eye_pos, in.normal) < 0.0) {
      ISpecular = vec4f(0.5, 0.5, 0.5, 1.0)
                  * pow(max(dot(Reflected, Eye), 0.0), 16.0) * 0.75;
    }

    let lit = (IAmbient + IDiffuse).rgb * in.color + ISpecular.rgb;
    return vec4f(lit, 1.0);
  }
);
// clang-format on

/* ---- Mirror shader --------------------------------------------------------
 *
 * Renders the mirror plane using projective texturing: the clip-space
 * position of each fragment is converted to UV coordinates to sample the
 * offscreen texture (the reflection image).
 *
 * WebGPU Y-axis correction:
 *   Vulkan clip Y is top=+1 → bottom=-1, but NDC Y is +down.
 *   WebGPU clip Y is +up.  After dividing by w, we get NDC in [-1,1].
 *   Vulkan bias:  u = ndc.x*0.5+0.5,  v = ndc.y*0.5+0.5  (ndc.y+1→bottom)
 *   WebGPU fix:   u = ndc.x*0.5+0.5,  v = -ndc.y*0.5+0.5  (flip V)
 *
 * Uses @builtin(front_facing) to only render reflection on the top face.
 * Applies a 7×7 box blur (radius 3) for a soft-edge effect.
 * -------------------------------------------------------------------------- */
// clang-format off
static const char* offscreen_mirror_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    light_pos  : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  @group(0) @binding(1) var offscreen_sampler : sampler;
  @group(0) @binding(2) var offscreen_tex     : texture_2d<f32>;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv0      : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       proj_pos : vec4f,
  };

  @vertex
  fn vs_mirror(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    let clip = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);
    out.position = clip;
    out.proj_pos = clip;
    return out;
  }

  @fragment
  fn fs_mirror(
    in : VertexOutput,
    @builtin(front_facing) front_facing : bool
  ) -> @location(0) vec4f {
    // Perspective divide → NDC [-1, 1]
    let inv_w  = 1.0 / in.proj_pos.w;
    let ndc_x  = in.proj_pos.x * inv_w;
    let ndc_y  = in.proj_pos.y * inv_w;

    // Projective UV: flip V for WebGPU Y-up clip space
    let u = ndc_x *  0.5 + 0.5;
    let v = ndc_y * -0.5 + 0.5;   // negate Y to match texture orientation

    // 7×7 box blur — must be outside any non-uniform branch (WGSL uniform CF)
    let blur = 1.0 / 512.0;
    var reflection = vec4f(0.0);
    for (var x = -3; x <= 3; x++) {
      for (var y = -3; y <= 3; y++) {
        reflection += textureSample(
          offscreen_tex, offscreen_sampler,
          vec2f(u + f32(x) * blur, v + f32(y) * blur)
        );
      }
    }
    reflection /= 49.0;

    // Only show the reflection on the front face; back face → black
    let mask = select(0.0, 1.0, front_facing);
    return vec4f(reflection.rgb * mask, 1.0);
  }
);
// clang-format on

/* ---- Quad shader (debug: display offscreen texture fullscreen) ------------
 *
 * Generates a fullscreen triangle from vertex_index (no vertex buffer).
 * V coordinate is flipped for WebGPU Y-up clip space so the texture is
 * displayed right-side up (texture UV origin is top-left in both APIs).
 * -------------------------------------------------------------------------- */
// clang-format off
static const char* offscreen_quad_shader_wgsl = CODE(
  @group(0) @binding(1) var quad_sampler : sampler;
  @group(0) @binding(2) var quad_tex     : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv       : vec2f,
  };

  @vertex
  fn vs_quad(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out : VertexOutput;
    // Generate fullscreen triangle UV in [0,2] then clip-map to [-1,1]
    let uv       = vec2f(
      f32((vertex_index << 1u) & 2u),
      f32( vertex_index        & 2u)
    );
    // Flip V so texture top maps to clip top (+Y in WebGPU)
    out.uv       = vec2f(uv.x, 1.0 - uv.y);
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_quad(in : VertexOutput) -> @location(0) vec4f {
    return textureSample(quad_tex, quad_sampler, in.uv);
  }
);
// clang-format on
