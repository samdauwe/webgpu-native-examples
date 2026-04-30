/* -------------------------------------------------------------------------- *
 * WebGPU Example - Screen Space Ambient Occlusion (SSAO)
 *
 * Port of the Vulkan SSAO example by Sascha Willems to C99 WebGPU.
 *
 * The example demonstrates Screen Space Ambient Occlusion (SSAO) using a
 * multi-pass deferred rendering pipeline:
 *
 *  Pass 1 – G-Buffer: Renders view-space positions + linear depth, normals,
 *           and albedo into multiple render targets.
 *  Pass 2 – SSAO generation: Calculates ambient occlusion from the G-buffer
 *           using 64 hemisphere kernel samples oriented by a small noise
 *           texture (Gram-Schmidt reorthogonalization / TBN).
 *  Pass 3 – SSAO blur: A simple 5×5 box filter smooths the raw AO.
 *  Pass 4 – Composition: Combines the lit scene colour with the AO factor
 *           and renders to the swapchain.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/ssao/ssao.cpp
 * -------------------------------------------------------------------------- */

#include "core/camera.h"
#include "core/gltf_model.h"
#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

/* Note: camera.c uses cglm's default [-1,1] NDC depth projection (OpenGL).
 * The linearDepth shader function is written to match that convention.
 * CGLM_FORCE_DEPTH_ZERO_TO_ONE is NOT needed here because no cglm projection
 * calls are made in this file. */
#include <cglm/cglm.h>
#include <stdlib.h>
#include <string.h>

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
 * Constants
 * -------------------------------------------------------------------------- */

#define SSAO_KERNEL_SIZE (64)
#define SSAO_RADIUS (0.3f)
#define SSAO_NOISE_DIM (8)

/* -------------------------------------------------------------------------- *
 * WGSL Shader sources (forward declarations – definitions at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* gbuffer_shader_wgsl;
static const char* ssao_shader_wgsl;
static const char* blur_shader_wgsl;
static const char* composition_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * State structure
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Scene model */
  gltf_model_t model;
  bool model_loaded;
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;

  /* Offscreen G-Buffer textures */
  struct {
    WGPUTexture position_texture; /* RGBA32Float – view pos + linear depth */
    WGPUTextureView position_view;
    WGPUTexture normal_texture; /* RGBA8Unorm – packed normals */
    WGPUTextureView normal_view;
    WGPUTexture albedo_texture; /* RGBA8Unorm – material color */
    WGPUTextureView albedo_view;
    WGPUTexture depth_texture; /* Depth24PlusStencil8 */
    WGPUTextureView depth_view;
  } gbuffer;

  /* SSAO framebuffer */
  struct {
    WGPUTexture color_texture; /* R8Unorm */
    WGPUTextureView color_view;
  } ssao_fb;

  /* SSAO blur framebuffer */
  struct {
    WGPUTexture color_texture; /* R8Unorm */
    WGPUTextureView color_view;
  } ssao_blur_fb;

  /* Shared sampler for all offscreen textures */
  WGPUSampler color_sampler;

  /* SSAO noise texture */
  WGPUTexture noise_texture;
  WGPUTextureView noise_view;
  WGPUSampler noise_sampler;

  /* Uniform buffers */
  WGPUBuffer scene_ubo;       /* UBOSceneParams */
  WGPUBuffer ssao_kernel_ubo; /* SSAO kernel samples */
  WGPUBuffer ssao_params_ubo; /* UBOSSAOParams */

  /* Bind group layouts */
  WGPUBindGroupLayout gbuffer_bgl;
  WGPUBindGroupLayout ssao_bgl;
  WGPUBindGroupLayout ssao_blur_bgl;
  WGPUBindGroupLayout composition_bgl;

  /* Bind groups */
  WGPUBindGroup gbuffer_bg;
  WGPUBindGroup ssao_bg;
  WGPUBindGroup ssao_blur_bg;
  WGPUBindGroup composition_bg;

  /* Pipeline layouts */
  WGPUPipelineLayout gbuffer_pipeline_layout;
  WGPUPipelineLayout ssao_pipeline_layout;
  WGPUPipelineLayout ssao_blur_pipeline_layout;
  WGPUPipelineLayout composition_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline gbuffer_pipeline;
  WGPURenderPipeline ssao_pipeline;
  WGPURenderPipeline ssao_blur_pipeline;
  WGPURenderPipeline composition_pipeline;

  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_atts[3];
    WGPURenderPassDepthStencilAttachment depth_att;
    WGPURenderPassDescriptor desc;
  } gbuffer_pass;

  struct {
    WGPURenderPassColorAttachment color_att;
    WGPURenderPassDescriptor desc;
  } ssao_pass;

  struct {
    WGPURenderPassColorAttachment color_att;
    WGPURenderPassDescriptor desc;
  } ssao_blur_pass;

  struct {
    WGPURenderPassColorAttachment color_att;
    WGPURenderPassDepthStencilAttachment depth_att;
    WGPURenderPassDescriptor desc;
  } composition_pass;

  /* CPU-side uniform data */
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
    float near_plane;
    float far_plane;
    float _pad[2]; /* align to 16 bytes */
  } ubo_scene;

  struct {
    mat4 projection;
    int32_t ssao;
    int32_t ssao_only;
    int32_t ssao_blur;
    int32_t _pad;
  } ubo_ssao_params;

  /* GUI settings */
  struct {
    bool enable_ssao;
    bool ssao_blur;
    bool ssao_only;
  } settings;

  /* Per-material textures for the G-Buffer pass (group 1) */
  struct {
    WGPUTexture* gpu_textures;
    WGPUTextureView* views;
    uint32_t count;
  } mat_textures;
  WGPUSampler material_sampler;
  WGPUTexture default_color_texture;
  WGPUTextureView default_color_view;
  WGPUBindGroupLayout material_bgl;
  WGPUBindGroup* material_bgs;
  uint32_t material_bg_count;
  WGPUBindGroup default_material_bg;

  /* Timing */
  uint64_t last_frame_time;

  /* Lifecycle */
  WGPUBool initialized;
} state = {
  .ubo_scene = {
    .near_plane = 0.1f,
    .far_plane  = 64.0f,
  },
  .ubo_ssao_params = {
    .ssao      = 1,
    .ssao_only = 0,
    .ssao_blur = 1,
  },
  .settings = {
    .enable_ssao = true,
    .ssao_blur   = true,
    .ssao_only   = false,
  },
  .initialized = false,
};

/* -------------------------------------------------------------------------- *
 * Loading descriptor for the glTF model
 * -------------------------------------------------------------------------- */

static const gltf_model_desc_t model_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

/* -------------------------------------------------------------------------- *
 * Forward declarations
 * -------------------------------------------------------------------------- */

static void create_offscreen_framebuffers(wgpu_context_t* wgpu_context);
static void destroy_offscreen_framebuffers(void);
static void create_uniform_buffers(wgpu_context_t* wgpu_context);
static void create_noise_texture(wgpu_context_t* wgpu_context);
static void setup_bind_group_layouts(wgpu_context_t* wgpu_context);
static void setup_bind_groups(wgpu_context_t* wgpu_context);
static void setup_pipeline_layouts(wgpu_context_t* wgpu_context);
static void setup_render_pipelines(wgpu_context_t* wgpu_context);
static void update_uniform_buffers(wgpu_context_t* wgpu_context);

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  bool ok = gltf_model_load_from_file(
    &state.model, "assets/models/Sponza/glTF/Sponza.gltf", 1.0f);
  if (!ok) {
    fprintf(stderr, "Failed to load Sponza.gltf\n");
    return;
  }
  state.model_loaded = true;
}

static void create_model_buffers(wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.model;

  /* Bake node transforms into vertex positions/normals */
  size_t vb_size     = m->vertex_count * sizeof(gltf_vertex_t);
  gltf_vertex_t* buf = (gltf_vertex_t*)malloc(vb_size);
  memcpy(buf, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, buf, &model_load_desc);

  /* Vertex buffer */
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("SSAO - Vertex Buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, buf, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);
  free(buf);

  /* Index buffer */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("SSAO - Index Buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = true,
              });
    void* idata = wgpuBufferGetMappedRange(state.index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.index_buffer);
  }
}

/* -------------------------------------------------------------------------- *
 * Per-material textures for the G-Buffer pass
 * -------------------------------------------------------------------------- */

static void create_material_textures(wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.model;

  /* Filtering sampler for material textures */
  state.material_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Material Sampler"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .mipmapFilter  = WGPUMipmapFilterMode_Linear,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 1.0f,
              .maxAnisotropy = 1,
            });

  /* Default 1x1 white texture for materials without a base color texture */
  {
    uint8_t white_pixel[4]      = {255, 255, 255, 255};
    state.default_color_texture = wgpuDeviceCreateTexture(
      device,
      &(WGPUTextureDescriptor){
        .label     = STRVIEW("Default White Texture"),
        .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size      = {1, 1, 1},
        .format    = WGPUTextureFormat_RGBA8UnormSrgb,
        .mipLevelCount = 1,
        .sampleCount   = 1,
      });
    wgpuQueueWriteTexture(wgpu_context->queue,
                          &(WGPUTexelCopyTextureInfo){
                            .texture  = state.default_color_texture,
                            .mipLevel = 0,
                            .origin   = {0, 0, 0},
                            .aspect   = WGPUTextureAspect_All,
                          },
                          white_pixel, sizeof(white_pixel),
                          &(WGPUTexelCopyBufferLayout){
                            .offset       = 0,
                            .bytesPerRow  = 4,
                            .rowsPerImage = 1,
                          },
                          &(WGPUExtent3D){1, 1, 1});
    state.default_color_view = wgpuTextureCreateView(
      state.default_color_texture, &(WGPUTextureViewDescriptor){
                                     .format = WGPUTextureFormat_RGBA8UnormSrgb,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .mipLevelCount   = 1,
                                     .arrayLayerCount = 1,
                                   });
  }

  /* Upload model textures to GPU */
  state.mat_textures.count = m->texture_count;
  if (m->texture_count > 0) {
    state.mat_textures.gpu_textures
      = (WGPUTexture*)calloc(m->texture_count, sizeof(WGPUTexture));
    state.mat_textures.views
      = (WGPUTextureView*)calloc(m->texture_count, sizeof(WGPUTextureView));
    for (uint32_t i = 0; i < m->texture_count; i++) {
      gltf_texture_t* tex = &m->textures[i];
      if (!tex->data || tex->width == 0 || tex->height == 0) {
        continue;
      }
      state.mat_textures.gpu_textures[i] = wgpuDeviceCreateTexture(
        device,
        &(WGPUTextureDescriptor){
          .label = STRVIEW("Material Texture"),
          .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
          .dimension     = WGPUTextureDimension_2D,
          .size          = {tex->width, tex->height, 1},
          .format        = WGPUTextureFormat_RGBA8UnormSrgb,
          .mipLevelCount = 1,
          .sampleCount   = 1,
        });
      uint32_t data_size = tex->width * tex->height * 4;
      wgpuQueueWriteTexture(wgpu_context->queue,
                            &(WGPUTexelCopyTextureInfo){
                              .texture  = state.mat_textures.gpu_textures[i],
                              .mipLevel = 0,
                              .origin   = {0, 0, 0},
                              .aspect   = WGPUTextureAspect_All,
                            },
                            tex->data, data_size,
                            &(WGPUTexelCopyBufferLayout){
                              .offset       = 0,
                              .bytesPerRow  = tex->width * 4,
                              .rowsPerImage = tex->height,
                            },
                            &(WGPUExtent3D){tex->width, tex->height, 1});
      state.mat_textures.views[i]
        = wgpuTextureCreateView(state.mat_textures.gpu_textures[i],
                                &(WGPUTextureViewDescriptor){
                                  .format    = WGPUTextureFormat_RGBA8UnormSrgb,
                                  .dimension = WGPUTextureViewDimension_2D,
                                  .mipLevelCount   = 1,
                                  .arrayLayerCount = 1,
                                });
    }
  }

  /* Bind group layout for per-material texture (group 1):
   * binding 0 = texture, binding 1 = sampler */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
    };
    state.material_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Material BGL"),
                .entryCount = 2,
                .entries    = entries,
              });
  }

  /* Default material bind group (white texture) */
  {
    WGPUBindGroupEntry entries[2] = {
      {.binding = 0, .textureView = state.default_color_view},
      {.binding = 1, .sampler = state.material_sampler},
    };
    state.default_material_bg = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Default Material BG"),
                .layout     = state.material_bgl,
                .entryCount = 2,
                .entries    = entries,
              });
  }

  /* Per-material bind groups */
  state.material_bg_count = m->material_count;
  if (m->material_count > 0) {
    state.material_bgs
      = (WGPUBindGroup*)calloc(m->material_count, sizeof(WGPUBindGroup));
    for (uint32_t i = 0; i < m->material_count; i++) {
      gltf_material_t* mat = &m->materials[i];
      WGPUTextureView view = state.default_color_view;
      if (mat->base_color_tex_index >= 0
          && (uint32_t)mat->base_color_tex_index < state.mat_textures.count
          && state.mat_textures.views[mat->base_color_tex_index] != NULL) {
        view = state.mat_textures.views[mat->base_color_tex_index];
      }
      WGPUBindGroupEntry entries[2] = {
        {.binding = 0, .textureView = view},
        {.binding = 1, .sampler = state.material_sampler},
      };
      state.material_bgs[i]
        = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                              .label  = STRVIEW("Material BG"),
                                              .layout = state.material_bgl,
                                              .entryCount = 2,
                                              .entries    = entries,
                                            });
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Offscreen framebuffers
 * -------------------------------------------------------------------------- */

static void create_offscreen_framebuffers(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  /* --- G-Buffer textures --- */

  /* Position + linear depth (RGBA32Float) */
  state.gbuffer.position_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("GBuffer Position"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_RGBA32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.gbuffer.position_view = wgpuTextureCreateView(
    state.gbuffer.position_texture, &(WGPUTextureViewDescriptor){
                                      .format = WGPUTextureFormat_RGBA32Float,
                                      .dimension = WGPUTextureViewDimension_2D,
                                      .mipLevelCount   = 1,
                                      .arrayLayerCount = 1,
                                    });

  /* Normals (RGBA8Unorm) */
  state.gbuffer.normal_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("GBuffer Normal"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_RGBA8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.gbuffer.normal_view = wgpuTextureCreateView(
    state.gbuffer.normal_texture, &(WGPUTextureViewDescriptor){
                                    .format    = WGPUTextureFormat_RGBA8Unorm,
                                    .dimension = WGPUTextureViewDimension_2D,
                                    .mipLevelCount   = 1,
                                    .arrayLayerCount = 1,
                                  });

  /* Albedo (RGBA8Unorm) */
  state.gbuffer.albedo_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("GBuffer Albedo"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_RGBA8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.gbuffer.albedo_view = wgpuTextureCreateView(
    state.gbuffer.albedo_texture, &(WGPUTextureViewDescriptor){
                                    .format    = WGPUTextureFormat_RGBA8Unorm,
                                    .dimension = WGPUTextureViewDimension_2D,
                                    .mipLevelCount   = 1,
                                    .arrayLayerCount = 1,
                                  });

  /* Depth */
  state.gbuffer.depth_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("GBuffer Depth"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_Depth24PlusStencil8,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.gbuffer.depth_view
    = wgpuTextureCreateView(state.gbuffer.depth_texture,
                            &(WGPUTextureViewDescriptor){
                              .format = WGPUTextureFormat_Depth24PlusStencil8,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .mipLevelCount   = 1,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });

  /* --- SSAO texture (R8Unorm) --- */
  state.ssao_fb.color_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("SSAO Color"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_R8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.ssao_fb.color_view = wgpuTextureCreateView(
    state.ssao_fb.color_texture, &(WGPUTextureViewDescriptor){
                                   .format        = WGPUTextureFormat_R8Unorm,
                                   .dimension     = WGPUTextureViewDimension_2D,
                                   .mipLevelCount = 1,
                                   .arrayLayerCount = 1,
                                 });

  /* --- SSAO blur texture (R8Unorm) --- */
  state.ssao_blur_fb.color_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("SSAO Blur Color"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_R8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.ssao_blur_fb.color_view
    = wgpuTextureCreateView(state.ssao_blur_fb.color_texture,
                            &(WGPUTextureViewDescriptor){
                              .format          = WGPUTextureFormat_R8Unorm,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .mipLevelCount   = 1,
                              .arrayLayerCount = 1,
                            });

  /* --- Shared color sampler (NEAREST for G-Buffer to preserve exact values) */
  state.color_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Color Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Nearest,
              .minFilter     = WGPUFilterMode_Nearest,
              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 1.0f,
              .maxAnisotropy = 1,
            });
}

static void destroy_offscreen_framebuffers(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.position_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.position_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.normal_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.normal_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.albedo_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.albedo_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.depth_texture)

  WGPU_RELEASE_RESOURCE(TextureView, state.ssao_fb.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.ssao_fb.color_texture)

  WGPU_RELEASE_RESOURCE(TextureView, state.ssao_blur_fb.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.ssao_blur_fb.color_texture)

  WGPU_RELEASE_RESOURCE(Sampler, state.color_sampler)
}

/* -------------------------------------------------------------------------- *
 * SSAO noise texture & kernel
 * -------------------------------------------------------------------------- */

/* Simple pseudo-random: deterministic for reproducible results */
static uint32_t rng_state = 42u;

static float rand_float(void)
{
  /* xorshift32 */
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return (float)(rng_state & 0x00FFFFFFu) / (float)0x00FFFFFFu;
}

static float lerp_f(float a, float b, float f)
{
  return a + f * (b - a);
}

static void create_noise_texture(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Generate 2D noise vectors (xy random in [-1,1], zw = 0) */
  const uint32_t noise_count = SSAO_NOISE_DIM * SSAO_NOISE_DIM;
  float noise_data[SSAO_NOISE_DIM * SSAO_NOISE_DIM * 4];
  for (uint32_t i = 0; i < noise_count; i++) {
    noise_data[i * 4 + 0] = rand_float() * 2.0f - 1.0f;
    noise_data[i * 4 + 1] = rand_float() * 2.0f - 1.0f;
    noise_data[i * 4 + 2] = 0.0f;
    noise_data[i * 4 + 3] = 0.0f;
  }

  /* Create noise texture (RGBA32Float for precision, matches Vulkan) */
  state.noise_texture = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("SSAO Noise Texture"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {SSAO_NOISE_DIM, SSAO_NOISE_DIM, 1},
      .format    = WGPUTextureFormat_RGBA32Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.noise_texture,
                          .mipLevel = 0,
                          .origin   = {0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        noise_data, sizeof(noise_data),
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = SSAO_NOISE_DIM * 4 * sizeof(float),
                          .rowsPerImage = SSAO_NOISE_DIM,
                        },
                        &(WGPUExtent3D){SSAO_NOISE_DIM, SSAO_NOISE_DIM, 1});

  state.noise_view = wgpuTextureCreateView(
    state.noise_texture, &(WGPUTextureViewDescriptor){
                           .format          = WGPUTextureFormat_RGBA32Float,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });

  state.noise_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Noise Sampler"),
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_Repeat,
              .addressModeW  = WGPUAddressMode_Repeat,
              .magFilter     = WGPUFilterMode_Nearest,
              .minFilter     = WGPUFilterMode_Nearest,
              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
              .lodMinClamp   = 0.0f,
              .lodMaxClamp   = 1.0f,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void create_uniform_buffers(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- Scene UBO ---- */
  state.scene_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.ubo_scene),
            });

  /* ---- SSAO kernel UBO ---- */
  /* Generate hemisphere kernel samples */
  float ssao_kernel[SSAO_KERNEL_SIZE * 4];
  for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; i++) {
    vec3 sample_v;
    sample_v[0] = rand_float() * 2.0f - 1.0f;
    sample_v[1] = rand_float() * 2.0f - 1.0f;
    sample_v[2] = rand_float(); /* hemisphere: z in [0, 1] */
    glm_vec3_normalize(sample_v);
    glm_vec3_scale(sample_v, rand_float(), sample_v);

    /* Importance weighting: more samples close to the origin */
    float scale = (float)i / (float)SSAO_KERNEL_SIZE;
    scale       = lerp_f(0.1f, 1.0f, scale * scale);
    glm_vec3_scale(sample_v, scale, sample_v);

    ssao_kernel[i * 4 + 0] = sample_v[0];
    ssao_kernel[i * 4 + 1] = sample_v[1];
    ssao_kernel[i * 4 + 2] = sample_v[2];
    ssao_kernel[i * 4 + 3] = 0.0f;
  }

  state.ssao_kernel_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("SSAO Kernel UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(ssao_kernel),
              .mappedAtCreation = true,
            });
  void* kdata
    = wgpuBufferGetMappedRange(state.ssao_kernel_ubo, 0, sizeof(ssao_kernel));
  memcpy(kdata, ssao_kernel, sizeof(ssao_kernel));
  wgpuBufferUnmap(state.ssao_kernel_ubo);

  /* ---- SSAO params UBO ---- */
  state.ssao_params_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("SSAO Params UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.ubo_ssao_params),
            });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* Scene uniforms */
  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_scene.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_scene.view);
  glm_mat4_identity(state.ubo_scene.model);
  wgpuQueueWriteBuffer(queue, state.scene_ubo, 0, &state.ubo_scene,
                       sizeof(state.ubo_scene));

  /* SSAO params */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.ubo_ssao_params.projection);
  state.ubo_ssao_params.ssao      = state.settings.enable_ssao ? 1 : 0;
  state.ubo_ssao_params.ssao_only = state.settings.ssao_only ? 1 : 0;
  state.ubo_ssao_params.ssao_blur = state.settings.ssao_blur ? 1 : 0;
  wgpuQueueWriteBuffer(queue, state.ssao_params_ubo, 0, &state.ubo_ssao_params,
                       sizeof(state.ubo_ssao_params));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void setup_bind_group_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- G-Buffer: binding 0 = scene UBO (vert+frag) ---- */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
    };
    state.gbuffer_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("GBuffer BGL"),
                .entryCount = 1,
                .entries    = entries,
              });
  }

  /* ---- SSAO: 0=position tex, 1=position sampler, 2=normal tex, 3=normal
   * sampler, 4=noise tex, 5=noise sampler, 6=kernel UBO, 7=ssao_params UBO */
  {
    WGPUBindGroupLayoutEntry entries[8] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_UnfilterableFloat,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_UnfilterableFloat,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
      {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
    };
    state.ssao_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("SSAO BGL"),
                .entryCount = 8,
                .entries    = entries,
              });
  }

  /* ---- SSAO Blur: 0=ssao tex, 1=ssao sampler ---- */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
    };
    state.ssao_blur_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("SSAO Blur BGL"),
                .entryCount = 2,
                .entries    = entries,
              });
  }

  /* ---- Composition: 0=position tex, 1=position samp, 2=normal tex,
   * 3=normal samp, 4=albedo tex, 5=albedo samp, 6=ssao tex, 7=ssao samp,
   * 8=ssao blur tex, 9=ssao blur samp, 10=params UBO ---- */
  {
    WGPUBindGroupLayoutEntry entries[11] = {
      /* Position texture + sampler */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_UnfilterableFloat,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      /* Normal texture + sampler */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      /* Albedo texture + sampler */
      {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      /* SSAO texture + sampler */
      {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 7,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      /* SSAO Blur texture + sampler */
      {
        .binding    = 8,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D},
      },
      {
        .binding    = 9,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      /* Params UBO */
      {
        .binding    = 10,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
    };
    state.composition_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition BGL"),
                .entryCount = 11,
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void setup_bind_groups(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ---- G-Buffer ---- */
  {
    WGPUBindGroupEntry entries[1] = {
      {
        .binding = 0,
        .buffer  = state.scene_ubo,
        .size    = sizeof(state.ubo_scene),
      },
    };
    state.gbuffer_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW("GBuffer BG"),
                                            .layout     = state.gbuffer_bgl,
                                            .entryCount = 1,
                                            .entries    = entries,
                                          });
  }

  /* ---- SSAO ---- */
  {
    WGPUBindGroupEntry entries[8] = {
      {.binding = 0, .textureView = state.gbuffer.position_view},
      {.binding = 1, .sampler = state.color_sampler},
      {.binding = 2, .textureView = state.gbuffer.normal_view},
      {.binding = 3, .sampler = state.color_sampler},
      {.binding = 4, .textureView = state.noise_view},
      {.binding = 5, .sampler = state.noise_sampler},
      {.binding = 6,
       .buffer  = state.ssao_kernel_ubo,
       .size    = SSAO_KERNEL_SIZE * 4 * sizeof(float)},
      {.binding = 7,
       .buffer  = state.ssao_params_ubo,
       .size    = sizeof(state.ubo_ssao_params)},
    };
    state.ssao_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW("SSAO BG"),
                                            .layout     = state.ssao_bgl,
                                            .entryCount = 8,
                                            .entries    = entries,
                                          });
  }

  /* ---- SSAO Blur ---- */
  {
    WGPUBindGroupEntry entries[2] = {
      {.binding = 0, .textureView = state.ssao_fb.color_view},
      {.binding = 1, .sampler = state.color_sampler},
    };
    state.ssao_blur_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("SSAO Blur BG"),
                                            .layout = state.ssao_blur_bgl,
                                            .entryCount = 2,
                                            .entries    = entries,
                                          });
  }

  /* ---- Composition ---- */
  {
    WGPUBindGroupEntry entries[11] = {
      {.binding = 0, .textureView = state.gbuffer.position_view},
      {.binding = 1, .sampler = state.color_sampler},
      {.binding = 2, .textureView = state.gbuffer.normal_view},
      {.binding = 3, .sampler = state.color_sampler},
      {.binding = 4, .textureView = state.gbuffer.albedo_view},
      {.binding = 5, .sampler = state.color_sampler},
      {.binding = 6, .textureView = state.ssao_fb.color_view},
      {.binding = 7, .sampler = state.color_sampler},
      {.binding = 8, .textureView = state.ssao_blur_fb.color_view},
      {.binding = 9, .sampler = state.color_sampler},
      {.binding = 10,
       .buffer  = state.ssao_params_ubo,
       .size    = sizeof(state.ubo_ssao_params)},
    };
    state.composition_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Composition BG"),
                                            .layout = state.composition_bgl,
                                            .entryCount = 11,
                                            .entries    = entries,
                                          });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline layouts
 * -------------------------------------------------------------------------- */

static void setup_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.gbuffer_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("GBuffer Pipeline Layout"),
              .bindGroupLayoutCount = 2,
              .bindGroupLayouts
              = (WGPUBindGroupLayout[]){state.gbuffer_bgl, state.material_bgl},
            });

  state.ssao_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("SSAO Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.ssao_bgl,
            });

  state.ssao_blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("SSAO Blur Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.ssao_blur_bgl,
            });

  state.composition_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Composition Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.composition_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void setup_render_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Create shader modules */
  WGPUShaderModule gbuffer_sm
    = wgpu_create_shader_module(device, gbuffer_shader_wgsl);
  WGPUShaderModule ssao_sm
    = wgpu_create_shader_module(device, ssao_shader_wgsl);
  WGPUShaderModule blur_sm
    = wgpu_create_shader_module(device, blur_shader_wgsl);
  WGPUShaderModule composition_sm
    = wgpu_create_shader_module(device, composition_shader_wgsl);

  /* ---- G-Buffer pipeline ---- */
  {
    /* Vertex buffer layout matching gltf_vertex_t:
     *   position (vec3f) + normal (vec3f) + uv0 (vec2f) + uv1 (vec2f)
     *   + tangent (vec4f) + color (vec4f) + joint0 (4xu32) + weight0 (vec4f)
     * Total stride = sizeof(gltf_vertex_t) */
    WGPUVertexAttribute attrs[4] = {
      {/* position */
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position),
       .shaderLocation = 0},
      {/* normal */
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal),
       .shaderLocation = 1},
      {/* uv0 */
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0),
       .shaderLocation = 2},
      {/* color */
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color),
       .shaderLocation = 3},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 4,
      .attributes     = attrs,
    };

    /* 3 color targets for MRT */
    WGPUColorTargetState color_targets[3] = {
      {.format    = WGPUTextureFormat_RGBA32Float,
       .writeMask = WGPUColorWriteMask_All},
      {.format    = WGPUTextureFormat_RGBA8Unorm,
       .writeMask = WGPUColorWriteMask_All},
      {.format    = WGPUTextureFormat_RGBA8Unorm,
       .writeMask = WGPUColorWriteMask_All},
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("GBuffer Pipeline"),
      .layout = state.gbuffer_pipeline_layout,
      .vertex = {
        .module      = gbuffer_sm,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &vb_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = gbuffer_sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 3,
        .targets     = color_targets,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Back,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = WGPUTextureFormat_Depth24PlusStencil8,
        .depthWriteEnabled   = WGPUOptionalBool_True,
        .depthCompare        = WGPUCompareFunction_LessEqual,
        .stencilFront        = {.compare = WGPUCompareFunction_Always},
        .stencilBack         = {.compare = WGPUCompareFunction_Always},
      },
      .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    };

    state.gbuffer_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
  }

  /* ---- SSAO generation pipeline (fullscreen triangle) ---- */
  {
    WGPUColorTargetState color_target = {
      .format    = WGPUTextureFormat_R8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("SSAO Pipeline"),
      .layout = state.ssao_pipeline_layout,
      .vertex = {
        .module      = ssao_sm,
        .entryPoint  = STRVIEW("vs_fullscreen"),
        .bufferCount = 0,
      },
      .fragment = &(WGPUFragmentState){
        .module      = ssao_sm,
        .entryPoint  = STRVIEW("fs_ssao"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    };

    state.ssao_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
  }

  /* ---- SSAO blur pipeline (fullscreen triangle) ---- */
  {
    WGPUColorTargetState color_target = {
      .format    = WGPUTextureFormat_R8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("SSAO Blur Pipeline"),
      .layout = state.ssao_blur_pipeline_layout,
      .vertex = {
        .module      = blur_sm,
        .entryPoint  = STRVIEW("vs_fullscreen"),
        .bufferCount = 0,
      },
      .fragment = &(WGPUFragmentState){
        .module      = blur_sm,
        .entryPoint  = STRVIEW("fs_blur"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    };

    state.ssao_blur_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
  }

  /* ---- Composition pipeline (fullscreen triangle) ---- */
  {
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Composition Pipeline"),
      .layout = state.composition_pipeline_layout,
      .vertex = {
        .module      = composition_sm,
        .entryPoint  = STRVIEW("vs_fullscreen"),
        .bufferCount = 0,
      },
      .fragment = &(WGPUFragmentState){
        .module      = composition_sm,
        .entryPoint  = STRVIEW("fs_composition"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format              = wgpu_context->depth_stencil_format,
        .depthWriteEnabled   = WGPUOptionalBool_False,
        .depthCompare        = WGPUCompareFunction_Always,
        .stencilFront        = {.compare = WGPUCompareFunction_Always},
        .stencilBack         = {.compare = WGPUCompareFunction_Always},
      },
      .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    };

    state.composition_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
  }

  /* Release shader modules */
  wgpuShaderModuleRelease(gbuffer_sm);
  wgpuShaderModuleRelease(ssao_sm);
  wgpuShaderModuleRelease(blur_sm);
  wgpuShaderModuleRelease(composition_sm);
}

/* -------------------------------------------------------------------------- *
 * Resize handling – recreate resolution-dependent resources
 * -------------------------------------------------------------------------- */

static void on_resize(wgpu_context_t* wgpu_context)
{
  /* Release old bind groups that reference old texture views */
  WGPU_RELEASE_RESOURCE(BindGroup, state.ssao_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.ssao_blur_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bg)

  /* Recreate offscreen framebuffers at new size */
  destroy_offscreen_framebuffers();
  create_offscreen_framebuffers(wgpu_context);

  /* Recreate bind groups with new texture views */
  setup_bind_groups(wgpu_context);

  /* Update camera aspect ratio */
  camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                              / (float)wgpu_context->height);
}

/* -------------------------------------------------------------------------- *
 * Draw helpers
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  if (!state.model_loaded) {
    return;
  }

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }

  gltf_model_t* m = &state.model;
  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
    if (node->mesh == NULL) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];

      /* Bind per-material texture (group 1) */
      if (prim->material_index >= 0
          && (uint32_t)prim->material_index < state.material_bg_count) {
        wgpuRenderPassEncoderSetBindGroup(
          pass, 1, state.material_bgs[prim->material_index], 0, NULL);
      }
      else {
        wgpuRenderPassEncoderSetBindGroup(pass, 1, state.default_material_bg, 0,
                                          NULL);
      }

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

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Enable SSAO", &state.settings.enable_ssao);
    igCheckbox("SSAO blur", &state.settings.ssao_blur);
    igCheckbox("SSAO pass only", &state.settings.ssao_only);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event callback
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    on_resize(wgpu_context);
    return;
  }

  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }
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

  /* Camera setup – first person, matching Vulkan example */
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.rotation_speed = 0.25f;
  state.camera.movement_speed = 1.0f;
  camera_set_position(&state.camera, (vec3){1.0f, 0.75f, 0.0f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 90.0f, 0.0f});
  camera_set_perspective(&state.camera, 60.0f,
                         (float)wgpu_context->width
                           / (float)wgpu_context->height,
                         state.ubo_scene.near_plane, state.ubo_scene.far_plane);

  /* Load model */
  load_model();
  create_model_buffers(wgpu_context);
  create_material_textures(wgpu_context);

  /* Create offscreen resources */
  create_offscreen_framebuffers(wgpu_context);
  create_noise_texture(wgpu_context);
  create_uniform_buffers(wgpu_context);

  /* Descriptor sets / pipelines */
  setup_bind_group_layouts(wgpu_context);
  setup_bind_groups(wgpu_context);
  setup_pipeline_layouts(wgpu_context);
  setup_render_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

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

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Render ---- */
  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* ============ Pass 1: G-Buffer (MRT) ============ */
  {
    WGPURenderPassColorAttachment color_atts[3] = {
      {
        .view       = state.gbuffer.position_view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0.0, 0.0, 0.0, 1.0},
      },
      {
        .view       = state.gbuffer.normal_view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0.0, 0.0, 0.0, 1.0},
      },
      {
        .view       = state.gbuffer.albedo_view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0.0, 0.0, 0.0, 1.0},
      },
    };
    WGPURenderPassDepthStencilAttachment depth_att = {
      .view              = state.gbuffer.depth_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };
    WGPURenderPassDescriptor rp_desc = {
      .colorAttachmentCount   = 3,
      .colorAttachments       = color_atts,
      .depthStencilAttachment = &depth_att,
    };

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &rp_desc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.gbuffer_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.gbuffer_bg, 0, NULL);
    draw_model(pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 2: SSAO Generation ============ */
  {
    WGPURenderPassColorAttachment color_att = {
      .view       = state.ssao_fb.color_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
    };
    WGPURenderPassDescriptor rp_desc = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &color_att,
    };

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &rp_desc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.ssao_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.ssao_bg, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 3: SSAO Blur ============ */
  {
    WGPURenderPassColorAttachment color_att = {
      .view       = state.ssao_blur_fb.color_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
    };
    WGPURenderPassDescriptor rp_desc = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &color_att,
    };

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &rp_desc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.ssao_blur_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.ssao_blur_bg, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 4: Composition (to swapchain) ============ */
  {
    WGPURenderPassColorAttachment color_att = {
      .view       = wgpu_context->swapchain_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0, 0.0, 0.0, 1.0},
    };
    WGPURenderPassDepthStencilAttachment depth_att = {
      .view              = wgpu_context->depth_stencil_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };
    WGPURenderPassDescriptor rp_desc = {
      .colorAttachmentCount   = 1,
      .colorAttachments       = &color_att,
      .depthStencilAttachment = &depth_att,
    };

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &rp_desc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.composition_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.composition_bg, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Model */
  gltf_model_destroy(&state.model);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Material textures */
  for (uint32_t i = 0; i < state.mat_textures.count; i++) {
    WGPU_RELEASE_RESOURCE(TextureView, state.mat_textures.views[i])
    WGPU_RELEASE_RESOURCE(Texture, state.mat_textures.gpu_textures[i])
  }
  free(state.mat_textures.views);
  free(state.mat_textures.gpu_textures);
  WGPU_RELEASE_RESOURCE(TextureView, state.default_color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.default_color_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.material_sampler)
  WGPU_RELEASE_RESOURCE(BindGroup, state.default_material_bg)
  for (uint32_t i = 0; i < state.material_bg_count; i++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.material_bgs[i])
  }
  free(state.material_bgs);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.material_bgl)

  /* Offscreen */
  destroy_offscreen_framebuffers();

  /* Noise texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.noise_view)
  WGPU_RELEASE_RESOURCE(Texture, state.noise_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.noise_sampler)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.ssao_kernel_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.ssao_params_ubo)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.gbuffer_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.ssao_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.ssao_blur_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bg)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.gbuffer_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.ssao_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.ssao_blur_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_bgl)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.gbuffer_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.ssao_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.ssao_blur_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_pipeline_layout)

  /* Render pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gbuffer_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.ssao_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.ssao_blur_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Screen Space Ambient Occlusion",
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

/* ---- G-Buffer shader ---- */
// clang-format off
static const char* gbuffer_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model      : mat4x4f,
    view       : mat4x4f,
    nearPlane  : f32,
    farPlane   : f32,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;

  @group(1) @binding(0) var colorMap     : texture_2d<f32>;
  @group(1) @binding(1) var colorSampler : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) color    : vec4f,
  }

  struct VertexOutput {
    @builtin(position) clip_position : vec4f,
    @location(0) normal   : vec3f,
    @location(1) uv       : vec2f,
    @location(2) color    : vec3f,
    @location(3) view_pos : vec3f,
  }

  struct GBufferOutput {
    @location(0) position : vec4f,
    @location(1) normal   : vec4f,
    @location(2) albedo   : vec4f,
  }

  @vertex
  fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    let pos4 = vec4f(input.position, 1.0);
    let mv = ubo.view * ubo.model;
    output.clip_position = ubo.projection * mv * pos4;
    output.uv = input.uv;
    output.view_pos = (mv * pos4).xyz;
    // Normal matrix = transpose(inverse(mat3(modelView)))
    // For orthogonal matrices (no non-uniform scale) this equals mat3(modelView)
    // The Vulkan reference uses transpose(inverse(mat3(view * model)))
    let mv3 = mat3x3f(mv[0].xyz, mv[1].xyz, mv[2].xyz);
    output.normal = mv3 * input.normal;
    output.color = input.color.rgb;
    return output;
  }

  fn linearDepth(depth : f32, near : f32, far : f32) -> f32 {
    // camera.c produces an OpenGL-style [-1,1] NDC projection (cglm default,
    // no CGLM_FORCE_DEPTH_ZERO_TO_ONE).  WebGPU clips NDC z to [0,1], so
    // @builtin(position).z is already the clipped NDC value.  The correct
    // inversion for a [-1,1] projection with [0,1] depth buffer is:
    //   d = 2*n*f / ((f+n) - depth*(f-n))
    return (2.0 * near * far) / (far + near - depth * (far - near));
  }

  @fragment
  fn fs_main(input : VertexOutput) -> GBufferOutput {
    var output : GBufferOutput;
    // builtin(position).z is the interpolated depth in [0,1]
    let depth = input.clip_position.z;
    output.position = vec4f(input.view_pos, linearDepth(depth, ubo.nearPlane, ubo.farPlane));
    output.normal = vec4f(normalize(input.normal) * 0.5 + 0.5, 1.0);
    output.albedo = textureSample(colorMap, colorSampler, input.uv) * vec4f(input.color, 1.0);
    return output;
  }
);
// clang-format on

/* ---- SSAO generation shader ---- */
// clang-format off
static const char* ssao_shader_wgsl = CODE(
  const SSAO_KERNEL_SIZE : i32 = 64;
  const SSAO_RADIUS : f32 = 0.3;

  @group(0) @binding(0) var samplerPositionDepth : texture_2d<f32>;
  @group(0) @binding(1) var positionSampler      : sampler;
  @group(0) @binding(2) var samplerNormal        : texture_2d<f32>;
  @group(0) @binding(3) var normalSampler        : sampler;
  @group(0) @binding(4) var ssaoNoiseTex         : texture_2d<f32>;
  @group(0) @binding(5) var noiseSampler         : sampler;

  struct SSAOKernel {
    samples : array<vec4f, 64>,
  }
  @group(0) @binding(6) var<uniform> uboSSAOKernel : SSAOKernel;

  struct SSAOParams {
    projection : mat4x4f,
  }
  @group(0) @binding(7) var<uniform> uboParams : SSAOParams;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_fullscreen(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    output.uv = vec2f(u, 1.0 - v);
    output.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return output;
  }

  @fragment
  fn fs_ssao(input : VertexOutput) -> @location(0) f32 {
    let fragPos = textureSample(samplerPositionDepth, positionSampler, input.uv).rgb;
    let normal = normalize(textureSample(samplerNormal, normalSampler, input.uv).rgb * 2.0 - 1.0);

    let texDim = textureDimensions(samplerPositionDepth, 0);
    let noiseDim = textureDimensions(ssaoNoiseTex, 0);
    let noiseUV = vec2f(f32(texDim.x) / f32(noiseDim.x), f32(texDim.y) / f32(noiseDim.y)) * input.uv;
    let randomVec = textureSample(ssaoNoiseTex, noiseSampler, noiseUV).xyz * 2.0 - 1.0;

    let tangent = normalize(randomVec - normal * dot(randomVec, normal));
    let bitangent = cross(tangent, normal);
    let TBN = mat3x3f(tangent, bitangent, normal);

    var occlusion = 0.0;
    let bias = 0.025;

    for (var i = 0; i < SSAO_KERNEL_SIZE; i++) {
      var samplePos = TBN * uboSSAOKernel.samples[i].xyz;
      samplePos = fragPos + samplePos * SSAO_RADIUS;

      var offset = vec4f(samplePos, 1.0);
      offset = uboParams.projection * offset;
      let ndc = offset.xyz / offset.w;
      // Map from NDC [-1,1] to UV [0,1]; flip Y for WebGPU
      let sampleUV = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

      let sampleDepth = -textureSample(samplerPositionDepth, positionSampler, sampleUV).w;

      let rangeCheck = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(fragPos.z - sampleDepth));
      let occ = select(0.0, 1.0, sampleDepth >= samplePos.z + bias);
      occlusion += occ * rangeCheck;
    }

    return 1.0 - (occlusion / f32(SSAO_KERNEL_SIZE));
  }
);
// clang-format on

/* ---- SSAO blur shader ---- */
// clang-format off
static const char* blur_shader_wgsl = CODE(
  @group(0) @binding(0) var samplerSSAO : texture_2d<f32>;
  @group(0) @binding(1) var ssaoSampler : sampler;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_fullscreen(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    output.uv = vec2f(u, 1.0 - v);
    output.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return output;
  }

  @fragment
  fn fs_blur(input : VertexOutput) -> @location(0) f32 {
    let texDim = textureDimensions(samplerSSAO, 0);
    let texelSize = vec2f(1.0 / f32(texDim.x), 1.0 / f32(texDim.y));
    var result = 0.0;
    var n = 0;
    for (var x = -2; x <= 2; x++) {
      for (var y = -2; y <= 2; y++) {
        let offset = vec2f(f32(x), f32(y)) * texelSize;
        result += textureSample(samplerSSAO, ssaoSampler, input.uv + offset).r;
        n++;
      }
    }
    return result / f32(n);
  }
);
// clang-format on

/* ---- Composition shader ---- */
// clang-format off
static const char* composition_shader_wgsl = CODE(
  @group(0) @binding(0) var samplerPosition   : texture_2d<f32>;
  @group(0) @binding(1) var positionSampler   : sampler;
  @group(0) @binding(2) var samplerNormal     : texture_2d<f32>;
  @group(0) @binding(3) var normalSampler     : sampler;
  @group(0) @binding(4) var samplerAlbedo     : texture_2d<f32>;
  @group(0) @binding(5) var albedoSampler     : sampler;
  @group(0) @binding(6) var samplerSSAO       : texture_2d<f32>;
  @group(0) @binding(7) var ssaoSampler       : sampler;
  @group(0) @binding(8) var samplerSSAOBlur   : texture_2d<f32>;
  @group(0) @binding(9) var ssaoBlurSampler   : sampler;

  struct UBOParams {
    projection : mat4x4f,
    ssao       : i32,
    ssaoOnly   : i32,
    ssaoBlur   : i32,
  }
  @group(0) @binding(10) var<uniform> uboParams : UBOParams;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  }

  @vertex
  fn vs_fullscreen(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    output.uv = vec2f(u, 1.0 - v);
    output.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return output;
  }

  @fragment
  fn fs_composition(input : VertexOutput) -> @location(0) vec4f {
    let fragPos = textureSample(samplerPosition, positionSampler, input.uv).rgb;
    let normal = normalize(textureSample(samplerNormal, normalSampler, input.uv).rgb * 2.0 - 1.0);
    let albedo = textureSample(samplerAlbedo, albedoSampler, input.uv);

    var ssao : f32;
    if (uboParams.ssaoBlur == 1) {
      ssao = textureSample(samplerSSAOBlur, ssaoBlurSampler, input.uv).r;
    } else {
      ssao = textureSample(samplerSSAO, ssaoSampler, input.uv).r;
    }

    let lightPos = vec3f(0.0, 0.0, 0.0);
    let L = normalize(lightPos - fragPos);
    let NdotL = max(0.5, dot(normal, L));

    var outColor : vec3f;

    if (uboParams.ssaoOnly == 1) {
      outColor = vec3f(ssao, ssao, ssao);
    } else {
      let baseColor = albedo.rgb * NdotL;
      if (uboParams.ssao == 1) {
        outColor = vec3f(ssao, ssao, ssao) * baseColor;
      } else {
        outColor = baseColor;
      }
    }

    // Apply sRGB gamma encoding (matches Vulkan's sRGB swapchain behavior)
    outColor = pow(outColor, vec3f(1.0 / 2.2));

    return vec4f(outColor, 1.0);
  }
);
// clang-format on
