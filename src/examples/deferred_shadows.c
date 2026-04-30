/**
 * @brief Deferred shading with shadows from multiple light sources.
 *
 * Ported from the Vulkan deferred shadows example. The Vulkan version uses
 * geometry shader instancing to write all 3 shadow map layers in one pass.
 * Since WebGPU does not support geometry shaders, this port renders to each
 * shadow map layer in a separate render pass.
 *
 * Three-pass rendering pipeline:
 *   1. Shadow map generation (3x depth-only, one per light)
 *   2. G-Buffer generation (MRT: position, normal, albedo)
 *   3. Composition (fullscreen deferred lighting + PCF shadow sampling)
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/deferredshadows
 */

#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
#include <math.h>
#include <string.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>
#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>
#define SOKOL_LOG_IMPL
#include <sokol_log.h>

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define GBUFFER_DIM 2048u
#define SHADOW_MAP_DIM 2048u
#define LIGHT_COUNT 3u
#define NUM_INSTANCES 3u
#define NUM_TEXTURES 4u

/* Texture fetch buffer size */
#define TEXTURE_FILE_BUFFER_SIZE (5 * 1024 * 1024)

/* -------------------------------------------------------------------------- *
 * WGSL shader source forward declarations
 * -------------------------------------------------------------------------- */

static const char* mrt_shader_wgsl;
static const char* shadow_shader_wgsl;
static const char* composition_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

/* Light with shadow mapping support (must match WGSL layout exactly) */
typedef struct {
  vec4 position;    /* xyz = world-space position, w = 1.0            */
  vec4 tgt;         /* xyz = look-at target, w = 0.0                  */
  vec4 color;       /* RGB color, w = unused                          */
  mat4 view_matrix; /* Light MVP for shadow coordinate computation    */
} shadow_light_t;   /* 112 bytes, align 4 */

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Timer / animation */
  float timer;
  float animation_speed;
  bool paused;
  uint64_t last_frame_time;

  /* Shadow mapping parameters */
  float z_near;
  float z_far;
  float light_fov;

  /* Models */
  gltf_model_t model;
  gltf_model_t background;
  bool models_loaded;

  /* GPU buffers for models */
  WGPUBuffer model_vertex_buffer;
  WGPUBuffer model_index_buffer;
  WGPUBuffer bg_vertex_buffer;
  WGPUBuffer bg_index_buffer;

  /* Textures: 0=model color, 1=model normal, 2=bg color, 3=bg normal */
  struct {
    wgpu_texture_t textures[NUM_TEXTURES];
    WGPUSampler sampler;
    uint8_t file_buffers[NUM_TEXTURES][TEXTURE_FILE_BUFFER_SIZE];
    int load_count;
    bool all_loaded;
  } textures;

  /* G-Buffer */
  struct {
    WGPUTexture position_texture;
    WGPUTexture normal_texture;
    WGPUTexture albedo_texture;
    WGPUTexture depth_texture;
    WGPUTextureView position_view;
    WGPUTextureView normal_view;
    WGPUTextureView albedo_view;
    WGPUTextureView depth_view;
    WGPUSampler sampler;
  } gbuffer;

  /* Shadow maps */
  struct {
    WGPUTexture depth_texture;
    WGPUTextureView layer_views[LIGHT_COUNT];
    WGPUTextureView array_view;
    WGPUSampler comparison_sampler;
  } shadow_map;

  /* Uniform buffers */
  WGPUBuffer offscreen_ubo;
  WGPUBuffer composition_ubo;
  WGPUBuffer shadow_ubos[LIGHT_COUNT];

  /* Offscreen UBO data */
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
    vec4 instance_pos[NUM_INSTANCES];
  } offscreen_ubo_data;

  /* Shadow UBO data (per light) */
  struct {
    mat4 mvp;
    vec4 instance_pos[NUM_INSTANCES];
  } shadow_ubo_data[LIGHT_COUNT];

  /* Composition UBO data (must match WGSL layout) */
  struct {
    vec4 view_pos;
    shadow_light_t lights[LIGHT_COUNT];
    uint32_t use_shadows;
    int32_t debug_display_target;
    float _pad[2];
  } composition_ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout offscreen_bgl;
  WGPUBindGroupLayout shadow_bgl;
  WGPUBindGroupLayout composition_bgl;

  /* Bind groups */
  WGPUBindGroup model_bind_group;
  WGPUBindGroup bg_bind_group;
  WGPUBindGroup shadow_bind_groups[LIGHT_COUNT];
  WGPUBindGroup composition_bind_group;

  /* Pipeline layouts */
  WGPUPipelineLayout offscreen_pipeline_layout;
  WGPUPipelineLayout shadow_pipeline_layout;
  WGPUPipelineLayout composition_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline offscreen_pipeline;
  WGPURenderPipeline shadow_pipeline;
  WGPURenderPipeline composition_pipeline;

  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_attachments[3];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } offscreen_pass;

  struct {
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } shadow_pass;

  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } composition_pass;

  /* GUI settings */
  struct {
    int32_t debug_display_target;
    bool enable_shadows;
  } settings;

  WGPUBool initialized;
} state = {
  .animation_speed               = 0.0625f,
  .timer                         = 0.0f,
  .paused                        = false,
  .z_near                        = 0.1f,
  .z_far                         = 64.0f,
  .light_fov                     = 100.0f,
  .settings.debug_display_target = 0,
  .settings.enable_shadows       = true,
};

/* -------------------------------------------------------------------------- *
 * G-Buffer creation
 * -------------------------------------------------------------------------- */

static void destroy_gbuffer_textures(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.position_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.normal_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.albedo_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.position_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.normal_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.albedo_texture)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.depth_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.gbuffer.sampler)
}

static void init_gbuffer_textures(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  destroy_gbuffer_textures();

  /* Position: RGBA16Float */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer Position"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = WGPUTextureFormat_RGBA16Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.gbuffer.position_texture = wgpuDeviceCreateTexture(device, &desc);
    state.gbuffer.position_view
      = wgpuTextureCreateView(state.gbuffer.position_texture,
                              &(WGPUTextureViewDescriptor){
                                .format         = WGPUTextureFormat_RGBA16Float,
                                .dimension      = WGPUTextureViewDimension_2D,
                                .baseMipLevel   = 0,
                                .mipLevelCount  = 1,
                                .baseArrayLayer = 0,
                                .arrayLayerCount = 1,
                                .aspect          = WGPUTextureAspect_All,
                              });
  }

  /* Normal: RGBA16Float */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer Normal"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = WGPUTextureFormat_RGBA16Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.gbuffer.normal_texture = wgpuDeviceCreateTexture(device, &desc);
    state.gbuffer.normal_view    = wgpuTextureCreateView(
      state.gbuffer.normal_texture, &(WGPUTextureViewDescriptor){
                                         .format = WGPUTextureFormat_RGBA16Float,
                                         .dimension = WGPUTextureViewDimension_2D,
                                         .baseMipLevel    = 0,
                                         .mipLevelCount   = 1,
                                         .baseArrayLayer  = 0,
                                         .arrayLayerCount = 1,
                                         .aspect          = WGPUTextureAspect_All,
                                    });
  }

  /* Albedo: RGBA8Unorm (RGB = diffuse, A = specular) */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer Albedo"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.gbuffer.albedo_texture = wgpuDeviceCreateTexture(device, &desc);
    state.gbuffer.albedo_view    = wgpuTextureCreateView(
      state.gbuffer.albedo_texture, &(WGPUTextureViewDescriptor){
                                         .format    = WGPUTextureFormat_RGBA8Unorm,
                                         .dimension = WGPUTextureViewDimension_2D,
                                         .baseMipLevel    = 0,
                                         .mipLevelCount   = 1,
                                         .baseArrayLayer  = 0,
                                         .arrayLayerCount = 1,
                                         .aspect          = WGPUTextureAspect_All,
                                    });
  }

  /* Depth: Depth24PlusStencil8 */
  {
    WGPUTextureDescriptor desc = {
      .label         = STRVIEW("GBuffer Depth"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.gbuffer.depth_texture = wgpuDeviceCreateTexture(device, &desc);
    state.gbuffer.depth_view
      = wgpuTextureCreateView(state.gbuffer.depth_texture,
                              &(WGPUTextureViewDescriptor){
                                .format = WGPUTextureFormat_Depth24PlusStencil8,
                                .dimension       = WGPUTextureViewDimension_2D,
                                .baseMipLevel    = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = 0,
                                .arrayLayerCount = 1,
                                .aspect          = WGPUTextureAspect_All,
                              });
  }

  /* Nearest-neighbor sampler for G-Buffer */
  {
    state.gbuffer.sampler = wgpuDeviceCreateSampler(
      device, &(WGPUSamplerDescriptor){
                .label         = STRVIEW("GBuffer Sampler"),
                .addressModeU  = WGPUAddressMode_ClampToEdge,
                .addressModeV  = WGPUAddressMode_ClampToEdge,
                .addressModeW  = WGPUAddressMode_ClampToEdge,
                .magFilter     = WGPUFilterMode_Nearest,
                .minFilter     = WGPUFilterMode_Nearest,
                .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                .lodMinClamp   = 0.0f,
                .lodMaxClamp   = 1.0f,
                .maxAnisotropy = 1,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Shadow map creation
 * -------------------------------------------------------------------------- */

static void destroy_shadow_map(void)
{
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(TextureView, state.shadow_map.layer_views[i])
  }
  WGPU_RELEASE_RESOURCE(TextureView, state.shadow_map.array_view)
  WGPU_RELEASE_RESOURCE(Texture, state.shadow_map.depth_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.shadow_map.comparison_sampler)
}

static void init_shadow_map(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  destroy_shadow_map();

  /* Layered depth texture (2D array with LIGHT_COUNT layers) */
  state.shadow_map.depth_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Shadow Map Depth Array"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {SHADOW_MAP_DIM, SHADOW_MAP_DIM, LIGHT_COUNT},
              .format        = WGPUTextureFormat_Depth32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  /* Per-layer views for render targets */
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    state.shadow_map.layer_views[i]
      = wgpuTextureCreateView(state.shadow_map.depth_texture,
                              &(WGPUTextureViewDescriptor){
                                .format        = WGPUTextureFormat_Depth32Float,
                                .dimension     = WGPUTextureViewDimension_2D,
                                .baseMipLevel  = 0,
                                .mipLevelCount = 1,
                                .baseArrayLayer  = i,
                                .arrayLayerCount = 1,
                                .aspect          = WGPUTextureAspect_DepthOnly,
                              });
  }

  /* Array view for sampling in composition shader */
  state.shadow_map.array_view
    = wgpuTextureCreateView(state.shadow_map.depth_texture,
                            &(WGPUTextureViewDescriptor){
                              .format        = WGPUTextureFormat_Depth32Float,
                              .dimension     = WGPUTextureViewDimension_2DArray,
                              .baseMipLevel  = 0,
                              .mipLevelCount = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = LIGHT_COUNT,
                              .aspect          = WGPUTextureAspect_DepthOnly,
                            });

  /* Comparison sampler for PCF shadow sampling */
  state.shadow_map.comparison_sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .label         = STRVIEW("Shadow Comparison Sampler"),
              .addressModeU  = WGPUAddressMode_ClampToEdge,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_ClampToEdge,
              .magFilter     = WGPUFilterMode_Linear,
              .minFilter     = WGPUFilterMode_Linear,
              .compare       = WGPUCompareFunction_Less,
              .maxAnisotropy = 1,
            });
}

/* -------------------------------------------------------------------------- *
 * Render pass descriptors
 * -------------------------------------------------------------------------- */

static void init_render_passes(void)
{
  /* Offscreen G-Buffer pass: 3 color attachments + depth */
  {
    state.offscreen_pass.color_attachments[0] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.position_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.color_attachments[1] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.normal_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.color_attachments[2] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.albedo_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = state.gbuffer.depth_view,
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };
    state.offscreen_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("G-Buffer Render Pass"),
      .colorAttachmentCount   = 3,
      .colorAttachments       = state.offscreen_pass.color_attachments,
      .depthStencilAttachment = &state.offscreen_pass.depth_stencil_attachment,
    };
  }

  /* Shadow pass: depth-only (view set per light per frame) */
  {
    state.shadow_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view            = NULL, /* Set per light */
        .depthLoadOp     = WGPULoadOp_Clear,
        .depthStoreOp    = WGPUStoreOp_Store,
        .depthClearValue = 1.0f,
      };
    state.shadow_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("Shadow Map Render Pass"),
      .colorAttachmentCount   = 0,
      .colorAttachments       = NULL,
      .depthStencilAttachment = &state.shadow_pass.depth_stencil_attachment,
    };
  }

  /* Composition pass: 1 color (swapchain) + depth */
  {
    state.composition_pass.color_attachment = (WGPURenderPassColorAttachment){
      .view       = NULL, /* Set per frame */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.2f, 1.0f},
    };
    state.composition_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = NULL, /* Set per frame */
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };
    state.composition_pass.descriptor = (WGPURenderPassDescriptor){
      .label                = STRVIEW("Composition Render Pass"),
      .colorAttachmentCount = 1,
      .colorAttachments     = &state.composition_pass.color_attachment,
      .depthStencilAttachment
      = &state.composition_pass.depth_stencil_attachment,
    };
  }
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  const gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  bool ok = gltf_model_load_from_file_ext(
    &state.model, "assets/models/armor/armor.gltf", 1.0f, &desc);
  if (!ok) {
    printf("Failed to load armor.gltf\n");
    return;
  }

  ok = gltf_model_load_from_file_ext(
    &state.background, "assets/models/deferred_box.gltf", 1.0f, &desc);
  if (!ok) {
    printf("Failed to load deferred_box.gltf\n");
    return;
  }

  state.models_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  if (!state.models_loaded) {
    return;
  }

  struct {
    gltf_model_t* model;
    WGPUBuffer* vb;
    WGPUBuffer* ib;
    const char* vb_label;
    const char* ib_label;
  } items[2] = {
    {&state.model, &state.model_vertex_buffer, &state.model_index_buffer,
     "Model VB", "Model IB"},
    {&state.background, &state.bg_vertex_buffer, &state.bg_index_buffer,
     "Background VB", "Background IB"},
  };

  for (int i = 0; i < 2; i++) {
    gltf_model_t* m = items[i].model;
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);

    *items[i].vb = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW(items[i].vb_label),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = true,
              });
    void* vdata = wgpuBufferGetMappedRange(*items[i].vb, 0, vb_size);
    memcpy(vdata, m->vertices, vb_size);
    wgpuBufferUnmap(*items[i].vb);

    if (m->index_count > 0) {
      size_t ib_size = m->index_count * sizeof(uint32_t);
      *items[i].ib   = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                    .label = STRVIEW(items[i].ib_label),
                    .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                    .size  = ib_size,
                    .mappedAtCreation = true,
                });
      void* idata = wgpuBufferGetMappedRange(*items[i].ib, 0, ib_size);
      memcpy(idata, m->indices, ib_size);
      wgpuBufferUnmap(*items[i].ib);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Texture loading
 * -------------------------------------------------------------------------- */

static void texture_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_w, img_h, num_ch;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_w, &img_h, &num_ch, 4);

  if (pixels) {
    int tex_index       = *(int*)response->user_data;
    wgpu_texture_t* tex = &state.textures.textures[tex_index];

    tex->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_w,
        .height             = (uint32_t)img_h,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = (size_t)(img_w * img_h * 4),
      },
    };
    tex->desc.is_dirty = true;
    state.textures.load_count++;
  }
}

static void fetch_textures(void)
{
  static const char* texture_paths[NUM_TEXTURES] = {
    "assets/models/armor/colormap_rgba.png",
    "assets/models/armor/normalmap_rgba.png",
    "assets/textures/stonefloor02_color_rgba.png",
    "assets/textures/stonefloor02_normal_rgba.png",
  };

  static int tex_indices[NUM_TEXTURES] = {0, 1, 2, 3};

  state.textures.load_count = 0;
  state.textures.all_loaded = false;

  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    sfetch_send(&(sfetch_request_t){
      .path      = texture_paths[i],
      .callback  = texture_fetch_callback,
      .buffer    = SFETCH_RANGE(state.textures.file_buffers[i]),
      .user_data = {
        .ptr  = &tex_indices[i],
        .size = sizeof(int),
      },
    });
  }
}

/* Forward declarations */
static void init_bind_groups(struct wgpu_context_t* wgpu_context);

static void init_textures(struct wgpu_context_t* wgpu_context)
{
  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    state.textures.textures[i]
      = wgpu_create_color_bars_texture(wgpu_context, NULL);
  }

  state.textures.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Model Texture Sampler"),
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
}

static void update_textures(struct wgpu_context_t* wgpu_context)
{
  bool any_updated = false;
  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    wgpu_texture_t* tex = &state.textures.textures[i];
    if (tex->desc.is_dirty) {
      wgpu_recreate_texture(wgpu_context, tex);
      FREE_TEXTURE_PIXELS(*tex);
      any_updated = true;
    }
  }

  if (any_updated && state.textures.load_count >= (int)NUM_TEXTURES) {
    state.textures.all_loaded = true;
    init_bind_groups(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Instance positions (Vulkan: (0,0,0), (-7,0,-4), (4,0,-6)) */
  /* Y=0 in Vulkan → Y=0 in WebGPU (no change needed) */
  glm_vec4_copy((vec4){0.0f, 0.0f, 0.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[0]);
  glm_vec4_copy((vec4){-7.0f, 0.0f, -4.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[1]);
  glm_vec4_copy((vec4){4.0f, 0.0f, -6.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[2]);

  /* Offscreen UBO */
  state.offscreen_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Offscreen UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.offscreen_ubo_data),
            });

  /* Composition UBO */
  state.composition_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Composition UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.composition_ubo_data),
            });

  /* Shadow UBOs (one per light) */
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    state.shadow_ubos[i] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Shadow UBO"),
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(state.shadow_ubo_data[0]),
              });
  }
}

static void init_lights(void)
{
  /* Vulkan positions (negate Y for WebGPU):
   * Light 0: pos (-14, -0.5, 15)  → (-14, 0.5, 15)
   * Light 1: pos (14, -4, 12)     → (14, 4, 12)
   * Light 2: pos (0, -10, 4)      → (0, 10, 4)
   */
  shadow_light_t* lights = state.composition_ubo_data.lights;

  lights[0].position[0] = -14.0f;
  lights[0].position[1] = 0.5f;
  lights[0].position[2] = 15.0f;
  lights[0].position[3] = 1.0f;
  lights[0].tgt[0]      = -2.0f;
  lights[0].tgt[1]      = 0.0f;
  lights[0].tgt[2]      = 0.0f;
  lights[0].tgt[3]      = 0.0f;
  lights[0].color[0]    = 1.0f;
  lights[0].color[1]    = 0.5f;
  lights[0].color[2]    = 0.5f;
  lights[0].color[3]    = 0.0f;

  lights[1].position[0] = 14.0f;
  lights[1].position[1] = 4.0f;
  lights[1].position[2] = 12.0f;
  lights[1].position[3] = 1.0f;
  lights[1].tgt[0]      = 2.0f;
  lights[1].tgt[1]      = 0.0f;
  lights[1].tgt[2]      = 0.0f;
  lights[1].tgt[3]      = 0.0f;
  lights[1].color[0]    = 0.0f;
  lights[1].color[1]    = 0.0f;
  lights[1].color[2]    = 1.0f;
  lights[1].color[3]    = 0.0f;

  lights[2].position[0] = 0.0f;
  lights[2].position[1] = 10.0f;
  lights[2].position[2] = 4.0f;
  lights[2].position[3] = 1.0f;
  lights[2].tgt[0]      = 0.0f;
  lights[2].tgt[1]      = 0.0f;
  lights[2].tgt[2]      = 0.0f;
  lights[2].tgt[3]      = 0.0f;
  lights[2].color[0]    = 1.0f;
  lights[2].color[1]    = 1.0f;
  lights[2].color[2]    = 1.0f;
  lights[2].color[3]    = 0.0f;

  state.composition_ubo_data.use_shadows = 1;
}

static void update_uniform_buffer_offscreen(struct wgpu_context_t* wgpu_context)
{
  glm_mat4_copy(state.camera.matrices.perspective,
                state.offscreen_ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.offscreen_ubo_data.view);
  glm_mat4_identity(state.offscreen_ubo_data.model);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.offscreen_ubo, 0,
                       &state.offscreen_ubo_data,
                       sizeof(state.offscreen_ubo_data));
}

static void
update_uniform_buffer_composition(struct wgpu_context_t* wgpu_context)
{
  shadow_light_t* lights = state.composition_ubo_data.lights;

  /* Animate light positions (X and Z only, same as Vulkan) */
  if (!state.paused) {
    float t  = state.timer * 360.0f;
    float tr = glm_rad(t);

    lights[0].position[0] = -14.0f + fabsf(sinf(tr) * 20.0f);
    lights[0].position[2] = 15.0f + cosf(tr) * 1.0f;

    lights[1].position[0] = 14.0f - fabsf(sinf(tr) * 2.5f);
    lights[1].position[2] = 13.0f + cosf(tr) * 4.0f;

    lights[2].position[0] = sinf(tr) * 4.0f;
    lights[2].position[2] = 4.0f + cosf(tr) * 2.0f;
  }

  /* Compute shadow MVP matrices for each light */
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    mat4 shadow_proj, shadow_view, shadow_mvp;

    glm_perspective(glm_rad(state.light_fov), 1.0f, state.z_near, state.z_far,
                    shadow_proj);
    glm_lookat((vec3){lights[i].position[0], lights[i].position[1],
                      lights[i].position[2]},
               (vec3){lights[i].tgt[0], lights[i].tgt[1], lights[i].tgt[2]},
               (vec3){0.0f, 1.0f, 0.0f}, shadow_view);
    glm_mat4_mul(shadow_proj, shadow_view, shadow_mvp);

    /* Store MVP in light for composition shader */
    glm_mat4_copy(shadow_mvp, lights[i].view_matrix);

    /* Store MVP in shadow UBO for shadow pass */
    glm_mat4_copy(shadow_mvp, state.shadow_ubo_data[i].mvp);
    memcpy(state.shadow_ubo_data[i].instance_pos,
           state.offscreen_ubo_data.instance_pos,
           sizeof(state.offscreen_ubo_data.instance_pos));
  }

  /* View position */
  glm_vec4_copy(state.camera.view_pos, state.composition_ubo_data.view_pos);

  /* Settings */
  state.composition_ubo_data.use_shadows
    = state.settings.enable_shadows ? 1 : 0;
  state.composition_ubo_data.debug_display_target
    = state.settings.debug_display_target;

  /* Upload all uniform data */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.composition_ubo, 0,
                       &state.composition_ubo_data,
                       sizeof(state.composition_ubo_data));

  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    wgpuQueueWriteBuffer(wgpu_context->queue, state.shadow_ubos[i], 0,
                         &state.shadow_ubo_data[i],
                         sizeof(state.shadow_ubo_data[i]));
  }
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Offscreen BGL: UBO + colorMap + normalMap + sampler */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.offscreen_ubo_data),
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
    };

    state.offscreen_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Offscreen BGL"),
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Shadow BGL: UBO only */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.shadow_ubo_data[0]),
        },
      },
    };

    state.shadow_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Shadow BGL"),
                .entryCount = 1,
                .entries    = entries,
              });
  }

  /* Composition BGL: G-Buffer textures + sampler + UBO + shadow map + cmp
   * sampler */
  {
    WGPUBindGroupLayoutEntry entries[7] = {
      /* Binding 0: Position texture */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 1: Normal texture */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 2: Albedo texture */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 3: G-Buffer sampler */
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
      /* Binding 4: Composition UBO */
      {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.composition_ubo_data),
        },
      },
      /* Binding 5: Shadow map (depth 2D array) */
      {
        .binding    = 5,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Depth,
          .viewDimension = WGPUTextureViewDimension_2DArray,
        },
      },
      /* Binding 6: Shadow comparison sampler */
      {
        .binding    = 6,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Comparison},
      },
    };

    state.composition_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition BGL"),
                .entryCount = 7,
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  WGPU_RELEASE_RESOURCE(BindGroup, state.model_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_bind_group)
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.shadow_bind_groups[i])
  }
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)

  /* Model bind group: UBO + model color + model normal + sampler */
  {
    WGPUBindGroupEntry entries[4] = {
      {.binding = 0,
       .buffer  = state.offscreen_ubo,
       .offset  = 0,
       .size    = sizeof(state.offscreen_ubo_data)},
      {.binding = 1, .textureView = state.textures.textures[0].view},
      {.binding = 2, .textureView = state.textures.textures[1].view},
      {.binding = 3, .sampler = state.textures.sampler},
    };
    state.model_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Model Bind Group"),
                .layout     = state.offscreen_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Background bind group: UBO + bg color + bg normal + sampler */
  {
    WGPUBindGroupEntry entries[4] = {
      {.binding = 0,
       .buffer  = state.offscreen_ubo,
       .offset  = 0,
       .size    = sizeof(state.offscreen_ubo_data)},
      {.binding = 1, .textureView = state.textures.textures[2].view},
      {.binding = 2, .textureView = state.textures.textures[3].view},
      {.binding = 3, .sampler = state.textures.sampler},
    };
    state.bg_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Background Bind Group"),
                .layout     = state.offscreen_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Shadow bind groups (one per light) */
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    WGPUBindGroupEntry entries[1] = {
      {.binding = 0,
       .buffer  = state.shadow_ubos[i],
       .offset  = 0,
       .size    = sizeof(state.shadow_ubo_data[0])},
    };
    state.shadow_bind_groups[i] = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Shadow Bind Group"),
                .layout     = state.shadow_bgl,
                .entryCount = 1,
                .entries    = entries,
              });
  }

  /* Composition bind group */
  {
    WGPUBindGroupEntry entries[7] = {
      {.binding = 0, .textureView = state.gbuffer.position_view},
      {.binding = 1, .textureView = state.gbuffer.normal_view},
      {.binding = 2, .textureView = state.gbuffer.albedo_view},
      {.binding = 3, .sampler = state.gbuffer.sampler},
      {.binding = 4,
       .buffer  = state.composition_ubo,
       .offset  = 0,
       .size    = sizeof(state.composition_ubo_data)},
      {.binding = 5, .textureView = state.shadow_map.array_view},
      {.binding = 6, .sampler = state.shadow_map.comparison_sampler},
    };
    state.composition_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Composition Bind Group"),
                .layout     = state.composition_bgl,
                .entryCount = 7,
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline layouts
 * -------------------------------------------------------------------------- */

static void init_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.offscreen_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Offscreen Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.offscreen_bgl,
            });

  state.shadow_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Shadow Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.shadow_bgl,
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

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ===== Offscreen (MRT) Pipeline ===== */
  {
    WGPUVertexAttribute vert_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, tangent)},
      {.shaderLocation = 4,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vert_attrs),
      .attributes     = vert_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, mrt_shader_wgsl);

    WGPUBlendState no_blend               = wgpu_create_blend_state(false);
    WGPUColorTargetState color_targets[3] = {
      {.format    = WGPUTextureFormat_RGBA16Float,
       .blend     = &no_blend,
       .writeMask = WGPUColorWriteMask_All},
      {.format    = WGPUTextureFormat_RGBA16Float,
       .blend     = &no_blend,
       .writeMask = WGPUColorWriteMask_All},
      {.format    = WGPUTextureFormat_RGBA8Unorm,
       .blend     = &no_blend,
       .writeMask = WGPUColorWriteMask_All},
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.offscreen_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Offscreen Pipeline"),
                .layout = state.offscreen_pipeline_layout,
                .vertex = (WGPUVertexState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("vs_main"),
                  .bufferCount = 1,
                  .buffers     = &vb_layout,
                },
                .primitive = (WGPUPrimitiveState){
                  .topology  = WGPUPrimitiveTopology_TriangleList,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode  = WGPUCullMode_Back,
                },
                .depthStencil = &depth_stencil,
                .multisample  = (WGPUMultisampleState){
                  .count = 1,
                  .mask  = 0xFFFFFFFF,
                },
                .fragment = &(WGPUFragmentState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("fs_main"),
                  .targetCount = 3,
                  .targets     = color_targets,
                },
              });
    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Shadow Pipeline (depth-only) ===== */
  {
    /* Only need position attribute for shadow rendering */
    WGPUVertexAttribute shadow_attr = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
    };

    WGPUVertexBufferLayout shadow_vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &shadow_attr,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, shadow_shader_wgsl);

    WGPUDepthStencilState shadow_depth_stencil = {
      .format              = WGPUTextureFormat_Depth32Float,
      .depthWriteEnabled   = WGPUOptionalBool_True,
      .depthCompare        = WGPUCompareFunction_LessEqual,
      .depthBias           = 2,
      .depthBiasSlopeScale = 1.75f,
      .depthBiasClamp      = 0.0f,
    };

    state.shadow_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Shadow Pipeline"),
                .layout = state.shadow_pipeline_layout,
                .vertex = (WGPUVertexState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("vs_main"),
                  .bufferCount = 1,
                  .buffers     = &shadow_vb_layout,
                },
                .primitive = (WGPUPrimitiveState){
                  .topology  = WGPUPrimitiveTopology_TriangleList,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode  = WGPUCullMode_Front, /* Front-face cull for shadows */
                },
                .depthStencil = &shadow_depth_stencil,
                .multisample  = (WGPUMultisampleState){
                  .count = 1,
                  .mask  = 0xFFFFFFFF,
                },
                /* No fragment shader — depth-only */
              });
    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }

  /* ===== Composition Pipeline ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, composition_shader_wgsl);

    WGPUBlendState no_blend           = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &no_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = wgpu_context->depth_stencil_format,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.composition_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Composition Pipeline"),
                .layout = state.composition_pipeline_layout,
                .vertex = (WGPUVertexState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("vs_main"),
                  .bufferCount = 0,
                  .buffers     = NULL,
                },
                .primitive = (WGPUPrimitiveState){
                  .topology  = WGPUPrimitiveTopology_TriangleList,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode  = WGPUCullMode_None,
                },
                .depthStencil = &depth_stencil,
                .multisample  = (WGPUMultisampleState){
                  .count = 1,
                  .mask  = 0xFFFFFFFF,
                },
                .fragment = &(WGPUFragmentState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("fs_main"),
                  .targetCount = 1,
                  .targets     = &color_target,
                },
              });
    WGPU_RELEASE_RESOURCE(ShaderModule, shader);
  }
}

/* -------------------------------------------------------------------------- *
 * Helper: draw a gltf model
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib, uint32_t instance_count)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  if (ib) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(
          pass, prim->index_count, instance_count, prim->first_index, 0, 0);
      }
      else if (prim->vertex_count > 0) {
        wgpuRenderPassEncoderDraw(pass, prim->vertex_count, instance_count, 0,
                                  0);
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * ImGui overlay
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Deferred Shadows", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    static const char* display_modes[] = {
      "Final composition", "Shadows", "Position",
      "Normals",           "Albedo",  "Specular",
    };
    imgui_overlay_combo_box("Display", &state.settings.debug_display_target,
                            display_modes, ARRAY_SIZE(display_modes));
    igCheckbox("Shadows", &state.settings.enable_shadows);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event callback
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
  camera_on_input_event(&state.camera, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_CHAR) {
    switch (input_event->char_code) {
      case 'p':
      case 'P':
        state.paused = !state.paused;
        break;
    }
  }
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
    .max_requests = NUM_TEXTURES,
    .num_channels = 1,
    .num_lanes    = NUM_TEXTURES,
    .logger.func  = slog_func,
  });

  /* Camera (first-person) */
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.movement_speed = 5.0f;
  state.camera.rotation_speed = 0.25f;
  camera_set_position(&state.camera, (vec3){2.15f, 0.3f, -8.75f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-0.75f, 12.5f, 0.0f));
  camera_set_perspective(&state.camera, 60.0f,
                         (float)wgpu_context->width
                           / (float)wgpu_context->height,
                         state.z_near, state.z_far);

  /* Load models */
  load_models();
  create_model_buffers(wgpu_context);

  /* Create G-Buffer textures and shadow map */
  init_gbuffer_textures(wgpu_context);
  init_shadow_map(wgpu_context);

  /* Create render pass descriptors */
  init_render_passes();

  /* Create placeholder textures and start async fetch */
  init_textures(wgpu_context);
  fetch_textures();

  /* Create uniform buffers and initialize lights */
  init_uniform_buffers(wgpu_context);
  init_lights();

  /* Bind group layouts → pipeline layouts */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Create initial bind groups */
  init_bind_groups(wgpu_context);

  /* Create render pipelines */
  init_pipelines(wgpu_context);

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

  /* Timing */
  uint64_t now          = stm_now();
  float delta           = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Pump async file loading */
  sfetch_dowork();
  update_textures(wgpu_context);

  /* Advance animation timer */
  if (!state.paused) {
    state.timer += delta * state.animation_speed;
    if (state.timer > 1.0f) {
      state.timer -= 1.0f;
    }
  }

  /* Update camera */
  camera_update(&state.camera, delta);

  /* Update uniforms */
  update_uniform_buffer_offscreen(wgpu_context);
  update_uniform_buffer_composition(wgpu_context);

  /* ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, delta);
  render_gui(wgpu_context);

  /* Command encoding */
  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* ===== Pass 1: Shadow Map Generation (one pass per light) ===== */
  if (state.models_loaded) {
    for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
      state.shadow_pass.depth_stencil_attachment.view
        = state.shadow_map.layer_views[i];

      WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        cmd_enc, &state.shadow_pass.descriptor);

      wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)SHADOW_MAP_DIM,
                                       (float)SHADOW_MAP_DIM, 0.0f, 1.0f);
      wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, SHADOW_MAP_DIM,
                                          SHADOW_MAP_DIM);
      wgpuRenderPassEncoderSetPipeline(pass, state.shadow_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.shadow_bind_groups[i], 0,
                                        0);

      /* Draw background (1 instance) */
      draw_model(pass, &state.background, state.bg_vertex_buffer,
                 state.bg_index_buffer, 1);

      /* Draw models (3 instances) */
      draw_model(pass, &state.model, state.model_vertex_buffer,
                 state.model_index_buffer, NUM_INSTANCES);

      wgpuRenderPassEncoderEnd(pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
    }
  }

  /* ===== Pass 2: Offscreen G-Buffer ===== */
  if (state.models_loaded) {
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)GBUFFER_DIM,
                                     (float)GBUFFER_DIM, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, GBUFFER_DIM, GBUFFER_DIM);
    wgpuRenderPassEncoderSetPipeline(pass, state.offscreen_pipeline);

    /* Draw background (single instance) */
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bg_bind_group, 0, 0);
    draw_model(pass, &state.background, state.bg_vertex_buffer,
               state.bg_index_buffer, 1);

    /* Draw armor (3 instances) */
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.model_bind_group, 0, 0);
    draw_model(pass, &state.model, state.model_vertex_buffer,
               state.model_index_buffer, NUM_INSTANCES);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  }

  /* ===== Pass 3: Composition (deferred lighting + shadows) ===== */
  {
    state.composition_pass.color_attachment.view = wgpu_context->swapchain_view;
    state.composition_pass.depth_stencil_attachment.view
      = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.composition_pass.descriptor);

    wgpuRenderPassEncoderSetPipeline(pass, state.composition_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.composition_bind_group, 0,
                                      0);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay render */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  gltf_model_destroy(&state.model);
  gltf_model_destroy(&state.background);

  WGPU_RELEASE_RESOURCE(Buffer, state.model_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.model_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.bg_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.bg_index_buffer)

  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    wgpu_destroy_texture(&state.textures.textures[i]);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)

  destroy_gbuffer_textures();
  destroy_shadow_map();

  WGPU_RELEASE_RESOURCE(Buffer, state.offscreen_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.composition_ubo)
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(Buffer, state.shadow_ubos[i])
  }

  WGPU_RELEASE_RESOURCE(BindGroup, state.model_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bg_bind_group)
  for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.shadow_bind_groups[i])
  }
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.offscreen_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.shadow_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_bgl)

  WGPU_RELEASE_RESOURCE(PipelineLayout, state.offscreen_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.shadow_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_pipeline_layout)

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.offscreen_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.shadow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_pipeline)

  imgui_overlay_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Deferred Shading with Shadows",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL shader code
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* mrt_shader_wgsl = CODE(
  struct UBO {
    projection  : mat4x4f,
    model       : mat4x4f,
    view        : mat4x4f,
    instancePos : array<vec4f, 3>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var colorMap    : texture_2d<f32>;
  @group(0) @binding(2) var normalMap   : texture_2d<f32>;
  @group(0) @binding(3) var texSampler  : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) tangent  : vec4f,
    @location(4) color    : vec4f,
    @builtin(instance_index) instanceIndex : u32,
  };

  struct VertexOutput {
    @builtin(position) position   : vec4f,
    @location(0)       worldNormal : vec3f,
    @location(1)       uv         : vec2f,
    @location(2)       color      : vec3f,
    @location(3)       worldPos   : vec3f,
    @location(4)       tangent    : vec3f,
  };

  struct FragmentOutput {
    @location(0) gPosition : vec4f,
    @location(1) gNormal   : vec4f,
    @location(2) gAlbedo   : vec4f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    let instanceOffset = ubo.instancePos[in.instanceIndex];
    let worldPos = vec4f(in.position, 1.0) + instanceOffset;

    out.position    = ubo.projection * ubo.view * ubo.model * worldPos;
    out.uv          = in.uv;
    out.worldPos    = (ubo.model * worldPos).xyz;

    let normalMatrix = mat3x3f(
      ubo.model[0].xyz,
      ubo.model[1].xyz,
      ubo.model[2].xyz,
    );
    out.worldNormal = normalMatrix * normalize(in.normal);
    out.tangent     = normalMatrix * normalize(in.tangent.xyz);
    out.color       = in.color.rgb;

    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    out.gPosition = vec4f(in.worldPos, 1.0);

    let N = normalize(in.worldNormal);
    let T = normalize(in.tangent);
    let B = cross(N, T);
    let TBN = mat3x3f(T, B, N);

    let sampledNormal = textureSample(normalMap, texSampler, in.uv).xyz * 2.0 - vec3f(1.0);
    let tnorm = TBN * normalize(sampledNormal);
    out.gNormal = vec4f(tnorm, 1.0);

    out.gAlbedo = textureSample(colorMap, texSampler, in.uv);

    return out;
  }
);

static const char* shadow_shader_wgsl = CODE(
  struct ShadowUBO {
    mvp         : mat4x4f,
    instancePos : array<vec4f, 3>,
  };

  @group(0) @binding(0) var<uniform> shadow_ubo : ShadowUBO;

  @vertex
  fn vs_main(
    @location(0) position : vec3f,
    @builtin(instance_index) instanceIndex : u32,
  ) -> @builtin(position) vec4f {
    let worldPos = vec4f(position, 1.0) + shadow_ubo.instancePos[instanceIndex];
    return shadow_ubo.mvp * worldPos;
  }
);

static const char* composition_shader_wgsl = CODE(
  const LIGHT_COUNT : u32 = 3u;
  const SHADOW_FACTOR : f32 = 0.25;
  const AMBIENT_LIGHT : f32 = 0.1;

  struct Light {
    position   : vec4f,
    tgt        : vec4f,
    color      : vec4f,
    viewMatrix : mat4x4f,
  };

  struct UBO {
    viewPos            : vec4f,
    lights             : array<Light, 3>,
    useShadows         : u32,
    debugDisplayTarget : i32,
  };

  @group(0) @binding(0) var gPosition     : texture_2d<f32>;
  @group(0) @binding(1) var gNormal       : texture_2d<f32>;
  @group(0) @binding(2) var gAlbedo       : texture_2d<f32>;
  @group(0) @binding(3) var gSampler      : sampler;
  @group(0) @binding(4) var<uniform> ubo  : UBO;
  @group(0) @binding(5) var shadowMap     : texture_depth_2d_array;
  @group(0) @binding(6) var shadowSampler : sampler_comparison;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(
      f32((vertexIndex << 1u) & 2u),
      f32(vertexIndex & 2u),
    );
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  /* Single shadow sample with comparison */
  fn textureProj(P : vec4f, layer : i32, offset : vec2f) -> f32 {
    var shadow : f32 = 1.0;
    let sc = P.xyz / P.w;

    /* NDC to UV (WebGPU: negate Y for texture coordinate) */
    let uv = vec2f(sc.x * 0.5 + 0.5, -sc.y * 0.5 + 0.5);

    if (sc.z >= 0.0 && sc.z <= 1.0 && P.w > 0.0) {
      let cmp = textureSampleCompareLevel(shadowMap, shadowSampler, uv + offset, layer, sc.z);
      shadow = mix(SHADOW_FACTOR, 1.0, cmp);
    }
    return shadow;
  }

  /* 3x3 PCF filtering */
  fn filterPCF(sc : vec4f, layer : i32) -> f32 {
    let texDim = textureDimensions(shadowMap);
    let scale : f32 = 1.5;
    let dx = scale / f32(texDim.x);
    let dy = scale / f32(texDim.y);

    var shadowFactor : f32 = 0.0;
    var count : i32 = 0;

    for (var x : i32 = -1; x <= 1; x++) {
      for (var y : i32 = -1; y <= 1; y++) {
        shadowFactor += textureProj(sc, layer, vec2f(f32(x) * dx, f32(y) * dy));
        count++;
      }
    }
    return shadowFactor / f32(count);
  }

  /* Apply shadows from all lights to fragment color */
  fn shadow(fragcolor : vec3f, fragpos : vec3f) -> vec3f {
    var result = fragcolor;
    for (var i : i32 = 0; i < i32(LIGHT_COUNT); i++) {
      let shadowClip = ubo.lights[i].viewMatrix * vec4f(fragpos, 1.0);
      let shadowFactor = filterPCF(shadowClip, i);
      result *= shadowFactor;
    }
    return result;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let fragPos = textureSample(gPosition, gSampler, in.uv).rgb;
    let normal  = textureSample(gNormal, gSampler, in.uv).rgb;
    let albedo  = textureSample(gAlbedo, gSampler, in.uv);

    /* Debug display modes */
    if (ubo.debugDisplayTarget > 0) {
      switch (ubo.debugDisplayTarget) {
        case 1: { return vec4f(shadow(vec3f(1.0), fragPos), 1.0); }
        case 2: { return vec4f(fragPos.x, -fragPos.y, fragPos.z, 1.0); }
        case 3: { return vec4f(normal.x, -normal.y, normal.z, 1.0); }
        case 4: { return vec4f(albedo.rgb, 1.0); }
        case 5: { return vec4f(albedo.aaa, 1.0); }
        default: { return vec4f(fragPos, 1.0); }
      }
    }

    /* Ambient */
    var fragcolor = albedo.rgb * AMBIENT_LIGHT;

    /* Per-light calculations */
    let N = normalize(normal);

    for (var i : i32 = 0; i < i32(LIGHT_COUNT); i++) {
      let L = ubo.lights[i].position.xyz - fragPos;
      let dist = length(L);
      let Ln = normalize(L);

      let V = normalize(ubo.viewPos.xyz - fragPos);

      /* Spot light parameters */
      let lightCosInnerAngle = cos(radians(15.0));
      let lightCosOuterAngle = cos(radians(25.0));
      let lightRange : f32 = 100.0;

      /* Spot light direction */
      let dir = normalize(ubo.lights[i].position.xyz - ubo.lights[i].tgt.xyz);

      /* Dual-cone spot light with soft falloff */
      let cosDir = dot(Ln, dir);
      let spotEffect = smoothstep(lightCosOuterAngle, lightCosInnerAngle, cosDir);
      let heightAttenuation = smoothstep(lightRange, 0.0, dist);

      /* Diffuse (Lambertian) */
      let NdotL = max(0.0, dot(N, Ln));
      let diff = vec3f(NdotL);

      /* Specular (Phong) */
      let R = reflect(-Ln, N);
      let NdotR = max(0.0, dot(R, V));
      let spec = vec3f(pow(NdotR, 16.0) * albedo.a * 2.5);

      fragcolor += (diff + spec) * spotEffect * heightAttenuation * ubo.lights[i].color.rgb * albedo.rgb;
    }

    /* Apply shadows */
    if (ubo.useShadows > 0u) {
      fragcolor = shadow(fragcolor, fragPos);
    }

    return vec4f(fragcolor, 1.0);
  }
);
// clang-format on
