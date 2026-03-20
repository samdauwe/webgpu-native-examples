/**
 * @brief Multi-sampled deferred shading with explicit MSAA resolve.
 *
 * Ported from the Vulkan deferred multisampling example. Renders the scene
 * into multisampled G-Buffer render targets (position, normal, albedo) and
 * performs per-sample lighting in the composition pass with explicit MSAA
 * resolve in the fragment shader.
 *
 * Features:
 *  - 4x MSAA for the off-screen G-Buffer pass
 *  - Per-sample deferred lighting with manual resolve
 *  - 6 point lights with attenuation
 *  - Normal mapping via TBN matrix
 *  - Instanced rendering (3 armor models + background box)
 *  - Debug visualization of individual G-Buffer channels
 *  - Toggle MSAA on/off at runtime
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/deferredmultisampling
 */

#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"
#include "webgpu/imgui_overlay.h"

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
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

#define GBUFFER_DIM (2048u)
#define NUM_LIGHTS (6u)
#define NUM_INSTANCES (3u)
#define NUM_TEXTURES (4u)
#define SAMPLE_COUNT (4u)

/* Texture fetch buffer size (max compressed size per texture) */
#define TEXTURE_FILE_BUFFER_SIZE (5 * 1024 * 1024)

/* -------------------------------------------------------------------------- *
 * WGSL shader source forward declarations
 * -------------------------------------------------------------------------- */

static const char* mrt_shader_wgsl;
static const char* composition_msaa_shader_wgsl;
static const char* composition_no_msaa_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Types
 * -------------------------------------------------------------------------- */

typedef struct {
  vec4 position; /* xyz = world-space position, w = unused */
  vec3 color;    /* RGB light color                        */
  float radius;  /* Attenuation radius                     */
} light_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Timer / animation */
  uint64_t last_frame_time;

  /* Models */
  gltf_model_t armor_model;
  gltf_model_t box_model;
  bool models_loaded;

  /* GPU buffers for models */
  WGPUBuffer armor_vertex_buffer;
  WGPUBuffer armor_index_buffer;
  WGPUBuffer box_vertex_buffer;
  WGPUBuffer box_index_buffer;

  /* Textures: 0=armor color, 1=armor normal, 2=bg color, 3=bg normal */
  struct {
    wgpu_texture_t textures[NUM_TEXTURES];
    WGPUSampler sampler;
    uint8_t file_buffers[NUM_TEXTURES][TEXTURE_FILE_BUFFER_SIZE];
    int load_count;
    bool all_loaded;
  } textures;

  /* G-Buffer (multisampled) */
  struct {
    WGPUTexture position_ms; /* RGBA16Float multisampled */
    WGPUTexture normal_ms;   /* RGBA16Float multisampled */
    WGPUTexture albedo_ms;   /* RGBA8Unorm multisampled  */
    WGPUTexture depth_ms;    /* Depth24PlusStencil8 ms   */
    WGPUTextureView position_ms_view;
    WGPUTextureView normal_ms_view;
    WGPUTextureView albedo_ms_view;
    WGPUTextureView depth_ms_view;
    /* Resolve targets (sampleCount=1, for sampling in composition) */
    WGPUTexture position_resolve;
    WGPUTexture normal_resolve;
    WGPUTexture albedo_resolve;
    WGPUTextureView position_resolve_view;
    WGPUTextureView normal_resolve_view;
    WGPUTextureView albedo_resolve_view;
    WGPUSampler sampler;
  } gbuffer;

  /* Uniform buffers */
  WGPUBuffer offscreen_ubo;
  WGPUBuffer composition_ubo;

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
    vec4 instance_pos[NUM_INSTANCES];
  } offscreen_ubo_data;

  struct {
    light_t lights[NUM_LIGHTS];
    vec4 view_pos;
    int32_t debug_display_target;
    int32_t num_samples;
    float _pad[2]; /* Align to 16 bytes */
  } composition_ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout offscreen_bgl;
  WGPUBindGroupLayout composition_msaa_bgl;
  WGPUBindGroupLayout composition_no_msaa_bgl;

  /* Bind groups */
  WGPUBindGroup armor_bind_group;
  WGPUBindGroup box_bind_group;
  WGPUBindGroup composition_msaa_bind_group;
  WGPUBindGroup composition_no_msaa_bind_group;

  /* Pipeline layouts */
  WGPUPipelineLayout offscreen_pipeline_layout;
  WGPUPipelineLayout composition_msaa_pipeline_layout;
  WGPUPipelineLayout composition_no_msaa_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline offscreen_pipeline;
  WGPURenderPipeline composition_msaa_pipeline;
  WGPURenderPipeline composition_no_msaa_pipeline;

  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_attachments[3];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } offscreen_pass;

  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
    WGPURenderPassDescriptor descriptor;
  } composition_pass;

  /* GUI settings */
  struct {
    int32_t debug_display_target;
    bool use_msaa;
  } settings;

  WGPUBool initialized;
} state = {
  .settings.debug_display_target = 0,
  .settings.use_msaa             = true,
};

/* -------------------------------------------------------------------------- *
 * G-Buffer creation
 * -------------------------------------------------------------------------- */

static void destroy_gbuffer_textures(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.position_ms_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.normal_ms_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.albedo_ms_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.depth_ms_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.position_ms)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.normal_ms)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.albedo_ms)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.depth_ms)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.position_resolve_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.normal_resolve_view)
  WGPU_RELEASE_RESOURCE(TextureView, state.gbuffer.albedo_resolve_view)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.position_resolve)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.normal_resolve)
  WGPU_RELEASE_RESOURCE(Texture, state.gbuffer.albedo_resolve)
  WGPU_RELEASE_RESOURCE(Sampler, state.gbuffer.sampler)
}

static WGPUTextureView create_texture_view_2d(WGPUTexture texture,
                                              WGPUTextureFormat format)
{
  return wgpuTextureCreateView(texture,
                               &(WGPUTextureViewDescriptor){
                                 .format          = format,
                                 .dimension       = WGPUTextureViewDimension_2D,
                                 .baseMipLevel    = 0,
                                 .mipLevelCount   = 1,
                                 .baseArrayLayer  = 0,
                                 .arrayLayerCount = 1,
                                 .aspect          = WGPUTextureAspect_All,
                               });
}

static void init_gbuffer_textures(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  destroy_gbuffer_textures();

  const WGPUTextureUsage ms_usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
  const WGPUTextureUsage resolve_usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;

  /* Helper struct for creating paired MS + resolve textures */
  struct {
    WGPUTextureFormat format;
    WGPUTexture* ms_tex;
    WGPUTextureView* ms_view;
    WGPUTexture* resolve_tex;
    WGPUTextureView* resolve_view;
    const char* label;
  } color_attachments[] = {
    {WGPUTextureFormat_RGBA16Float, &state.gbuffer.position_ms,
     &state.gbuffer.position_ms_view, &state.gbuffer.position_resolve,
     &state.gbuffer.position_resolve_view, "GBuffer Position"},
    {WGPUTextureFormat_RGBA16Float, &state.gbuffer.normal_ms,
     &state.gbuffer.normal_ms_view, &state.gbuffer.normal_resolve,
     &state.gbuffer.normal_resolve_view, "GBuffer Normal"},
    {WGPUTextureFormat_RGBA8Unorm, &state.gbuffer.albedo_ms,
     &state.gbuffer.albedo_ms_view, &state.gbuffer.albedo_resolve,
     &state.gbuffer.albedo_resolve_view, "GBuffer Albedo"},
  };

  for (uint32_t i = 0; i < ARRAY_SIZE(color_attachments); i++) {
    /* Multisample texture */
    WGPUTextureDescriptor ms_desc = {
      .label         = STRVIEW("MS - Texture"),
      .usage         = ms_usage,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = color_attachments[i].format,
      .mipLevelCount = 1,
      .sampleCount   = SAMPLE_COUNT,
    };
    *color_attachments[i].ms_tex  = wgpuDeviceCreateTexture(device, &ms_desc);
    *color_attachments[i].ms_view = create_texture_view_2d(
      *color_attachments[i].ms_tex, color_attachments[i].format);

    /* Resolve texture (sampleCount = 1) */
    WGPUTextureDescriptor resolve_desc = {
      .label         = STRVIEW("Resolve Texture"),
      .usage         = resolve_usage,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = color_attachments[i].format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    *color_attachments[i].resolve_tex
      = wgpuDeviceCreateTexture(device, &resolve_desc);
    *color_attachments[i].resolve_view = create_texture_view_2d(
      *color_attachments[i].resolve_tex, color_attachments[i].format);
  }

  /* Depth: Depth24PlusStencil8, multisampled */
  {
    WGPUTextureDescriptor desc = {
      .label         = STRVIEW("GBuffer Depth MS"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {GBUFFER_DIM, GBUFFER_DIM, 1},
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .mipLevelCount = 1,
      .sampleCount   = SAMPLE_COUNT,
    };
    state.gbuffer.depth_ms      = wgpuDeviceCreateTexture(device, &desc);
    state.gbuffer.depth_ms_view = create_texture_view_2d(
      state.gbuffer.depth_ms, WGPUTextureFormat_Depth24PlusStencil8);
  }

  /* Nearest-neighbor sampler for G-Buffer sampling */
  {
    WGPUSamplerDescriptor desc = {
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
    };
    state.gbuffer.sampler = wgpuDeviceCreateSampler(device, &desc);
  }
}

/* -------------------------------------------------------------------------- *
 * Render pass descriptors
 * -------------------------------------------------------------------------- */

static void init_render_passes(void)
{
  /* Offscreen G-Buffer pass: 3 multisampled color + depth */
  {
    state.offscreen_pass.color_attachments[0] = (WGPURenderPassColorAttachment){
      .view          = state.gbuffer.position_ms_view,
      .resolveTarget = state.gbuffer.position_resolve_view,
      .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue    = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.color_attachments[1] = (WGPURenderPassColorAttachment){
      .view          = state.gbuffer.normal_ms_view,
      .resolveTarget = state.gbuffer.normal_resolve_view,
      .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue    = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.color_attachments[2] = (WGPURenderPassColorAttachment){
      .view          = state.gbuffer.albedo_ms_view,
      .resolveTarget = state.gbuffer.albedo_resolve_view,
      .depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp        = WGPULoadOp_Clear,
      .storeOp       = WGPUStoreOp_Store,
      .clearValue    = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.offscreen_pass.depth_stencil_attachment
      = (WGPURenderPassDepthStencilAttachment){
        .view              = state.gbuffer.depth_ms_view,
        .depthLoadOp       = WGPULoadOp_Clear,
        .depthStoreOp      = WGPUStoreOp_Store,
        .depthClearValue   = 1.0f,
        .stencilLoadOp     = WGPULoadOp_Clear,
        .stencilStoreOp    = WGPUStoreOp_Store,
        .stencilClearValue = 0,
      };
    state.offscreen_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("G-Buffer MSAA Render Pass"),
      .colorAttachmentCount   = 3,
      .colorAttachments       = state.offscreen_pass.color_attachments,
      .depthStencilAttachment = &state.offscreen_pass.depth_stencil_attachment,
    };
  }

  /* Composition pass: 1 color attachment (swapchain) + depth */
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
    &state.armor_model, "assets/models/armor/armor.gltf", 1.0f, &desc);
  if (!ok) {
    printf("Failed to load armor.gltf\n");
    return;
  }

  ok = gltf_model_load_from_file_ext(
    &state.box_model, "assets/models/deferred_box.gltf", 1.0f, &desc);
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
    {&state.armor_model, &state.armor_vertex_buffer, &state.armor_index_buffer,
     "Armor VB", "Armor IB"},
    {&state.box_model, &state.box_vertex_buffer, &state.box_index_buffer,
     "Box VB", "Box IB"},
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

  WGPUSamplerDescriptor desc = {
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
  };
  state.textures.sampler = wgpuDeviceCreateSampler(wgpu_context->device, &desc);
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

  /* Instance positions */
  glm_vec4_copy((vec4){0.0f, 0.0f, 0.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[0]);
  glm_vec4_copy((vec4){-4.0f, 0.0f, -4.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[1]);
  glm_vec4_copy((vec4){4.0f, 0.0f, -4.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[2]);

  state.offscreen_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Offscreen UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.offscreen_ubo_data),
            });

  state.composition_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Composition UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.composition_ubo_data),
            });

  /* Initialize lights - matching Vulkan positions */
  /* Light 0: White */
  state.composition_ubo_data.lights[0] = (light_t){
    .position = {0.0f, 0.0f, 5.0f, 0.0f},
    .color    = {1.5f, 1.5f, 1.5f},
    .radius   = 15.0f * 0.25f,
  };
  /* Light 1: Red */
  state.composition_ubo_data.lights[1] = (light_t){
    .position = {-2.30f, 0.0f, 1.05f, 0.0f},
    .color    = {1.0f, 0.0f, 0.0f},
    .radius   = 15.0f,
  };
  /* Light 2: Blue */
  state.composition_ubo_data.lights[2] = (light_t){
    .position = {4.0f, 1.0f, 2.0f, 0.0f},
    .color    = {0.0f, 0.0f, 2.5f},
    .radius   = 5.0f,
  };
  /* Light 3: Yellow */
  state.composition_ubo_data.lights[3] = (light_t){
    .position = {0.0f, 0.9f, 0.5f, 0.0f},
    .color    = {1.0f, 1.0f, 0.0f},
    .radius   = 2.0f,
  };
  /* Light 4: Green */
  state.composition_ubo_data.lights[4] = (light_t){
    .position = {5.0f, 0.5f, -3.53f, 0.0f},
    .color    = {0.0f, 1.0f, 0.2f},
    .radius   = 5.0f,
  };
  /* Light 5: Orange */
  state.composition_ubo_data.lights[5] = (light_t){
    .position = {7.07f, 1.0f, 7.07f, 0.0f},
    .color    = {1.0f, 0.7f, 0.3f},
    .radius   = 25.0f,
  };

  state.composition_ubo_data.num_samples = SAMPLE_COUNT;
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
  /* View position from camera (Vulkan viewPos calculation:
   * camera.position * vec4(-1, 1, -1, 1) ) */
  vec4 vp;
  glm_vec4_copy(state.camera.view_pos, vp);
  glm_vec4_copy(vp, state.composition_ubo_data.view_pos);

  state.composition_ubo_data.debug_display_target
    = state.settings.debug_display_target;
  state.composition_ubo_data.num_samples
    = state.settings.use_msaa ? (int32_t)SAMPLE_COUNT : 1;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.composition_ubo, 0,
                       &state.composition_ubo_data,
                       sizeof(state.composition_ubo_data));
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
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.offscreen_ubo_data),
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };

    state.offscreen_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Offscreen BGL"),
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Composition MSAA BGL: multisampled position/normal/albedo + UBO */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = true,
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = true,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = true,
        },
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.composition_ubo_data),
        },
      },
    };

    state.composition_msaa_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition MSAA BGL"),
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Composition No-MSAA BGL: resolved textures + sampler + UBO */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.composition_ubo_data),
        },
      },
    };

    state.composition_no_msaa_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition No-MSAA BGL"),
                .entryCount = 5,
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

  WGPU_RELEASE_RESOURCE(BindGroup, state.armor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.box_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_msaa_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_no_msaa_bind_group)

  /* Armor bind group: UBO + armor color + armor normal + sampler */
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
    state.armor_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Armor Bind Group"),
                .layout     = state.offscreen_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Box (background) bind group */
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
    state.box_bind_group
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Box Bind Group"),
                                            .layout = state.offscreen_bgl,
                                            .entryCount = 4,
                                            .entries    = entries,
                                          });
  }

  /* Composition MSAA bind group: multisampled textures + UBO */
  {
    WGPUBindGroupEntry entries[4] = {
      {.binding = 0, .textureView = state.gbuffer.position_ms_view},
      {.binding = 1, .textureView = state.gbuffer.normal_ms_view},
      {.binding = 2, .textureView = state.gbuffer.albedo_ms_view},
      {.binding = 3,
       .buffer  = state.composition_ubo,
       .offset  = 0,
       .size    = sizeof(state.composition_ubo_data)},
    };
    state.composition_msaa_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Composition MSAA Bind Group"),
                .layout     = state.composition_msaa_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Composition No-MSAA bind group: resolved textures + sampler + UBO */
  {
    WGPUBindGroupEntry entries[5] = {
      {.binding = 0, .textureView = state.gbuffer.position_resolve_view},
      {.binding = 1, .textureView = state.gbuffer.normal_resolve_view},
      {.binding = 2, .textureView = state.gbuffer.albedo_resolve_view},
      {.binding = 3, .sampler = state.gbuffer.sampler},
      {.binding = 4,
       .buffer  = state.composition_ubo,
       .offset  = 0,
       .size    = sizeof(state.composition_ubo_data)},
    };
    state.composition_no_msaa_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Composition No-MSAA Bind Group"),
                .layout     = state.composition_no_msaa_bgl,
                .entryCount = 5,
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

  state.composition_msaa_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label = STRVIEW("Composition MSAA Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.composition_msaa_bgl,
            });

  state.composition_no_msaa_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label = STRVIEW("Composition No-MSAA Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.composition_no_msaa_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ===== Offscreen (MRT) Pipeline with MSAA ===== */
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

    WGPUDepthStencilState depth_stencil_state = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.offscreen_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Offscreen MSAA Pipeline"),
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
                .depthStencil = &depth_stencil_state,
                .multisample  = (WGPUMultisampleState){
                  .count                  = SAMPLE_COUNT,
                  .mask                   = 0xFFFFFFFF,
                  .alphaToCoverageEnabled = true,
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

  /* ===== Composition Pipeline (MSAA - per sample lighting) ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, composition_msaa_shader_wgsl);

    WGPUBlendState no_blend           = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &no_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil_state = {
      .format            = wgpu_context->depth_stencil_format,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.composition_msaa_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Composition MSAA Pipeline"),
                .layout = state.composition_msaa_pipeline_layout,
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
                .depthStencil = &depth_stencil_state,
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

  /* ===== Composition Pipeline (No MSAA - standard sampling) ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, composition_no_msaa_shader_wgsl);

    WGPUBlendState no_blend           = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &no_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil_state = {
      .format            = wgpu_context->depth_stencil_format,
      .depthWriteEnabled = WGPUOptionalBool_True,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.composition_no_msaa_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Composition No-MSAA Pipeline"),
                .layout = state.composition_no_msaa_pipeline_layout,
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
                .depthStencil = &depth_stencil_state,
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
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Multi-sampled Deferred Shading", NULL,
          ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    static const char* display_items[]
      = {"Final composition", "Position", "Normals", "Albedo", "Specular"};
    imgui_overlay_combo_box("Display", &state.settings.debug_display_target,
                            display_items, ARRAY_SIZE(display_items));
    imgui_overlay_checkbox("MSAA", &state.settings.use_msaa);
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Callbacks
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 8,
      .num_channels = 1,
      .num_lanes    = 4,
      .logger.func  = slog_func,
    });

    /* Camera setup (first-person, matching Vulkan reference) */
    camera_init(&state.camera);
    state.camera.type           = CameraType_FirstPerson;
    state.camera.movement_speed = 5.0f;
    state.camera.rotation_speed = 0.25f;
    camera_set_position(&state.camera, (vec3){2.15f, 0.3f, -8.75f});
    camera_set_rotation(&state.camera, (vec3){-0.75f, 12.5f, 0.0f});
    camera_set_perspective(
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

    load_models();
    create_model_buffers(wgpu_context);
    init_gbuffer_textures(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_textures(wgpu_context);
    init_bind_group_layouts(wgpu_context);
    init_pipeline_layouts(wgpu_context);
    init_pipelines(wgpu_context);
    init_bind_groups(wgpu_context);
    init_render_passes();

    fetch_textures();

    imgui_overlay_init(wgpu_context);

    state.last_frame_time = stm_now();
    state.initialized     = true;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();
  update_textures(wgpu_context);

  /* Delta time */
  uint64_t now          = stm_now();
  float delta_time      = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  camera_update(&state.camera, delta_time);

  /* Update uniform buffers */
  update_uniform_buffer_offscreen(wgpu_context);
  update_uniform_buffer_composition(wgpu_context);

  /* ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ===== Pass 1: Offscreen G-Buffer (multisampled) ===== */
  {
    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_pass.descriptor);

    wgpuRenderPassEncoderSetPipeline(rpass, state.offscreen_pipeline);

    /* Background box */
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.box_bind_group, 0, NULL);
    draw_model(rpass, &state.box_model, state.box_vertex_buffer,
               state.box_index_buffer, 1);

    /* Armor instances */
    wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.armor_bind_group, 0,
                                      NULL);
    draw_model(rpass, &state.armor_model, state.armor_vertex_buffer,
               state.armor_index_buffer, NUM_INSTANCES);

    wgpuRenderPassEncoderEnd(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

  /* ===== Pass 2: Composition (fullscreen quad with deferred lighting) ===== */
  {
    state.composition_pass.color_attachment.view = wgpu_context->swapchain_view;
    state.composition_pass.depth_stencil_attachment.view
      = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder rpass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.composition_pass.descriptor);

    if (state.settings.use_msaa) {
      wgpuRenderPassEncoderSetPipeline(rpass, state.composition_msaa_pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        rpass, 0, state.composition_msaa_bind_group, 0, NULL);
    }
    else {
      wgpuRenderPassEncoderSetPipeline(rpass,
                                       state.composition_no_msaa_pipeline);
      wgpuRenderPassEncoderSetBindGroup(
        rpass, 0, state.composition_no_msaa_bind_group, 0, NULL);
    }

    /* Fullscreen triangle */
    wgpuRenderPassEncoderDraw(rpass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(rpass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer)
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc)

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
  camera_on_input_event(&state.camera, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                                / (float)wgpu_context->height);
  }
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  sfetch_shutdown();
  imgui_overlay_shutdown();

  /* Release G-Buffer textures */
  destroy_gbuffer_textures();

  /* Release model textures */
  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    wgpu_texture_t* tex = &state.textures.textures[i];
    WGPU_RELEASE_RESOURCE(Texture, tex->handle)
    WGPU_RELEASE_RESOURCE(TextureView, tex->view)
    FREE_TEXTURE_PIXELS(*tex);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.armor_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.armor_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.box_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.box_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.offscreen_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.composition_ubo)

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.armor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.box_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_msaa_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_no_msaa_bind_group)

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.offscreen_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_msaa_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_no_msaa_bgl)

  /* Release pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.offscreen_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_msaa_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout,
                        state.composition_no_msaa_pipeline_layout)

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.offscreen_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_msaa_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_no_msaa_pipeline)

  /* Release models */
  gltf_model_destroy(&state.armor_model);
  gltf_model_destroy(&state.box_model);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Multi-sampled Deferred Shading",
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

/* MRT shader: renders geometry into G-Buffer (position, normal, albedo) */
static const char* mrt_shader_wgsl = CODE(
  struct UBO {
    projection  : mat4x4f,
    model       : mat4x4f,
    view        : mat4x4f,
    instancePos : array<vec4f, 3>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var colorMap   : texture_2d<f32>;
  @group(0) @binding(2) var normalMap  : texture_2d<f32>;
  @group(0) @binding(3) var texSampler : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) tangent  : vec4f,
    @location(4) color    : vec4f,
    @builtin(instance_index) instanceIndex : u32,
  };

  struct VertexOutput {
    @builtin(position) position    : vec4f,
    @location(0)       worldNormal : vec3f,
    @location(1)       uv          : vec2f,
    @location(2)       color       : vec3f,
    @location(3)       worldPos    : vec3f,
    @location(4)       tangent     : vec3f,
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

/* Composition shader with MSAA: per-sample lighting with explicit resolve */
static const char* composition_msaa_shader_wgsl = CODE(
  struct Light {
    position : vec4f,
    color    : vec3f,
    radius   : f32,
  };

  struct UBO {
    lights             : array<Light, 6>,
    viewPos            : vec4f,
    debugDisplayTarget : i32,
    numSamples         : i32,
  };

  @group(0) @binding(0) var gPosition : texture_multisampled_2d<f32>;
  @group(0) @binding(1) var gNormal   : texture_multisampled_2d<f32>;
  @group(0) @binding(2) var gAlbedo   : texture_multisampled_2d<f32>;
  @group(0) @binding(3) var<uniform> ubo : UBO;

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

  fn calculateLighting(pos : vec3f, normal : vec3f, albedo : vec4f) -> vec3f {
    var result = vec3f(0.0);

    for (var i = 0; i < 6; i++) {
      let L = ubo.lights[i].position.xyz - pos;
      let dist = length(L);

      let V = normalize(ubo.viewPos.xyz - pos);
      let Ln = normalize(L);
      let N = normalize(normal);

      let atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

      let NdotL = max(0.0, dot(N, Ln));
      let diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

      let R = reflect(-Ln, N);
      let NdotR = max(0.0, dot(R, V));
      let spec = ubo.lights[i].color * albedo.a * pow(NdotR, 8.0) * atten;

      result += diff + spec;
    }
    return result;
  }

  fn resolve(tex : texture_multisampled_2d<f32>, coords : vec2i) -> vec4f {
    var result = vec4f(0.0);
    let sampleCount = ubo.numSamples;
    for (var i = 0; i < sampleCount; i++) {
      result += textureLoad(tex, coords, i);
    }
    return result / f32(sampleCount);
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let attDim = textureDimensions(gPosition);
    let UV = vec2i(in.uv * vec2f(attDim));

    if (ubo.debugDisplayTarget > 0) {
      switch (ubo.debugDisplayTarget) {
        case 1: { return vec4f(textureLoad(gPosition, UV, 0).rgb, 1.0); }
        case 2: { return vec4f(textureLoad(gNormal, UV, 0).rgb, 1.0); }
        case 3: { return vec4f(textureLoad(gAlbedo, UV, 0).rgb, 1.0); }
        case 4: { return vec4f(textureLoad(gAlbedo, UV, 0).aaa, 1.0); }
        default: { return vec4f(textureLoad(gPosition, UV, 0).rgb, 1.0); }
      }
    }

    let ambient = 0.15;
    let alb = resolve(gAlbedo, UV);
    var fragColor = vec3f(0.0);

    let sampleCount = ubo.numSamples;
    for (var i = 0; i < sampleCount; i++) {
      let pos = textureLoad(gPosition, UV, i).rgb;
      let normal = textureLoad(gNormal, UV, i).rgb;
      let albedo = textureLoad(gAlbedo, UV, i);
      fragColor += calculateLighting(pos, normal, albedo);
    }

    fragColor = (alb.rgb * f32(ambient)) + fragColor / f32(sampleCount);

    return vec4f(fragColor, 1.0);
  }
);

/* Composition shader without MSAA: standard texture sampling */
static const char* composition_no_msaa_shader_wgsl = CODE(
  struct Light {
    position : vec4f,
    color    : vec3f,
    radius   : f32,
  };

  struct UBO {
    lights             : array<Light, 6>,
    viewPos            : vec4f,
    debugDisplayTarget : i32,
    numSamples         : i32,
  };

  @group(0) @binding(0) var gPosition : texture_2d<f32>;
  @group(0) @binding(1) var gNormal   : texture_2d<f32>;
  @group(0) @binding(2) var gAlbedo   : texture_2d<f32>;
  @group(0) @binding(3) var gSampler  : sampler;
  @group(0) @binding(4) var<uniform> ubo : UBO;

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

  fn calculateLighting(pos : vec3f, normal : vec3f, albedo : vec4f) -> vec3f {
    var result = vec3f(0.0);

    for (var i = 0; i < 6; i++) {
      let L = ubo.lights[i].position.xyz - pos;
      let dist = length(L);

      let V = normalize(ubo.viewPos.xyz - pos);
      let Ln = normalize(L);
      let N = normalize(normal);

      let atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

      let NdotL = max(0.0, dot(N, Ln));
      let diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

      let R = reflect(-Ln, N);
      let NdotR = max(0.0, dot(R, V));
      let spec = ubo.lights[i].color * albedo.a * pow(NdotR, 8.0) * atten;

      result += diff + spec;
    }
    return result;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let fragPos = textureSample(gPosition, gSampler, in.uv).rgb;
    let normal  = textureSample(gNormal, gSampler, in.uv).rgb;
    let albedo  = textureSample(gAlbedo, gSampler, in.uv);

    if (ubo.debugDisplayTarget > 0) {
      switch (ubo.debugDisplayTarget) {
        case 1: { return vec4f(fragPos, 1.0); }
        case 2: { return vec4f(normal, 1.0); }
        case 3: { return vec4f(albedo.rgb, 1.0); }
        case 4: { return vec4f(albedo.aaa, 1.0); }
        default: { return vec4f(fragPos, 1.0); }
      }
    }

    let ambient = 0.15;
    var fragColor = albedo.rgb * ambient;
    fragColor += calculateLighting(fragPos, normal, albedo);

    return vec4f(fragColor, 1.0);
  }
);

// clang-format on
