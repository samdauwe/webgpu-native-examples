/**
 * @brief Subpasses - G-Buffer compositing with forward transparency.
 *
 * Ported from the Vulkan subpasses example. Implements a deferred rendering
 * setup with a forward transparency pass. In Vulkan this uses three subpasses
 * within a single render pass with input attachments. In WebGPU (which has no
 * subpass concept) this is implemented as three separate render passes:
 *
 *   Pass 0: G-Buffer fill (MRT - position, normal, albedo + depth)
 *   Pass 1: Deferred composition (fullscreen, reads G-Buffer via textureLoad)
 *   Pass 2: Forward transparency (alpha-blended glass, reuses depth)
 *
 * 64 randomized point lights illuminate the scene with per-pixel evaluation.
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/subpasses
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

#define NUM_LIGHTS (64u)

/* Texture fetch buffer size */
#define TEXTURE_FILE_BUFFER_SIZE (2 * 1024 * 1024)

/* -------------------------------------------------------------------------- *
 * WGSL shader source forward declarations
 * -------------------------------------------------------------------------- */

static const char* gbuffer_shader_wgsl;
static const char* composition_shader_wgsl;
static const char* transparent_shader_wgsl;

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

  /* Models */
  gltf_model_t scene_model;
  gltf_model_t transparent_model;
  bool models_loaded;

  /* GPU buffers for models */
  WGPUBuffer scene_vertex_buffer;
  WGPUBuffer scene_index_buffer;
  WGPUBuffer transparent_vertex_buffer;
  WGPUBuffer transparent_index_buffer;

  /* Glass texture */
  struct {
    wgpu_texture_t texture;
    WGPUSampler sampler;
    uint8_t file_buffer[TEXTURE_FILE_BUFFER_SIZE];
    bool loaded;
  } glass_texture;

  /* G-Buffer */
  struct {
    WGPUTexture
      position_texture;         /* RGBA16Float: world XYZ + linearized depth */
    WGPUTexture normal_texture; /* RGBA16Float: world normal                 */
    WGPUTexture albedo_texture; /* RGBA8Unorm:  vertex color RGB + A         */
    WGPUTexture depth_texture;  /* Depth24PlusStencil8                       */
    WGPUTextureView position_view;
    WGPUTextureView normal_view;
    WGPUTextureView albedo_view;
    WGPUTextureView depth_view;
    uint32_t width;
    uint32_t height;
  } gbuffer;

  /* Uniform buffers */
  WGPUBuffer gbuffer_ubo;

  /* Lights SSBO */
  WGPUBuffer lights_ssbo;

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
  } gbuffer_ubo_data;

  /* Lights data */
  light_t lights[NUM_LIGHTS];

  /* Bind group layouts */
  WGPUBindGroupLayout gbuffer_bgl;     /* G-Buffer pass (scene)      */
  WGPUBindGroupLayout composition_bgl; /* Composition pass           */
  WGPUBindGroupLayout transparent_bgl; /* Transparent forward pass   */

  /* Bind groups */
  WGPUBindGroup scene_bind_group;
  WGPUBindGroup composition_bind_group;
  WGPUBindGroup transparent_bind_group;

  /* Pipeline layouts */
  WGPUPipelineLayout gbuffer_pipeline_layout;
  WGPUPipelineLayout composition_pipeline_layout;
  WGPUPipelineLayout transparent_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline gbuffer_pipeline;
  WGPURenderPipeline composition_pipeline;
  WGPURenderPipeline transparent_pipeline;

  /* Render pass descriptors */
  struct {
    WGPURenderPassColorAttachment color_attachments[3];
    WGPURenderPassDepthStencilAttachment depth_stencil;
    WGPURenderPassDescriptor descriptor;
  } gbuffer_pass;

  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDescriptor descriptor;
  } composition_pass;

  struct {
    WGPURenderPassColorAttachment color_attachment;
    WGPURenderPassDepthStencilAttachment depth_stencil;
    WGPURenderPassDescriptor descriptor;
  } transparent_pass;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {0};

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
}

static void init_gbuffer_textures(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = wgpu_context->width;
  uint32_t h        = wgpu_context->height;

  if (state.gbuffer.width == w && state.gbuffer.height == h) {
    return;
  }

  destroy_gbuffer_textures();

  state.gbuffer.width  = w;
  state.gbuffer.height = h;

  /* Position: RGBA16Float (xyz = world pos, a = linearized depth) */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer Position"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {w, h, 1},
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
      .size          = {w, h, 1},
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

  /* Albedo: RGBA8Unorm */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer Albedo"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {w, h, 1},
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

  /* Depth: Depth24PlusStencil8 (shared across G-buffer and transparent pass) */
  {
    WGPUTextureDescriptor desc = {
      .label         = STRVIEW("GBuffer Depth"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {w, h, 1},
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
}

/* -------------------------------------------------------------------------- *
 * Render pass descriptors
 * -------------------------------------------------------------------------- */

static void init_render_passes(void)
{
  /* Pass 0: G-Buffer fill (3 color MRT + depth, clear all) */
  {
    state.gbuffer_pass.color_attachments[0] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.position_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.gbuffer_pass.color_attachments[1] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.normal_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.gbuffer_pass.color_attachments[2] = (WGPURenderPassColorAttachment){
      .view       = state.gbuffer.albedo_view,
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.gbuffer_pass.depth_stencil = (WGPURenderPassDepthStencilAttachment){
      .view              = state.gbuffer.depth_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };
    state.gbuffer_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("G-Buffer Pass"),
      .colorAttachmentCount   = 3,
      .colorAttachments       = state.gbuffer_pass.color_attachments,
      .depthStencilAttachment = &state.gbuffer_pass.depth_stencil,
    };
  }

  /* Pass 1: Composition (1 color target = swapchain, no depth) */
  {
    state.composition_pass.color_attachment = (WGPURenderPassColorAttachment){
      .view       = NULL, /* Set per frame */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    state.composition_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("Composition Pass"),
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.composition_pass.color_attachment,
      .depthStencilAttachment = NULL,
    };
  }

  /* Pass 2: Transparent forward (1 color = swapchain, load depth from pass 0)*/
  {
    state.transparent_pass.color_attachment = (WGPURenderPassColorAttachment){
      .view       = NULL, /* Set per frame */
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      .loadOp     = WGPULoadOp_Load, /* Preserve composition result */
      .storeOp    = WGPUStoreOp_Store,
    };
    state.transparent_pass.depth_stencil
      = (WGPURenderPassDepthStencilAttachment){
        .view           = state.gbuffer.depth_view,
        .depthLoadOp    = WGPULoadOp_Load, /* Reuse G-Buffer depth */
        .depthStoreOp   = WGPUStoreOp_Store,
        .stencilLoadOp  = WGPULoadOp_Load,
        .stencilStoreOp = WGPUStoreOp_Store,
      };
    state.transparent_pass.descriptor = (WGPURenderPassDescriptor){
      .label                  = STRVIEW("Transparent Pass"),
      .colorAttachmentCount   = 1,
      .colorAttachments       = &state.transparent_pass.color_attachment,
      .depthStencilAttachment = &state.transparent_pass.depth_stencil,
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
    &state.scene_model, "assets/models/samplebuilding.gltf", 1.0f, &desc);
  if (!ok) {
    printf("Failed to load samplebuilding.gltf\n");
    return;
  }

  ok = gltf_model_load_from_file_ext(&state.transparent_model,
                                     "assets/models/samplebuilding_glass.gltf",
                                     1.0f, &desc);
  if (!ok) {
    printf("Failed to load samplebuilding_glass.gltf\n");
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
    {&state.scene_model, &state.scene_vertex_buffer, &state.scene_index_buffer,
     "Scene VB", "Scene IB"},
    {&state.transparent_model, &state.transparent_vertex_buffer,
     &state.transparent_index_buffer, "Transparent VB", "Transparent IB"},
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
 * Glass texture loading
 * -------------------------------------------------------------------------- */

static void glass_texture_fetch_cb(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Glass texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_w, img_h, num_ch;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_w, &img_h, &num_ch, 4);

  if (pixels) {
    wgpu_texture_t* tex = &state.glass_texture.texture;
    tex->desc           = (wgpu_texture_desc_t){
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
    tex->desc.is_dirty         = true;
    state.glass_texture.loaded = true;
  }
}

static void fetch_glass_texture(void)
{
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/colored_glass_rgba.png",
    .callback = glass_texture_fetch_cb,
    .buffer   = SFETCH_RANGE(state.glass_texture.file_buffer),
  });
}

/* Forward declaration */
static void init_bind_groups(struct wgpu_context_t* wgpu_context);

static void init_glass_texture(struct wgpu_context_t* wgpu_context)
{
  state.glass_texture.texture
    = wgpu_create_color_bars_texture(wgpu_context, NULL);

  WGPUSamplerDescriptor desc = {
    .label         = STRVIEW("Glass Sampler"),
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
  state.glass_texture.sampler
    = wgpuDeviceCreateSampler(wgpu_context->device, &desc);
}

static void update_glass_texture(struct wgpu_context_t* wgpu_context)
{
  wgpu_texture_t* tex = &state.glass_texture.texture;
  if (tex->desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, tex);
    FREE_TEXTURE_PIXELS(*tex);
    init_bind_groups(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Lights initialization
 * -------------------------------------------------------------------------- */

static uint32_t rng_state = 0;

static float rnd_float(float min_val, float max_val)
{
  /* Simple xorshift32 for reproducible randomness */
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  float t = (float)(rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
  return min_val + t * (max_val - min_val);
}

static void init_lights(void)
{
  rng_state = 42u; /* Fixed seed for reproducibility */

  for (uint32_t i = 0; i < NUM_LIGHTS; i++) {
    state.lights[i].position[0] = rnd_float(-1.0f, 1.0f) * 8.0f;
    state.lights[i].position[1] = 0.25f + fabsf(rnd_float(-1.0f, 1.0f)) * 4.0f;
    state.lights[i].position[2] = rnd_float(-1.0f, 1.0f) * 8.0f;
    state.lights[i].position[3] = 1.0f;

    state.lights[i].color[0] = rnd_float(0.0f, 0.5f) * 2.0f;
    state.lights[i].color[1] = rnd_float(0.0f, 0.5f) * 2.0f;
    state.lights[i].color[2] = rnd_float(0.0f, 0.5f) * 2.0f;

    state.lights[i].radius = 1.0f + fabsf(rnd_float(-1.0f, 1.0f));
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* G-Buffer UBO (MVP matrices) */
  state.gbuffer_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("GBuffer UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.gbuffer_ubo_data),
            });

  /* Lights SSBO */
  state.lights_ssbo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Lights SSBO"),
              .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.lights),
            });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Update MVP */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.gbuffer_ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.gbuffer_ubo_data.view);
  glm_mat4_identity(state.gbuffer_ubo_data.model);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.gbuffer_ubo, 0,
                       &state.gbuffer_ubo_data, sizeof(state.gbuffer_ubo_data));

  /* Upload lights */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.lights_ssbo, 0, state.lights,
                       sizeof(state.lights));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* G-Buffer BGL: binding 0 = UBO (vertex) */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.gbuffer_ubo_data),
        },
      },
    };
    state.gbuffer_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("GBuffer BGL"),
                .entryCount = 1,
                .entries    = entries,
              });
  }

  /* Composition BGL: 3 G-Buffer textures + lights SSBO */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      /* Binding 0: Position texture (fragment) */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 1: Normal texture (fragment) */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 2: Albedo texture (fragment) */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 3: Lights SSBO (fragment) */
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {
          .type           = WGPUBufferBindingType_ReadOnlyStorage,
          .minBindingSize = sizeof(state.lights),
        },
      },
    };
    state.composition_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition BGL"),
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Transparent BGL: UBO + position texture + glass texture + sampler */
  {
    WGPUBindGroupLayoutEntry entries[4] = {
      /* Binding 0: UBO (vertex) */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.gbuffer_ubo_data),
        },
      },
      /* Binding 1: Position/depth texture (fragment) */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 2: Glass texture (fragment) */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      /* Binding 3: Sampler (fragment) */
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    state.transparent_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Transparent BGL"),
                .entryCount = 4,
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

  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.transparent_bind_group)

  /* Scene bind group (G-Buffer pass): UBO */
  {
    WGPUBindGroupEntry entries[1] = {
      {
        .binding = 0,
        .buffer  = state.gbuffer_ubo,
        .offset  = 0,
        .size    = sizeof(state.gbuffer_ubo_data),
      },
    };
    state.scene_bind_group
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label      = STRVIEW("Scene BG"),
                                            .layout     = state.gbuffer_bgl,
                                            .entryCount = 1,
                                            .entries    = entries,
                                          });
  }

  /* Composition bind group: G-Buffer textures + lights SSBO */
  {
    WGPUBindGroupEntry entries[4] = {
      {.binding = 0, .textureView = state.gbuffer.position_view},
      {.binding = 1, .textureView = state.gbuffer.normal_view},
      {.binding = 2, .textureView = state.gbuffer.albedo_view},
      {
        .binding = 3,
        .buffer  = state.lights_ssbo,
        .offset  = 0,
        .size    = sizeof(state.lights),
      },
    };
    state.composition_bind_group
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Composition BG"),
                                            .layout = state.composition_bgl,
                                            .entryCount = 4,
                                            .entries    = entries,
                                          });
  }

  /* Transparent bind group: UBO + position texture + glass + sampler */
  {
    WGPUBindGroupEntry entries[4] = {
      {
        .binding = 0,
        .buffer  = state.gbuffer_ubo,
        .offset  = 0,
        .size    = sizeof(state.gbuffer_ubo_data),
      },
      {.binding = 1, .textureView = state.gbuffer.position_view},
      {.binding = 2, .textureView = state.glass_texture.texture.view},
      {.binding = 3, .sampler = state.glass_texture.sampler},
    };
    state.transparent_bind_group
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label  = STRVIEW("Transparent BG"),
                                            .layout = state.transparent_bgl,
                                            .entryCount = 4,
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

  state.gbuffer_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("GBuffer PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.gbuffer_bgl,
            });

  state.composition_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Composition PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.composition_bgl,
            });

  state.transparent_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Transparent PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.transparent_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* ===== G-Buffer Pipeline (MRT, 3 color targets) ===== */
  {
    WGPUVertexAttribute vert_attrs[] = {
      /* position: vec3f */
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      /* color: vec4f */
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
      /* normal: vec3f */
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vert_attrs),
      .attributes     = vert_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, gbuffer_shader_wgsl);

    WGPUBlendState no_blend               = wgpu_create_blend_state(false);
    WGPUColorTargetState color_targets[3] = {
      {/* Position: RGBA16Float */
       .format    = WGPUTextureFormat_RGBA16Float,
       .blend     = &no_blend,
       .writeMask = WGPUColorWriteMask_All},
      {/* Normal: RGBA16Float */
       .format    = WGPUTextureFormat_RGBA16Float,
       .blend     = &no_blend,
       .writeMask = WGPUColorWriteMask_All},
      {/* Albedo: RGBA8Unorm */
       .format    = WGPUTextureFormat_RGBA8Unorm,
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

    state.gbuffer_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("GBuffer Pipeline"),
                .layout = state.gbuffer_pipeline_layout,
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
    WGPU_RELEASE_RESOURCE(ShaderModule, shader)
  }

  /* ===== Composition Pipeline (fullscreen, no vertex input) ===== */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, composition_shader_wgsl);

    WGPUBlendState no_blend           = wgpu_create_blend_state(false);
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &no_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.composition_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Composition Pipeline"),
                .layout = state.composition_pipeline_layout,
                .vertex = (WGPUVertexState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("vs_main"),
                  .bufferCount = 0,
                },
                .primitive = (WGPUPrimitiveState){
                  .topology  = WGPUPrimitiveTopology_TriangleList,
                  .frontFace = WGPUFrontFace_CCW,
                  .cullMode  = WGPUCullMode_None,
                },
                .multisample = (WGPUMultisampleState){
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
    WGPU_RELEASE_RESOURCE(ShaderModule, shader)
  }

  /* ===== Transparent Pipeline (alpha blending, reuse depth) ===== */
  {
    WGPUVertexAttribute vert_attrs[] = {
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, color)},
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vert_attrs),
      .attributes     = vert_attrs,
    };

    WGPUShaderModule shader
      = wgpu_create_shader_module(device, transparent_shader_wgsl);

    /* Alpha blending */
    WGPUBlendState blend = {
      .color = {
        .srcFactor = WGPUBlendFactor_SrcAlpha,
        .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
        .operation = WGPUBlendOperation_Add,
      },
      .alpha = {
        .srcFactor = WGPUBlendFactor_One,
        .dstFactor = WGPUBlendFactor_Zero,
        .operation = WGPUBlendOperation_Add,
      },
    };
    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPUDepthStencilState depth_stencil = {
      .format            = WGPUTextureFormat_Depth24PlusStencil8,
      .depthWriteEnabled = WGPUOptionalBool_False,
      .depthCompare      = WGPUCompareFunction_LessEqual,
      .stencilFront      = {.compare = WGPUCompareFunction_Always},
      .stencilBack       = {.compare = WGPUCompareFunction_Always},
    };

    state.transparent_pipeline = wgpuDeviceCreateRenderPipeline(
      device, &(WGPURenderPipelineDescriptor){
                .label  = STRVIEW("Transparent Pipeline"),
                .layout = state.transparent_pipeline_layout,
                .vertex = (WGPUVertexState){
                  .module      = shader,
                  .entryPoint  = STRVIEW("vs_main"),
                  .bufferCount = 1,
                  .buffers     = &vb_layout,
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
    WGPU_RELEASE_RESOURCE(ShaderModule, shader)
  }
}

/* -------------------------------------------------------------------------- *
 * Helper: draw a gltf model
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib)
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
 * ImGui overlay
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Subpasses", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Subpasses", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igText("0: Deferred G-Buffer creation");
    igText("1: Deferred composition");
    igText("2: Forward transparency");
  }

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    if (igButton("Randomize lights", (ImVec2){0, 0})) {
      rng_state = (uint32_t)stm_now();
      init_lights();
    }
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
  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Window resize handler
 * -------------------------------------------------------------------------- */

static void on_resize(struct wgpu_context_t* wgpu_context)
{
  init_gbuffer_textures(wgpu_context);
  init_render_passes();
  init_bind_groups(wgpu_context);
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

  /* Camera (first-person) */
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.movement_speed = 5.0f;
  state.camera.rotation_speed = 0.25f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  /* Vulkan: pos (-3.2, 1.0, 5.9), rot (0.5, 210.05, 0) */
  camera_set_position(&state.camera, (vec3){-3.2f, 1.0f, 5.9f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(0.5f, 210.05f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load models */
  load_models();
  create_model_buffers(wgpu_context);

  /* Initialize lights */
  init_lights();

  /* G-Buffer textures */
  init_gbuffer_textures(wgpu_context);

  /* Render pass descriptors */
  init_render_passes();

  /* Glass texture */
  init_glass_texture(wgpu_context);
  fetch_glass_texture();

  /* Uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Bind group layouts → pipeline layouts */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Bind groups */
  init_bind_groups(wgpu_context);

  /* Render pipelines */
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
  update_glass_texture(wgpu_context);

  /* Handle window resize */
  if (state.gbuffer.width != (uint32_t)wgpu_context->width
      || state.gbuffer.height != (uint32_t)wgpu_context->height) {
    on_resize(wgpu_context);
  }

  /* Update camera */
  camera_update(&state.camera, delta);

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, delta);
  render_gui(wgpu_context);

  uint32_t w = wgpu_context->width;
  uint32_t h = wgpu_context->height;

  WGPUCommandEncoder cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* ===== Pass 0: G-Buffer fill ===== */
  if (state.models_loaded) {
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.gbuffer_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.gbuffer_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0, 0);
    draw_model(pass, &state.scene_model, state.scene_vertex_buffer,
               state.scene_index_buffer);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  }

  /* ===== Pass 1: Composition (deferred lighting) ===== */
  {
    state.composition_pass.color_attachment.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.composition_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.composition_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.composition_bind_group, 0,
                                      0);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  }

  /* ===== Pass 2: Transparent forward ===== */
  if (state.models_loaded) {
    state.transparent_pass.color_attachment.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.transparent_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);
    wgpuRenderPassEncoderSetPipeline(pass, state.transparent_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.transparent_bind_group, 0,
                                      0);
    draw_model(pass, &state.transparent_model, state.transparent_vertex_buffer,
               state.transparent_index_buffer);

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

  gltf_model_destroy(&state.scene_model);
  gltf_model_destroy(&state.transparent_model);

  WGPU_RELEASE_RESOURCE(Buffer, state.scene_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.transparent_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.transparent_index_buffer)

  wgpu_destroy_texture(&state.glass_texture.texture);
  WGPU_RELEASE_RESOURCE(Sampler, state.glass_texture.sampler)

  destroy_gbuffer_textures();

  WGPU_RELEASE_RESOURCE(Buffer, state.gbuffer_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.lights_ssbo)

  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.transparent_bind_group)

  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.gbuffer_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.transparent_bgl)

  WGPU_RELEASE_RESOURCE(PipelineLayout, state.gbuffer_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.transparent_pipeline_layout)

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.gbuffer_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.transparent_pipeline)

  imgui_overlay_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Subpasses",
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

/* G-Buffer vertex + fragment shader */
static const char* gbuffer_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model      : mat4x4f,
    view       : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) color    : vec4f,
    @location(2) normal   : vec3f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       normal   : vec3f,
    @location(1)       color    : vec3f,
    @location(2)       worldPos : vec3f,
  };

  struct FragmentOutput {
    @location(0) gPosition : vec4f,
    @location(1) gNormal   : vec4f,
    @location(2) gAlbedo   : vec4f,
  };

  const NEAR_PLANE : f32 = 0.1;
  const FAR_PLANE  : f32 = 256.0;

  fn linearDepth(depth : f32) -> f32 {
    let z = depth * 2.0 - 1.0;
    return (2.0 * NEAR_PLANE * FAR_PLANE)
         / (FAR_PLANE + NEAR_PLANE - z * (FAR_PLANE - NEAR_PLANE));
  }

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    out.position = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);

    // World-space position
    out.worldPos = (ubo.model * vec4f(in.position, 1.0)).xyz;

    // Normal in world space
    let normalMatrix = mat3x3f(
      ubo.model[0].xyz,
      ubo.model[1].xyz,
      ubo.model[2].xyz,
    );
    out.normal = normalMatrix * normalize(in.normal);

    out.color = in.color.rgb;

    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    out.gPosition = vec4f(in.worldPos, 1.0);

    let N = normalize(in.normal);
    out.gNormal = vec4f(N, 1.0);

    out.gAlbedo.r = in.color.r;
    out.gAlbedo.g = in.color.g;
    out.gAlbedo.b = in.color.b;
    out.gAlbedo.a = 1.0;

    // Store linearized depth in position alpha
    out.gPosition.a = linearDepth(in.position.z);

    return out;
  }
);

/* Composition vertex + fragment shader (fullscreen deferred lighting) */
static const char* composition_shader_wgsl = CODE(
  struct Light {
    position : vec4f,
    color    : vec3f,
    radius   : f32,
  };

  @group(0) @binding(0) var inputPosition : texture_2d<f32>;
  @group(0) @binding(1) var inputNormal   : texture_2d<f32>;
  @group(0) @binding(2) var inputAlbedo   : texture_2d<f32>;
  @group(0) @binding(3) var<storage, read> lights : array<Light, 64>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(
      f32((vertexIndex << 1u) & 2u),
      f32(vertexIndex & 2u),
    );
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let coords = vec2i(floor(in.position.xy));

    let fragPos = textureLoad(inputPosition, coords, 0).rgb;
    let normal  = textureLoad(inputNormal, coords, 0).rgb;
    let albedo  = textureLoad(inputAlbedo, coords, 0);

    let ambient = 0.05;

    var fragcolor = albedo.rgb * ambient;

    for (var i = 0u; i < 64u; i++) {
      let L    = lights[i].position.xyz - fragPos;
      let dist = length(L);
      let Ln   = normalize(L);

      let atten = lights[i].radius / (pow(dist, 3.0) + 1.0);

      let N     = normalize(normal);
      let NdotL = max(0.0, dot(N, Ln));
      let diff  = lights[i].color * albedo.rgb * NdotL * atten;

      fragcolor += diff;
    }

    return vec4f(fragcolor, 1.0);
  }
);

/* Transparent vertex + fragment shader */
static const char* transparent_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model      : mat4x4f,
    view       : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var samplerPositionDepth : texture_2d<f32>;
  @group(0) @binding(2) var samplerTexture : texture_2d<f32>;
  @group(0) @binding(3) var texSampler : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) color    : vec4f,
    @location(2) normal   : vec3f,
    @location(3) uv       : vec2f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       color    : vec3f,
    @location(1)       uv      : vec2f,
  };

  const NEAR_PLANE : f32 = 0.1;
  const FAR_PLANE  : f32 = 256.0;

  fn linearDepth(depth : f32) -> f32 {
    let z = depth * 2.0 - 1.0;
    return (2.0 * NEAR_PLANE * FAR_PLANE)
         / (FAR_PLANE + NEAR_PLANE - z * (FAR_PLANE - NEAR_PLANE));
  }

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color = in.color.rgb;
    out.uv    = in.uv;
    out.position = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let coords = vec2i(floor(in.position.xy));

    // Read linearized depth from G-Buffer position.a
    let depth = textureLoad(samplerPositionDepth, coords, 0).a;

    // Sample the glass texture before discard to avoid
    // implicit derivatives in non-uniform control flow
    let sampledColor = textureSample(samplerTexture, texSampler, in.uv);

    // Discard fragments behind opaque geometry
    if (depth != 0.0 && linearDepth(in.position.z) > depth) {
      discard;
    }

    return sampledColor;
  }
);

// clang-format on
