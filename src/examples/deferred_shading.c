/**
 * @brief Deferred shading with multiple render targets (G-Buffer).
 *
 * Ported from the Vulkan deferred shading example. Renders the scene into
 * three off-screen render targets (position, normal, albedo+specular) which
 * are then sampled in a composition pass that evaluates 6 animated point
 * lights per pixel using Blinn-Phong shading.
 *
 * Debug visualization modes allow inspecting individual G-Buffer channels.
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/deferred
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

/* Texture fetch buffer size (1024x1024 RGBA = 4 MB max compressed) */
#define TEXTURE_FILE_BUFFER_SIZE (5 * 1024 * 1024)

/* -------------------------------------------------------------------------- *
 * WGSL shader source forward declarations
 * -------------------------------------------------------------------------- */

static const char* mrt_shader_wgsl;
static const char* composition_shader_wgsl;

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
  float timer;
  float animation_speed;
  bool paused;
  uint64_t last_frame_time;

  /* Models */
  gltf_model_t armor_model;
  gltf_model_t floor_model;
  bool models_loaded;

  /* GPU buffers for models */
  WGPUBuffer armor_vertex_buffer;
  WGPUBuffer armor_index_buffer;
  WGPUBuffer floor_vertex_buffer;
  WGPUBuffer floor_index_buffer;

  /* Textures: 0=armor color, 1=armor normal, 2=floor color, 3=floor normal */
  struct {
    wgpu_texture_t textures[NUM_TEXTURES];
    WGPUSampler sampler;
    uint8_t file_buffers[NUM_TEXTURES][TEXTURE_FILE_BUFFER_SIZE];
    int load_count;
    bool all_loaded;
  } textures;

  /* G-Buffer */
  struct {
    WGPUTexture position_texture; /* RGBA16Float: world-space XYZ + padding */
    WGPUTexture normal_texture;   /* RGBA16Float: world-space normal + pad  */
    WGPUTexture albedo_texture;   /* RGBA8Unorm:  diffuse RGB + specular A  */
    WGPUTexture depth_texture;    /* Depth24PlusStencil8                    */
    WGPUTextureView position_view;
    WGPUTextureView normal_view;
    WGPUTextureView albedo_view;
    WGPUTextureView depth_view;
    WGPUSampler sampler; /* Nearest-neighbor for G-Buffer sampling */
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
    float _pad[3]; /* Align to 16 bytes */
  } composition_ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout offscreen_bgl;   /* For MRT pass (model + floor) */
  WGPUBindGroupLayout composition_bgl; /* For composition pass         */

  /* Bind groups */
  WGPUBindGroup armor_bind_group;
  WGPUBindGroup floor_bind_group;
  WGPUBindGroup composition_bind_group;

  /* Pipeline layouts */
  WGPUPipelineLayout offscreen_pipeline_layout;
  WGPUPipelineLayout composition_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline offscreen_pipeline;
  WGPURenderPipeline composition_pipeline;

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
  } settings;

  WGPUBool initialized;
} state = {
  .animation_speed               = 0.1f,
  .timer                         = 0.0f,
  .paused                        = false,
  .settings.debug_display_target = 0,
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

  /* Destroy previous textures if resizing */
  destroy_gbuffer_textures();

  /* Position: RGBA16Float */
  {
    WGPUTextureDescriptor desc = {
      .label = STRVIEW("GBuffer - Position"),
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

  /* Albedo: RGBA8Unorm (RGB = diffuse color, A = specular intensity) */
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
    &state.floor_model, "assets/models/deferred_floor.gltf", 1.0f, &desc);
  if (!ok) {
    printf("Failed to load deferred_floor.gltf\n");
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
    {&state.floor_model, &state.floor_vertex_buffer, &state.floor_index_buffer,
     "Floor VB", "Floor IB"},
  };

  for (int i = 0; i < 2; i++) {
    gltf_model_t* m = items[i].model;
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);

    /* Upload vertices */
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

    /* Upload indices */
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
    /* Determine which texture index from the user_data */
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
    "assets/textures/stonefloor01_color_rgba.png",
    "assets/textures/stonefloor01_normal_rgba.png",
  };

  /* Texture index storage for callbacks */
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

/* Forward declaration */
static void init_bind_groups(struct wgpu_context_t* wgpu_context);

static void init_textures(struct wgpu_context_t* wgpu_context)
{
  /* Create placeholder textures (color bars until actual data loads) */
  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    state.textures.textures[i]
      = wgpu_create_color_bars_texture(wgpu_context, NULL);
  }

  /* Sampler for model textures (linear filtering) */
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
    /* Rebuild bind groups with the newly loaded textures */
    init_bind_groups(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Instance positions (WebGPU Y-axis: negated Y from Vulkan) */
  glm_vec4_copy((vec4){0.0f, 0.0f, 0.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[0]);
  glm_vec4_copy((vec4){-4.0f, 0.0f, -4.0f, 0.0f},
                state.offscreen_ubo_data.instance_pos[1]);
  glm_vec4_copy((vec4){4.0f, 0.0f, -4.0f, 0.0f},
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

  /* Initialize lights (WebGPU Y-axis: negate Y from Vulkan) */
  /* Light 0: White */
  state.composition_ubo_data.lights[0] = (light_t){
    .position = {0.0f, -0.0f, 1.0f, 0.0f},
    .color    = {1.5f, 1.5f, 1.5f},
    .radius   = 3.75f,
  };
  /* Light 1: Red */
  state.composition_ubo_data.lights[1] = (light_t){
    .position = {-2.0f, -0.0f, 0.0f, 0.0f},
    .color    = {1.0f, 0.0f, 0.0f},
    .radius   = 15.0f,
  };
  /* Light 2: Blue */
  state.composition_ubo_data.lights[2] = (light_t){
    .position = {2.0f, 1.0f, 0.0f, 0.0f},
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
    .position = {0.0f, 0.5f, 0.0f, 0.0f},
    .color    = {0.0f, 1.0f, 0.2f},
    .radius   = 5.0f,
  };
  /* Light 5: Orange */
  state.composition_ubo_data.lights[5] = (light_t){
    .position = {0.0f, 1.0f, 0.0f, 0.0f},
    .color    = {1.0f, 0.7f, 0.3f},
    .radius   = 25.0f,
  };
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
  /* Animate lights */
  if (!state.paused) {
    float t  = state.timer * 360.0f;
    float tr = glm_rad(t);

    /* Light 0: Center rotating */
    state.composition_ubo_data.lights[0].position[0] = sinf(tr) * 5.0f;
    state.composition_ubo_data.lights[0].position[2] = cosf(tr) * 5.0f;

    /* Light 1: Orbits left instance */
    state.composition_ubo_data.lights[1].position[0]
      = -4.0f + sinf(glm_rad(t + 45.0f)) * 2.0f;

    /* Light 2: Orbits right instance */
    state.composition_ubo_data.lights[2].position[0] = 4.0f + sinf(tr) * 2.0f;

    /* Light 4: Figure-8 motion */
    state.composition_ubo_data.lights[4].position[0]
      = sinf(glm_rad(t + 90.0f)) * 5.0f;
    state.composition_ubo_data.lights[4].position[2]
      = -cosf(glm_rad(t + 45.0f)) * 5.0f;

    /* Light 5: Reverse rotation, larger radius */
    state.composition_ubo_data.lights[5].position[0]
      = sinf(glm_rad(-t + 135.0f)) * 10.0f;
    state.composition_ubo_data.lights[5].position[2]
      = -cosf(glm_rad(-t - 45.0f)) * 10.0f;
  }

  /* View position from camera */
  glm_vec4_copy(state.camera.view_pos, state.composition_ubo_data.view_pos);

  /* Debug display target from settings */
  state.composition_ubo_data.debug_display_target
    = state.settings.debug_display_target;

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

  /* Offscreen BGL: UBO + colorMap + normalMap */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      /* Binding 0: UBO (vertex) */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {
          .type             = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = false,
          .minBindingSize   = sizeof(state.offscreen_ubo_data),
        },
      },
      /* Binding 1: Color map texture (fragment) */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      /* Binding 2: Normal map texture (fragment) */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
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

    state.offscreen_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Offscreen BGL"),
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Composition BGL: position + normal + albedo textures + sampler + UBO */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      /* Binding 0: Position texture (fragment) */
      {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      /* Binding 1: Normal texture (fragment) */
      {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      /* Binding 2: Albedo texture (fragment) */
      {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      /* Binding 3: G-Buffer sampler (fragment) */
      {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      /* Binding 4: Composition UBO (fragment) */
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

    state.composition_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Composition BGL"),
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

  /* Release existing bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.armor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.floor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)

  /* Armor bind group: UBO + armor color + armor normal + sampler */
  {
    WGPUBindGroupEntry entries[4] = {
      {
        .binding = 0,
        .buffer  = state.offscreen_ubo,
        .offset  = 0,
        .size    = sizeof(state.offscreen_ubo_data),
      },
      {
        .binding     = 1,
        .textureView = state.textures.textures[0].view, /* Armor color */
      },
      {
        .binding     = 2,
        .textureView = state.textures.textures[1].view, /* Armor normal */
      },
      {
        .binding = 3,
        .sampler = state.textures.sampler,
      },
    };

    state.armor_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Armor Bind Group"),
                .layout     = state.offscreen_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Floor bind group: UBO + floor color + floor normal + sampler */
  {
    WGPUBindGroupEntry entries[4] = {
      {
        .binding = 0,
        .buffer  = state.offscreen_ubo,
        .offset  = 0,
        .size    = sizeof(state.offscreen_ubo_data),
      },
      {
        .binding     = 1,
        .textureView = state.textures.textures[2].view, /* Floor color */
      },
      {
        .binding     = 2,
        .textureView = state.textures.textures[3].view, /* Floor normal */
      },
      {
        .binding = 3,
        .sampler = state.textures.sampler,
      },
    };

    state.floor_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Floor Bind Group"),
                .layout     = state.offscreen_bgl,
                .entryCount = 4,
                .entries    = entries,
              });
  }

  /* Composition bind group: G-Buffer textures + sampler + UBO */
  {
    WGPUBindGroupEntry entries[5] = {
      {
        .binding     = 0,
        .textureView = state.gbuffer.position_view,
      },
      {
        .binding     = 1,
        .textureView = state.gbuffer.normal_view,
      },
      {
        .binding     = 2,
        .textureView = state.gbuffer.albedo_view,
      },
      {
        .binding = 3,
        .sampler = state.gbuffer.sampler,
      },
      {
        .binding = 4,
        .buffer  = state.composition_ubo,
        .offset  = 0,
        .size    = sizeof(state.composition_ubo_data),
      },
    };

    state.composition_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Composition Bind Group"),
                .layout     = state.composition_bgl,
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

  /* Offscreen pipeline layout */
  state.offscreen_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Offscreen Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.offscreen_bgl,
            });

  /* Composition pipeline layout */
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
    /* Vertex buffer layout matching gltf_vertex_t */
    WGPUVertexAttribute vert_attrs[] = {
      /* position: vec3f @ offset 0 */
      {.shaderLocation = 0,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, position)},
      /* normal: vec3f */
      {.shaderLocation = 1,
       .format         = WGPUVertexFormat_Float32x3,
       .offset         = offsetof(gltf_vertex_t, normal)},
      /* uv: vec2f */
      {.shaderLocation = 2,
       .format         = WGPUVertexFormat_Float32x2,
       .offset         = offsetof(gltf_vertex_t, uv0)},
      /* tangent: vec4f */
      {.shaderLocation = 3,
       .format         = WGPUVertexFormat_Float32x4,
       .offset         = offsetof(gltf_vertex_t, tangent)},
      /* color: vec4f */
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

    /* Shader module */
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, mrt_shader_wgsl);

    /* 3 color targets for MRT (no blending) */
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

    WGPUDepthStencilState depth_stencil_state = {
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
                .depthStencil = &depth_stencil_state,
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

    WGPUDepthStencilState depth_stencil_state = {
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
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Deferred Shading", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    static const char* display_modes[] = {
      "Final composition", "Position", "Normals", "Albedo", "Specular",
    };
    imgui_overlay_combo_box("Display", &state.settings.debug_display_target,
                            display_modes, ARRAY_SIZE(display_modes));
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
  /* Vulkan: pos (2.15, 0.3, -8.75), rot (-0.75, 12.5, 0)
   * WebGPU: negate Y for position and pitch */
  camera_set_position(&state.camera, (vec3){2.15f, 0.3f, -8.75f});
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-0.75f, 12.5f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load models synchronously (small files) */
  load_models();
  create_model_buffers(wgpu_context);

  /* Create G-Buffer textures */
  init_gbuffer_textures(wgpu_context);

  /* Create render pass descriptors */
  init_render_passes();

  /* Create placeholder textures and start async fetch */
  init_textures(wgpu_context);
  fetch_textures();

  /* Create uniform buffers */
  init_uniform_buffers(wgpu_context);

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

  /* Update textures if any loaded */
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

  /* ===== Pass 1: Offscreen G-Buffer ===== */
  if (state.models_loaded) {
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_pass.descriptor);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)GBUFFER_DIM,
                                     (float)GBUFFER_DIM, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, GBUFFER_DIM, GBUFFER_DIM);

    /* Bind offscreen pipeline */
    wgpuRenderPassEncoderSetPipeline(pass, state.offscreen_pipeline);

    /* Draw floor (single instance) */
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.floor_bind_group, 0, 0);
    draw_model(pass, &state.floor_model, state.floor_vertex_buffer,
               state.floor_index_buffer, 1);

    /* Draw armor (3 instances) */
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.armor_bind_group, 0, 0);
    draw_model(pass, &state.armor_model, state.armor_vertex_buffer,
               state.armor_index_buffer, NUM_INSTANCES);

    wgpuRenderPassEncoderEnd(pass);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
  }

  /* ===== Pass 2: Composition (deferred lighting) ===== */
  {
    state.composition_pass.color_attachment.view = wgpu_context->swapchain_view;
    state.composition_pass.depth_stencil_attachment.view
      = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.composition_pass.descriptor);

    wgpuRenderPassEncoderSetPipeline(pass, state.composition_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.composition_bind_group, 0,
                                      0);
    /* Fullscreen triangle (3 vertices generated in vertex shader) */
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

  /* Destroy models */
  gltf_model_destroy(&state.armor_model);
  gltf_model_destroy(&state.floor_model);

  /* Destroy GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.armor_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.armor_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.floor_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.floor_index_buffer)

  /* Destroy textures */
  for (uint32_t i = 0; i < NUM_TEXTURES; i++) {
    wgpu_destroy_texture(&state.textures.textures[i]);
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler)

  /* Destroy G-Buffer */
  destroy_gbuffer_textures();

  /* Destroy uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.offscreen_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.composition_ubo)

  /* Destroy bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.armor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.floor_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.composition_bind_group)

  /* Destroy bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.offscreen_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_bgl)

  /* Destroy pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.offscreen_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_pipeline_layout)

  /* Destroy render pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.offscreen_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_pipeline)

  /* ImGui */
  imgui_overlay_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Deferred Shading",
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
    projection : mat4x4f,
    model      : mat4x4f,
    view       : mat4x4f,
    instancePos : array<vec4f, 3>,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var colorMap : texture_2d<f32>;
  @group(0) @binding(2) var normalMap : texture_2d<f32>;
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

    // World-space position
    out.gPosition = vec4f(in.worldPos, 1.0);

    // Normal mapping via TBN matrix
    let N = normalize(in.worldNormal);
    let T = normalize(in.tangent);
    let B = cross(N, T);
    let TBN = mat3x3f(T, B, N);

    let sampledNormal = textureSample(normalMap, texSampler, in.uv).xyz * 2.0 - vec3f(1.0);
    let tnorm = TBN * normalize(sampledNormal);
    out.gNormal = vec4f(tnorm, 1.0);

    // Albedo + specular (alpha channel)
    out.gAlbedo = textureSample(colorMap, texSampler, in.uv);

    return out;
  }
);

static const char* composition_shader_wgsl = CODE(
  struct Light {
    position : vec4f,
    color    : vec3f,
    radius   : f32,
  };

  struct UBO {
    lights             : array<Light, 6>,
    viewPos            : vec4f,
    debugDisplayTarget : i32,
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
    // Generate fullscreen triangle
    let uv = vec2f(
      f32((vertexIndex << 1u) & 2u),
      f32(vertexIndex & 2u),
    );
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let fragPos = textureSample(gPosition, gSampler, in.uv).rgb;
    let normal  = textureSample(gNormal, gSampler, in.uv).rgb;
    let albedo  = textureSample(gAlbedo, gSampler, in.uv);

    // Debug display modes
    if (ubo.debugDisplayTarget > 0) {
      switch (ubo.debugDisplayTarget) {
        case 1: { return vec4f(fragPos, 1.0); }
        case 2: { return vec4f(normal, 1.0); }
        case 3: { return vec4f(albedo.rgb, 1.0); }
        case 4: { return vec4f(albedo.aaa, 1.0); }
        default: { return vec4f(fragPos, 1.0); }
      }
    }

    // Deferred shading: accumulate lighting from 6 point lights
    let ambient = 0.0;
    var fragcolor = albedo.rgb * ambient;

    for (var i = 0; i < 6; i++) {
      let L = ubo.lights[i].position.xyz - fragPos;
      let dist = length(L);

      let V = normalize(ubo.viewPos.xyz - fragPos);
      let Ln = normalize(L);
      let N = normalize(normal);

      // Attenuation
      let atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

      // Diffuse (Lambertian)
      let NdotL = max(0.0, dot(N, Ln));
      let diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

      // Specular (Phong)
      let R = reflect(-Ln, N);
      let NdotR = max(0.0, dot(R, V));
      let spec = ubo.lights[i].color * albedo.a * pow(NdotR, 16.0) * atten;

      fragcolor += diff + spec;
    }

    return vec4f(fragcolor, 1.0);
  }
);
// clang-format on
