/* -------------------------------------------------------------------------- *
 * WebGPU Example - Radial Blur (Fullscreen Post-Processing)
 *
 * Implements a single-pass fullscreen radial blur post-processing effect.
 * A glTF "glow sphere" model is first rendered to an offscreen texture,
 * extracting bright emitters via vertex color detection. The offscreen color
 * texture is then sampled in a fullscreen radial blur fragment shader (32
 * samples along radial lines from a configurable origin) and composited
 * onto the main scene with additive blending.
 *
 * Rendering passes:
 *   1. Offscreen color pass: Render glow sphere bright colors to FB[0]
 *   2. Main scene: Phong-lit sphere + radial blur composite (additive)
 *
 * Features:
 * - 32-sample radial blur with configurable scale, strength, and origin
 * - Offscreen rendering for glow extraction
 * - Additive blending for blur composition
 * - Phong lighting with gradient-ramp glow for bright vertex colors
 * - Animated gradient position cycling through the gradient texture
 * - GUI controls for blur toggle, debug view, and blur parameters
 * - LookAt camera with mouse orbit and auto-rotation
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/radialblur
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations - defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* radial_blur_colorpass_shader_wgsl;
static const char* radial_blur_phongpass_shader_wgsl;
static const char* radial_blur_radialblur_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define OFFSCREEN_SIZE (512)
#define GRADIENT_TEX_WIDTH (256)
#define GRADIENT_TEX_HEIGHT (1)
#define GRADIENT_TEX_NUM_BYTES (GRADIENT_TEX_WIDTH * GRADIENT_TEX_HEIGHT * 4)

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Timer for animation */
  float timer;
  float timer_speed;

  /* Model */
  gltf_model_t scene_model;
  bool model_loaded;

  /* GPU vertex/index buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;

  /* Gradient texture (1D ramp, 256x1 RGBA) */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    bool loaded;
  } gradient_texture;
  uint8_t gradient_pixels[GRADIENT_TEX_NUM_BYTES];

  /* Offscreen framebuffer */
  struct {
    WGPUTexture color_texture;
    WGPUTextureView color_view;
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
  } offscreen_fb;
  WGPUSampler offscreen_sampler;

  /* Uniform buffers */
  WGPUBuffer scene_ubo;       /* projection + modelView + gradientPos */
  WGPUBuffer blur_params_ubo; /* blur scale, strength, origin */

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 model_view;
    float gradient_pos;
    float _pad[3]; /* Align to 16 bytes */
  } scene_ubo_data;

  struct {
    float radial_blur_scale;
    float radial_blur_strength;
    float radial_origin_x;
    float radial_origin_y;
  } blur_params_data;

  /* Bind group layouts */
  WGPUBindGroupLayout scene_bgl;       /* scene (UBO + gradient texture) */
  WGPUBindGroupLayout radial_blur_bgl; /* blur (UBO + offscreen texture) */

  /* Pipeline layouts */
  WGPUPipelineLayout scene_pipeline_layout;
  WGPUPipelineLayout radial_blur_pipeline_layout;

  /* Render pipelines (4 total) */
  WGPURenderPipeline color_pass_pipeline;  /* Offscreen color extraction */
  WGPURenderPipeline phong_pass_pipeline;  /* Phong-lit scene rendering */
  WGPURenderPipeline radial_blur_pipeline; /* Radial blur (additive) */
  WGPURenderPipeline offscreen_display_pipeline; /* Debug: display offscreen */

  /* Bind groups */
  WGPUBindGroup scene_bind_group;
  WGPUBindGroup radial_blur_bind_group;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment offscreen_color_att;
  WGPURenderPassDepthStencilAttachment offscreen_depth_att;
  WGPURenderPassDescriptor offscreen_render_pass_desc;

  WGPURenderPassColorAttachment main_color_att;
  WGPURenderPassDepthStencilAttachment main_depth_att;
  WGPURenderPassDescriptor main_render_pass_desc;

  /* Settings (GUI) */
  struct {
    bool blur;
    bool display_texture;
    float radial_blur_scale;
    float radial_blur_strength;
    float radial_origin_x;
    float radial_origin_y;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .timer_speed = 0.5f,
  .settings = {
    .blur                = true,
    .display_texture     = false,
    .radial_blur_scale   = 0.35f,
    .radial_blur_strength = 0.75f,
    .radial_origin_x     = 0.5f,
    .radial_origin_y     = 0.5f,
  },
  .main_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
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
  .main_render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.main_color_att,
    .depthStencilAttachment = &state.main_depth_att,
  },
  .offscreen_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 0.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .offscreen_depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .offscreen_render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.offscreen_color_att,
    .depthStencilAttachment = &state.offscreen_depth_att,
  },
};

/* -------------------------------------------------------------------------- *
 * Offscreen framebuffer setup
 * -------------------------------------------------------------------------- */

static void init_offscreen_framebuffer(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Color texture (RGBA8, used as render target + sampled in blur shader) */
  WGPUTextureDescriptor color_desc = {
    .label = STRVIEW("Offscreen Color"),
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {OFFSCREEN_SIZE, OFFSCREEN_SIZE, 1},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.offscreen_fb.color_texture
    = wgpuDeviceCreateTexture(device, &color_desc);

  state.offscreen_fb.color_view
    = wgpuTextureCreateView(state.offscreen_fb.color_texture,
                            &(WGPUTextureViewDescriptor){
                              .format          = WGPUTextureFormat_RGBA8Unorm,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });

  /* Depth texture */
  WGPUTextureDescriptor depth_desc = {
    .label         = STRVIEW("Offscreen Depth"),
    .usage         = WGPUTextureUsage_RenderAttachment,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {OFFSCREEN_SIZE, OFFSCREEN_SIZE, 1},
    .format        = WGPUTextureFormat_Depth24PlusStencil8,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.offscreen_fb.depth_texture
    = wgpuDeviceCreateTexture(device, &depth_desc);

  state.offscreen_fb.depth_view
    = wgpuTextureCreateView(state.offscreen_fb.depth_texture,
                            &(WGPUTextureViewDescriptor){
                              .format = WGPUTextureFormat_Depth24PlusStencil8,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });

  /* Sampler for offscreen color texture */
  WGPUSamplerDescriptor sampler_desc = {
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
  };
  state.offscreen_sampler = wgpuDeviceCreateSampler(device, &sampler_desc);
}

static void destroy_offscreen_framebuffer(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen_fb.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen_fb.color_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.offscreen_fb.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.offscreen_fb.depth_texture)
  WGPU_RELEASE_RESOURCE(Sampler, state.offscreen_sampler)
}

/* -------------------------------------------------------------------------- *
 * Gradient texture loading (async via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void init_gradient_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Create 2D texture (256x1 RGBA8) */
  WGPUTextureDescriptor tex_desc = {
    .label         = STRVIEW("Gradient Texture"),
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {GRADIENT_TEX_WIDTH, GRADIENT_TEX_HEIGHT, 1},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.gradient_texture.handle = wgpuDeviceCreateTexture(device, &tex_desc);

  state.gradient_texture.view = wgpuTextureCreateView(
    state.gradient_texture.handle, &(WGPUTextureViewDescriptor){
                                     .format    = WGPUTextureFormat_RGBA8Unorm,
                                     .dimension = WGPUTextureViewDimension_2D,
                                     .baseMipLevel    = 0,
                                     .mipLevelCount   = 1,
                                     .baseArrayLayer  = 0,
                                     .arrayLayerCount = 1,
                                     .aspect          = WGPUTextureAspect_All,
                                   });

  /* Sampler for gradient ramp — use Repeat to cycle the gradient continuously
   * (matching the Vulkan KTX loader default sampler address mode) */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Gradient Sampler"),
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
  state.gradient_texture.sampler
    = wgpuDeviceCreateSampler(device, &sampler_desc);
}

static void gradient_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Gradient texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_w, img_h, num_ch;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_w, &img_h, &num_ch, 4);
  if (pixels) {
    memcpy(state.gradient_pixels, pixels, (size_t)(img_w * img_h * 4));
    image_free(pixels);
    state.gradient_texture.loaded = true;
  }
}

static void fetch_gradient_texture(void)
{
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/particle_gradient_rgba.png",
    .callback = gradient_fetch_callback,
    .buffer   = SFETCH_RANGE(state.gradient_pixels),
    .channel  = 0,
  });
}

static void upload_gradient_pixels(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  WGPUTexelCopyTextureInfo dst = {
    .texture  = state.gradient_texture.handle,
    .mipLevel = 0,
    .origin   = {0, 0, 0},
    .aspect   = WGPUTextureAspect_All,
  };
  WGPUTexelCopyBufferLayout layout = {
    .offset       = 0,
    .bytesPerRow  = GRADIENT_TEX_WIDTH * 4,
    .rowsPerImage = GRADIENT_TEX_HEIGHT,
  };
  WGPUExtent3D size = {GRADIENT_TEX_WIDTH, GRADIENT_TEX_HEIGHT, 1};

  wgpuQueueWriteTexture(queue, &dst, state.gradient_pixels,
                        GRADIENT_TEX_NUM_BYTES, &layout, &size);

  state.gradient_texture.loaded = false; /* Mark as uploaded */
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static const gltf_model_desc_t model_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

static void load_model(void)
{
  bool ok = gltf_model_load_from_file(&state.scene_model,
                                      "assets/models/glowsphere.gltf", 1.0f);
  if (!ok) {
    printf("Failed to load glowsphere.gltf\n");
    return;
  }
  state.model_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  if (!state.model_loaded) {
    return;
  }

  gltf_model_t* m = &state.scene_model;
  size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);

  /* Bake transforms into vertex data */
  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  memcpy(xformed, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, xformed, &model_load_desc);

  /* Upload vertices */
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene Vertex Buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, xformed, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);
  free(xformed);

  /* Upload indices */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Scene Index Buffer"),
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
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene UBO: projection (64) + modelView (64) + gradientPos (4) + pad (12)
   * = 144 bytes, round up to 256 for alignment safety */
  state.scene_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.scene_ubo_data),
            });

  /* Blur params UBO: 4 floats = 16 bytes */
  state.blur_params_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Blur Params UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.blur_params_data),
            });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context,
                                   float delta_time)
{
  WGPUQueue queue = wgpu_context->queue;
  float aspect    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Scene UBO */
  glm_perspective(glm_rad(45.0f), aspect, 1.0f, 256.0f,
                  state.scene_ubo_data.projection);

  glm_mat4_copy(state.camera.matrices.view, state.scene_ubo_data.model_view);

  /* Animate gradient position */
  state.scene_ubo_data.gradient_pos += delta_time * 0.1f;

  wgpuQueueWriteBuffer(queue, state.scene_ubo, 0, &state.scene_ubo_data,
                       sizeof(state.scene_ubo_data));

  /* Blur params UBO */
  state.blur_params_data.radial_blur_scale = state.settings.radial_blur_scale;
  state.blur_params_data.radial_blur_strength
    = state.settings.radial_blur_strength;
  state.blur_params_data.radial_origin_x = state.settings.radial_origin_x;
  state.blur_params_data.radial_origin_y = state.settings.radial_origin_y;

  wgpuQueueWriteBuffer(queue, state.blur_params_ubo, 0, &state.blur_params_data,
                       sizeof(state.blur_params_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene bind group layout:
   * binding 0: UBO (vertex) - projection + modelView + gradientPos
   * binding 1: sampler (fragment) - gradient texture sampler
   * binding 2: texture_2d (fragment) - gradient texture */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.scene_ubo_data),
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    state.scene_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Scene BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Radial blur bind group layout:
   * binding 0: UBO (fragment) - blur params
   * binding 1: sampler (fragment) - offscreen texture sampler
   * binding 2: texture_2d (fragment) - offscreen color texture */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = sizeof(state.blur_params_data),
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    state.radial_blur_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Radial Blur BGL"),
                .entryCount = ARRAY_SIZE(entries),
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

  state.scene_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Scene Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.scene_bgl,
            });

  state.radial_blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Radial Blur Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.radial_blur_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.scene_ubo,
        .offset  = 0,
        .size    = sizeof(state.scene_ubo_data),
      },
      [1] = {
        .binding = 1,
        .sampler = state.gradient_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.gradient_texture.view,
      },
    };
    state.scene_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Scene Bind Group"),
                .layout     = state.scene_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Radial blur bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.blur_params_ubo,
        .offset  = 0,
        .size    = sizeof(state.blur_params_data),
      },
      [1] = {
        .binding = 1,
        .sampler = state.offscreen_sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.offscreen_fb.color_view,
      },
    };
    state.radial_blur_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Radial Blur Bind Group"),
                .layout     = state.radial_blur_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Vertex buffer layout for glTF model */
  WGPUVertexAttribute vertex_attrs[] = {
    {
      .shaderLocation = 0,
      .offset         = offsetof(gltf_vertex_t, position),
      .format         = WGPUVertexFormat_Float32x3,
    },
    {
      .shaderLocation = 1,
      .offset         = offsetof(gltf_vertex_t, normal),
      .format         = WGPUVertexFormat_Float32x3,
    },
    {
      .shaderLocation = 2,
      .offset         = offsetof(gltf_vertex_t, uv0),
      .format         = WGPUVertexFormat_Float32x2,
    },
    {
      .shaderLocation = 3,
      .offset         = offsetof(gltf_vertex_t, color),
      .format         = WGPUVertexFormat_Float32x4,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  /* Common depth stencil state */
  WGPUDepthStencilState depth_stencil_on = {
    .format               = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled    = WGPUOptionalBool_True,
    .depthCompare         = WGPUCompareFunction_LessEqual,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  WGPUDepthStencilState depth_stencil_main = depth_stencil_on;
  depth_stencil_main.format                = wgpu_context->depth_stencil_format;

  /* ------ Color pass pipeline (offscreen, bright color extraction) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, radial_blur_colorpass_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.color_pass_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Color Pass Pipeline"),
        .layout = state.scene_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vertex_buffer_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_on,
        .multisample = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &color_target,
        },
      });

    wgpuShaderModuleRelease(sm);
  }

  /* ------ Phong pass pipeline (main scene, opaque) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, radial_blur_phongpass_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.phong_pass_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Phong Pass Pipeline"),
        .layout = state.scene_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vertex_buffer_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_main,
        .multisample = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &color_target,
        },
      });

    wgpuShaderModuleRelease(sm);
  }

  /* ------ Radial blur pipeline (main scene, additive blend) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, radial_blur_radialblur_shader_wgsl);

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

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .blend     = &additive_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.radial_blur_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Radial Blur Pipeline"),
        .layout = state.radial_blur_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_main,
        .multisample = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &color_target,
        },
      });

    wgpuShaderModuleRelease(sm);
  }

  /* ------ Offscreen display pipeline (radial blur without additive blend) --
   * In Vulkan, this uses the same radial blur shaders as the blur pipeline,
   * but with blending disabled, so you see the blur result only (no scene
   * underneath). This is the "Display render target" debug mode. ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, radial_blur_radialblur_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.offscreen_display_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Offscreen Display Pipeline"),
        .layout = state.radial_blur_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0,
          .buffers     = NULL,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_None,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &depth_stencil_main,
        .multisample = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = sm,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &color_target,
        },
      });

    wgpuShaderModuleRelease(sm);
  }
}

/* -------------------------------------------------------------------------- *
 * Draw a glTF model's primitives
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  if (!state.model_loaded) {
    return;
  }

  gltf_model_t* model = &state.scene_model;

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (node->mesh == NULL) {
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
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Radial Blur Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Radial blur", &state.settings.blur);
    igCheckbox("Display render target only", &state.settings.display_texture);
  }

  if (state.settings.blur) {
    if (igCollapsingHeaderBoolPtr("Blur parameters", NULL,
                                  ImGuiTreeNodeFlags_DefaultOpen)) {
      imgui_overlay_slider_float("Scale", &state.settings.radial_blur_scale,
                                 0.1f, 1.0f, "%.2f");
      imgui_overlay_slider_float(
        "Strength", &state.settings.radial_blur_strength, 0.1f, 2.0f, "%.2f");
      imgui_overlay_slider_float(
        "Horiz. origin", &state.settings.radial_origin_x, 0.0f, 1.0f, "%.2f");
      imgui_overlay_slider_float(
        "Vert. origin", &state.settings.radial_origin_y, 0.0f, 1.0f, "%.2f");
    }
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
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  /* sokol_fetch: gradient texture + model */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 1,
    .num_lanes    = 4,
    .logger.func  = slog_func,
  });

  /* Camera setup (Vulkan: pos=(0,0,-17.5), rot=(-16.25,-28.75,0))
   * WebGPU Y-up: negate Y for position and rotation pitch */
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -17.5f});
  camera_set_rotation(&state.camera, (vec3){16.25f, -28.75f, 0.0f});
  camera_set_perspective(
    &state.camera, 45.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 1.0f, 256.0f);

  /* Load model synchronously (it's small) */
  load_model();
  create_model_buffers(wgpu_context);

  /* Create offscreen framebuffer */
  init_offscreen_framebuffer(wgpu_context);

  /* Create gradient texture (initially empty, populated by fetch) */
  init_gradient_texture(wgpu_context);

  /* Start gradient texture loading */
  fetch_gradient_texture();

  /* Create uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Create bind group layouts + pipeline layouts */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Create bind groups */
  init_bind_groups(wgpu_context);

  /* Create render pipelines */
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

  /* Pump async file loading */
  sfetch_dowork();

  /* Upload gradient pixels once loaded */
  if (state.gradient_texture.loaded) {
    upload_gradient_pixels(wgpu_context);
  }

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Auto-rotate camera (10 degrees/sec around Y, matching Vulkan) */
  vec3 cur_rot;
  glm_vec3_copy(state.camera.rotation, cur_rot);
  cur_rot[1] += delta_time * 10.0f;
  camera_set_rotation(&state.camera, cur_rot);

  /* Update animation timer */
  state.timer += delta_time * state.timer_speed;

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffers(wgpu_context, delta_time);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Render ---- */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ============ Pass 1: Offscreen color extraction ============ */
  if (state.model_loaded) {
    state.offscreen_color_att.view = state.offscreen_fb.color_view;
    state.offscreen_depth_att.view = state.offscreen_fb.depth_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_render_pass_desc);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, OFFSCREEN_SIZE, OFFSCREEN_SIZE,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, OFFSCREEN_SIZE,
                                        OFFSCREEN_SIZE);

    wgpuRenderPassEncoderSetPipeline(pass, state.color_pass_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0, 0);
    draw_model(pass);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 2: Main scene + radial blur composite ============ */
  {
    state.main_color_att.view = wgpu_context->swapchain_view;
    state.main_depth_att.view = wgpu_context->depth_stencil_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.main_render_pass_desc);

    uint32_t w = (uint32_t)wgpu_context->width;
    uint32_t h = (uint32_t)wgpu_context->height;
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

    /* (a) 3D Phong-lit scene */
    if (state.model_loaded) {
      wgpuRenderPassEncoderSetPipeline(pass, state.phong_pass_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0, 0);
      draw_model(pass);
    }

    /* (b) Fullscreen radial blur composite (or debug display) */
    if (state.settings.blur) {
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.radial_blur_bind_group,
                                        0, 0);
      if (state.settings.display_texture) {
        wgpuRenderPassEncoderSetPipeline(pass,
                                         state.offscreen_display_pipeline);
      }
      else {
        wgpuRenderPassEncoderSetPipeline(pass, state.radial_blur_pipeline);
      }
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* Submit */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ImGui overlay render */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Destroy model */
  gltf_model_destroy(&state.scene_model);

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_params_ubo)

  /* Offscreen resources */
  destroy_offscreen_framebuffer();

  /* Gradient texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.gradient_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.gradient_texture.handle)
  WGPU_RELEASE_RESOURCE(Sampler, state.gradient_texture.sampler)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.radial_blur_bind_group)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.scene_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.radial_blur_bgl)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.scene_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.radial_blur_pipeline_layout)

  /* Render pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.color_pass_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.phong_pass_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.radial_blur_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.offscreen_display_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Full screen radial blur effect",
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

/* ---- Color pass: extracts bright glow colors to offscreen FB ---- */
// clang-format off
static const char* radial_blur_colorpass_shader_wgsl = CODE(
  struct SceneUBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
    gradientPos : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : SceneUBO;
  @group(0) @binding(1) var gradientSampler : sampler;
  @group(0) @binding(2) var gradientTexture : texture_2d<f32>;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec3f,
    @location(1) uv    : vec2f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color = in.color.rgb;
    out.uv = vec2f(ubo.gradientPos, 0.0);
    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    /* Sample gradient unconditionally (WGSL uniform control flow requirement) */
    let gradient = textureSample(gradientTexture, gradientSampler, in.uv);
    /* Detect bright emitters by max color channel >= 0.9 */
    if (in.color.r >= 0.9 || in.color.g >= 0.9 || in.color.b >= 0.9) {
      return vec4f(gradient.rgb, 1.0);
    } else {
      return vec4f(in.color, 1.0);
    }
  }
);
// clang-format on

/* ---- Phong pass: scene rendering with Phong lighting and glow ---- */
// clang-format off
static const char* radial_blur_phongpass_shader_wgsl = CODE(
  struct SceneUBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
    gradientPos : f32,
  };

  @group(0) @binding(0) var<uniform> ubo : SceneUBO;
  @group(0) @binding(1) var gradientSampler : sampler;
  @group(0) @binding(2) var gradientTexture : texture_2d<f32>;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal   : vec3f,
    @location(1) color    : vec3f,
    @location(2) eyePos   : vec3f,
    @location(3) lightVec : vec3f,
    @location(4) uv       : vec2f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.normal = in.normal;
    out.color = in.color.rgb;
    out.uv = vec2f(ubo.gradientPos, 0.0);
    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);
    out.eyePos = (ubo.modelView * vec4f(in.position, 1.0)).xyz;
    let lightPos = vec4f(0.0, 0.0, -5.0, 1.0);
    out.lightVec = normalize(lightPos.xyz - in.position);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    /* Sample gradient unconditionally (WGSL uniform control flow requirement) */
    let gradient = textureSample(gradientTexture, gradientSampler, in.uv);
    /* Detect bright glow emitters */
    if (in.color.r >= 0.9 || in.color.g >= 0.9 || in.color.b >= 0.9) {
      return vec4f(gradient.rgb, 1.0);
    }

    let Eye = normalize(-in.eyePos);
    let Reflected = normalize(reflect(-in.lightVec, in.normal));

    let IAmbient = vec4f(0.2, 0.2, 0.2, 1.0);
    let IDiffuse = vec4f(0.5, 0.5, 0.5, 0.5) * max(dot(in.normal, in.lightVec), 0.0);
    let specular = 0.25;
    let ISpecular = vec4f(0.5, 0.5, 0.5, 1.0) * pow(max(dot(Reflected, Eye), 0.0), 4.0) * specular;
    return vec4f((IAmbient + IDiffuse).rgb * in.color + ISpecular.rgb, 1.0);
  }
);
// clang-format on

/* ---- Radial blur: fullscreen triangle with 32-sample radial blur ---- */
// clang-format off
static const char* radial_blur_radialblur_shader_wgsl = CODE(
  struct BlurParams {
    radialBlurScale    : f32,
    radialBlurStrength : f32,
    radialOriginX      : f32,
    radialOriginY      : f32,
  };

  @group(0) @binding(0) var<uniform> params : BlurParams;
  @group(0) @binding(1) var texSampler : sampler;
  @group(0) @binding(2) var texColor : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(
      f32((vertexIndex << 1u) & 2u),
      f32(vertexIndex & 2u)
    );
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let texDim = vec2f(textureDimensions(texColor, 0));
    let radialSize = vec2f(1.0 / texDim.x, 1.0 / texDim.y);
    let radialOrigin = vec2f(params.radialOriginX, params.radialOriginY);

    var UV = in.uv;
    UV = UV + radialSize * 0.5 - radialOrigin;

    var color = vec4f(0.0, 0.0, 0.0, 0.0);

    for (var i = 0; i < 32; i++) {
      let scale = 1.0 - params.radialBlurScale * (f32(i) / 31.0);
      color += textureSample(texColor, texSampler, UV * scale + radialOrigin);
    }

    return (color / 32.0) * params.radialBlurStrength;
  }
);
// clang-format on
