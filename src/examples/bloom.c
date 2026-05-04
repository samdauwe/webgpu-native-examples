/* -------------------------------------------------------------------------- *
 * WebGPU Example - Bloom (Fullscreen Blur)
 *
 * Implements a separable two-pass fullscreen Gaussian blur (bloom) effect.
 * Bright parts of a glTF model ("glow" mesh) are rendered to an offscreen
 * texture, then blurred in two passes (vertical + horizontal) using a 9-tap
 * Gaussian kernel. The blurred result is composited onto the main scene with
 * additive blending, creating a bloom halo around bright areas.
 *
 * Rendering passes:
 *   1. Offscreen glow render: Render glow mesh vertex colors to FB[0]
 *   2. Vertical blur: Fullscreen triangle reads FB[0], writes to FB[1]
 *   3. Main scene: Skybox + Phong-lit UFO + horizontal blur composite
 * (additive)
 *
 * Features:
 * - Separable 9-tap Gaussian blur (5 weights, 2 passes)
 * - Additive blending for bloom composition
 * - Cubemap skybox rendering (fullscreen triangle technique)
 * - Phong lighting with ambient glow boost for bright vertex colors
 * - GUI controls for bloom toggle and blur scale
 * - LookAt camera with mouse orbit
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/bloom
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

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations - defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* bloom_colorpass_shader_wgsl;
static const char* bloom_phongpass_shader_wgsl;
static const char* bloom_gaussblur_vert_shader_wgsl;
static const char* bloom_gaussblur_horz_shader_wgsl;
static const char* bloom_skybox_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define OFFSCREEN_WIDTH (256)
#define OFFSCREEN_HEIGHT (256)
#define NUM_CUBEMAP_FACES (6)
#define CUBEMAP_FACE_SIZE (512)
#define CUBEMAP_FACE_NUM_BYTES (CUBEMAP_FACE_SIZE * CUBEMAP_FACE_SIZE * 4)

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Timer for UFO animation */
  float timer;
  float animation_speed;

  /* Models */
  gltf_model_t ufo_model;
  gltf_model_t ufo_glow_model;
  bool models_loaded;

  /* GPU vertex/index buffers for UFO body */
  WGPUBuffer ufo_vertex_buffer;
  WGPUBuffer ufo_index_buffer;

  /* GPU vertex/index buffers for UFO glow */
  WGPUBuffer glow_vertex_buffer;
  WGPUBuffer glow_index_buffer;

  /* Cubemap texture */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    bool is_dirty;
  } cubemap_texture;
  uint8_t* cubemap_pixels[NUM_CUBEMAP_FACES];
  int cubemap_load_count;

  /* Offscreen framebuffers (2 for ping-pong blur) */
  struct {
    WGPUTexture color_texture;
    WGPUTextureView color_view;
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
  } offscreen_fb[2];
  WGPUSampler offscreen_sampler;

  /* Uniform buffers */
  WGPUBuffer scene_ubo;       /* MVP for scene objects */
  WGPUBuffer skybox_ubo;      /* MVP for skybox (no translation) */
  WGPUBuffer blur_params_ubo; /* blur scale + strength */

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
  } scene_ubo_data;
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
  } skybox_ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout scene_bgl;
  WGPUBindGroupLayout blur_bgl;
  WGPUBindGroupLayout skybox_bgl;

  /* Pipeline layouts */
  WGPUPipelineLayout scene_pipeline_layout;
  WGPUPipelineLayout blur_pipeline_layout;
  WGPUPipelineLayout skybox_pipeline_layout;

  /* Render pipelines (5 total) */
  WGPURenderPipeline glow_pipeline;      /* Offscreen glow render */
  WGPURenderPipeline blur_vert_pipeline; /* Vertical blur */
  WGPURenderPipeline blur_horz_pipeline; /* Horizontal blur (additive) */
  WGPURenderPipeline phong_pipeline;     /* Phong-lit scene */
  WGPURenderPipeline skybox_pipeline;    /* Cubemap skybox */

  /* Bind groups */
  WGPUBindGroup scene_bind_group;
  WGPUBindGroup blur_vert_bind_group; /* reads FB[0] */
  WGPUBindGroup blur_horz_bind_group; /* reads FB[1] */
  WGPUBindGroup skybox_bind_group;

  /* Render pass descriptors for offscreen */
  WGPURenderPassColorAttachment offscreen_color_att;
  WGPURenderPassDepthStencilAttachment offscreen_depth_att;
  WGPURenderPassDescriptor offscreen_render_pass_desc;

  /* Main render pass */
  WGPURenderPassColorAttachment main_color_att;
  WGPURenderPassDepthStencilAttachment main_depth_att;
  WGPURenderPassDescriptor main_render_pass_desc;

  /* Settings */
  struct {
    bool bloom;
    float blur_scale;
    float blur_strength;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .animation_speed = 0.25f,
  .settings = {
    .bloom         = true,
    .blur_scale    = 1.0f,
    .blur_strength = 1.5f,
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
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
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

static void init_offscreen_framebuffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  for (int i = 0; i < 2; i++) {
    /* Color texture */
    WGPUTextureDescriptor color_desc = {
      .label = STRVIEW("Offscreen Color - Texture"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 1},
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.offscreen_fb[i].color_texture
      = wgpuDeviceCreateTexture(device, &color_desc);

    state.offscreen_fb[i].color_view = wgpuTextureCreateView(
      state.offscreen_fb[i].color_texture,
      &(WGPUTextureViewDescriptor){
        .label           = STRVIEW("Offscreen Color - Texture View"),
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
      .label         = STRVIEW("Offscreen Depth - Texture"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 1},
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.offscreen_fb[i].depth_texture
      = wgpuDeviceCreateTexture(device, &depth_desc);

    state.offscreen_fb[i].depth_view
      = wgpuTextureCreateView(state.offscreen_fb[i].depth_texture,
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

  /* Shared sampler for offscreen textures */
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

static void destroy_offscreen_framebuffers(void)
{
  for (int i = 0; i < 2; i++) {
    WGPU_RELEASE_RESOURCE(TextureView, state.offscreen_fb[i].color_view)
    WGPU_RELEASE_RESOURCE(Texture, state.offscreen_fb[i].color_texture)
    WGPU_RELEASE_RESOURCE(TextureView, state.offscreen_fb[i].depth_view)
    WGPU_RELEASE_RESOURCE(Texture, state.offscreen_fb[i].depth_texture)
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.offscreen_sampler)
}

/* -------------------------------------------------------------------------- *
 * Cubemap loading
 * -------------------------------------------------------------------------- */

static void init_cubemap_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Create cubemap texture (6 array layers) */
  WGPUTextureDescriptor tex_desc = {
    .label = STRVIEW("Cubemap Texture"),
    .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
             | WGPUTextureUsage_RenderAttachment,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {CUBEMAP_FACE_SIZE, CUBEMAP_FACE_SIZE, NUM_CUBEMAP_FACES},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.cubemap_texture.handle = wgpuDeviceCreateTexture(device, &tex_desc);

  /* Create cube view */
  WGPUTextureViewDescriptor view_desc = {
    .label           = STRVIEW("Cubemap View"),
    .format          = WGPUTextureFormat_RGBA8Unorm,
    .dimension       = WGPUTextureViewDimension_Cube,
    .baseMipLevel    = 0,
    .mipLevelCount   = 1,
    .baseArrayLayer  = 0,
    .arrayLayerCount = NUM_CUBEMAP_FACES,
    .aspect          = WGPUTextureAspect_All,
  };
  state.cubemap_texture.view
    = wgpuTextureCreateView(state.cubemap_texture.handle, &view_desc);

  /* Sampler */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Cubemap Sampler"),
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
  state.cubemap_texture.sampler
    = wgpuDeviceCreateSampler(device, &sampler_desc);
}

static void cubemap_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Cubemap face fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_w, img_h, num_ch;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_w, &img_h, &num_ch, 4);
  if (pixels) {
    ASSERT(img_w == CUBEMAP_FACE_SIZE && img_h == CUBEMAP_FACE_SIZE);
    memcpy((void*)response->buffer.ptr, pixels, (size_t)(img_w * img_h * 4));
    image_free(pixels);
    state.cubemap_load_count++;
  }
}

static void fetch_cubemap_faces(void)
{
  /* Face order: +X, -X, +Y, -Y, +Z, -Z */
  static const char* face_paths[NUM_CUBEMAP_FACES] = {
    "assets/textures/cubemaps/cubemap_space_px.png",
    "assets/textures/cubemaps/cubemap_space_nx.png",
    "assets/textures/cubemaps/cubemap_space_py.png",
    "assets/textures/cubemaps/cubemap_space_ny.png",
    "assets/textures/cubemaps/cubemap_space_pz.png",
    "assets/textures/cubemaps/cubemap_space_nz.png",
  };

  state.cubemap_texture.is_dirty = true;
  state.cubemap_load_count       = 0;

  for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
    state.cubemap_pixels[i] = (uint8_t*)malloc(CUBEMAP_FACE_NUM_BYTES);
    sfetch_send(&(sfetch_request_t){
      .path     = face_paths[i],
      .callback = cubemap_fetch_callback,
      .buffer
      = {.ptr = state.cubemap_pixels[i], .size = CUBEMAP_FACE_NUM_BYTES},
      .channel = 0,
    });
  }
}

static void upload_cubemap_pixels(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
    uint32_t bytes_per_row = CUBEMAP_FACE_SIZE * 4;
    uint32_t data_size     = CUBEMAP_FACE_NUM_BYTES;

    WGPUBufferDescriptor staging_desc = {
      .usage            = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
      .size             = data_size,
      .mappedAtCreation = true,
    };
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &staging_desc);

    void* mapped = wgpuBufferGetMappedRange(staging, 0, data_size);
    memcpy(mapped, state.cubemap_pixels[face], data_size);
    wgpuBufferUnmap(staging);

    WGPUTexelCopyBufferLayout src_layout = {
      .offset       = 0,
      .bytesPerRow  = bytes_per_row,
      .rowsPerImage = CUBEMAP_FACE_SIZE,
    };
    WGPUTexelCopyTextureInfo dst_info = {
      .texture  = state.cubemap_texture.handle,
      .mipLevel = 0,
      .origin   = {0, 0, (uint32_t)face},
      .aspect   = WGPUTextureAspect_All,
    };
    WGPUExtent3D copy_size = {CUBEMAP_FACE_SIZE, CUBEMAP_FACE_SIZE, 1};

    wgpuCommandEncoderCopyBufferToTexture(enc,
                                          &(WGPUTexelCopyBufferInfo){
                                            .buffer = staging,
                                            .layout = src_layout,
                                          },
                                          &dst_info, &copy_size);

    wgpuBufferRelease(staging);
  }

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);

  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  state.cubemap_texture.is_dirty = false;

  /* Free face pixel buffers - data uploaded to GPU */
  for (int face = 0; face < NUM_CUBEMAP_FACES; face++) {
    free(state.cubemap_pixels[face]);
    state.cubemap_pixels[face] = NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  /* Load UFO body (Phong-lit) */
  bool ok = gltf_model_load_from_file(&state.ufo_model,
                                      "assets/models/retroufo.gltf", 1.0f);
  if (!ok) {
    printf("Failed to load retroufo.gltf\n");
    return;
  }

  /* Load UFO glow mesh */
  ok = gltf_model_load_from_file(&state.ufo_glow_model,
                                 "assets/models/retroufo_glow.gltf", 1.0f);
  if (!ok) {
    printf("Failed to load retroufo_glow.gltf\n");
    return;
  }

  state.models_loaded = true;
}

/* Loading descriptor: pre-transform vertices by node world matrices and
 * pre-multiply vertex colors by material baseColorFactor.
 *
 * The Vulkan reference uses PreTransformVertices | PreMultiplyVertexColors
 * | FlipY. FlipY is omitted here (WebGPU uses Y-up like OpenGL). */
static const gltf_model_desc_t ufo_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  if (!state.models_loaded) {
    return;
  }

  /* Process both UFO body and UFO glow models */
  struct {
    gltf_model_t* model;
    WGPUBuffer* vb;
    WGPUBuffer* ib;
    const char* vb_label;
    const char* ib_label;
  } items[2] = {
    {&state.ufo_model, &state.ufo_vertex_buffer, &state.ufo_index_buffer,
     "UFO Vertex Buffer", "UFO Index Buffer"},
    {&state.ufo_glow_model, &state.glow_vertex_buffer, &state.glow_index_buffer,
     "Glow Vertex Buffer", "Glow Index Buffer"},
  };

  for (int mi = 0; mi < 2; mi++) {
    gltf_model_t* m = items[mi].model;
    size_t vb_size  = m->vertex_count * sizeof(gltf_vertex_t);

    /* Create a copy of vertex data and bake node world transforms */
    gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
    memcpy(xformed, m->vertices, vb_size);
    gltf_model_bake_node_transforms(m, xformed, &ufo_load_desc);

    /* Upload transformed vertices to GPU */
    *items[mi].vb = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW(items[mi].vb_label),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = true,
              });
    void* vdata = wgpuBufferGetMappedRange(*items[mi].vb, 0, vb_size);
    memcpy(vdata, xformed, vb_size);
    wgpuBufferUnmap(*items[mi].vb);
    free(xformed);

    /* Upload index buffer (unchanged) */
    if (m->index_count > 0) {
      size_t ib_size = m->index_count * sizeof(uint32_t);
      *items[mi].ib  = wgpuDeviceCreateBuffer(
        device, &(WGPUBufferDescriptor){
                   .label = STRVIEW(items[mi].ib_label),
                   .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                   .size  = ib_size,
                   .mappedAtCreation = true,
                });
      void* idata = wgpuBufferGetMappedRange(*items[mi].ib, 0, ib_size);
      memcpy(idata, m->indices, ib_size);
      wgpuBufferUnmap(*items[mi].ib);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene UBO: projection + view + model (3 × mat4 = 192 bytes) */
  state.scene_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Scene UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = 3 * sizeof(mat4),
            });

  /* Skybox UBO: projection + view + model (3 × mat4 = 192 bytes) */
  state.skybox_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Skybox UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = 3 * sizeof(mat4),
            });

  /* Blur params UBO: blur_scale + blur_strength (2 × float = 8 bytes) */
  /* Pad to 16 bytes for alignment */
  state.blur_params_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Blur Params UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = 16,
            });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;
  float aspect    = (float)wgpu_context->width / (float)wgpu_context->height;

  /* ---- Scene UBO ---- */
  glm_perspective(glm_rad(45.0f), aspect, 0.1f, 256.0f,
                  state.scene_ubo_data.projection);

  glm_mat4_copy(state.camera.matrices.view, state.scene_ubo_data.view);

  /* UFO animation: bob + rotate */
  float angle_rad = state.timer * GLM_PIf * 2.0f;
  mat4 model;
  glm_mat4_identity(model);
  glm_translate(
    model, (vec3){sinf(angle_rad) * 0.25f, -1.0f, cosf(angle_rad) * 0.25f});
  glm_rotate(model, -sinf(angle_rad) * 0.15f, (vec3){1, 0, 0});
  glm_rotate(model, angle_rad, (vec3){0, 1, 0});
  glm_mat4_copy(model, state.scene_ubo_data.model);

  wgpuQueueWriteBuffer(queue, state.scene_ubo, 0, &state.scene_ubo_data,
                       3 * sizeof(mat4));

  /* ---- Skybox UBO ---- */
  glm_mat4_copy(state.scene_ubo_data.projection,
                state.skybox_ubo_data.projection);

  /* Strip translation from the view matrix for skybox */
  mat4 view_no_translate;
  glm_mat4_copy(state.camera.matrices.view, view_no_translate);
  view_no_translate[3][0] = 0.0f;
  view_no_translate[3][1] = 0.0f;
  view_no_translate[3][2] = 0.0f;
  glm_mat4_copy(view_no_translate, state.skybox_ubo_data.view);

  glm_mat4_identity(state.skybox_ubo_data.model);

  wgpuQueueWriteBuffer(queue, state.skybox_ubo, 0, &state.skybox_ubo_data,
                       3 * sizeof(mat4));

  /* ---- Blur params UBO ---- */
  float blur_data[4] = {
    state.settings.blur_scale, state.settings.blur_strength, 0.0f,
    0.0f /* padding */
  };
  wgpuQueueWriteBuffer(queue, state.blur_params_ubo, 0, blur_data,
                       sizeof(blur_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene bind group layout: UBO (vert) */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 3 * sizeof(mat4),
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

  /* Blur bind group layout: UBO (frag) + sampler (frag) + texture (frag) */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 16,
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
    state.blur_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Blur BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Skybox bind group layout: UBO (vert) + sampler (frag) + cubemap (frag) */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 3 * sizeof(mat4),
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
          .viewDimension = WGPUTextureViewDimension_Cube,
        },
      },
    };
    state.skybox_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Skybox BGL"),
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

  /* Scene pipeline layout */
  state.scene_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Scene Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.scene_bgl,
            });

  /* Blur pipeline layout */
  state.blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Blur Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.blur_bgl,
            });

  /* Skybox pipeline layout */
  state.skybox_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Skybox Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.skybox_bgl,
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
    WGPUBindGroupEntry entries[1] = {
      [0] = {
        .binding = 0,
        .buffer  = state.scene_ubo,
        .offset  = 0,
        .size    = 3 * sizeof(mat4),
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

  /* Blur vertical bind group: reads FB[0] */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.blur_params_ubo,
        .offset  = 0,
        .size    = 16,
      },
      [1] = {
        .binding = 1,
        .sampler = state.offscreen_sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.offscreen_fb[0].color_view,
      },
    };
    state.blur_vert_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Blur Vert Bind Group"),
                .layout     = state.blur_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Blur horizontal bind group: reads FB[1] */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.blur_params_ubo,
        .offset  = 0,
        .size    = 16,
      },
      [1] = {
        .binding = 1,
        .sampler = state.offscreen_sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.offscreen_fb[1].color_view,
      },
    };
    state.blur_horz_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Blur Horz Bind Group"),
                .layout     = state.blur_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Skybox bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {
        .binding = 0,
        .buffer  = state.skybox_ubo,
        .offset  = 0,
        .size    = 3 * sizeof(mat4),
      },
      [1] = {
        .binding = 1,
        .sampler = state.cubemap_texture.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.cubemap_texture.view,
      },
    };
    state.skybox_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Skybox Bind Group"),
                .layout     = state.skybox_bgl,
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

  /* Vertex buffer layout for glTF model (gltf_vertex_t) */
  WGPUVertexAttribute vertex_attrs[] = {
    /* position: vec3 at offset 0 */
    {
      .shaderLocation = 0,
      .offset         = offsetof(gltf_vertex_t, position),
      .format         = WGPUVertexFormat_Float32x3,
    },
    /* normal: vec3 */
    {
      .shaderLocation = 1,
      .offset         = offsetof(gltf_vertex_t, normal),
      .format         = WGPUVertexFormat_Float32x3,
    },
    /* uv0: vec2 */
    {
      .shaderLocation = 2,
      .offset         = offsetof(gltf_vertex_t, uv0),
      .format         = WGPUVertexFormat_Float32x2,
    },
    /* color: vec4 */
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

  /* ------ Glow pass pipeline (offscreen, renders glow mesh colors) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, bloom_colorpass_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.glow_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Glow Pipeline"),
        .layout = state.scene_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vertex_buffer_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &(WGPUDepthStencilState){
          .format              = WGPUTextureFormat_Depth24PlusStencil8,
          .depthWriteEnabled   = WGPUOptionalBool_True,
          .depthCompare        = WGPUCompareFunction_LessEqual,
          .stencilFront.compare = WGPUCompareFunction_Always,
          .stencilBack.compare  = WGPUCompareFunction_Always,
        },
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

  /* ------ Vertical blur pipeline (offscreen FB[0] → FB[1]) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, bloom_gaussblur_vert_shader_wgsl);

    /* Additive blending: src=ONE, dst=ONE */
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
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .blend     = &additive_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.blur_vert_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Blur Vert Pipeline"),
        .layout = state.blur_pipeline_layout,
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
        .depthStencil = &(WGPUDepthStencilState){
          .format              = WGPUTextureFormat_Depth24PlusStencil8,
          .depthWriteEnabled   = WGPUOptionalBool_True,
          .depthCompare        = WGPUCompareFunction_LessEqual,
          .stencilFront.compare = WGPUCompareFunction_Always,
          .stencilBack.compare  = WGPUCompareFunction_Always,
        },
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

  /* ------ Horizontal blur pipeline (main pass, additive on scene) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, bloom_gaussblur_horz_shader_wgsl);

    /* Additive blending */
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

    state.blur_horz_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Blur Horz Pipeline"),
        .layout = state.blur_pipeline_layout,
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
        .depthStencil = &(WGPUDepthStencilState){
          .format              = wgpu_context->depth_stencil_format,
          .depthWriteEnabled   = WGPUOptionalBool_True,
          .depthCompare        = WGPUCompareFunction_LessEqual,
          .stencilFront.compare = WGPUCompareFunction_Always,
          .stencilBack.compare  = WGPUCompareFunction_Always,
        },
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
      = wgpu_create_shader_module(device, bloom_phongpass_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.phong_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Phong Pipeline"),
        .layout = state.scene_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vertex_buffer_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &(WGPUDepthStencilState){
          .format              = wgpu_context->depth_stencil_format,
          .depthWriteEnabled   = WGPUOptionalBool_True,
          .depthCompare        = WGPUCompareFunction_LessEqual,
          .stencilFront.compare = WGPUCompareFunction_Always,
          .stencilBack.compare  = WGPUCompareFunction_Always,
        },
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

  /* ------ Skybox pipeline (main scene, no depth write, front-cull) ------ */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, bloom_skybox_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.skybox_pipeline = wgpuDeviceCreateRenderPipeline(device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Skybox - Render pipeline"),
        .layout = state.skybox_pipeline_layout,
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
        .depthStencil = &(WGPUDepthStencilState){
          .format              = wgpu_context->depth_stencil_format,
          .depthWriteEnabled   = WGPUOptionalBool_False,
          .depthCompare        = WGPUCompareFunction_LessEqual,
          .stencilFront.compare = WGPUCompareFunction_Always,
          .stencilBack.compare  = WGPUCompareFunction_Always,
        },
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

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* model,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  if (ib) {
    wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                        WGPU_WHOLE_SIZE);
  }

  /* Iterate through all nodes → meshes → primitives */
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
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Bloom Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("Bloom", &state.settings.bloom);
    imgui_overlay_slider_float("Blur Scale", &state.settings.blur_scale, 0.1f,
                               4.0f, "%.1f");
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

  /* sokol_fetch: 6 cubemap faces + 2 model files */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 8,
    .num_channels = 1,
    .num_lanes    = 8,
    .logger.func  = slog_func,
  });

  /* Camera setup */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera, (vec3){0.0f, -1.5f, -10.25f});
  camera_set_rotation(&state.camera, (vec3){-7.5f, -343.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 45.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load models synchronously (they're small) */
  load_models();
  create_model_buffers(wgpu_context);

  /* Create offscreen framebuffers */
  init_offscreen_framebuffers(wgpu_context);

  /* Create cubemap texture (initially black, populated by fetch) */
  init_cubemap_texture(wgpu_context);

  /* Start cubemap face loading */
  fetch_cubemap_faces();

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

  /* Upload cubemap pixels once all 6 faces are loaded */
  if (state.cubemap_texture.is_dirty
      && state.cubemap_load_count == NUM_CUBEMAP_FACES) {
    upload_cubemap_pixels(wgpu_context);
  }

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Update animation timer */
  state.timer += delta_time * state.animation_speed;
  if (state.timer > 1.0f) {
    state.timer -= 1.0f;
  }

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffers(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Render ---- */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ============ Pass 1: Glow render to offscreen FB[0] ============ */
  if (state.settings.bloom && state.models_loaded) {
    state.offscreen_color_att.view = state.offscreen_fb[0].color_view;
    state.offscreen_depth_att.view = state.offscreen_fb[0].depth_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_render_pass_desc);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, OFFSCREEN_WIDTH,
                                     OFFSCREEN_HEIGHT, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, OFFSCREEN_WIDTH,
                                        OFFSCREEN_HEIGHT);

    wgpuRenderPassEncoderSetPipeline(pass, state.glow_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0, 0);
    draw_model(pass, &state.ufo_glow_model, state.glow_vertex_buffer,
               state.glow_index_buffer);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 2: Vertical blur FB[0] → FB[1] ============ */
  if (state.settings.bloom) {
    state.offscreen_color_att.view = state.offscreen_fb[1].color_view;
    state.offscreen_depth_att.view = state.offscreen_fb[1].depth_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_render_pass_desc);

    wgpuRenderPassEncoderSetViewport(pass, 0, 0, OFFSCREEN_WIDTH,
                                     OFFSCREEN_HEIGHT, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, OFFSCREEN_WIDTH,
                                        OFFSCREEN_HEIGHT);

    wgpuRenderPassEncoderSetPipeline(pass, state.blur_vert_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.blur_vert_bind_group, 0,
                                      0);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ============ Pass 3: Main scene ============ */
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

    /* (a) Skybox */
    wgpuRenderPassEncoderSetPipeline(pass, state.skybox_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.skybox_bind_group, 0, 0);
    wgpuRenderPassEncoderDraw(pass, 36, 1, 0, 0);

    /* (b) Phong-lit UFO */
    if (state.models_loaded) {
      wgpuRenderPassEncoderSetPipeline(pass, state.phong_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bind_group, 0, 0);
      draw_model(pass, &state.ufo_model, state.ufo_vertex_buffer,
                 state.ufo_index_buffer);
    }

    /* (c) Horizontal blur composite (additive blend on scene) */
    if (state.settings.bloom) {
      wgpuRenderPassEncoderSetPipeline(pass, state.blur_horz_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.blur_horz_bind_group, 0,
                                        0);
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

  /* Free any cubemap pixel buffers not yet released */
  for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
    free(state.cubemap_pixels[i]);
    state.cubemap_pixels[i] = NULL;
  }

  /* Destroy models */
  gltf_model_destroy(&state.ufo_model);
  gltf_model_destroy(&state.ufo_glow_model);

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.ufo_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.ufo_index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.glow_vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.glow_index_buffer)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.scene_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.blur_params_ubo)

  /* Offscreen resources */
  destroy_offscreen_framebuffers();

  /* Cubemap */
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap_texture.view)
  WGPU_RELEASE_RESOURCE(Texture, state.cubemap_texture.handle)
  WGPU_RELEASE_RESOURCE(Sampler, state.cubemap_texture.sampler)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_vert_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blur_horz_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.skybox_bind_group)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.scene_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.blur_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.skybox_bgl)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.scene_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.blur_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.skybox_pipeline_layout)

  /* Render pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.glow_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blur_vert_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blur_horz_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.phong_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Bloom (Fullscreen Blur)",
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

/* ---- Color pass: renders glow mesh vertex colors to offscreen FB ---- */
// clang-format off
static const char* bloom_colorpass_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

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
    out.uv = in.uv;
    out.color = in.color.rgb;
    out.position = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
  }
);
// clang-format on

/* ---- Phong pass: lit scene rendering with ambient glow boost ---- */
// clang-format off
static const char* bloom_phongpass_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal   : vec3f,
    @location(1) uv       : vec2f,
    @location(2) color    : vec3f,
    @location(3) viewVec  : vec3f,
    @location(4) lightVec : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.normal = in.normal;
    out.color = in.color.rgb;
    out.uv = in.uv;
    out.position = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);

    let lightPos = vec3f(-5.0, -5.0, 0.0);
    let pos = ubo.view * ubo.model * vec4f(in.position, 1.0);
    out.normal = (ubo.view * ubo.model * vec4f(in.normal, 0.0)).xyz;
    out.lightVec = lightPos - pos.xyz;
    out.viewVec = -pos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    var ambient = vec3f(0.0);

    // Boost ambient for bright (glow) vertex colors
    if (in.color.r >= 0.9 || in.color.g >= 0.9 || in.color.b >= 0.9) {
      ambient = in.color * 0.25;
    }

    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(in.viewVec);
    let R = reflect(-L, N);
    let diffuse = max(dot(N, L), 0.0) * in.color;
    let specular = pow(max(dot(R, V), 0.0), 8.0) * vec3f(0.75);
    return vec4f(ambient + diffuse + specular, 1.0);
  }
);
// clang-format on

/* ---- Gaussian blur vertex shader (fullscreen triangle) ---- */
// clang-format off
static const char* bloom_gaussblur_vert_shader_wgsl = CODE(
  struct BlurParams {
    blurScale    : f32,
    blurStrength : f32,
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
    out.uv = uv;
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  // Vertical blur (direction = 0)
  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let weight = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    let texDim = vec2f(textureDimensions(texColor, 0));
    let tex_offset = 1.0 / texDim * params.blurScale;
    var result = textureSample(texColor, texSampler, in.uv).rgb * weight[0];

    // Vertical direction: offset along Y
    for (var i = 1; i < 5; i++) {
      let offset = vec2f(0.0, tex_offset.y * f32(i));
      result += textureSample(texColor, texSampler, in.uv + offset).rgb * weight[i] * params.blurStrength;
      result += textureSample(texColor, texSampler, in.uv - offset).rgb * weight[i] * params.blurStrength;
    }
    return vec4f(result, 1.0);
  }
);
// clang-format on

/* ---- Gaussian blur horizontal direction ---- */
// clang-format off
static const char* bloom_gaussblur_horz_shader_wgsl = CODE(
  struct BlurParams {
    blurScale    : f32,
    blurStrength : f32,
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
    out.uv = uv;
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  // Horizontal blur (direction = 1)
  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let weight = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    let texDim = vec2f(textureDimensions(texColor, 0));
    let tex_offset = 1.0 / texDim * params.blurScale;
    var result = textureSample(texColor, texSampler, in.uv).rgb * weight[0];

    // Horizontal direction: offset along X
    for (var i = 1; i < 5; i++) {
      let offset = vec2f(tex_offset.x * f32(i), 0.0);
      result += textureSample(texColor, texSampler, in.uv + offset).rgb * weight[i] * params.blurStrength;
      result += textureSample(texColor, texSampler, in.uv - offset).rgb * weight[i] * params.blurStrength;
    }
    return vec4f(result, 1.0);
  }
);
// clang-format on

/* ---- Skybox: cubemap rendering with cube mesh ---- */
// clang-format off
static const char* bloom_skybox_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var cubeSampler : sampler;
  @group(0) @binding(2) var cubeTexture : texture_cube<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uvw : vec3f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    // Hardcoded cube vertices (36 vertices for 12 triangles)
    var positions = array<vec3f, 36>(
      // +X face
      vec3f( 1, -1, -1), vec3f( 1, -1,  1), vec3f( 1,  1,  1),
      vec3f( 1, -1, -1), vec3f( 1,  1,  1), vec3f( 1,  1, -1),
      // -X face
      vec3f(-1, -1,  1), vec3f(-1, -1, -1), vec3f(-1,  1, -1),
      vec3f(-1, -1,  1), vec3f(-1,  1, -1), vec3f(-1,  1,  1),
      // +Y face
      vec3f(-1,  1, -1), vec3f( 1,  1, -1), vec3f( 1,  1,  1),
      vec3f(-1,  1, -1), vec3f( 1,  1,  1), vec3f(-1,  1,  1),
      // -Y face
      vec3f(-1, -1,  1), vec3f( 1, -1,  1), vec3f( 1, -1, -1),
      vec3f(-1, -1,  1), vec3f( 1, -1, -1), vec3f(-1, -1, -1),
      // +Z face
      vec3f(-1, -1,  1), vec3f(-1,  1,  1), vec3f( 1,  1,  1),
      vec3f(-1, -1,  1), vec3f( 1,  1,  1), vec3f( 1, -1,  1),
      // -Z face
      vec3f( 1, -1, -1), vec3f( 1,  1, -1), vec3f(-1,  1, -1),
      vec3f( 1, -1, -1), vec3f(-1,  1, -1), vec3f(-1, -1, -1),
    );

    var out : VertexOutput;
    let pos = positions[vertexIndex];
    // Negate Y to match Vulkan's FlipY convention on the skybox cube model.
    // The space cubemap PNG faces are stored in Vulkan's convention (-Y face =
    // sky when looking up). Vulkan compensates via FlipY on the cube geometry
    // (negating Y in the sample direction). WebGPU uses Y-up NDC without any
    // projection Y-flip, so we must explicitly negate Y here to sample the
    // correct cubemap face when looking up.
    out.uvw = vec3f(pos.x, -pos.y, pos.z);
    out.position = ubo.projection * ubo.view * ubo.model * vec4f(pos, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return textureSample(cubeTexture, cubeSampler, in.uvw);
  }
);
// clang-format on
