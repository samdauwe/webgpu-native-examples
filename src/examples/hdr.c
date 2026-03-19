/* -------------------------------------------------------------------------- *
 * WebGPU Example - High Dynamic Range Rendering
 *
 * Implements an HDR rendering pipeline that uses floating-point textures for
 * wider color range. The scene (skybox + reflective object) is rendered to
 * offscreen RGBA16Float attachments with exposure-based tone mapping and bloom
 * extraction. A separable two-pass 25-tap Gaussian blur is applied to the
 * bright pixels, and the final image composites the tone-mapped scene with
 * the blurred bloom overlay via additive blending.
 *
 * Rendering passes:
 *   1. Offscreen G-Buffer: Skybox + reflective 3D object → 2 float16 color
 *      attachments (tone-mapped scene + bright pixels) + depth
 *   2. Bloom filter (horizontal): Fullscreen Gaussian blur on bright pixels
 *   3. Final composition: Tone-mapped scene + additive bloom overlay
 *
 * Features:
 * - HDR rendering with RGBA16Float offscreen buffers
 * - Exposure-based tone mapping (1 - exp(-color * exposure))
 * - Cook-Torrance specular BRDF with environment map reflection
 * - Separable 25-tap Gaussian bloom blur
 * - Cubemap skybox rendering
 * - Multiple selectable 3D objects (sphere, teapot, torusknot, venus)
 * - GUI controls for object selection, exposure, bloom, and skybox toggle
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/hdr
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

static const char* hdr_gbuffer_skybox_shader_wgsl;
static const char* hdr_gbuffer_reflect_shader_wgsl;
static const char* hdr_bloom_horz_shader_wgsl;
static const char* hdr_bloom_vert_shader_wgsl;
static const char* hdr_composition_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define NUM_CUBEMAP_FACES (6)
#define CUBEMAP_FACE_SIZE (1024)
#define CUBEMAP_FACE_NUM_BYTES (CUBEMAP_FACE_SIZE * CUBEMAP_FACE_SIZE * 4)

#define NUM_OBJECTS (4)

static const char* object_model_paths[NUM_OBJECTS] = {
  "assets/models/sphere.gltf",
  "assets/models/teapot.gltf",
  "assets/models/torusknot.gltf",
  "assets/models/venus.gltf",
};

static const char* object_names[NUM_OBJECTS] = {
  "Sphere",
  "Teapot",
  "Torusknot",
  "Venus",
};

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Models */
  gltf_model_t skybox_model;
  gltf_model_t object_models[NUM_OBJECTS];
  bool models_loaded;

  /* GPU vertex/index buffers for skybox */
  WGPUBuffer skybox_vertex_buffer;
  WGPUBuffer skybox_index_buffer;

  /* GPU vertex/index buffers for objects */
  WGPUBuffer object_vertex_buffers[NUM_OBJECTS];
  WGPUBuffer object_index_buffers[NUM_OBJECTS];

  /* Cubemap texture */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    bool is_dirty;
  } cubemap_texture;
  uint8_t cubemap_pixels[NUM_CUBEMAP_FACES][CUBEMAP_FACE_NUM_BYTES];
  int cubemap_load_count;

  /* Offscreen framebuffer (2 color attachments + depth) */
  struct {
    WGPUTexture color_textures[2];
    WGPUTextureView color_views[2];
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
    WGPUSampler sampler;
    uint32_t width;
    uint32_t height;
  } offscreen;

  /* Filter pass framebuffer (1 color attachment, no depth) */
  struct {
    WGPUTexture color_texture;
    WGPUTextureView color_view;
    WGPUSampler sampler;
    uint32_t width;
    uint32_t height;
  } filter_pass;

  /* Uniform buffer */
  struct {
    mat4 projection;
    mat4 modelview;
    mat4 inverse_modelview;
    float exposure;
    float _pad[3];
  } ubo_data;
  WGPUBuffer uniform_buffer;

  /* Bind group layouts */
  WGPUBindGroupLayout models_bgl;
  WGPUBindGroupLayout bloom_filter_bgl;
  WGPUBindGroupLayout composition_bgl;

  /* Pipeline layouts */
  WGPUPipelineLayout models_pipeline_layout;
  WGPUPipelineLayout bloom_filter_pipeline_layout;
  WGPUPipelineLayout composition_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline skybox_pipeline;
  WGPURenderPipeline reflect_pipeline;
  WGPURenderPipeline composition_pipeline;
  WGPURenderPipeline bloom_vert_pipeline; /* vertical blur (final pass) */
  WGPURenderPipeline bloom_horz_pipeline; /* horizontal blur (filter pass) */

  /* Bind groups */
  WGPUBindGroup skybox_bind_group;
  WGPUBindGroup object_bind_group;
  WGPUBindGroup bloom_filter_bind_group;
  WGPUBindGroup composition_bind_group;

  /* GUI settings */
  struct {
    int32_t object_index;
    float exposure;
    bool bloom;
    bool display_skybox;
  } settings;

  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .settings = {
    .object_index  = 1,
    .exposure      = 1.0f,
    .bloom         = true,
    .display_skybox = true,
  },
};

/* -------------------------------------------------------------------------- *
 * Cubemap loading
 * -------------------------------------------------------------------------- */

static void init_cubemap_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  WGPUTextureDescriptor tex_desc = {
    .label         = STRVIEW("HDR Cubemap Texture"),
    .usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {CUBEMAP_FACE_SIZE, CUBEMAP_FACE_SIZE, NUM_CUBEMAP_FACES},
    .format        = WGPUTextureFormat_RGBA8Unorm,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.cubemap_texture.handle = wgpuDeviceCreateTexture(device, &tex_desc);

  WGPUTextureViewDescriptor view_desc = {
    .label           = STRVIEW("HDR Cubemap View"),
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

  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("HDR Cubemap Sampler"),
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
    "assets/textures/cubemaps/uffizi_cube_px.png",
    "assets/textures/cubemaps/uffizi_cube_nx.png",
    "assets/textures/cubemaps/uffizi_cube_py.png",
    "assets/textures/cubemaps/uffizi_cube_ny.png",
    "assets/textures/cubemaps/uffizi_cube_pz.png",
    "assets/textures/cubemaps/uffizi_cube_nz.png",
  };

  state.cubemap_texture.is_dirty = true;
  state.cubemap_load_count       = 0;

  for (int i = 0; i < NUM_CUBEMAP_FACES; i++) {
    sfetch_send(&(sfetch_request_t){
      .path     = face_paths[i],
      .callback = cubemap_fetch_callback,
      .buffer   = SFETCH_RANGE(state.cubemap_pixels[i]),
      .channel  = 0,
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
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_models(void)
{
  /* Load skybox cube */
  bool ok = gltf_model_load_from_file(&state.skybox_model,
                                      "assets/models/cube.gltf", 1.0f);
  if (!ok) {
    printf("Failed to load cube.gltf\n");
    return;
  }

  /* Load all object models */
  for (int i = 0; i < NUM_OBJECTS; i++) {
    ok = gltf_model_load_from_file(&state.object_models[i],
                                   object_model_paths[i], 1.0f);
    if (!ok) {
      printf("Failed to load %s\n", object_model_paths[i]);
      return;
    }
  }

  state.models_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  gltf_model_desc_t bake_desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices,
  };

  /* ----- skybox ----- */
  {
    gltf_model_t* m      = &state.skybox_model;
    uint32_t vb_size     = m->vertex_count * sizeof(gltf_vertex_t);
    gltf_vertex_t* baked = (gltf_vertex_t*)malloc(vb_size);
    memcpy(baked, m->vertices, vb_size);
    gltf_model_bake_node_transforms(m, baked, &bake_desc);
    WGPUBufferDescriptor vb_desc = {
      .label            = STRVIEW("Skybox VB"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = vb_size,
      .mappedAtCreation = true,
    };
    state.skybox_vertex_buffer = wgpuDeviceCreateBuffer(device, &vb_desc);
    void* dst
      = wgpuBufferGetMappedRange(state.skybox_vertex_buffer, 0, vb_size);
    memcpy(dst, baked, vb_size);
    wgpuBufferUnmap(state.skybox_vertex_buffer);
    free(baked);

    uint32_t ib_size             = m->index_count * sizeof(uint32_t);
    WGPUBufferDescriptor ib_desc = {
      .label            = STRVIEW("Skybox IB"),
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = ib_size,
      .mappedAtCreation = true,
    };
    state.skybox_index_buffer = wgpuDeviceCreateBuffer(device, &ib_desc);
    dst = wgpuBufferGetMappedRange(state.skybox_index_buffer, 0, ib_size);
    memcpy(dst, m->indices, ib_size);
    wgpuBufferUnmap(state.skybox_index_buffer);
  }

  /* ----- objects ----- */
  for (int i = 0; i < NUM_OBJECTS; i++) {
    gltf_model_t* m      = &state.object_models[i];
    uint32_t vb_size     = m->vertex_count * sizeof(gltf_vertex_t);
    gltf_vertex_t* baked = (gltf_vertex_t*)malloc(vb_size);
    memcpy(baked, m->vertices, vb_size);
    gltf_model_bake_node_transforms(m, baked, &bake_desc);
    WGPUBufferDescriptor vb_desc = {
      .label            = STRVIEW("Object VB"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = vb_size,
      .mappedAtCreation = true,
    };
    state.object_vertex_buffers[i] = wgpuDeviceCreateBuffer(device, &vb_desc);
    void* dst
      = wgpuBufferGetMappedRange(state.object_vertex_buffers[i], 0, vb_size);
    memcpy(dst, baked, vb_size);
    wgpuBufferUnmap(state.object_vertex_buffers[i]);
    free(baked);

    uint32_t ib_size             = m->index_count * sizeof(uint32_t);
    WGPUBufferDescriptor ib_desc = {
      .label            = STRVIEW("Object IB"),
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = ib_size,
      .mappedAtCreation = true,
    };
    state.object_index_buffers[i] = wgpuDeviceCreateBuffer(device, &ib_desc);
    dst = wgpuBufferGetMappedRange(state.object_index_buffers[i], 0, ib_size);
    memcpy(dst, m->indices, ib_size);
    wgpuBufferUnmap(state.object_index_buffers[i]);
  }
}

static void draw_model(WGPURenderPassEncoder pass, const gltf_model_t* model,
                       WGPUBuffer vertex_buffer, WGPUBuffer index_buffer)
{
  wgpuRenderPassEncoderSetVertexBuffer(
    pass, 0, vertex_buffer, 0, model->vertex_count * sizeof(gltf_vertex_t));
  wgpuRenderPassEncoderSetIndexBuffer(pass, index_buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      model->index_count * sizeof(uint32_t));

  for (uint32_t n = 0; n < model->linear_node_count; n++) {
    gltf_node_t* node = model->linear_nodes[n];
    if (node->mesh) {
      for (uint32_t p = 0; p < node->mesh->primitive_count; p++) {
        gltf_primitive_t* prim = &node->mesh->primitives[p];
        if (prim->index_count > 0) {
          wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                           prim->first_index, 0, 0);
        }
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Offscreen framebuffer setup
 * -------------------------------------------------------------------------- */

static void destroy_offscreen_resources(void)
{
  for (int i = 0; i < 2; i++) {
    if (state.offscreen.color_views[i]) {
      wgpuTextureViewRelease(state.offscreen.color_views[i]);
      state.offscreen.color_views[i] = NULL;
    }
    if (state.offscreen.color_textures[i]) {
      wgpuTextureDestroy(state.offscreen.color_textures[i]);
      wgpuTextureRelease(state.offscreen.color_textures[i]);
      state.offscreen.color_textures[i] = NULL;
    }
  }
  if (state.offscreen.depth_view) {
    wgpuTextureViewRelease(state.offscreen.depth_view);
    state.offscreen.depth_view = NULL;
  }
  if (state.offscreen.depth_texture) {
    wgpuTextureDestroy(state.offscreen.depth_texture);
    wgpuTextureRelease(state.offscreen.depth_texture);
    state.offscreen.depth_texture = NULL;
  }
  if (state.offscreen.sampler) {
    wgpuSamplerRelease(state.offscreen.sampler);
    state.offscreen.sampler = NULL;
  }
}

static void destroy_filter_pass_resources(void)
{
  if (state.filter_pass.color_view) {
    wgpuTextureViewRelease(state.filter_pass.color_view);
    state.filter_pass.color_view = NULL;
  }
  if (state.filter_pass.color_texture) {
    wgpuTextureDestroy(state.filter_pass.color_texture);
    wgpuTextureRelease(state.filter_pass.color_texture);
    state.filter_pass.color_texture = NULL;
  }
  if (state.filter_pass.sampler) {
    wgpuSamplerRelease(state.filter_pass.sampler);
    state.filter_pass.sampler = NULL;
  }
}

static void init_offscreen_framebuffer(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.offscreen.width  = wgpu_context->width;
  state.offscreen.height = wgpu_context->height;

  /* Two RGBA16Float color attachments */
  for (int i = 0; i < 2; i++) {
    WGPUTextureDescriptor td = {
      .label = STRVIEW("Offscreen Color"),
      .usage
      = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {state.offscreen.width, state.offscreen.height, 1},
      .format        = WGPUTextureFormat_RGBA16Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.offscreen.color_textures[i] = wgpuDeviceCreateTexture(device, &td);
    state.offscreen.color_views[i]
      = wgpuTextureCreateView(state.offscreen.color_textures[i], NULL);
  }

  /* Depth attachment */
  {
    WGPUTextureDescriptor td = {
      .label         = STRVIEW("Offscreen Depth"),
      .usage         = WGPUTextureUsage_RenderAttachment,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {state.offscreen.width, state.offscreen.height, 1},
      .format        = WGPUTextureFormat_Depth24PlusStencil8,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    state.offscreen.depth_texture = wgpuDeviceCreateTexture(device, &td);
    state.offscreen.depth_view
      = wgpuTextureCreateView(state.offscreen.depth_texture, NULL);
  }

  /* Sampler (nearest filter, same as Vulkan version) */
  WGPUSamplerDescriptor sd = {
    .label         = STRVIEW("Offscreen Sampler"),
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .magFilter     = WGPUFilterMode_Nearest,
    .minFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  state.offscreen.sampler = wgpuDeviceCreateSampler(device, &sd);
}

static void init_filter_pass_framebuffer(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.filter_pass.width  = wgpu_context->width;
  state.filter_pass.height = wgpu_context->height;

  WGPUTextureDescriptor td = {
    .label = STRVIEW("FilterPass Color"),
    .usage
    = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
    .dimension     = WGPUTextureDimension_2D,
    .size          = {state.filter_pass.width, state.filter_pass.height, 1},
    .format        = WGPUTextureFormat_RGBA16Float,
    .mipLevelCount = 1,
    .sampleCount   = 1,
  };
  state.filter_pass.color_texture = wgpuDeviceCreateTexture(device, &td);
  state.filter_pass.color_view
    = wgpuTextureCreateView(state.filter_pass.color_texture, NULL);

  WGPUSamplerDescriptor sd = {
    .label         = STRVIEW("FilterPass Sampler"),
    .addressModeU  = WGPUAddressMode_ClampToEdge,
    .addressModeV  = WGPUAddressMode_ClampToEdge,
    .addressModeW  = WGPUAddressMode_ClampToEdge,
    .magFilter     = WGPUFilterMode_Nearest,
    .minFilter     = WGPUFilterMode_Nearest,
    .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
    .lodMinClamp   = 0.0f,
    .lodMaxClamp   = 1.0f,
    .maxAnisotropy = 1,
  };
  state.filter_pass.sampler = wgpuDeviceCreateSampler(device, &sd);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  WGPUBufferDescriptor bd = {
    .label = STRVIEW("HDR UBO"),
    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    .size  = sizeof(state.ubo_data),
  };
  state.uniform_buffer = wgpuDeviceCreateBuffer(wgpu_context->device, &bd);
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Projection and view from camera */
  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_data.modelview);

  /* Compute inverse modelview for world-space cubemap lookups */
  glm_mat4_inv(state.ubo_data.modelview, state.ubo_data.inverse_modelview);

  state.ubo_data.exposure = state.settings.exposure;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.ubo_data, sizeof(state.ubo_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts, pipeline layouts, bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Models layout: binding(0)=UBO, binding(1)=sampler, binding(2)=textureCube
   */
  {
    WGPUBindGroupLayoutEntry entries[3] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform,
                       .minBindingSize = sizeof(state.ubo_data)},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_Cube,
                       .multisampled  = false},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Models BGL"),
      .entryCount = 3,
      .entries    = entries,
    };
    state.models_bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
  }

  /* Bloom filter layout: binding(0)=sampler, binding(1)=texture2D (bright) */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D,
                       .multisampled  = false},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Bloom Filter BGL"),
      .entryCount = 2,
      .entries    = entries,
    };
    state.bloom_filter_bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
  }

  /* Composition layout: binding(0)=sampler, binding(1)=texture2D (scene) */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {.sampleType    = WGPUTextureSampleType_Float,
                       .viewDimension = WGPUTextureViewDimension_2D,
                       .multisampled  = false},
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Composition BGL"),
      .entryCount = 2,
      .entries    = entries,
    };
    state.composition_bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
  }
}

static void init_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  {
    WGPUPipelineLayoutDescriptor desc = {
      .label                = STRVIEW("Models PL"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.models_bgl,
    };
    state.models_pipeline_layout
      = wgpuDeviceCreatePipelineLayout(device, &desc);
  }
  {
    WGPUPipelineLayoutDescriptor desc = {
      .label                = STRVIEW("Bloom Filter PL"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bloom_filter_bgl,
    };
    state.bloom_filter_pipeline_layout
      = wgpuDeviceCreatePipelineLayout(device, &desc);
  }
  {
    WGPUPipelineLayoutDescriptor desc = {
      .label                = STRVIEW("Composition PL"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.composition_bgl,
    };
    state.composition_pipeline_layout
      = wgpuDeviceCreatePipelineLayout(device, &desc);
  }
}

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Skybox bind group (same layout as object, same UBO + cubemap) */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_buffer,
             .size    = sizeof(state.ubo_data)},
      [1] = {.binding = 1, .sampler = state.cubemap_texture.sampler},
      [2] = {.binding = 2, .textureView = state.cubemap_texture.view},
    };
    WGPUBindGroupDescriptor desc = {
      .label      = STRVIEW("Skybox BG"),
      .layout     = state.models_bgl,
      .entryCount = 3,
      .entries    = entries,
    };
    state.skybox_bind_group = wgpuDeviceCreateBindGroup(device, &desc);
  }

  /* Object bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0,
             .buffer  = state.uniform_buffer,
             .size    = sizeof(state.ubo_data)},
      [1] = {.binding = 1, .sampler = state.cubemap_texture.sampler},
      [2] = {.binding = 2, .textureView = state.cubemap_texture.view},
    };
    WGPUBindGroupDescriptor desc = {
      .label      = STRVIEW("Object BG"),
      .layout     = state.models_bgl,
      .entryCount = 3,
      .entries    = entries,
    };
    state.object_bind_group = wgpuDeviceCreateBindGroup(device, &desc);
  }

  /* Bloom filter bind group: samples from offscreen color[1] (bright pixels) */
  {
    WGPUBindGroupEntry entries[2] = {
      [0] = {.binding = 0, .sampler = state.offscreen.sampler},
      [1] = {.binding = 1, .textureView = state.offscreen.color_views[1]},
    };
    WGPUBindGroupDescriptor desc = {
      .label      = STRVIEW("Bloom Filter BG"),
      .layout     = state.bloom_filter_bgl,
      .entryCount = 2,
      .entries    = entries,
    };
    state.bloom_filter_bind_group = wgpuDeviceCreateBindGroup(device, &desc);
  }

  /* Composition bind group: samples from offscreen color[0] (tone-mapped) */
  {
    WGPUBindGroupEntry entries[2] = {
      [0] = {.binding = 0, .sampler = state.offscreen.sampler},
      [1] = {.binding = 1, .textureView = state.offscreen.color_views[0]},
    };
    WGPUBindGroupDescriptor desc = {
      .label      = STRVIEW("Composition BG"),
      .layout     = state.composition_bgl,
      .entryCount = 2,
      .entries    = entries,
    };
    state.composition_bind_group = wgpuDeviceCreateBindGroup(device, &desc);
  }
}

static void destroy_bind_groups(void)
{
  if (state.bloom_filter_bind_group) {
    wgpuBindGroupRelease(state.bloom_filter_bind_group);
    state.bloom_filter_bind_group = NULL;
  }
  if (state.composition_bind_group) {
    wgpuBindGroupRelease(state.composition_bind_group);
    state.composition_bind_group = NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

/* Vertex layout: position(vec3) + normal(vec3) from gltf_vertex_t */
static WGPUVertexAttribute model_vertex_attrs[2];
static WGPUVertexBufferLayout model_vertex_layout;

static void init_vertex_layout(void)
{
  model_vertex_attrs[0] = (WGPUVertexAttribute){
    .format         = WGPUVertexFormat_Float32x3,
    .offset         = offsetof(gltf_vertex_t, position),
    .shaderLocation = 0,
  };
  model_vertex_attrs[1] = (WGPUVertexAttribute){
    .format         = WGPUVertexFormat_Float32x3,
    .offset         = offsetof(gltf_vertex_t, normal),
    .shaderLocation = 1,
  };
  model_vertex_layout = (WGPUVertexBufferLayout){
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = 2,
    .attributes     = model_vertex_attrs,
  };
}

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  WGPUTextureFormat surface_format = wgpu_context->render_format;

  WGPUColorTargetState offscreen_targets[2] = {
    [0] = {.format    = WGPUTextureFormat_RGBA16Float,
           .writeMask = WGPUColorWriteMask_All},
    [1] = {.format    = WGPUTextureFormat_RGBA16Float,
           .writeMask = WGPUColorWriteMask_All},
  };

  /* Depth-stencil state shared by offscreen pipelines */
  WGPUDepthStencilState offscreen_ds = {
    .format            = WGPUTextureFormat_Depth24PlusStencil8,
    .depthWriteEnabled = WGPUOptionalBool_True,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
    .stencilReadMask   = 0xFF,
    .stencilWriteMask  = 0xFF,
  };

  /* ----- Skybox pipeline (offscreen, cull front, depth test) ----- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, hdr_gbuffer_skybox_shader_wgsl);
    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Skybox Pipeline"),
      .layout = state.models_pipeline_layout,
      .vertex = {
        .module      = sm,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &model_vertex_layout,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Front,
      },
      .depthStencil = &offscreen_ds,
      .multisample  = {.count = 1, .mask = ~0u},
      .fragment     = &(WGPUFragmentState){
        .module      = sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 2,
        .targets     = offscreen_targets,
      },
    };
    state.skybox_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    wgpuShaderModuleRelease(sm);
  }

  /* ----- Reflect pipeline (offscreen, cull back, depth test+write) ----- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, hdr_gbuffer_reflect_shader_wgsl);

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Reflect Pipeline"),
      .layout = state.models_pipeline_layout,
      .vertex = {
        .module      = sm,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &model_vertex_layout,
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Back,
      },
      .depthStencil = &offscreen_ds,
      .multisample  = {.count = 1, .mask = ~0u},
      .fragment     = &(WGPUFragmentState){
        .module      = sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 2,
        .targets     = offscreen_targets,
      },
    };
    state.reflect_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    wgpuShaderModuleRelease(sm);
  }

  /* Additive blend state for bloom */
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

  /* ----- Bloom horizontal pipeline (into filterPass, additive) ----- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, hdr_bloom_horz_shader_wgsl);

    WGPUColorTargetState target = {
      .format    = WGPUTextureFormat_RGBA16Float,
      .blend     = &additive_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Bloom Horz Pipeline"),
      .layout = state.bloom_filter_pipeline_layout,
      .vertex = {
        .module     = sm,
        .entryPoint = STRVIEW("vs_main"),
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_None,
      },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment    = &(WGPUFragmentState){
        .module      = sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &target,
      },
    };
    state.bloom_horz_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    wgpuShaderModuleRelease(sm);
  }

  /* ----- Bloom vertical pipeline (swapchain, additive) ----- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, hdr_bloom_vert_shader_wgsl);

    WGPUColorTargetState target = {
      .format    = surface_format,
      .blend     = &additive_blend,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Bloom Vert Pipeline"),
      .layout = state.bloom_filter_pipeline_layout,
      .vertex = {
        .module     = sm,
        .entryPoint = STRVIEW("vs_main"),
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_None,
      },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment    = &(WGPUFragmentState){
        .module      = sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &target,
      },
    };
    state.bloom_vert_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    wgpuShaderModuleRelease(sm);
  }

  /* ----- Composition pipeline (swapchain, no blend) ----- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, hdr_composition_shader_wgsl);

    WGPUColorTargetState target = {
      .format    = surface_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    WGPURenderPipelineDescriptor desc = {
      .label  = STRVIEW("Composition Pipeline"),
      .layout = state.composition_pipeline_layout,
      .vertex = {
        .module     = sm,
        .entryPoint = STRVIEW("vs_main"),
      },
      .primitive = {
        .topology = WGPUPrimitiveTopology_TriangleList,
        .cullMode = WGPUCullMode_None,
      },
      .multisample = {.count = 1, .mask = ~0u},
      .fragment    = &(WGPUFragmentState){
        .module      = sm,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets     = &target,
      },
    };
    state.composition_pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
    wgpuShaderModuleRelease(sm);
  }
}

/* -------------------------------------------------------------------------- *
 * Resize handling
 * -------------------------------------------------------------------------- */

static void on_resize(struct wgpu_context_t* wgpu_context)
{
  uint32_t w = wgpu_context->width;
  uint32_t h = wgpu_context->height;

  if (w == 0 || h == 0) {
    return;
  }

  /* Recreate offscreen + filter pass at new size */
  destroy_offscreen_resources();
  init_offscreen_framebuffer(wgpu_context);

  destroy_filter_pass_resources();
  init_filter_pass_framebuffer(wgpu_context);

  /* Recreate bind groups that reference those textures */
  destroy_bind_groups();
  init_bind_groups(wgpu_context);

  /* Update camera aspect ratio */
  camera_set_perspective(&state.camera, 60.0f, (float)w / (float)h, 0.1f,
                         256.0f);
}

/* -------------------------------------------------------------------------- *
 * GUI
 * -------------------------------------------------------------------------- */

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  (void)wgpu_context;

  igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_FirstUseEver, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){260, 0}, ImGuiCond_FirstUseEver);
  igBegin("HDR Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_combo_box("Object type", &state.settings.object_index,
                            object_names, NUM_OBJECTS);
    imgui_overlay_slider_float("Exposure", &state.settings.exposure, 0.1f,
                               10.0f, "%.3f");
    igCheckbox("Bloom", &state.settings.bloom);
    igCheckbox("Skybox", &state.settings.display_skybox);
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
  camera_on_input_event(&state.camera, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    on_resize(wgpu_context);
  }
}

/* -------------------------------------------------------------------------- *
 * Frame rendering
 * -------------------------------------------------------------------------- */

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Upload cubemap when all faces loaded */
  if (state.cubemap_texture.is_dirty
      && state.cubemap_load_count >= NUM_CUBEMAP_FACES) {
    upload_cubemap_pixels(wgpu_context);
  }

  /* Timing */
  uint64_t now          = stm_now();
  float delta_time      = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  camera_update(&state.camera, delta_time);
  update_uniform_buffers(wgpu_context);

  /* ImGui new frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* Need models loaded to render */
  if (!state.models_loaded) {
    /* Just draw ImGui overlay on a cleared screen */
    WGPUTextureView back_buffer = wgpu_context->swapchain_view;
    WGPUCommandEncoder enc
      = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
    WGPURenderPassColorAttachment ca = {
      .view       = back_buffer,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 1},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rpd);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    imgui_overlay_render(wgpu_context);
    return EXIT_SUCCESS;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  int32_t obj_idx = state.settings.object_index;

  /* ===== Pass 1: Offscreen G-Buffer ===== */
  {
    WGPURenderPassColorAttachment color_atts[2] = {
      [0] = {
        .view       = state.offscreen.color_views[0],
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
      [1] = {
        .view       = state.offscreen.color_views[1],
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      },
    };
    WGPURenderPassDepthStencilAttachment ds_att = {
      .view              = state.offscreen.depth_view,
      .depthLoadOp       = WGPULoadOp_Clear,
      .depthStoreOp      = WGPUStoreOp_Store,
      .depthClearValue   = 1.0f,
      .stencilLoadOp     = WGPULoadOp_Clear,
      .stencilStoreOp    = WGPUStoreOp_Store,
      .stencilClearValue = 0,
    };
    WGPURenderPassDescriptor rpd = {
      .colorAttachmentCount   = 2,
      .colorAttachments       = color_atts,
      .depthStencilAttachment = &ds_att,
    };

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rpd);

    /* Skybox */
    if (state.settings.display_skybox) {
      wgpuRenderPassEncoderSetPipeline(pass, state.skybox_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, state.skybox_bind_group, 0,
                                        NULL);
      draw_model(pass, &state.skybox_model, state.skybox_vertex_buffer,
                 state.skybox_index_buffer);
    }

    /* 3D object */
    wgpuRenderPassEncoderSetPipeline(pass, state.reflect_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.object_bind_group, 0,
                                      NULL);
    draw_model(pass, &state.object_models[obj_idx],
               state.object_vertex_buffers[obj_idx],
               state.object_index_buffers[obj_idx]);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ===== Pass 2: Bloom filter (horizontal blur) ===== */
  if (state.settings.bloom) {
    WGPURenderPassColorAttachment ca = {
      .view       = state.filter_pass.color_view,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 0},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rpd);

    wgpuRenderPassEncoderSetPipeline(pass, state.bloom_horz_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bloom_filter_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ===== Pass 3: Final composition + bloom overlay ===== */
  {
    WGPUTextureView back_buffer = wgpu_context->swapchain_view;

    WGPURenderPassColorAttachment ca = {
      .view       = back_buffer,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = {0, 0, 0, 1},
      .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };
    WGPURenderPassDescriptor rpd = {
      .colorAttachmentCount = 1,
      .colorAttachments     = &ca,
    };

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rpd);

    /* Composition: draw tone-mapped scene */
    wgpuRenderPassEncoderSetPipeline(pass, state.composition_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.composition_bind_group, 0,
                                      NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    /* Bloom overlay: additive vertical blur */
    if (state.settings.bloom) {
      /* Need a bind group for the filter_pass output */
      WGPUBindGroupEntry entries[2] = {
        [0] = {.binding = 0, .sampler = state.filter_pass.sampler},
        [1] = {.binding = 1, .textureView = state.filter_pass.color_view},
      };
      WGPUBindGroupDescriptor desc = {
        .layout     = state.bloom_filter_bgl,
        .entryCount = 2,
        .entries    = entries,
      };
      WGPUBindGroup bloom_vert_bg = wgpuDeviceCreateBindGroup(device, &desc);

      wgpuRenderPassEncoderSetPipeline(pass, state.bloom_vert_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bloom_vert_bg, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

      wgpuBindGroupRelease(bloom_vert_bg);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Init / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = NUM_CUBEMAP_FACES + NUM_OBJECTS + 2,
    .num_channels = 1,
    .num_lanes    = NUM_CUBEMAP_FACES,
    .logger.func  = slog_func,
  });

  /* Camera: Vulkan had pos(0,0,-6), rot(0,0,0), 60° FOV */
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;
  camera_set_position(&state.camera, (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -6.0f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(0.0f, 0.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load models synchronously */
  load_models();

  /* Cubemap texture + async face loading */
  init_cubemap_texture(wgpu_context);
  fetch_cubemap_faces();

  /* Create GPU buffers for models */
  if (state.models_loaded) {
    create_model_buffers(wgpu_context);
  }

  /* Offscreen framebuffers */
  init_offscreen_framebuffer(wgpu_context);
  init_filter_pass_framebuffer(wgpu_context);

  /* UBO */
  init_uniform_buffer(wgpu_context);

  /* Layouts */
  init_vertex_layout();
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);

  /* Bind groups */
  init_bind_groups(wgpu_context);

  /* Pipelines */
  init_pipelines(wgpu_context);

  /* ImGui */
  imgui_overlay_init(wgpu_context);

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  (void)wgpu_context;

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.reflect_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.composition_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.bloom_vert_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.bloom_horz_pipeline);

  /* Release pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.models_pipeline_layout);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.bloom_filter_pipeline_layout);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.composition_pipeline_layout);

  /* Release bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.models_bgl);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bloom_filter_bgl);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.composition_bgl);

  /* Release bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.skybox_bind_group);
  WGPU_RELEASE_RESOURCE(BindGroup, state.object_bind_group);
  destroy_bind_groups();

  /* Release UBO */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer);

  /* Release offscreen resources */
  destroy_offscreen_resources();
  destroy_filter_pass_resources();

  /* Release cubemap */
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap_texture.view);
  if (state.cubemap_texture.handle) {
    wgpuTextureDestroy(state.cubemap_texture.handle);
    wgpuTextureRelease(state.cubemap_texture.handle);
    state.cubemap_texture.handle = NULL;
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.cubemap_texture.sampler);

  /* Release model buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_vertex_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.skybox_index_buffer);
  for (int i = 0; i < NUM_OBJECTS; i++) {
    WGPU_RELEASE_RESOURCE(Buffer, state.object_vertex_buffers[i]);
    WGPU_RELEASE_RESOURCE(Buffer, state.object_index_buffers[i]);
  }

  /* Destroy models */
  gltf_model_destroy(&state.skybox_model);
  for (int i = 0; i < NUM_OBJECTS; i++) {
    gltf_model_destroy(&state.object_models[i]);
  }
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "High Dynamic Range Rendering",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader Code
 * -------------------------------------------------------------------------- */

/* clang-format off */

/* G-Buffer vertex + fragment shader for skybox (type=0) */
static const char* hdr_gbuffer_skybox_shader_wgsl = CODE(
  struct UBO {
    projection       : mat4x4f,
    modelview        : mat4x4f,
    inverseModelview : mat4x4f,
    exposure         : f32,
  }
  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var envSampler   : sampler;
  @group(0) @binding(2) var envTexture   : texture_cube<f32>;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uvw     : vec3f,
  }

  @vertex fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.uvw = in.position;

    /* Strip translation from modelview for skybox */
    let mv3 = mat3x3f(
      ubo.modelview[0].xyz,
      ubo.modelview[1].xyz,
      ubo.modelview[2].xyz
    );
    let pos = mv3 * in.position;
    out.position = ubo.projection * vec4f(pos, 1.0);

    return out;
  }

  struct FragmentOutput {
    @location(0) color0 : vec4f,
    @location(1) color1 : vec4f,
  }

  @fragment fn fs_main(in : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    let normal = normalize(in.uvw);
    let color = textureSample(envTexture, envSampler, normal);

    /* Tone mapping: exposure-based */
    out.color0 = vec4f(vec3f(1.0) - exp(-color.rgb * ubo.exposure), 1.0);

    /* Bloom extraction: luminance threshold */
    let l = dot(out.color0.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let threshold = 0.75;
    if (l > threshold) {
      out.color1 = vec4f(out.color0.rgb, 1.0);
    } else {
      out.color1 = vec4f(0.0, 0.0, 0.0, 1.0);
    }

    return out;
  }
);

/* G-Buffer vertex + fragment shader for reflective object (type=1) */
static const char* hdr_gbuffer_reflect_shader_wgsl = CODE(
  struct UBO {
    projection       : mat4x4f,
    modelview        : mat4x4f,
    inverseModelview : mat4x4f,
    exposure         : f32,
  }
  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var envSampler   : sampler;
  @group(0) @binding(2) var envTexture   : texture_cube<f32>;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uvw     : vec3f,
    @location(1)       posVS   : vec3f,
    @location(2)       normalVS: vec3f,
    @location(3)       viewVec : vec3f,
    @location(4)       lightVec: vec3f,
  }

  @vertex fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.uvw = in.position;

    let posVS = ubo.modelview * vec4f(in.position, 1.0);
    out.posVS = posVS.xyz;
    out.position = ubo.projection * posVS;

    out.normalVS = (mat3x3f(
      ubo.modelview[0].xyz,
      ubo.modelview[1].xyz,
      ubo.modelview[2].xyz
    ) * in.normal);

    /* Light at (0, 5, 5) in WebGPU coordinates (Y negated from Vulkan) */
    let lightPos = vec3f(0.0, 5.0, 5.0);
    out.lightVec = lightPos - posVS.xyz;
    out.viewVec  = -posVS.xyz;

    return out;
  }

  struct FragmentOutput {
    @location(0) color0 : vec4f,
    @location(1) color1 : vec4f,
  }

  @fragment fn fs_main(in : VertexOutput) -> FragmentOutput {
    var out : FragmentOutput;

    let invMV3 = mat3x3f(
      ubo.inverseModelview[0].xyz,
      ubo.inverseModelview[1].xyz,
      ubo.inverseModelview[2].xyz
    );

    let wViewVec = invMV3 * normalize(in.viewVec);
    let normal   = normalize(in.normalVS);
    let wNormal  = invMV3 * normal;

    let lightVec = normalize(in.lightVec);
    let NdotL    = max(dot(normal, lightVec), 0.0);

    let eyeDir  = normalize(in.viewVec);
    let halfVec = normalize(lightVec + eyeDir);
    let NdotH   = max(dot(normal, halfVec), 0.0);
    let NdotV   = max(dot(normal, eyeDir), 0.0);
    let VdotH   = max(dot(eyeDir, halfVec), 0.0);

    /* Geometric attenuation (Cook-Torrance) */
    let NH2    = 2.0 * NdotH;
    let g1     = (NH2 * NdotV) / max(VdotH, 0.001);
    let g2     = (NH2 * NdotL) / max(VdotH, 0.001);
    let geoAtt = min(1.0, min(g1, g2));

    let F0 = 0.6;
    let k  = 0.2;

    /* Fresnel (Schlick approximation) */
    var fresnel = pow(1.0 - VdotH, 5.0);
    fresnel = fresnel * (1.0 - F0) + F0;

    let spec = (fresnel * geoAtt)
             / max(NdotV * NdotL * 3.14, 0.001);

    let envColor = textureSample(envTexture, envSampler,
                                  reflect(-wViewVec, wNormal));

    let color = vec4f(envColor.rgb * NdotL * (k + spec * (1.0 - k)), 1.0);

    /* Tone mapping */
    out.color0 = vec4f(vec3f(1.0) - exp(-color.rgb * ubo.exposure), 1.0);

    /* Bloom extraction */
    let l = dot(out.color0.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let threshold = 0.75;
    if (l > threshold) {
      out.color1 = vec4f(out.color0.rgb, 1.0);
    } else {
      out.color1 = vec4f(0.0, 0.0, 0.0, 1.0);
    }

    return out;
  }
);

/* Bloom horizontal blur (into filterPass) - dir=0 in Vulkan */
static const char* hdr_bloom_horz_shader_wgsl = CODE(
  @group(0) @binding(0) var bloomSampler : sampler;
  @group(0) @binding(1) var bloomTexture : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
  }

  @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.uv = uv;
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let weights = array<f32, 25>(
      0.0024499299678342, 0.0043538453346397, 0.0073599963704157,
      0.0118349786570722, 0.0181026699707781, 0.0263392293891488,
      0.0364543006660986, 0.0479932050577658, 0.0601029809166942,
      0.0715974486241365, 0.0811305381519717, 0.0874493212267511,
      0.0896631113333857,
      0.0874493212267511, 0.0811305381519717, 0.0715974486241365,
      0.0601029809166942, 0.0479932050577658, 0.0364543006660986,
      0.0263392293891488, 0.0181026699707781, 0.0118349786570722,
      0.0073599963704157, 0.0043538453346397, 0.0024499299678342
    );

    let blurScale    = 0.003;
    let blurStrength = 1.0;

    /* Horizontal blur: dir=0 in Vulkan uses inUV.yx with no aspect correction
     * This means the sweep is along the second component of P.
     * With yx swap: P.x = inUV.y, P.y = inUV.x
     * Offset is along P.y (= original x), so this is horizontal. */
    let P = in.uv.yx - vec2f(0.0, f32(25 >> 1) * blurScale);

    var color = vec4f(0.0);
    for (var i = 0; i < 25; i++) {
      let dv = vec2f(0.0, f32(i) * blurScale);
      color += textureSample(bloomTexture, bloomSampler, P + dv)
               * weights[i] * blurStrength;
    }

    return color;
  }
);

/* Bloom vertical blur (onto swapchain, additive) - dir=1 in Vulkan */
static const char* hdr_bloom_vert_shader_wgsl = CODE(
  @group(0) @binding(0) var bloomSampler : sampler;
  @group(0) @binding(1) var bloomTexture : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
  }

  @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.uv = uv;
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let weights = array<f32, 25>(
      0.0024499299678342, 0.0043538453346397, 0.0073599963704157,
      0.0118349786570722, 0.0181026699707781, 0.0263392293891488,
      0.0364543006660986, 0.0479932050577658, 0.0601029809166942,
      0.0715974486241365, 0.0811305381519717, 0.0874493212267511,
      0.0896631113333857,
      0.0874493212267511, 0.0811305381519717, 0.0715974486241365,
      0.0601029809166942, 0.0479932050577658, 0.0364543006660986,
      0.0263392293891488, 0.0181026699707781, 0.0118349786570722,
      0.0073599963704157, 0.0043538453346397, 0.0024499299678342
    );

    let blurScale    = 0.003;
    let blurStrength = 1.0;

    /* Vertical blur: dir=1 in Vulkan adds aspect ratio correction */
    let texDim = vec2f(textureDimensions(bloomTexture, 0));
    let ar     = texDim.y / texDim.x;

    let P = in.uv.yx - vec2f(0.0, f32(25 >> 1) * ar * blurScale);

    var color = vec4f(0.0);
    for (var i = 0; i < 25; i++) {
      let dv = vec2f(0.0, f32(i) * blurScale) * ar;
      color += textureSample(bloomTexture, bloomSampler, P + dv)
               * weights[i] * blurStrength;
    }

    return color;
  }
);

/* Composition shader: fullscreen triangle sampling tone-mapped scene */
static const char* hdr_composition_shader_wgsl = CODE(
  @group(0) @binding(0) var sceneSampler : sampler;
  @group(0) @binding(1) var sceneTexture : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
  }

  @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    var out : VertexOutput;
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.uv = uv;
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return textureSample(sceneTexture, sceneSampler, in.uv);
  }
);

/* clang-format on */
