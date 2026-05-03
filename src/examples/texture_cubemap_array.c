#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

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

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Texture Cubemap Array
 *
 * Loads a cubemap array (3 cubemaps) from three Horizontal Cross PNG images
 * and displays the selected cubemap as a skybox (background) and as a
 * reflection on a selectable 3D object.  The active cubemap is selected via
 * a GUI slider.
 *
 * Face extraction from each cross image is done entirely on the GPU via a
 * fragment-shader render pass (one per face × layer), writing into the
 * correct array layer of a cube-array texture.
 * Mipmaps are generated GPU-side using the built-in mipmap generator.
 *
 * Horizontal Cross layout (4W × 3H, face size = cross_w/4 × cross_h/3):
 *   col:  0    1    2    3
 *   row 0: .   +Y   .    .
 *   row 1: -X  +Z  +X   -Z
 *   row 2: .   -Y   .    .
 *
 * Face order in WebGPU cubemap: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
 * Array layer for face F of cubemap L: L * 6 + F
 *
 * Ported from Sascha Willems' Vulkan example "texturecubemaparray"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/texturecubemaparray
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

/* Main cubemap array rendering shader (skybox + reflection) */
static const char* texture_cubemap_array_shader_wgsl;
/* GPU face extraction: reads from a 2D cross image, writes to one cube face */
static const char* face_extract_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define NUM_CUBEMAP_FACES (6)
#define NUM_ARRAY_LAYERS (3)
#define NUM_TOTAL_LAYERS (NUM_CUBEMAP_FACES * NUM_ARRAY_LAYERS) /* 18 */
/* Horizontal cross source images: each 1024×768, face size = 1024/4 = 256 */
#define CUBEMAP_FACE_SIZE (256)
/* Buffer large enough for each compressed PNG cross file (~650 KB) */
#define CROSS_FILE_BUF_SIZE (2u * 1024u * 1024u)

#define NUM_OBJECTS (4)

/* -------------------------------------------------------------------------- *
 * Uniform data (must match WGSL layout, 16-byte aligned)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 projection;         /* 64 bytes, offset   0 */
  mat4 model_view;         /* 64 bytes, offset  64 */
  mat4 inverse_model_view; /* 64 bytes, offset 128 */
  float lod_bias;          /*  4 bytes, offset 192 */
  int32_t cube_map_index;  /*  4 bytes, offset 196 */
  float _pad[2];           /*  8 bytes, offset 200 */
} uniform_data_t;          /* 208 bytes total */

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Skybox model */
  gltf_model_t skybox_model;
  bool skybox_model_loaded;
  struct {
    WGPUBuffer vertex;
    WGPUBuffer index;
  } skybox_buffers;

  /* Selectable 3D object models */
  gltf_model_t objects[NUM_OBJECTS];
  bool objects_loaded[NUM_OBJECTS];
  struct {
    WGPUBuffer vertex;
    WGPUBuffer index;
  } object_buffers[NUM_OBJECTS];

  /* Cubemap array texture */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
    WGPUSampler sampler;
    bool is_ready;
  } cubemap;

  /* Per-layer raw PNG file buffers for sokol_fetch */
  uint8_t* cross_file_buf[NUM_ARRAY_LAYERS];
  /* Decoded cross image pixels per layer (freed after GPU upload) */
  uint8_t* cross_pixels[NUM_ARRAY_LAYERS];
  int cross_width[NUM_ARRAY_LAYERS];
  int cross_height[NUM_ARRAY_LAYERS];
  int cross_loaded_count; /* incremented atomically by sfetch callbacks */

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo;

  /* Depth texture (recreated on resize) */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Bind group / layout */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  bool bind_group_dirty; /* rebuild when cubemap is ready */

  /* Render pipelines */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline skybox_pipeline;
  WGPURenderPipeline reflect_pipeline;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI / settings */
  struct {
    bool display_skybox;
    float lod_bias;
    int32_t object_index;
    int32_t cube_map_index;
  } settings;
  const char* object_names[NUM_OBJECTS];

  uint64_t last_frame_time;
  bool initialized;
} state = {
  /* clang-format off */
  .settings = {
    .display_skybox  = true,
    .lod_bias        = 0.0f,
    .object_index    = 0,
    .cube_map_index  = 1,
  },
  .object_names = {
    "Sphere",
    "Teapot",
    "Torusknot",
    "Venus",
  },
  .color_attachment = {
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
  },
  .depth_stencil_attachment = {
    .depthLoadOp     = WGPULoadOp_Clear,
    .depthStoreOp    = WGPUStoreOp_Store,
    .depthClearValue = 1.0f,
    .stencilLoadOp   = WGPULoadOp_Undefined,
    .stencilStoreOp  = WGPUStoreOp_Undefined,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  /* clang-format on */
};

/* -------------------------------------------------------------------------- *
 * Depth texture (recreated on resize)
 * -------------------------------------------------------------------------- */

static void init_depth_texture(wgpu_context_t* wgpu_context)
{
  if (state.depth.view) {
    wgpuTextureViewRelease(state.depth.view);
    state.depth.view = NULL;
  }
  if (state.depth.texture) {
    wgpuTextureDestroy(state.depth.texture);
    wgpuTextureRelease(state.depth.texture);
    state.depth.texture = NULL;
  }

  state.depth.texture = wgpuDeviceCreateTexture(
    wgpu_context->device, &(WGPUTextureDescriptor){
                            .label         = STRVIEW("Depth texture"),
                            .usage         = WGPUTextureUsage_RenderAttachment,
                            .dimension     = WGPUTextureDimension_2D,
                            .size          = {(uint32_t)wgpu_context->width,
                                              (uint32_t)wgpu_context->height, 1},
                            .format        = WGPUTextureFormat_Depth24Plus,
                            .mipLevelCount = 1,
                            .sampleCount   = 1,
                          });
  ASSERT(state.depth.texture);

  state.depth.view = wgpuTextureCreateView(state.depth.texture, NULL);
  ASSERT(state.depth.view);

  state.depth_stencil_attachment.view = state.depth.view;
}

/* -------------------------------------------------------------------------- *
 * Cubemap array texture
 * -------------------------------------------------------------------------- */

/**
 * Extract all 18 cubemap faces (6 faces × 3 array layers) from the three
 * horizontal cross images that were decoded into state.cross_pixels[].
 *
 * Algorithm per layer:
 *  1. Upload the layer's cross image as a temporary 2D source texture.
 *  2. Run one render pass per face, writing to array layer = layer*6 + face.
 *  3. Free the CPU pixel buffer for that layer.
 * After all layers are done:
 *  4. Generate the full mip chain with wgpu_generate_mipmaps (CubeArray).
 *  5. Create the permanent cube-array view and sampler.
 *  6. Release all temporary resources.
 */
static void extract_cubemap_faces_from_crosses(wgpu_context_t* wgpu_context)
{
  /* Use the dimensions from layer 0; all three cross images are the same size
   */
  const uint32_t cross_w   = (uint32_t)state.cross_width[0];
  const uint32_t face_size = cross_w / 4u; /* = cross_h / 3 */

  /* ---------------------------------------------------------------------- *
   * 2. Create the destination cubemap array texture
   * ---------------------------------------------------------------------- */
  const uint32_t mip_count = wgpu_texture_mip_level_count(face_size, face_size);

  state.cubemap.handle = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Cubemap array texture"),
      .usage = WGPUTextureUsage_TextureBinding
               | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {face_size, face_size, NUM_TOTAL_LAYERS},
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = mip_count,
      .sampleCount   = 1,
    });
  ASSERT(state.cubemap.handle);

  /* ---------------------------------------------------------------------- *
   * 3. Build the face extraction render pipeline (shared for all layers)
   * ---------------------------------------------------------------------- */
  WGPUShaderModule extract_shader
    = wgpu_create_shader_module(wgpu_context->device, face_extract_shader_wgsl);

  WGPUBindGroupLayout extract_bgl = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Extract BGL"),
      .entryCount = 2,
      .entries    = (WGPUBindGroupLayoutEntry[]){
        {
          .binding    = 0,
          .visibility = WGPUShaderStage_Fragment,
          .sampler    = {.type = WGPUSamplerBindingType_Filtering},
        },
        {
          .binding    = 1,
          .visibility = WGPUShaderStage_Fragment,
          .texture = {
            .sampleType    = WGPUTextureSampleType_Float,
            .viewDimension = WGPUTextureViewDimension_2D,
            .multisampled  = false,
          },
        },
      },
    });
  ASSERT(extract_bgl);

  WGPUPipelineLayout extract_pl = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Extract PL"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &extract_bgl,
                          });
  ASSERT(extract_pl);

  WGPURenderPipeline extract_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Face extract pipeline"),
      .layout = extract_pl,
      .vertex = (WGPUVertexState){
        .module     = extract_shader,
        .entryPoint = STRVIEW("vs_extract"),
      },
      .primitive = (WGPUPrimitiveState){
        .topology = WGPUPrimitiveTopology_TriangleList,
      },
      .fragment = &(WGPUFragmentState){
        .module      = extract_shader,
        .entryPoint  = STRVIEW("fs_extract"),
        .targetCount = 1,
        .targets     = &(WGPUColorTargetState){
          .format    = WGPUTextureFormat_RGBA8Unorm,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .multisample = (WGPUMultisampleState){.count = 1, .mask = 0xFFFFFFFF},
    });
  ASSERT(extract_pipeline);

  /* A simple linear sampler for the extraction blit */
  WGPUSampler extract_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Cross extract sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                            .maxAnisotropy = 1,
                          });
  ASSERT(extract_sampler);

  /* ---------------------------------------------------------------------- *
   * 4. For each layer: upload cross → extract 6 faces
   * ---------------------------------------------------------------------- */

  /* Keep src textures alive until after the submit (destroyed below) */
  WGPUTexture src_textures[NUM_ARRAY_LAYERS] = {NULL};

  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  for (uint32_t layer = 0; layer < NUM_ARRAY_LAYERS; ++layer) {
    const uint32_t lw = (uint32_t)state.cross_width[layer];
    const uint32_t lh = (uint32_t)state.cross_height[layer];

    /* Upload this layer's cross pixels to a temporary 2D source texture */
    WGPUTexture src_tex = wgpuDeviceCreateTexture(
      wgpu_context->device,
      &(WGPUTextureDescriptor){
        .label     = STRVIEW("Cross source texture"),
        .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size      = {lw, lh, 1},
        .format    = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount   = 1,
      });
    ASSERT(src_tex);
    src_textures[layer] = src_tex;

    wgpuQueueWriteTexture(
      wgpu_context->queue,
      &(WGPUTexelCopyTextureInfo){
        .texture  = src_tex,
        .mipLevel = 0,
        .origin   = {0, 0, 0},
        .aspect   = WGPUTextureAspect_All,
      },
      state.cross_pixels[layer], (size_t)(lw * lh * 4u),
      &(WGPUTexelCopyBufferLayout){
        .offset       = 0,
        .bytesPerRow  = lw * 4u,
        .rowsPerImage = lh,
      },
      &(WGPUExtent3D){.width = lw, .height = lh, .depthOrArrayLayers = 1});

    /* Free CPU pixels now that they are on the GPU */
    image_free(state.cross_pixels[layer]);
    state.cross_pixels[layer] = NULL;

    WGPUTextureView src_view = wgpuTextureCreateView(src_tex, NULL);
    ASSERT(src_view);

    /* Build bind group for this source texture */
    WGPUBindGroup extract_bg = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Extract BG"),
        .layout     = extract_bgl,
        .entryCount = 2,
        .entries    = (WGPUBindGroupEntry[]){
          {.binding = 0, .sampler = extract_sampler},
          {.binding = 1, .textureView = src_view},
        },
      });
    ASSERT(extract_bg);

    /* Extract 6 faces for this layer */
    for (uint32_t face = 0; face < NUM_CUBEMAP_FACES; ++face) {
      const uint32_t array_layer = layer * NUM_CUBEMAP_FACES + face;

      /* 2D view targeting this face/layer at mip 0 */
      WGPUTextureView face_view = wgpuTextureCreateView(
        state.cubemap.handle, &(WGPUTextureViewDescriptor){
                                .label           = STRVIEW("Face dst view"),
                                .format          = WGPUTextureFormat_RGBA8Unorm,
                                .dimension       = WGPUTextureViewDimension_2D,
                                .baseMipLevel    = 0,
                                .mipLevelCount   = 1,
                                .baseArrayLayer  = array_layer,
                                .arrayLayerCount = 1,
                                .aspect          = WGPUTextureAspect_All,
                              });
      ASSERT(face_view);

      WGPURenderPassColorAttachment color_att = {
        .view       = face_view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 1},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      };
      WGPURenderPassDescriptor rp_desc = {
        .label                = STRVIEW("Face extract pass"),
        .colorAttachmentCount = 1,
        .colorAttachments     = &color_att,
      };

      WGPURenderPassEncoder pass
        = wgpuCommandEncoderBeginRenderPass(enc, &rp_desc);
      wgpuRenderPassEncoderSetPipeline(pass, extract_pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, extract_bg, 0, NULL);
      /* instance_index = face (passed via firstInstance) */
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, face);
      wgpuRenderPassEncoderEnd(pass);
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)
      WGPU_RELEASE_RESOURCE(TextureView, face_view)
    }

    WGPU_RELEASE_RESOURCE(BindGroup, extract_bg)
    WGPU_RELEASE_RESOURCE(TextureView, src_view)
    /* Do NOT destroy src_tex here — it is still referenced by the recorded
     * commands.  It will be destroyed after the submit below. */
  }

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, enc)
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd)

  /* Safe to destroy source textures now that the GPU work is submitted */
  for (uint32_t layer = 0; layer < NUM_ARRAY_LAYERS; ++layer) {
    if (src_textures[layer]) {
      wgpuTextureDestroy(src_textures[layer]);
      WGPU_RELEASE_RESOURCE(Texture, src_textures[layer])
    }
  }

  /* ---------------------------------------------------------------------- *
   * 5. Generate full mip chain GPU-side
   * ---------------------------------------------------------------------- */
  wgpu_generate_mipmaps(wgpu_context, state.cubemap.handle,
                        WGPU_MIPMAP_VIEW_CUBE_ARRAY);

  /* ---------------------------------------------------------------------- *
   * 6. Create the permanent cube-array texture view and sampler
   * ---------------------------------------------------------------------- */
  state.cubemap.view = wgpuTextureCreateView(
    state.cubemap.handle, &(WGPUTextureViewDescriptor){
                            .label         = STRVIEW("Cubemap array view"),
                            .format        = WGPUTextureFormat_RGBA8Unorm,
                            .dimension     = WGPUTextureViewDimension_CubeArray,
                            .baseMipLevel  = 0,
                            .mipLevelCount = mip_count,
                            .baseArrayLayer  = 0,
                            .arrayLayerCount = NUM_TOTAL_LAYERS,
                            .aspect          = WGPUTextureAspect_All,
                          });
  ASSERT(state.cubemap.view);

  state.cubemap.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Cubemap array sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .lodMinClamp   = 0.0f,
                            .lodMaxClamp   = (float)mip_count,
                            .compare       = WGPUCompareFunction_Undefined,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.cubemap.sampler);

  /* ---------------------------------------------------------------------- *
   * 7. Release shared extraction resources
   * ---------------------------------------------------------------------- */
  WGPU_RELEASE_RESOURCE(RenderPipeline, extract_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, extract_pl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, extract_bgl)
  WGPU_RELEASE_RESOURCE(ShaderModule, extract_shader)
  WGPU_RELEASE_RESOURCE(Sampler, extract_sampler)
}

/* -------------------------------------------------------------------------- *
 * Asynchronous horizontal-cross cubemap array loading (3 PNG fetches)
 * -------------------------------------------------------------------------- */

static void cross_fetch_callback(const sfetch_response_t* response)
{
  /* response->user_data holds the layer index (0, 1, or 2) */
  const uint32_t layer = *(const uint32_t*)response->user_data;

  if (!response->fetched || !response->data.ptr || !response->data.size) {
    printf(
      "[texture_cubemap_array] Failed to fetch cross image %u, error: %d\n",
      layer, response->error_code);
    free((void*)response->buffer.ptr);
    return;
  }

  int width = 0, height = 0, channels = 0;
  state.cross_pixels[layer]
    = image_pixels_from_memory(response->data.ptr, (int)response->data.size,
                               &width, &height, &channels, 4);
  free((void*)response->buffer.ptr);
  if (!state.cross_pixels[layer]) {
    printf("[texture_cubemap_array] Failed to decode cross PNG for layer %u\n",
           layer);
    return;
  }

  state.cross_width[layer]  = width;
  state.cross_height[layer] = height;
  state.cross_loaded_count++;

  printf(
    "[texture_cubemap_array] Layer %u cross image decoded: %dx%d (%d ch)\n",
    layer, width, height, channels);
}

static void fetch_cubemap_crosses(void)
{
  static const char* cross_paths[NUM_ARRAY_LAYERS] = {
    "assets/textures/cubemaps/cubemap_array_layer_0.png",
    "assets/textures/cubemaps/cubemap_array_layer_1.png",
    "assets/textures/cubemaps/cubemap_array_layer_2.png",
  };
  /* Store layer index in user_data so the callback knows which slot to fill */
  static const uint32_t layer_indices[NUM_ARRAY_LAYERS] = {0, 1, 2};

  for (uint32_t i = 0; i < NUM_ARRAY_LAYERS; ++i) {
    state.cross_file_buf[i] = (uint8_t*)malloc(CROSS_FILE_BUF_SIZE);
    sfetch_send(&(sfetch_request_t){
      .path     = cross_paths[i],
      .callback = cross_fetch_callback,
      .buffer   = {.ptr = state.cross_file_buf[i], .size = CROSS_FILE_BUF_SIZE},
      .user_data = SFETCH_RANGE(layer_indices[i]),
    });
  }
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void create_model_gpu_buffers(wgpu_context_t* wgpu_context,
                                     gltf_model_t* mdl, WGPUBuffer* vb_out,
                                     WGPUBuffer* ib_out, const char* label_vb,
                                     const char* label_ib)
{
  uint32_t vb_size = mdl->vertex_count * (uint32_t)sizeof(gltf_vertex_t);
  uint32_t ib_size = mdl->index_count * (uint32_t)sizeof(uint32_t);

  *vb_out = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = {.data = label_vb, .length = strlen(label_vb)},
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = vb_size,
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, *vb_out, 0, mdl->vertices, vb_size);

  *ib_out = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = {.data = label_ib, .length = strlen(label_ib)},
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = ib_size,
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, *ib_out, 0, mdl->indices, ib_size);
}

static void load_models(wgpu_context_t* wgpu_context)
{
  gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices,
  };

  /* Skybox cube */
  state.skybox_model_loaded = gltf_model_load_from_file_ext(
    &state.skybox_model, "assets/models/cube.gltf", 1.0f, &desc);
  if (state.skybox_model_loaded) {
    create_model_gpu_buffers(
      wgpu_context, &state.skybox_model, &state.skybox_buffers.vertex,
      &state.skybox_buffers.index, "Skybox VB", "Skybox IB");
  }

  /* Objects */
  static const char* object_paths[NUM_OBJECTS] = {
    "assets/models/sphere.gltf",
    "assets/models/teapot.gltf",
    "assets/models/torusknot.gltf",
    "assets/models/venus.gltf",
  };

  for (int i = 0; i < NUM_OBJECTS; ++i) {
    state.objects_loaded[i] = gltf_model_load_from_file_ext(
      &state.objects[i], object_paths[i], 1.0f, &desc);
    if (state.objects_loaded[i]) {
      char lbl_vb[64], lbl_ib[64];
      snprintf(lbl_vb, sizeof(lbl_vb), "Object[%d] VB", i);
      snprintf(lbl_ib, sizeof(lbl_ib), "Object[%d] IB", i);
      create_model_gpu_buffers(wgpu_context, &state.objects[i],
                               &state.object_buffers[i].vertex,
                               &state.object_buffers[i].index, lbl_vb, lbl_ib);
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(uniform_data_t),
      .mappedAtCreation = false,
    });
  ASSERT(state.uniform_buffer);
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;
  glm_perspective(glm_rad(60.0f), aspect, 0.1f, 256.0f, state.ubo.projection);

  glm_mat4_copy(state.camera.matrices.view, state.ubo.model_view);
  glm_mat4_inv(state.ubo.model_view, state.ubo.inverse_model_view);
  state.ubo.lod_bias       = state.settings.lod_bias;
  state.ubo.cube_map_index = state.settings.cube_map_index;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, &state.ubo,
                       sizeof(state.ubo));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout and bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      /* Binding 0: UBO (visible to both vertex and fragment) */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(uniform_data_t),
      },
    },
    [1] = {
      /* Binding 1: Sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [2] = {
      /* Binding 2: Cubemap array texture */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_CubeArray,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Cubemap array BGL"),
                            .entryCount = ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
  ASSERT(state.bind_group_layout);
}

static void create_bind_group(wgpu_context_t* wgpu_context)
{
  if (state.bind_group) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
    state.bind_group = NULL;
  }

  WGPUBindGroupEntry entries[3] = {
    [0] = {
      .binding = 0,
      .buffer  = state.uniform_buffer,
      .offset  = 0,
      .size    = sizeof(uniform_data_t),
    },
    [1] = {
      .binding = 1,
      .sampler = state.cubemap.sampler,
    },
    [2] = {
      .binding     = 2,
      .textureView = state.cubemap.view,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Cubemap array BG"),
                            .layout     = state.bind_group_layout,
                            .entryCount = ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
  ASSERT(state.bind_group);

  state.bind_group_dirty = false;
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Cubemap array pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout);

  WGPUShaderModule shader = wgpu_create_shader_module(
    wgpu_context->device, texture_cubemap_array_shader_wgsl);

  /* Vertex attributes: position (skybox + reflect) and normal (reflect only).
   */
  WGPUVertexAttribute attrs[2] = {
    [0] = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
    },
    [1] = {
      .shaderLocation = 1,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, normal),
    },
  };

  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(attrs),
    .attributes     = attrs,
  };

  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = NULL,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUDepthStencilState depth_no_write = {
    .format            = WGPUTextureFormat_Depth24Plus,
    .depthWriteEnabled = WGPUOptionalBool_False,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  WGPUDepthStencilState depth_write = {
    .format            = WGPUTextureFormat_Depth24Plus,
    .depthWriteEnabled = WGPUOptionalBool_True,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  /* Skybox pipeline: cull front faces, no depth write */
  state.skybox_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Skybox pipeline"),
      .layout = state.pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = shader,
        .entryPoint  = STRVIEW("vs_skybox"),
        .bufferCount = 1,
        .buffers     = &vb_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Front,
      },
      .depthStencil = &depth_no_write,
      .fragment = &(WGPUFragmentState){
        .module      = shader,
        .entryPoint  = STRVIEW("fs_skybox"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.skybox_pipeline);

  /* Reflect pipeline: cull back faces, depth write enabled */
  state.reflect_pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Reflect pipeline"),
      .layout = state.pipeline_layout,
      .vertex = (WGPUVertexState){
        .module      = shader,
        .entryPoint  = STRVIEW("vs_reflect"),
        .bufferCount = 1,
        .buffers     = &vb_layout,
      },
      .primitive = (WGPUPrimitiveState){
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_Back,
      },
      .depthStencil = &depth_write,
      .fragment = &(WGPUFragmentState){
        .module      = shader,
        .entryPoint  = STRVIEW("fs_reflect"),
        .targetCount = 1,
        .targets     = &color_target,
      },
      .multisample = (WGPUMultisampleState){
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });
  ASSERT(state.reflect_pipeline);

  WGPU_RELEASE_RESOURCE(ShaderModule, shader)
}

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass, gltf_model_t* mdl,
                       WGPUBuffer vb, WGPUBuffer ib)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vb, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(pass, ib, WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  for (uint32_t n = 0; n < mdl->linear_node_count; ++n) {
    gltf_node_t* node = mdl->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    for (uint32_t p = 0; p < node->mesh->primitive_count; ++p) {
      gltf_primitive_t* prim = &node->mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(pass, prim->index_count, 1,
                                         prim->first_index, 0, 0);
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
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Cube Map Array Textures", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  igCheckbox("Display skybox", &state.settings.display_skybox);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_int("Cube map", &state.settings.cube_map_index, 0,
                             NUM_ARRAY_LAYERS - 1);
    imgui_overlay_combo_box("##object_select", &state.settings.object_index,
                            state.object_names, NUM_OBJECTS);
    imgui_overlay_slider_float("LOD bias", &state.settings.lod_bias, 0.0f, 8.0f,
                               "%.1f");
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input event handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    camera_set_perspective(
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
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
  stm_setup();
  state.last_frame_time = stm_now();

  /* Camera: lookat, position at (0,0,-4) */
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.rotation_speed = 0.25f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -4.0f});
  camera_set_rotation(&state.camera, (vec3){0.0f, 0.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Sokol fetch — 3 lanes for parallel PNG downloads */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 1,
    .num_lanes    = 3,
    .logger.func  = slog_func,
  });

  /* Start async fetches for all three horizontal-cross PNGs */
  fetch_cubemap_crosses();

  /* Init GPU resources */
  init_depth_texture(wgpu_context);
  load_models(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_pipelines(wgpu_context);

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
  uint64_t now          = stm_now();
  float delta_time      = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Pump sokol-fetch */
  sfetch_dowork();

  /* When all 3 cross PNGs are decoded, extract faces and build the cubemap */
  if (!state.cubemap.is_ready && state.cross_loaded_count == NUM_ARRAY_LAYERS) {
    extract_cubemap_faces_from_crosses(wgpu_context);
    state.cubemap.is_ready = true;
    state.bind_group_dirty = true;
  }

  /* (Re)create the bind group when cubemap is ready */
  if (state.bind_group_dirty && state.cubemap.is_ready) {
    create_bind_group(wgpu_context);
  }

  /* Update camera */
  camera_update(&state.camera, delta_time);

  /* Update uniforms */
  update_uniform_buffer(wgpu_context);

  /* --- Render --- */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth.view;

  WGPUCommandEncoder enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Build ImGui frame before the render pass */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(enc, &state.render_pass_descriptor);

  wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, (uint32_t)wgpu_context->width,
                                      (uint32_t)wgpu_context->height);

  if (state.bind_group) {
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_group, 0, NULL);

    /* Skybox */
    if (state.settings.display_skybox && state.skybox_model_loaded) {
      wgpuRenderPassEncoderSetPipeline(pass, state.skybox_pipeline);
      draw_model(pass, &state.skybox_model, state.skybox_buffers.vertex,
                 state.skybox_buffers.index);
    }

    /* Reflected object */
    int32_t idx = state.settings.object_index;
    if (idx >= 0 && idx < NUM_OBJECTS && state.objects_loaded[idx]) {
      wgpuRenderPassEncoderSetPipeline(pass, state.reflect_pipeline);
      draw_model(pass, &state.objects[idx], state.object_buffers[idx].vertex,
                 state.object_buffers[idx].index);
    }
  }

  wgpuRenderPassEncoderEnd(pass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass)

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, enc)
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd)

  /* Render ImGui overlay in its own pass (after scene submit) */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown / cleanup
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Cubemap array GPU resources */
  WGPU_RELEASE_RESOURCE(TextureView, state.cubemap.view)
  if (state.cubemap.handle) {
    wgpuTextureDestroy(state.cubemap.handle);
    WGPU_RELEASE_RESOURCE(Texture, state.cubemap.handle)
  }
  WGPU_RELEASE_RESOURCE(Sampler, state.cubemap.sampler)

  /* Free any pending CPU pixel buffers (should already be NULL after loading)
   */
  for (int i = 0; i < NUM_ARRAY_LAYERS; ++i) {
    if (state.cross_pixels[i]) {
      image_free(state.cross_pixels[i]);
      state.cross_pixels[i] = NULL;
    }
  }

  /* Models */
  if (state.skybox_model_loaded) {
    gltf_model_destroy(&state.skybox_model);
    WGPU_RELEASE_RESOURCE(Buffer, state.skybox_buffers.vertex)
    WGPU_RELEASE_RESOURCE(Buffer, state.skybox_buffers.index)
  }
  for (int i = 0; i < NUM_OBJECTS; ++i) {
    if (state.objects_loaded[i]) {
      gltf_model_destroy(&state.objects[i]);
      WGPU_RELEASE_RESOURCE(Buffer, state.object_buffers[i].vertex)
      WGPU_RELEASE_RESOURCE(Buffer, state.object_buffers[i].index)
    }
  }

  /* Depth */
  WGPU_RELEASE_RESOURCE(TextureView, state.depth.view)
  if (state.depth.texture) {
    wgpuTextureDestroy(state.depth.texture);
    WGPU_RELEASE_RESOURCE(Texture, state.depth.texture)
  }

  /* Buffers, layouts, pipelines */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.skybox_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.reflect_pipeline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
}

/* -------------------------------------------------------------------------- *
 * Main
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Cube Map Array Textures",
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
static const char* face_extract_shader_wgsl = CODE(
  /*
   * GPU face extraction from a Horizontal Cross source texture.
   *
   * Cross layout (4 columns x 3 rows):
   *   col:  0    1    2    3
   *   row 0: .   +Y   .    .
   *   row 1: -X  +Z  +X   -Z
   *   row 2: .   -Y   .    .
   *
   * WebGPU cubemap face indices: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
   *
   * u_off[face] = left edge of the face in the cross, normalized [0,1]
   * v_off[face] = top  edge of the face in the cross, normalized [0,1]
   * face_du = 1/4  (face occupies 1/4 of the cross width)
   * face_dv = 1/3  (face occupies 1/3 of the cross height)
   *
   * The face index is passed via @builtin(instance_index) (firstInstance).
   */

  // u_off: left edge (in [0,1]) of each face in the cross image
  // Face order: +X(0)  -X(1)  +Y(2)  -Y(3)  +Z(4)  -Z(5)
  const u_off = array<f32, 6>(0.50, 0.00, 0.25, 0.25, 0.25, 0.75);
  // v_off: top edge (in [0,1]) of each face in the cross image
  const v_off = array<f32, 6>(1.0/3.0, 1.0/3.0, 0.0, 2.0/3.0, 1.0/3.0, 1.0/3.0);

  struct VSOutput {
    @builtin(position)                        pos  : vec4f,
    @location(0)                              uv   : vec2f,
    @location(1) @interpolate(flat, either)   face : u32,
  }

  @vertex
  fn vs_extract(
    @builtin(vertex_index)   vi   : u32,
    @builtin(instance_index) face : u32,
  ) -> VSOutput {
    var pts = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    let xy = pts[vi];
    var out : VSOutput;
    out.pos  = vec4f(xy, 0.0, 1.0);
    out.uv   = xy * vec2f(0.5, -0.5) + vec2f(0.5);
    out.face = face;
    return out;
  }

  @group(0) @binding(0) var src_sampler : sampler;
  @group(0) @binding(1) var src_tex     : texture_2d<f32>;

  @fragment
  fn fs_extract(in : VSOutput) -> @location(0) vec4f {
    let u = u_off[in.face] + in.uv.x * (1.0 / 4.0);
    let v = v_off[in.face] + in.uv.y * (1.0 / 3.0);
    return textureSample(src_tex, src_sampler, vec2f(u, v));
  }
);
// clang-format on

/* -------------------------------------------------------------------------- *
 * Main cubemap array rendering shader (skybox + reflection)
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* texture_cubemap_array_shader_wgsl = CODE(
  /* -------------- Shared UBO -------------------------------------------- */
  struct UBO {
    projection      : mat4x4f,
    model_view      : mat4x4f,
    inv_model_view  : mat4x4f,
    lod_bias        : f32,
    cube_map_index  : i32,
  }

  @group(0) @binding(0) var<uniform> ubo            : UBO;
  @group(0) @binding(1) var          cubeSampler     : sampler;
  @group(0) @binding(2) var          cubeArrayTexture : texture_cube_array<f32>;

  /* -------------- Skybox ------------------------------------------------- */
  struct SkyboxVertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uvw      : vec3f,
  }

  @vertex
  fn vs_skybox(@location(0) in_pos : vec3f) -> SkyboxVertexOutput {
    var out : SkyboxVertexOutput;

    // Use position as cubemap sampling direction.
    // Negate X to match the cubemap coordinate convention: the cglm right-handed
    // lookat matrix maps world +X to screen-left, so without negation the faces
    // appear horizontally mirrored.  Only X needs negation because WebGPU uses
    // Y-up NDC (no projection Y-flip, unlike the Vulkan port's flip_y=true).
    out.uvw = vec3f(-in_pos.x, in_pos.y, in_pos.z);

    // Remove translation from view matrix — only keep rotation
    let view_rot = mat4x4f(
      ubo.model_view[0],
      ubo.model_view[1],
      ubo.model_view[2],
      vec4f(0.0, 0.0, 0.0, 1.0)
    );
    out.position = ubo.projection * view_rot * vec4f(in_pos, 1.0);
    return out;
  }

  @fragment
  fn fs_skybox(in : SkyboxVertexOutput) -> @location(0) vec4f {
    return textureSample(cubeArrayTexture, cubeSampler, in.uvw,
                         u32(ubo.cube_map_index));
  }

  /* -------------- Reflect ------------------------------------------------ */
  struct ReflectVertexOutput {
    @builtin(position) position  : vec4f,
    @location(0)       pos       : vec3f,
    @location(1)       normal    : vec3f,
    @location(2)       view_vec  : vec3f,
    @location(3)       light_vec : vec3f,
  }

  @vertex
  fn vs_reflect(
    @location(0) in_pos    : vec3f,
    @location(1) in_normal : vec3f
  ) -> ReflectVertexOutput {
    var out : ReflectVertexOutput;

    out.position = ubo.projection * ubo.model_view * vec4f(in_pos, 1.0);

    // Position and normal in view space
    out.pos    = (ubo.model_view * vec4f(in_pos, 1.0)).xyz;
    let mv3   = mat3x3f(
      ubo.model_view[0].xyz,
      ubo.model_view[1].xyz,
      ubo.model_view[2].xyz
    );
    out.normal = mv3 * in_normal;

    let light_pos  = vec3f(0.0, -5.0, 5.0);
    out.light_vec  = light_pos - out.pos;
    out.view_vec   = -out.pos;

    return out;
  }

  @fragment
  fn fs_reflect(in : ReflectVertexOutput) -> @location(0) vec4f {
    // Reflection direction in view space
    let cI = normalize(in.pos);
    var cR = reflect(cI, normalize(in.normal));

    // Transform reflection vector back to world space
    cR = (ubo.inv_model_view * vec4f(cR, 0.0)).xyz;

    // Sample the cubemap array with an optional LOD bias.
    // Negate X for the same reason as vs_skybox: cglm right-handed view space
    // maps world +X to screen-left, so the reflection direction needs the same
    // X-flip to avoid a horizontally mirrored reflection.
    let cR_sample = vec3f(-cR.x, cR.y, cR.z);
    let color = textureSampleBias(cubeArrayTexture, cubeSampler, cR_sample,
                                  u32(ubo.cube_map_index), ubo.lod_bias);

    // Simple Phong-like lighting
    let N = normalize(in.normal);
    let L = normalize(in.light_vec);
    let V = normalize(in.view_vec);
    let R = reflect(-L, N);

    let ambient  = vec3f(0.5) * color.rgb;
    let diffuse  = max(dot(N, L), 0.0) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 16.0) * vec3f(0.5);

    return vec4f(ambient + diffuse * color.rgb + specular, 1.0);
  }
);
// clang-format on
