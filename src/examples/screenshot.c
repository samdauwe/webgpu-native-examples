/* -------------------------------------------------------------------------- *
 * WebGPU Example - Taking Screenshots
 *
 * Demonstrates how to capture the current framebuffer contents and save them
 * to disk as a PPM image file using WebGPU's CopyTextureToBuffer API.
 *
 * The scene renders a Chinese Dragon glTF model with a simple Phong shading
 * pipeline (per-vertex color + ambient/diffuse/specular lighting in view
 * space). The scene is rendered to an offscreen RGBA8Unorm texture with
 * CopySrc usage, then blitted to the swapchain each frame.
 *
 * On screenshot request (GUI button): the offscreen texture is copied to a
 * CPU-mappable staging buffer, mapped asynchronously, and written out as a
 * binary PPM file ("screenshot.ppm") in the working directory.
 *
 * Rendering passes:
 *   1. Scene pass: Dragon model → offscreen scene_tex (RGBA8Unorm + CopySrc)
 *   2. Blit pass:  Fullscreen triangle samples scene_tex → swapchain
 *   Screenshot copy (when requested):
 *      CopyTextureToBuffer(scene_tex → staging) → wgpuBufferMapAsync → PPM
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations — defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* screenshot_mesh_shader_wgsl;
static const char* screenshot_blit_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

/* Depth buffer format for the offscreen scene pass */
#define SCENE_DEPTH_FORMAT WGPUTextureFormat_Depth24PlusStencil8

/* WebGPU requires bytes-per-row for texture copies to be a multiple of 256 */
#define WGPU_COPY_BYTES_PER_ROW_ALIGNMENT (256u)

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Chinese Dragon model */
  gltf_model_t dragon_model;
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  bool model_loaded;

  /* Offscreen scene texture — rendered to each frame, copied for screenshots.
   * Usage: RenderAttachment | CopySrc | TextureBinding                      */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
  } scene_tex;

  /* Depth texture for the scene render pass */
  struct {
    WGPUTexture handle;
    WGPUTextureView view;
  } depth_tex;

  /* Blit sampler (shared between frames) */
  WGPUSampler blit_sampler;

  /* Uniform buffer: projection + view + model matrices (3 × mat4 = 192 B) */
  WGPUBuffer ubo;
  struct {
    mat4 projection;
    mat4 view;
    mat4 model;
  } ubo_data;

  /* Bind group layouts */
  WGPUBindGroupLayout scene_bgl; /* Mesh pass: UBO at binding 0           */
  WGPUBindGroupLayout blit_bgl;  /* Blit pass: sampler@0 + texture@1      */

  /* Pipeline layouts */
  WGPUPipelineLayout scene_pipeline_layout;
  WGPUPipelineLayout blit_pipeline_layout;

  /* Bind groups */
  WGPUBindGroup scene_bg;
  WGPUBindGroup blit_bg;

  /* Pipelines */
  WGPURenderPipeline scene_pipeline;
  WGPURenderPipeline blit_pipeline;

  /* Scene render pass descriptor (targets scene_tex) */
  WGPURenderPassColorAttachment scene_color_att;
  WGPURenderPassDepthStencilAttachment scene_depth_att;
  WGPURenderPassDescriptor scene_render_pass;

  /* Blit render pass descriptor (targets swapchain, no depth) */
  WGPURenderPassColorAttachment blit_color_att;
  WGPURenderPassDescriptor blit_render_pass;

  /* Screenshot state */
  bool screenshot_requested;    /* Set by GUI "Take Screenshot" button  */
  bool screenshot_saving;       /* True while buffer mapping is active  */
  bool screenshot_saved;        /* Set to true after PPM is written     */
  WGPUBuffer screenshot_buffer; /* Staging buffer (NULL when idle)       */
  uint32_t screenshot_width;
  uint32_t screenshot_height;
  uint32_t screenshot_bpr; /* Bytes per row (aligned to 256)       */

  /* Current window size — used to detect resize */
  int last_width;
  int last_height;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  /* clang-format off */
  .scene_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025f, 0.025f, 0.025f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .scene_depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .scene_render_pass = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.scene_color_att,
    .depthStencilAttachment = &state.scene_depth_att,
  },
  .blit_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .blit_render_pass = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.blit_color_att,
  },
  /* clang-format on */
};

/* -------------------------------------------------------------------------- *
 * Scene texture + depth texture
 * -------------------------------------------------------------------------- */

static void init_scene_textures(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  /* ---- Offscreen colour texture ---- */
  state.scene_tex.handle = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Scene Color Texture"),
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc
               | WGPUTextureUsage_TextureBinding,
      .dimension     = WGPUTextureDimension_2D,
      .size          = {w, h, 1},
      .format        = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  state.scene_tex.view = wgpuTextureCreateView(
    state.scene_tex.handle, &(WGPUTextureViewDescriptor){
                              .format          = WGPUTextureFormat_RGBA8Unorm,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });

  /* ---- Depth texture ---- */
  state.depth_tex.handle = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label         = STRVIEW("Scene Depth Texture"),
              .usage         = WGPUTextureUsage_RenderAttachment,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = SCENE_DEPTH_FORMAT,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });

  state.depth_tex.view = wgpuTextureCreateView(
    state.depth_tex.handle, &(WGPUTextureViewDescriptor){
                              .format          = SCENE_DEPTH_FORMAT,
                              .dimension       = WGPUTextureViewDimension_2D,
                              .baseMipLevel    = 0,
                              .mipLevelCount   = 1,
                              .baseArrayLayer  = 0,
                              .arrayLayerCount = 1,
                              .aspect          = WGPUTextureAspect_All,
                            });

  /* Point the render pass descriptors at the new views */
  state.scene_color_att.view = state.scene_tex.view;
  state.scene_depth_att.view = state.depth_tex.view;

  state.last_width  = (int)w;
  state.last_height = (int)h;
}

static void destroy_scene_textures(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.scene_tex.view)
  WGPU_RELEASE_RESOURCE(Texture, state.scene_tex.handle)
  WGPU_RELEASE_RESOURCE(TextureView, state.depth_tex.view)
  WGPU_RELEASE_RESOURCE(Texture, state.depth_tex.handle)
}

/* -------------------------------------------------------------------------- *
 * Blit sampler
 * -------------------------------------------------------------------------- */

static void init_blit_sampler(struct wgpu_context_t* wgpu_context)
{
  state.blit_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Blit Sampler"),
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

/* -------------------------------------------------------------------------- *
 * Model loading and GPU buffer creation
 * -------------------------------------------------------------------------- */

/* Vertex post-processing flags matching the Vulkan reference:
 * PreTransformVertices | PreMultiplyVertexColors.
 * FlipY is omitted because WebGPU uses the same Y-up convention as OpenGL. */
static const gltf_model_desc_t dragon_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

static void load_dragon_model(void)
{
  bool ok = gltf_model_load_from_file_ext(&state.dragon_model,
                                          "assets/models/chinesedragon.gltf",
                                          1.0f, &dragon_load_desc);
  if (!ok) {
    printf("[screenshot] Failed to load chinesedragon.gltf\n");
    return;
  }
  state.model_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.dragon_model;

  /* ---- Vertex buffer ---- */
  size_t vb_size      = m->vertex_count * sizeof(gltf_vertex_t);
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Dragon Vertex Buffer"),
              .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
              .size  = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, m->vertices, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);

  /* ---- Index buffer ---- */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Dragon Index Buffer"),
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
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  state.ubo = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Scene UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.ubo_data),
    });
}

static void update_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  float aspect = (float)wgpu_context->width / (float)wgpu_context->height;

  /* Perspective: FOV=60°, near=0.1, far=512 — matching the Vulkan example */
  glm_perspective(glm_rad(60.0f), aspect, 0.1f, 512.0f,
                  state.ubo_data.projection);

  /* View matrix from the orbiting look-at camera */
  glm_mat4_copy(state.camera.matrices.view, state.ubo_data.view);

  /* Model matrix: identity (dragon is at the origin) */
  glm_mat4_identity(state.ubo_data.model);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.ubo, 0, &state.ubo_data,
                       sizeof(state.ubo_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene BGL: single UBO visible to the vertex shader */
  {
    WGPUBindGroupLayoutEntry entry = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.ubo_data),
      },
    };
    state.scene_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Scene BGL"),
                .entryCount = 1,
                .entries    = &entry,
              });
  }

  /* Blit BGL: sampler at binding 0, texture at binding 1 */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = { .type = WGPUSamplerBindingType_NonFiltering },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
    };
    state.blit_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Blit BGL"),
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

  state.blit_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Blit Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.blit_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Scene bind group: UBO */
  {
    WGPUBindGroupEntry entry = {
      .binding = 0,
      .buffer  = state.ubo,
      .offset  = 0,
      .size    = sizeof(state.ubo_data),
    };
    state.scene_bg = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Scene Bind Group"),
                .layout     = state.scene_bgl,
                .entryCount = 1,
                .entries    = &entry,
              });
  }

  /* Blit bind group: sampler + scene_tex view */
  {
    WGPUBindGroupEntry entries[2] = {
      [0] = {
        .binding = 0,
        .sampler = state.blit_sampler,
      },
      [1] = {
        .binding     = 1,
        .textureView = state.scene_tex.view,
      },
    };
    state.blit_bg
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .label = STRVIEW("Blit Bind Group"),
                                            .layout     = state.blit_bgl,
                                            .entryCount = ARRAY_SIZE(entries),
                                            .entries    = entries,
                                          });
  }
}

static void destroy_bind_groups(void)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.scene_bg)
  WGPU_RELEASE_RESOURCE(BindGroup, state.blit_bg)
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Interleaved gltf_vertex_t layout — only 4 attributes consumed by shader */
  WGPUVertexAttribute scene_vertex_attrs[] = {
    /* position: vec3f at offsetof(gltf_vertex_t, position) */
    {
      .shaderLocation = 0,
      .offset         = offsetof(gltf_vertex_t, position),
      .format         = WGPUVertexFormat_Float32x3,
    },
    /* normal: vec3f */
    {
      .shaderLocation = 1,
      .offset         = offsetof(gltf_vertex_t, normal),
      .format         = WGPUVertexFormat_Float32x3,
    },
    /* uv0: vec2f (not used in the shader but present in the interleaved layout)
     */
    {
      .shaderLocation = 2,
      .offset         = offsetof(gltf_vertex_t, uv0),
      .format         = WGPUVertexFormat_Float32x2,
    },
    /* color: vec4f (pre-multiplied baseColorFactor via PreMultiplyVertexColors)
     */
    {
      .shaderLocation = 3,
      .offset         = offsetof(gltf_vertex_t, color),
      .format         = WGPUVertexFormat_Float32x4,
    },
  };

  WGPUVertexBufferLayout scene_vbl = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(scene_vertex_attrs),
    .attributes     = scene_vertex_attrs,
  };

  /* ---- Scene (Phong mesh) pipeline ---- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, screenshot_mesh_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.scene_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Scene Pipeline"),
        .layout = state.scene_pipeline_layout,
        .vertex = {
          .module      = sm,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &scene_vbl,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .cullMode  = WGPUCullMode_Back,
          .frontFace = WGPUFrontFace_CCW,
        },
        .depthStencil = &(WGPUDepthStencilState){
          .format              = SCENE_DEPTH_FORMAT,
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

  /* ---- Blit pipeline (scene_tex → swapchain, fullscreen triangle) ---- */
  {
    WGPUShaderModule sm
      = wgpu_create_shader_module(device, screenshot_blit_shader_wgsl);

    WGPUColorTargetState color_target = {
      .format    = wgpu_context->render_format,
      .writeMask = WGPUColorWriteMask_All,
    };

    state.blit_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Blit Pipeline"),
        .layout = state.blit_pipeline_layout,
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
        /* No depth needed for the blit pass */
        .depthStencil = NULL,
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
 * Draw model helpers
 * -------------------------------------------------------------------------- */

static void draw_dragon(WGPURenderPassEncoder pass)
{
  if (!state.model_loaded) {
    return;
  }

  gltf_model_t* m = &state.dragon_model;
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
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
 * Screenshot buffer mapping callback
 * -------------------------------------------------------------------------- */

/* Called by the Dawn GPU scheduler when the staging buffer mapping completes */
static void screenshot_map_callback(WGPUMapAsyncStatus status,
                                    WGPUStringView message, void* userdata1,
                                    void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);

  if (status == WGPUMapAsyncStatus_Success && state.screenshot_buffer) {
    uint32_t w   = state.screenshot_width;
    uint32_t h   = state.screenshot_height;
    uint32_t bpr = state.screenshot_bpr;

    const uint8_t* data = (const uint8_t*)wgpuBufferGetConstMappedRange(
      state.screenshot_buffer, 0, (size_t)bpr * h);

    if (data) {
      FILE* f = fopen("screenshot.ppm", "wb");
      if (f) {
        /* PPM P6 header */
        fprintf(f, "P6\n%u\n%u\n255\n", w, h);

        /* Write RGB rows (skip the alpha channel) */
        for (uint32_t y = 0; y < h; y++) {
          const uint8_t* row = data + (size_t)y * bpr;
          for (uint32_t x = 0; x < w; x++) {
            fwrite(row + x * 4u, 1, 3, f); /* R, G, B — skip A */
          }
        }
        fclose(f);
        state.screenshot_saved = true;
        printf("[screenshot] Screenshot saved to screenshot.ppm\n");
      }
      wgpuBufferUnmap(state.screenshot_buffer);
    }
  }
  else if (status != WGPUMapAsyncStatus_Success) {
    printf("[screenshot] Buffer mapping failed (status=%d)\n", (int)status);
  }

  if (state.screenshot_buffer) {
    wgpuBufferRelease(state.screenshot_buffer);
    state.screenshot_buffer = NULL;
  }
  state.screenshot_saving = false;
}

/* -------------------------------------------------------------------------- *
 * Screenshot capture
 *
 * Records a CopyTextureToBuffer command into the given encoder.  The actual
 * mapping and PPM write happen asynchronously via screenshot_map_callback
 * after the command buffer is submitted.
 * -------------------------------------------------------------------------- */

static void record_screenshot_copy(struct wgpu_context_t* wgpu_context,
                                   WGPUCommandEncoder enc)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  /* CopyTextureToBuffer requires bytes_per_row to be a multiple of 256 */
  uint32_t actual_bpr  = w * 4u;
  uint32_t aligned_bpr = (actual_bpr + WGPU_COPY_BYTES_PER_ROW_ALIGNMENT - 1u)
                         & ~(WGPU_COPY_BYTES_PER_ROW_ALIGNMENT - 1u);
  uint64_t buf_size = (uint64_t)aligned_bpr * h;

  /* Create the staging buffer (CPU-mappable, write-once from GPU) */
  state.screenshot_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Screenshot Staging Buffer"),
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
              .size  = buf_size,
            });

  /* Copy the offscreen scene texture into the staging buffer */
  wgpuCommandEncoderCopyTextureToBuffer(
    enc,
    &(WGPUTexelCopyTextureInfo){
      .texture  = state.scene_tex.handle,
      .mipLevel = 0,
      .origin   = {0, 0, 0},
      .aspect   = WGPUTextureAspect_All,
    },
    &(WGPUTexelCopyBufferInfo){
      .buffer = state.screenshot_buffer,
      .layout = {
        .offset       = 0,
        .bytesPerRow  = aligned_bpr,
        .rowsPerImage = h,
      },
    },
    &(WGPUExtent3D){w, h, 1});

  state.screenshot_width  = w;
  state.screenshot_height = h;
  state.screenshot_bpr    = aligned_bpr;
  state.screenshot_saving = true;
}

/* -------------------------------------------------------------------------- *
 * Window resize handling
 * -------------------------------------------------------------------------- */

static void on_resize(struct wgpu_context_t* wgpu_context)
{
  /* Recreate size-dependent textures */
  destroy_scene_textures();
  init_scene_textures(wgpu_context);

  /* Recreate the blit bind group (it holds a reference to scene_tex.view) */
  destroy_bind_groups();
  init_bind_groups(wgpu_context);

  /* Update the camera aspect ratio */
  camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                              / (float)wgpu_context->height);
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

  igBegin("Screenshot", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Functions", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    if (imgui_overlay_button("Take Screenshot")) {
      if (!state.screenshot_saving) {
        state.screenshot_requested = true;
        state.screenshot_saved     = false;
      }
    }

    if (state.screenshot_saving) {
      igTextUnformatted("Saving screenshot...", NULL);
    }
    else if (state.screenshot_saved) {
      igTextUnformatted("Screenshot saved as screenshot.ppm", NULL);
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

  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 512.0f);
  camera_set_rotation(&state.camera, (vec3){25.0f, 23.75f, 0.0f});
  camera_set_position(&state.camera, (vec3){0.0f, 0.0f, -3.0f});

  /* Load the dragon model (synchronous) */
  load_dragon_model();
  create_model_buffers(wgpu_context);

  /* Create the offscreen scene + depth textures */
  init_scene_textures(wgpu_context);

  /* Blit sampler */
  init_blit_sampler(wgpu_context);

  /* Uniform buffer */
  init_uniform_buffer(wgpu_context);

  /* Bind group layouts → pipeline layouts → bind groups → pipelines */
  init_bind_group_layouts(wgpu_context);
  init_pipeline_layouts(wgpu_context);
  init_bind_groups(wgpu_context);
  init_pipelines(wgpu_context);

  /* ImGui overlay */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Pump Dawn's internal event queue each frame so async callbacks (e.g. the
   * screenshot buffer mapping) can fire without blocking the render loop. */
  wgpuDeviceTick(wgpu_context->device);

  /* ---- Detect window resize ---- */
  if (wgpu_context->width != state.last_width
      || wgpu_context->height != state.last_height) {
    on_resize(wgpu_context);
  }

  /* ---- Timing ---- */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* ---- Update camera ---- */
  camera_update(&state.camera, delta_time);

  /* ---- Update uniforms ---- */
  update_uniform_buffer(wgpu_context);

  /* ---- ImGui new frame + draw GUI ---- */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Record render commands ---- */
  WGPUDevice device          = wgpu_context->device;
  WGPUQueue queue            = wgpu_context->queue;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ===== Pass 1: Scene → scene_tex ===== */
  {
    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.scene_render_pass);

    uint32_t w = (uint32_t)wgpu_context->width;
    uint32_t h = (uint32_t)wgpu_context->height;
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

    wgpuRenderPassEncoderSetPipeline(pass, state.scene_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.scene_bg, 0, NULL);
    draw_dragon(pass);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ===== Optional: copy scene_tex to staging buffer for screenshot ===== */
  bool doing_screenshot = false;
  if (state.screenshot_requested && !state.screenshot_saving) {
    record_screenshot_copy(wgpu_context, cmd_enc);
    state.screenshot_requested = false;
    doing_screenshot           = true;
  }

  /* ===== Pass 2: Blit scene_tex → swapchain ===== */
  {
    state.blit_color_att.view = wgpu_context->swapchain_view;

    WGPURenderPassEncoder pass
      = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.blit_render_pass);

    uint32_t w = (uint32_t)wgpu_context->width;
    uint32_t h = (uint32_t)wgpu_context->height;
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

    wgpuRenderPassEncoderSetPipeline(pass, state.blit_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.blit_bg, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ===== Submit ===== */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* ===== Start async buffer mapping (after submission) ===== *
   * The map request is issued here; the callback fires on a future
   * wgpuDeviceTick() call (at the top of the next frame or two).    */
  if (doing_screenshot && state.screenshot_buffer) {
    wgpuBufferMapAsync(state.screenshot_buffer, WGPUMapMode_Read, 0,
                       (size_t)state.screenshot_bpr * state.screenshot_height,
                       (WGPUBufferMapCallbackInfo){
                         .mode      = WGPUCallbackMode_AllowSpontaneous,
                         .callback  = screenshot_map_callback,
                         .userdata1 = NULL,
                         .userdata2 = NULL,
                       });
  }

  /* ===== ImGui overlay render (on top of blitted scene) ===== */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Release any pending screenshot buffer */
  if (state.screenshot_buffer) {
    wgpuBufferUnmap(state.screenshot_buffer);
    wgpuBufferRelease(state.screenshot_buffer);
    state.screenshot_buffer = NULL;
  }

  /* Destroy model resources */
  gltf_model_destroy(&state.dragon_model);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Destroy uniform buffer */
  WGPU_RELEASE_RESOURCE(Buffer, state.ubo)

  /* Destroy textures */
  destroy_scene_textures();
  WGPU_RELEASE_RESOURCE(Sampler, state.blit_sampler)

  /* Destroy bind groups */
  destroy_bind_groups();

  /* Destroy bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.scene_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.blit_bgl)

  /* Destroy pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.scene_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.blit_pipeline_layout)

  /* Destroy pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.scene_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.blit_pipeline)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Saving framebuffer to screenshot",
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

/* ---- Phong mesh shader (scene pass) ----
 *
 * Vertex attributes layout matches gltf_vertex_t offsets:
 *   location 0: position (vec3f)
 *   location 1: normal   (vec3f)
 *   location 2: uv0      (vec2f)  -- received but unused
 *   location 3: color    (vec4f)  -- pre-multiplied by baseColorFactor
 *
 * Lighting is computed in view space, matching the Vulkan reference shader.
 * The model matrix is identity (no model transform), so the model-view matrix
 * equals the view matrix.  The normal matrix is therefore mat3(view), which
 * maps world-space (== object-space) normals into camera/view space.
 */
// clang-format off
static const char* screenshot_mesh_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv0      : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) clip_pos  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       color     : vec3f,
    @location(2)       view_vec  : vec3f,
    @location(3)       light_vec : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    // Model-view matrix (model is identity, so MV = view)
    let mv = ubo.view * ubo.model;

    // Clip-space position
    out.clip_pos = ubo.projection * mv * vec4f(in.position, 1.0);

    // View-space position for lighting
    let pos_vs = mv * vec4f(in.position, 1.0);

    // Normal transformed into view space by the upper-left 3x3 of MV.
    // This is correct because model = identity, so the normal matrix is
    // mat3(view), which ortho-normalises world normals into camera space.
    let mv_mat3 = mat3x3f(mv[0].xyz, mv[1].xyz, mv[2].xyz);
    out.normal = mv_mat3 * in.normal;

    // Vertex colour is pre-multiplied by the material baseColorFactor
    out.color = in.color.rgb;

    // Light vector and view vector in camera space
    let light_pos = vec3f(1.0, -1.0, 1.0);
    out.light_vec = light_pos - pos_vs.xyz;
    out.view_vec  = -pos_vs.xyz;

    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let N = normalize(in.normal);
    let L = normalize(in.light_vec);
    let V = normalize(in.view_vec);
    let R = reflect(-L, N);

    let ambient  = vec3f(0.1);
    let diffuse  = max(dot(N, L), 0.0) * vec3f(1.0);
    let specular = pow(max(dot(R, V), 0.0), 16.0) * vec3f(0.75);

    return vec4f((ambient + diffuse) * in.color + specular, 1.0);
  }
);
// clang-format on

/* ---- Blit shader (fullscreen triangle: scene_tex → swapchain) ----
 *
 * Standard fullscreen-triangle technique:
 *   vertex_index 0 → NDC(-1,-1), UV(0, 1)
 *   vertex_index 1 → NDC( 3,-1), UV(2,-1)
 *   vertex_index 2 → NDC(-1, 3), UV(0,-1)
 *
 * The V-coordinate is flipped (1 - v) so that:
 *   - Screen top    (NDC y=+1) → texture UV y=0  (top of scene_tex)
 *   - Screen bottom (NDC y=-1) → texture UV y=1  (bottom of scene_tex)
 * Without the flip the image would appear vertically mirrored.
 */
// clang-format off
static const char* screenshot_blit_shader_wgsl = CODE(
  @group(0) @binding(0) var blit_sampler : sampler;
  @group(0) @binding(1) var blit_tex     : texture_2d<f32>;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv       : vec2f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out : VertexOutput;
    // Generate XY in [0,2] then remap to clip space
    let uv = vec2f(
      f32((vertex_index << 1u) & 2u),
      f32(vertex_index & 2u)
    );
    // Flip V: texture origin is top-left, NDC Y+ is up
    out.uv       = vec2f(uv.x, 1.0 - uv.y);
    out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    return textureSample(blit_tex, blit_sampler, in.uv);
  }
);
// clang-format on
