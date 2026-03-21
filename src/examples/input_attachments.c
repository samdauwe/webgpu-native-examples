/* -------------------------------------------------------------------------- *
 * WebGPU Example - Input Attachments
 *
 * Demonstrates the WebGPU equivalent of Vulkan input attachments: reading
 * attachment contents from a previous render pass at the same pixel position.
 * Since WebGPU does not have native subpasses / input attachments, this is
 * implemented as a two-pass approach using offscreen textures:
 *
 *   Pass 1 (Attachment Write): Render a glTF model with toon shading to
 *     offscreen color and depth textures.
 *   Pass 2 (Attachment Read): Fullscreen triangle reads those textures and
 *     applies either brightness/contrast (color) or depth range visualization.
 *
 * GUI controls let the user select which attachment to visualize and adjust
 * the post-processing parameters.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/inputattachments
 * -------------------------------------------------------------------------- */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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

#include "core/camera.h"
#include "core/gltf_model.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations - defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* attachment_write_shader_wgsl;
static const char* attachment_read_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t scene_model;
  bool model_loaded;

  /* GPU vertex/index buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;

  /* Offscreen attachments (color + depth written in pass 1, read in pass 2) */
  struct {
    WGPUTexture color_texture;
    WGPUTextureView color_view;
    WGPUTexture depth_texture;
    WGPUTextureView depth_view;
    WGPUSampler sampler;
    uint32_t width;
    uint32_t height;
  } attachments;

  /* Uniform buffers */
  WGPUBuffer matrices_ubo; /* MVP for pass 1 */
  WGPUBuffer params_ubo;   /* Post-process params for pass 2 */

  /* Uniform data */
  struct {
    mat4 projection;
    mat4 model;
    mat4 view;
  } ubo_matrices;

  struct {
    float brightness;
    float contrast;
    float range_min;
    float range_max;
    int32_t attachment_index;
    float _padding[3];
  } ubo_params;

  /* Bind group layouts */
  WGPUBindGroupLayout write_bgl;
  WGPUBindGroupLayout read_bgl;

  /* Pipeline layouts */
  WGPUPipelineLayout write_pipeline_layout;
  WGPUPipelineLayout read_pipeline_layout;

  /* Render pipelines */
  WGPURenderPipeline write_pipeline;
  WGPURenderPipeline read_pipeline;

  /* Bind groups */
  WGPUBindGroup write_bind_group;
  WGPUBindGroup read_bind_group;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment offscreen_color_att;
  WGPURenderPassDepthStencilAttachment offscreen_depth_att;
  WGPURenderPassDescriptor offscreen_render_pass_desc;

  WGPURenderPassColorAttachment main_color_att;
  WGPURenderPassDepthStencilAttachment main_depth_att;
  WGPURenderPassDescriptor main_render_pass_desc;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .ubo_params = {
    .brightness       = 0.5f,
    .contrast         = 1.8f,
    .range_min        = 0.6f,
    .range_max        = 1.0f,
    .attachment_index = 0,
  },
  .offscreen_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.2f, 0.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .offscreen_depth_att = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Undefined,
    .stencilStoreOp    = WGPUStoreOp_Undefined,
    .stencilClearValue = 0,
  },
  .offscreen_render_pass_desc = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.offscreen_color_att,
    .depthStencilAttachment = &state.offscreen_depth_att,
  },
  .main_color_att = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.2f, 1.0f},
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
};

/* -------------------------------------------------------------------------- *
 * Offscreen attachment setup
 * -------------------------------------------------------------------------- */

static void destroy_offscreen_attachments(void)
{
  WGPU_RELEASE_RESOURCE(TextureView, state.attachments.color_view)
  WGPU_RELEASE_RESOURCE(Texture, state.attachments.color_texture)
  WGPU_RELEASE_RESOURCE(TextureView, state.attachments.depth_view)
  WGPU_RELEASE_RESOURCE(Texture, state.attachments.depth_texture)
}

static void init_offscreen_attachments(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  uint32_t w        = (uint32_t)wgpu_context->width;
  uint32_t h        = (uint32_t)wgpu_context->height;

  /* Skip if size hasn't changed */
  if (state.attachments.width == w && state.attachments.height == h) {
    return;
  }

  /* Destroy previous if resized */
  destroy_offscreen_attachments();

  state.attachments.width  = w;
  state.attachments.height = h;

  /* Color attachment (RGBA8Unorm, used as render target + sampled texture) */
  state.attachments.color_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Attachment Color"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_RGBA8Unorm,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.attachments.color_view = wgpuTextureCreateView(
    state.attachments.color_texture, &(WGPUTextureViewDescriptor){
                                       .format = WGPUTextureFormat_RGBA8Unorm,
                                       .dimension = WGPUTextureViewDimension_2D,
                                       .baseMipLevel    = 0,
                                       .mipLevelCount   = 1,
                                       .baseArrayLayer  = 0,
                                       .arrayLayerCount = 1,
                                       .aspect          = WGPUTextureAspect_All,
                                     });

  /* Depth attachment (Depth24PlusStencil8, used as render target + sampled) */
  state.attachments.depth_texture = wgpuDeviceCreateTexture(
    device, &(WGPUTextureDescriptor){
              .label = STRVIEW("Attachment Depth"),
              .usage = WGPUTextureUsage_RenderAttachment
                       | WGPUTextureUsage_TextureBinding,
              .dimension     = WGPUTextureDimension_2D,
              .size          = {w, h, 1},
              .format        = WGPUTextureFormat_Depth32Float,
              .mipLevelCount = 1,
              .sampleCount   = 1,
            });
  state.attachments.depth_view = wgpuTextureCreateView(
    state.attachments.depth_texture, &(WGPUTextureViewDescriptor){
                                       .format = WGPUTextureFormat_Depth32Float,
                                       .dimension = WGPUTextureViewDimension_2D,
                                       .baseMipLevel    = 0,
                                       .mipLevelCount   = 1,
                                       .baseArrayLayer  = 0,
                                       .arrayLayerCount = 1,
                                       .aspect = WGPUTextureAspect_DepthOnly,
                                     });

  /* Sampler (only created once) */
  if (!state.attachments.sampler) {
    state.attachments.sampler = wgpuDeviceCreateSampler(
      device, &(WGPUSamplerDescriptor){
                .label         = STRVIEW("Attachment Sampler"),
                .addressModeU  = WGPUAddressMode_ClampToEdge,
                .addressModeV  = WGPUAddressMode_ClampToEdge,
                .addressModeW  = WGPUAddressMode_ClampToEdge,
                .magFilter     = WGPUFilterMode_Nearest,
                .minFilter     = WGPUFilterMode_Nearest,
                .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
                .maxAnisotropy = 1,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_load_from_file(&state.scene_model,
                            "assets/models/treasure_smooth.gltf", 1.0f);
  state.model_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.scene_model;
  size_t vb_size    = m->vertex_count * sizeof(gltf_vertex_t);

  /* Bake node transforms + pre-multiply vertex colors */
  gltf_model_desc_t load_desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  memcpy(xformed, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, xformed, &load_desc);

  /* Upload vertex buffer */
  state.vertex_buffer
    = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
                                       .label = STRVIEW("Scene Vertex Buffer"),
                                       .usage = WGPUBufferUsage_Vertex,
                                       .size  = vb_size,
                                       .mappedAtCreation = true,
                                     });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, xformed, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);
  free(xformed);

  /* Upload index buffer */
  if (m->index_count > 0) {
    size_t ib_size = m->index_count * sizeof(uint32_t);
    state.index_buffer
      = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
                                         .label = STRVIEW("Scene Index Buffer"),
                                         .usage = WGPUBufferUsage_Index,
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

  /* Matrices UBO: projection + model + view (3 x mat4 = 192 bytes) */
  state.matrices_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Matrices UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.ubo_matrices),
            });

  /* Params UBO: brightness, contrast, range, attachment_index + padding */
  state.params_ubo = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Params UBO"),
              .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
              .size  = sizeof(state.ubo_params),
            });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  WGPUQueue queue = wgpu_context->queue;

  /* Update matrices */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.ubo_matrices.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_matrices.view);
  glm_mat4_identity(state.ubo_matrices.model);

  wgpuQueueWriteBuffer(queue, state.matrices_ubo, 0, &state.ubo_matrices,
                       sizeof(state.ubo_matrices));

  /* Update params */
  wgpuQueueWriteBuffer(queue, state.params_ubo, 0, &state.ubo_params,
                       sizeof(state.ubo_params));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts & pipeline layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Attachment write: binding 0 = matrices UBO (vertex) */
  {
    WGPUBindGroupLayoutEntry entries[1] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
    };
    state.write_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Attachment Write BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Attachment read: b0=color texture, b1=color sampler,
     b2=depth texture, b3=depth sampler, b4=params UBO */
  {
    WGPUBindGroupLayoutEntry entries[5] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_Filtering},
      },
      [2] = {
        .binding    = 2,
        .visibility = WGPUShaderStage_Fragment,
        .texture    = {
          .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
          .viewDimension = WGPUTextureViewDimension_2D,
        },
      },
      [3] = {
        .binding    = 3,
        .visibility = WGPUShaderStage_Fragment,
        .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
      },
      [4] = {
        .binding    = 4,
        .visibility = WGPUShaderStage_Fragment,
        .buffer     = {.type = WGPUBufferBindingType_Uniform},
      },
    };
    state.read_bgl = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Attachment Read BGL"),
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

static void init_pipeline_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.write_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Attachment Write PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.write_bgl,
            });

  state.read_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Attachment Read PL"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.read_bgl,
            });
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Write bind group: matrices UBO */
  {
    WGPUBindGroupEntry entries[1] = {
      [0] = {
        .binding = 0,
        .buffer  = state.matrices_ubo,
        .offset  = 0,
        .size    = sizeof(state.ubo_matrices),
      },
    };
    WGPU_RELEASE_RESOURCE(BindGroup, state.write_bind_group)
    state.write_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Attachment Write BG"),
                .layout     = state.write_bgl,
                .entryCount = ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }

  /* Read bind group: color tex, color sampler, depth tex, depth sampler,
   * params UBO */
  {
    WGPUBindGroupEntry entries[5] = {
      [0] = {
        .binding     = 0,
        .textureView = state.attachments.color_view,
      },
      [1] = {
        .binding = 1,
        .sampler = state.attachments.sampler,
      },
      [2] = {
        .binding     = 2,
        .textureView = state.attachments.depth_view,
      },
      [3] = {
        .binding = 3,
        .sampler = state.attachments.sampler,
      },
      [4] = {
        .binding = 4,
        .buffer  = state.params_ubo,
        .offset  = 0,
        .size    = sizeof(state.ubo_params),
      },
    };
    WGPU_RELEASE_RESOURCE(BindGroup, state.read_bind_group)
    state.read_bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Attachment Read BG"),
                .layout     = state.read_bgl,
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

  /* ---- Attachment write pipeline ---- */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, attachment_write_shader_wgsl);

    /* Vertex buffer layout: position, color, normal from gltf_vertex_t */
    WGPUVertexAttribute vertex_attrs[] = {
      {
        .shaderLocation = 0,
        .offset         = offsetof(gltf_vertex_t, position),
        .format         = WGPUVertexFormat_Float32x3,
      },
      {
        .shaderLocation = 1,
        .offset         = offsetof(gltf_vertex_t, color),
        .format         = WGPUVertexFormat_Float32x4,
      },
      {
        .shaderLocation = 2,
        .offset         = offsetof(gltf_vertex_t, normal),
        .format         = WGPUVertexFormat_Float32x3,
      },
    };

    WGPUVertexBufferLayout vb_layout = {
      .arrayStride    = sizeof(gltf_vertex_t),
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = ARRAY_SIZE(vertex_attrs),
      .attributes     = vertex_attrs,
    };

    WGPUDepthStencilState depth_stencil
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = WGPUTextureFormat_Depth32Float,
        .depth_write_enabled = true,
      });
    depth_stencil.depthCompare = WGPUCompareFunction_LessEqual;

    WGPU_RELEASE_RESOURCE(RenderPipeline, state.write_pipeline)
    state.write_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Attachment Write Pipeline"),
        .layout = state.write_pipeline_layout,
        .vertex = {
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 1,
          .buffers     = &vb_layout,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_Back,
        },
        .depthStencil = &depth_stencil,
        .multisample  = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = shader,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = WGPUTextureFormat_RGBA8Unorm,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
      });

    wgpuShaderModuleRelease(shader);
  }

  /* ---- Attachment read pipeline (fullscreen triangle) ---- */
  {
    WGPUShaderModule shader
      = wgpu_create_shader_module(device, attachment_read_shader_wgsl);

    WGPUDepthStencilState depth_stencil
      = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
        .format              = wgpu_context->depth_stencil_format,
        .depth_write_enabled = false,
      });
    depth_stencil.depthCompare = WGPUCompareFunction_Always;

    WGPU_RELEASE_RESOURCE(RenderPipeline, state.read_pipeline)
    state.read_pipeline = wgpuDeviceCreateRenderPipeline(
      device,
      &(WGPURenderPipelineDescriptor){
        .label  = STRVIEW("Attachment Read Pipeline"),
        .layout = state.read_pipeline_layout,
        .vertex = {
          .module      = shader,
          .entryPoint  = STRVIEW("vs_main"),
          .bufferCount = 0,
        },
        .primitive = {
          .topology  = WGPUPrimitiveTopology_TriangleList,
          .frontFace = WGPUFrontFace_CCW,
          .cullMode  = WGPUCullMode_None,
        },
        .depthStencil = &depth_stencil,
        .multisample  = {
          .count = 1,
          .mask  = 0xFFFFFFFF,
        },
        .fragment = &(WGPUFragmentState){
          .module      = shader,
          .entryPoint  = STRVIEW("fs_main"),
          .targetCount = 1,
          .targets     = &(WGPUColorTargetState){
            .format    = wgpu_context->render_format,
            .writeMask = WGPUColorWriteMask_All,
          },
        },
      });

    wgpuShaderModuleRelease(shader);
  }
}

/* -------------------------------------------------------------------------- *
 * Model drawing
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  gltf_model_t* m = &state.scene_model;

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
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

static const char* attachment_names[] = {"Color", "Depth"};

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeaderBoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igText("Input attachment");
    imgui_overlay_combo_box("##attachment", &state.ubo_params.attachment_index,
                            attachment_names, 2);

    switch (state.ubo_params.attachment_index) {
      case 0: /* Color */
        igText("Brightness");
        imgui_overlay_slider_float("##b", &state.ubo_params.brightness, 0.0f,
                                   2.0f, "%.2f");
        igText("Contrast");
        imgui_overlay_slider_float("##c", &state.ubo_params.contrast, 0.0f,
                                   4.0f, "%.2f");
        break;
      case 1: /* Depth */
        igText("Visible range");
        imgui_overlay_slider_float("min", &state.ubo_params.range_min, 0.0f,
                                   state.ubo_params.range_max, "%.2f");
        imgui_overlay_slider_float("max", &state.ubo_params.range_max,
                                   state.ubo_params.range_min, 1.0f, "%.2f");
        break;
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
  camera_on_input_event(&state.camera, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    /* Recreate offscreen attachments + bind groups on resize */
    init_offscreen_attachments(wgpu_context);
    init_bind_groups(wgpu_context);
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

  /* Camera: first-person */
  camera_init(&state.camera);
  state.camera.type           = CameraType_FirstPerson;
  state.camera.movement_speed = 2.5f;
  camera_set_position(&state.camera, (vec3){1.65f, 1.75f, -6.15f});
  camera_set_rotation(&state.camera, (vec3){12.75f, 380.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load model synchronously (it's small) */
  load_model();
  create_model_buffers(wgpu_context);

  /* Create offscreen attachments */
  init_offscreen_attachments(wgpu_context);

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
  if (!state.initialized || !state.model_loaded) {
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
  UNUSED_VAR(queue);

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* ==== Pass 1: Attachment Write (offscreen) ==== */
  {
    state.offscreen_color_att.view = state.attachments.color_view;
    state.offscreen_depth_att.view = state.attachments.depth_view;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.offscreen_render_pass_desc);

    uint32_t w = state.attachments.width;
    uint32_t h = state.attachments.height;
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f,
                                     1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

    wgpuRenderPassEncoderSetPipeline(pass, state.write_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.write_bind_group, 0, NULL);
    draw_model(pass);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
  }

  /* ==== Pass 2: Attachment Read (main swap chain) ==== */
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

    wgpuRenderPassEncoderSetPipeline(pass, state.read_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, state.read_bind_group, 0, NULL);
    /* Fullscreen triangle: 3 vertices, no vertex buffer */
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
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

  imgui_overlay_shutdown();

  /* Destroy model */
  gltf_model_destroy(&state.scene_model);

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Uniform buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.matrices_ubo)
  WGPU_RELEASE_RESOURCE(Buffer, state.params_ubo)

  /* Offscreen attachments */
  destroy_offscreen_attachments();
  WGPU_RELEASE_RESOURCE(Sampler, state.attachments.sampler)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.write_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.read_bind_group)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.write_bgl)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.read_bgl)

  /* Pipeline layouts */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.write_pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.read_pipeline_layout)

  /* Render pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.write_pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.read_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Input Attachments",
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

/* Attachment write: toon shading with vertex colors */
static const char* attachment_write_shader_wgsl = CODE(
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
    @builtin(position) position  : vec4f,
    @location(0)       color     : vec3f,
    @location(1)       normal    : vec3f,
    @location(2)       view_vec  : vec3f,
    @location(3)       light_vec : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position  = ubo.projection * ubo.view * ubo.model * vec4f(in.position, 1.0);
    out.color     = in.color.rgb;
    out.normal    = in.normal;
    out.light_vec = vec3f(0.0, -5.0, 15.0) - in.position;
    out.view_vec  = -in.position;
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    /* Toon shading */
    let N = normalize(in.normal);
    let L = normalize(in.light_vec);
    let intensity = dot(N, L);

    var shade = 1.0;
    if (intensity < 0.5)  { shade = 0.75; }
    if (intensity < 0.35) { shade = 0.6; }
    if (intensity < 0.25) { shade = 0.5; }
    if (intensity < 0.1)  { shade = 0.25; }

    return vec4f(in.color * 3.0 * shade, 1.0);
  }
);

/* Attachment read: fullscreen triangle, brightness/contrast or depth viz */
static const char* attachment_read_shader_wgsl = CODE(
  struct Params {
    brightness       : f32,
    contrast         : f32,
    range_min        : f32,
    range_max        : f32,
    attachment_index : i32,
  };

  @group(0) @binding(0) var color_tex     : texture_2d<f32>;
  @group(0) @binding(1) var color_sampler : sampler;
  @group(0) @binding(2) var depth_tex     : texture_2d<f32>;
  @group(0) @binding(3) var depth_sampler : sampler;
  @group(0) @binding(4) var<uniform> params : Params;

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
  };

  @vertex
  fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out : VertexOutput;
    /* Fullscreen triangle: 3 vertices cover the whole screen */
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    /* Flip V: WebGPU clip space Y is up (-1=bottom, +1=top) but texture
       UV v=0 is the top row.  Without flipping, the offscreen texture
       would be displayed upside-down. */
    out.uv       = vec2f(u, 1.0 - v);
    out.position = vec4f(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    return out;
  }

  fn brightness_contrast(color : vec3f, brightness : f32, contrast : f32) -> vec3f {
    return (color - 0.5) * contrast + 0.5 + brightness;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    if (params.attachment_index == 0) {
      /* Color: apply brightness and contrast */
      let color = textureSample(color_tex, color_sampler, in.uv).rgb;
      return vec4f(brightness_contrast(color, params.brightness, params.contrast), 1.0);
    } else {
      /* Depth: visualize depth range */
      let depth = textureSample(depth_tex, depth_sampler, in.uv).r;
      let range = params.range_max - params.range_min;
      let normalized = (depth - params.range_min) / range;
      return vec4f(vec3f(normalized), 1.0);
    }
  }
);

// clang-format on
