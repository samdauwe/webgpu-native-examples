/**
 * @brief Multisampling using resolve attachments (MSAA).
 *
 * Ported from the Vulkan multisampling example. Renders a glTF model (Voyager
 * spacecraft) with 4× MSAA anti-aliasing via resolve attachments. The
 * multisampled color attachment is resolved to the swapchain image at the end
 * of the render pass. A GUI checkbox provides a "Sample rate shading" toggle
 * (note: WebGPU does not expose per-sample shading, so this is cosmetic only).
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/multisampling
 */

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
 * WGSL Shaders (forward declarations — defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* mesh_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define SAMPLE_COUNT (4u)
#define MAX_MATERIALS (8u)

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t model;
  bool model_loaded;

  /* GPU vertex/index buffers */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  struct {
    mat4 projection;
    mat4 model;
    vec4 light_pos;
  } ubo_data;

  /* Per-material texture bind groups */
  struct {
    WGPUBindGroup bind_group;
    WGPUTexture gpu_texture;
    WGPUTextureView gpu_texture_view;
  } materials[MAX_MATERIALS];
  uint32_t material_count;
  WGPUSampler texture_sampler;
  WGPUTexture default_texture;
  WGPUTextureView default_texture_view;

  /* Bind group layouts */
  WGPUBindGroupLayout ubo_bind_group_layout;
  WGPUBindGroupLayout texture_bind_group_layout;

  /* Bind group for UBO */
  WGPUBindGroup ubo_bind_group;

  /* Pipeline layout + render pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    bool sample_rate_shading;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .ubo_data = {
    /* Vulkan original: lightPos = (5.0, -5.0, 5.0, 1.0) */
    /* Negate Y for WebGPU: (5.0, 5.0, 5.0, 1.0) */
    .light_pos = {5.0f, 5.0f, 5.0f, 1.0f},
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    /* White background for higher contrast, matching Vulkan original */
    .clearValue = {1.0f, 1.0f, 1.0f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .depth_stencil_attachment = {
    .depthLoadOp       = WGPULoadOp_Clear,
    .depthStoreOp      = WGPUStoreOp_Store,
    .depthClearValue   = 1.0f,
    .stencilLoadOp     = WGPULoadOp_Clear,
    .stencilStoreOp    = WGPUStoreOp_Store,
    .stencilClearValue = 0,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
  .settings = {
    .sample_rate_shading = false,
  },
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_load_from_file(&state.model, "assets/models/voyager.gltf", 1.0f);
  state.model_loaded = true;
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.model;
  size_t vb_size    = m->vertex_count * sizeof(gltf_vertex_t);

  /* Bake node transforms into vertices (PreTransformVertices).
   * Do NOT use FlipY — this is WebGPU (Y-up). */
  gltf_model_desc_t load_desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  memcpy(xformed, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, xformed, &load_desc);

  /* Upload vertex buffer */
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label            = STRVIEW("Voyager Vertex Buffer"),
              .usage            = WGPUBufferUsage_Vertex,
              .size             = vb_size,
              .mappedAtCreation = true,
            });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, xformed, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);
  free(xformed);

  /* Upload index buffer */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label            = STRVIEW("Voyager Index Buffer"),
                .usage            = WGPUBufferUsage_Index,
                .size             = ib_size,
                .mappedAtCreation = true,
              });
    void* idata = wgpuBufferGetMappedRange(state.index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.index_buffer);
  }
}

/* -------------------------------------------------------------------------- *
 * Default 1x1 white texture (fallback for materials without a texture)
 * -------------------------------------------------------------------------- */

static void create_default_texture(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  state.default_texture = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .label     = STRVIEW("Default White Texture"),
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {1, 1, 1},
      .format    = WGPUTextureFormat_RGBA8Unorm,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });

  const uint8_t white_pixel[4] = {255, 255, 255, 255};
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo){
                          .texture  = state.default_texture,
                          .mipLevel = 0,
                          .origin   = {0, 0, 0},
                          .aspect   = WGPUTextureAspect_All,
                        },
                        white_pixel, sizeof(white_pixel),
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = 4,
                          .rowsPerImage = 1,
                        },
                        &(WGPUExtent3D){1, 1, 1});

  state.default_texture_view = wgpuTextureCreateView(
    state.default_texture, &(WGPUTextureViewDescriptor){
                             .label           = STRVIEW("Default White View"),
                             .format          = WGPUTextureFormat_RGBA8Unorm,
                             .dimension       = WGPUTextureViewDimension_2D,
                             .baseMipLevel    = 0,
                             .mipLevelCount   = 1,
                             .baseArrayLayer  = 0,
                             .arrayLayerCount = 1,
                           });
}

/* -------------------------------------------------------------------------- *
 * Texture sampler
 * -------------------------------------------------------------------------- */

static void create_texture_sampler(struct wgpu_context_t* wgpu_context)
{
  state.texture_sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Texture Sampler"),
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

/* -------------------------------------------------------------------------- *
 * Per-material texture bind groups
 * -------------------------------------------------------------------------- */

static void create_material_bind_groups(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  gltf_model_t* m   = &state.model;

  state.material_count
    = m->material_count < MAX_MATERIALS ? m->material_count : MAX_MATERIALS;

  for (uint32_t i = 0; i < state.material_count; ++i) {
    const gltf_material_t* mat = &m->materials[i];

    /* Upload material's base color texture to GPU, or use default */
    WGPUTextureView tex_view = state.default_texture_view;

    if (mat->base_color_tex_index >= 0
        && (uint32_t)mat->base_color_tex_index < m->texture_count) {
      const gltf_texture_t* tex = &m->textures[mat->base_color_tex_index];
      if (tex->data && tex->width > 0 && tex->height > 0) {
        /* Create GPU texture */
        state.materials[i].gpu_texture = wgpuDeviceCreateTexture(
          device,
          &(WGPUTextureDescriptor){
            .label = STRVIEW("Material Texture"),
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .dimension     = WGPUTextureDimension_2D,
            .size          = {tex->width, tex->height, 1},
            .format        = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1,
            .sampleCount   = 1,
          });

        /* Upload pixel data */
        wgpuQueueWriteTexture(wgpu_context->queue,
                              &(WGPUTexelCopyTextureInfo){
                                .texture  = state.materials[i].gpu_texture,
                                .mipLevel = 0,
                                .origin   = {0, 0, 0},
                                .aspect   = WGPUTextureAspect_All,
                              },
                              tex->data, (size_t)4 * tex->width * tex->height,
                              &(WGPUTexelCopyBufferLayout){
                                .offset       = 0,
                                .bytesPerRow  = 4 * tex->width,
                                .rowsPerImage = tex->height,
                              },
                              &(WGPUExtent3D){tex->width, tex->height, 1});

        /* Create view */
        state.materials[i].gpu_texture_view
          = wgpuTextureCreateView(state.materials[i].gpu_texture,
                                  &(WGPUTextureViewDescriptor){
                                    .label  = STRVIEW("Material Texture View"),
                                    .format = WGPUTextureFormat_RGBA8Unorm,
                                    .dimension    = WGPUTextureViewDimension_2D,
                                    .baseMipLevel = 0,
                                    .mipLevelCount   = 1,
                                    .baseArrayLayer  = 0,
                                    .arrayLayerCount = 1,
                                  });

        tex_view = state.materials[i].gpu_texture_view;
      }
    }

    WGPUBindGroupEntry entries[2] = {
      [0] = {.binding = 0, .textureView = tex_view},
      [1] = {.binding = 1, .sampler = state.texture_sampler},
    };

    state.materials[i].bind_group = wgpuDeviceCreateBindGroup(
      device, &(WGPUBindGroupDescriptor){
                .label      = STRVIEW("Material Bind Group"),
                .layout     = state.texture_bind_group_layout,
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.ubo_data),
    });
}

static void update_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  /* Projection from camera */
  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_data.projection);

  /* Model = view matrix from camera */
  glm_mat4_copy(state.camera.matrices.view, state.ubo_data.model);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.ubo_data, sizeof(state.ubo_data));
}

/* -------------------------------------------------------------------------- *
 * Bind group layouts
 * -------------------------------------------------------------------------- */

static void init_bind_group_layouts(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Set 0: UBO (projection + model + lightPos) */
  {
    WGPUBindGroupLayoutEntry entry = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.ubo_data),
      },
    };
    state.ubo_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("UBO Bind Group Layout"),
                .entryCount = 1,
                .entries    = &entry,
              });
  }

  /* Set 1: Texture + Sampler */
  {
    WGPUBindGroupLayoutEntry entries[2] = {
      [0] = {
        .binding    = 0,
        .visibility = WGPUShaderStage_Fragment,
        .texture = {
          .sampleType    = WGPUTextureSampleType_Float,
          .viewDimension = WGPUTextureViewDimension_2D,
          .multisampled  = false,
        },
      },
      [1] = {
        .binding    = 1,
        .visibility = WGPUShaderStage_Fragment,
        .sampler = {
          .type = WGPUSamplerBindingType_Filtering,
        },
      },
    };
    state.texture_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
      device, &(WGPUBindGroupLayoutDescriptor){
                .label      = STRVIEW("Texture Bind Group Layout"),
                .entryCount = (uint32_t)ARRAY_SIZE(entries),
                .entries    = entries,
              });
  }
}

/* -------------------------------------------------------------------------- *
 * Bind groups
 * -------------------------------------------------------------------------- */

static void init_ubo_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry entry = {
    .binding = 0,
    .buffer  = state.uniform_buffer,
    .size    = sizeof(state.ubo_data),
  };
  state.ubo_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("UBO Bind Group"),
                            .layout     = state.ubo_bind_group_layout,
                            .entryCount = 1,
                            .entries    = &entry,
                          });
}

/* -------------------------------------------------------------------------- *
 * Render pipeline
 * -------------------------------------------------------------------------- */

static void init_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Pipeline layout: set 0 = UBO, set 1 = texture */
  WGPUBindGroupLayout bg_layouts[2] = {
    state.ubo_bind_group_layout,
    state.texture_bind_group_layout,
  };
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Pipeline Layout"),
              .bindGroupLayoutCount = 2,
              .bindGroupLayouts     = bg_layouts,
            });

  /* Shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(device, mesh_shader_wgsl);

  /* Vertex buffer layout matching gltf_vertex_t */
  WGPUVertexAttribute vertex_attrs[] = {
    /* Position: vec3f */
    {.shaderLocation = 0,
     .offset         = offsetof(gltf_vertex_t, position),
     .format         = WGPUVertexFormat_Float32x3},
    /* Normal: vec3f */
    {.shaderLocation = 1,
     .offset         = offsetof(gltf_vertex_t, normal),
     .format         = WGPUVertexFormat_Float32x3},
    /* UV0: vec2f */
    {.shaderLocation = 2,
     .offset         = offsetof(gltf_vertex_t, uv0),
     .format         = WGPUVertexFormat_Float32x2},
    /* Color: vec4f */
    {.shaderLocation = 3,
     .offset         = offsetof(gltf_vertex_t, color),
     .format         = WGPUVertexFormat_Float32x4},
  };
  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  /* Blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state = {
    .format               = wgpu_context->depth_stencil_format,
    .depthWriteEnabled    = WGPUOptionalBool_True,
    .depthCompare         = WGPUCompareFunction_LessEqual,
    .stencilFront.compare = WGPUCompareFunction_Always,
    .stencilBack.compare  = WGPUCompareFunction_Always,
  };

  /* Render pipeline descriptor */
  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Multisampling - Render Pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &vb_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = shader_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_Back,
    },
    .depthStencil = &depth_stencil_state,
    .multisample  = {
      .count = SAMPLE_COUNT,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.pipeline = wgpuDeviceCreateRenderPipeline(device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
}

/* -------------------------------------------------------------------------- *
 * Draw model
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  gltf_model_t* m = &state.model;

  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  }

  for (uint32_t n = 0; n < m->linear_node_count; ++n) {
    gltf_node_t* node = m->linear_nodes[n];
    if (node->mesh == NULL) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; ++p) {
      gltf_primitive_t* prim = &mesh->primitives[p];

      /* Set per-material texture bind group */
      if (prim->material_index >= 0
          && (uint32_t)prim->material_index < state.material_count) {
        wgpuRenderPassEncoderSetBindGroup(
          pass, 1, state.materials[prim->material_index].bind_group, 0, NULL);
      }

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
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Settings", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_checkbox("Sample rate shading",
                           &state.settings.sample_rate_shading);
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

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                                / (float)wgpu_context->height);
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

  /* Camera: lookat type */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera, (vec3){2.5f, 2.5f, -7.5f});
  camera_set_rotation(&state.camera, (vec3){0.0f, -90.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load model synchronously */
  load_model();

  /* Create GPU buffers from model data */
  create_model_buffers(wgpu_context);

  /* Create default texture and sampler */
  create_default_texture(wgpu_context);
  create_texture_sampler(wgpu_context);

  /* Create uniform buffer */
  init_uniform_buffer(wgpu_context);

  /* Create bind group layouts */
  init_bind_group_layouts(wgpu_context);

  /* Create UBO bind group */
  init_ubo_bind_group(wgpu_context);

  /* Create per-material texture bind groups */
  create_material_bind_groups(wgpu_context);

  /* Create render pipeline */
  init_pipeline(wgpu_context);

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
  update_uniform_buffer(wgpu_context);

  /* ImGui */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* ---- Render ---- */
  WGPUDevice device = wgpu_context->device;

  /* Set target frame buffer for MSAA */
  if (SAMPLE_COUNT > 1) {
    state.color_attachment.view          = wgpu_context->msaa_view;
    state.color_attachment.resolveTarget = wgpu_context->swapchain_view;
  }
  else {
    state.color_attachment.view          = wgpu_context->swapchain_view;
    state.color_attachment.resolveTarget = NULL;
  }
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;
  wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

  wgpuRenderPassEncoderSetPipeline(pass, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.ubo_bind_group, 0, NULL);
  draw_model(pass);

  wgpuRenderPassEncoderEnd(pass);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  wgpuRenderPassEncoderRelease(pass);
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

  /* Destroy model CPU data */
  gltf_model_destroy(&state.model);

  /* Release GPU buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Material bind groups and textures */
  for (uint32_t i = 0; i < state.material_count; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.materials[i].bind_group)
    WGPU_RELEASE_RESOURCE(TextureView, state.materials[i].gpu_texture_view)
    WGPU_RELEASE_RESOURCE(Texture, state.materials[i].gpu_texture)
  }

  /* Default texture */
  WGPU_RELEASE_RESOURCE(TextureView, state.default_texture_view)
  WGPU_RELEASE_RESOURCE(Texture, state.default_texture)

  /* Sampler */
  WGPU_RELEASE_RESOURCE(Sampler, state.texture_sampler)

  /* Bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.ubo_bind_group)

  /* Bind group layouts */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.ubo_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.texture_bind_group_layout)

  /* Pipeline */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Multisampling",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
    .sample_count   = SAMPLE_COUNT,
  });
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* mesh_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model      : mat4x4f,
    lightPos   : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;

  @group(1) @binding(0) var colorTexture : texture_2d<f32>;
  @group(1) @binding(1) var colorSampler : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) normal   : vec3f,
    @location(2) uv       : vec2f,
    @location(3) color    : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position  : vec4f,
    @location(0)       normal    : vec3f,
    @location(1)       color     : vec3f,
    @location(2)       uv        : vec2f,
    @location(3)       viewVec   : vec3f,
    @location(4)       lightVec  : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = ubo.projection * ubo.model * vec4f(in.position, 1.0);
    out.color    = in.color.rgb;
    out.uv       = in.uv;

    let pos      = ubo.model * vec4f(in.position, 1.0);
    out.normal   = (ubo.model * vec4f(in.normal, 0.0)).xyz;
    let lPos     = (ubo.model * vec4f(ubo.lightPos.xyz, 0.0)).xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec  = -pos.xyz;
    return out;
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let color = textureSample(colorTexture, colorSampler, in.uv) * vec4f(in.color, 1.0);

    let N = normalize(in.normal);
    let L = normalize(in.lightVec);
    let V = normalize(in.viewVec);
    let R = reflect(-L, N);
    let diffuse  = max(dot(N, L), 0.15) * in.color;
    let specular = pow(max(dot(R, V), 0.0), 16.0) * vec3f(0.75);
    return vec4f(diffuse * color.rgb + specular, 1.0);
  }
);
// clang-format on
