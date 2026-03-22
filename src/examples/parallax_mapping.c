/**
 * @brief Parallax Mapping.
 *
 * Ported from the Vulkan parallax mapping example. Renders a textured plane
 * with a height map using different parallax mapping techniques: normal
 * mapping, basic parallax, steep parallax, and parallax occlusion mapping.
 * A GUI combo box lets you switch between mapping modes.
 *
 * @ref
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/parallaxmapping
 */

#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

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

#include "core/camera.h"
#include "core/gltf_model.h"
#include "core/image_loader.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (forward declarations — defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* parallax_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define TEXTURE_COUNT (2u)
#define FILE_BUFFER_SIZE (1024 * 1024 * 4) /* 4 MB per texture */

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
  uint32_t index_count;

  /* Textures */
  struct {
    wgpu_texture_t color_map;
    wgpu_texture_t normal_height_map;
  } textures;
  struct {
    const char* file;
    wgpu_texture_t* texture;
  } texture_mappings[TEXTURE_COUNT];
  uint8_t file_buffers[TEXTURE_COUNT][FILE_BUFFER_SIZE];

  /* Uniform buffers */
  WGPUBuffer uniform_buffer_vs;
  WGPUBuffer uniform_buffer_fs;

  struct {
    mat4 projection;
    mat4 view;
    mat4 model_mat;
    vec4 light_pos;
    vec4 camera_pos;
  } ubo_vs;

  struct {
    float height_scale;
    float parallax_bias;
    float num_layers;
    int32_t mapping_mode;
  } ubo_fs;

  /* Bind group layout + bind group */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Pipeline layout + render pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    int32_t mapping_mode;
  } settings;

  /* Mapping mode strings for combo box */
  const char* mapping_modes_str[5];

  /* Animation timer (Vulkan-style: 0..1 wrapping) */
  float timer;
  float timer_speed;
  bool paused;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .ubo_vs = {
    /* Vulkan original: lightPos = (0.0, -2.0, 0.0, 1.0)
     * Negate Y for WebGPU: (0.0, 2.0, 0.0, 1.0) */
    .light_pos = {0.0f, 2.0f, 0.0f, 1.0f},
  },
  .ubo_fs = {
    .height_scale  = 0.1f,
    .parallax_bias = -0.02f,
    .num_layers    = 48.0f,
    .mapping_mode  = 4,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
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
  // clang-format off
  .texture_mappings = {
    { .file = "assets/textures/rocks_color_rgba.png",         .texture = &state.textures.color_map         },
    { .file = "assets/textures/rocks_normal_height_rgba.png", .texture = &state.textures.normal_height_map },
  },
  // clang-format on
  .settings = {
    .mapping_mode = 4,
  },
  .mapping_modes_str = {
    "Color only",
    "Normal mapping",
    "Parallax mapping",
    "Steep parallax mapping",
    "Parallax occlusion mapping",
  },
  .timer       = 0.0f,
  .timer_speed = 0.125f, /* Vulkan: timerSpeed(0.25) *= 0.5 → 0.125 */
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_load_from_file(&state.model, "assets/models/plane.gltf", 1.0f);
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
  state.vertex_buffer
    = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
                                       .label = STRVIEW("Plane Vertex Buffer"),
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
                                         .label = STRVIEW("Plane Index Buffer"),
                                         .usage = WGPUBufferUsage_Index,
                                         .size  = ib_size,
                                         .mappedAtCreation = true,
                                       });
    void* idata = wgpuBufferGetMappedRange(state.index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.index_buffer);
    state.index_count = m->index_count;
  }
}

/* -------------------------------------------------------------------------- *
 * Texture loading (asynchronous via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed: %s (error: %d)\n",
           response->path ? response->path : "?", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  uint8_t* pixels            = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc            = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = (uint32_t)img_width,
        .height             = (uint32_t)img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = (size_t)(img_width * img_height * 4),
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_textures(struct wgpu_context_t* wgpu_context)
{
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(state.texture_mappings); ++i) {
    wgpu_texture_t* texture = state.texture_mappings[i].texture;
    /* Create placeholder texture */
    *(texture) = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });
    /* Start async file load */
    sfetch_send(&(sfetch_request_t){
      .path     = state.texture_mappings[i].file,
      .callback = fetch_callback,
      .buffer   = { .ptr = state.file_buffers[i], .size = FILE_BUFFER_SIZE },
      .user_data = {
        .ptr  = &texture,
        .size = sizeof(wgpu_texture_t*),
      },
    });
  }
}

static bool textures_loaded(void)
{
  for (uint8_t i = 0; i < TEXTURE_COUNT; ++i) {
    if (!state.texture_mappings[i].texture->desc.is_dirty) {
      return false;
    }
  }
  return true;
}

/* Forward declaration */
static void init_bind_group(struct wgpu_context_t* wgpu_context);

static void update_textures(struct wgpu_context_t* wgpu_context)
{
  if (!textures_loaded()) {
    return;
  }

  for (uint8_t i = 0; i < TEXTURE_COUNT; ++i) {
    wgpu_recreate_texture(wgpu_context, state.texture_mappings[i].texture);
    FREE_TEXTURE_PIXELS(*state.texture_mappings[i].texture);
  }

  /* Recreate the bind group with the new textures */
  if (state.bind_group) {
    wgpuBindGroupRelease(state.bind_group);
    state.bind_group = NULL;
  }
  init_bind_group(wgpu_context);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffers
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Vertex shader UBO */
  state.uniform_buffer_vs = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("VS UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.ubo_vs),
    });

  /* Fragment shader UBO */
  state.uniform_buffer_fs = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("FS UBO"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(state.ubo_fs),
    });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  /* Vertex shader UBO */
  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_vs.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_vs.view);

  /* Model matrix: scale 0.2 (matches Vulkan) */
  glm_mat4_identity(state.ubo_vs.model_mat);
  glm_scale_uni(state.ubo_vs.model_mat, 0.2f);

  /* Animate light position */
  if (!state.paused) {
    state.ubo_vs.light_pos[0] = sinf(state.timer * 2.0f * GLM_PIf) * 1.5f;
    state.ubo_vs.light_pos[2] = cosf(state.timer * 2.0f * GLM_PIf) * 1.5f;
  }

  /* Camera position: extract actual world-space position.
   * Vulkan does: cameraPos = vec4(pos, 0) * vec4(-1, 1, -1, 1)
   * In the WebGPU camera, set_position negates Y internally, so all three
   * components must be negated to recover the true world position. */
  state.ubo_vs.camera_pos[0] = -state.camera.position[0];
  state.ubo_vs.camera_pos[1] = -state.camera.position[1];
  state.ubo_vs.camera_pos[2] = -state.camera.position[2];
  state.ubo_vs.camera_pos[3] = 1.0f;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs, 0,
                       &state.ubo_vs, sizeof(state.ubo_vs));

  /* Fragment shader UBO */
  state.ubo_fs.mapping_mode = state.settings.mapping_mode;
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_fs, 0,
                       &state.ubo_fs, sizeof(state.ubo_fs));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout + bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[5] = {
    /* Binding 0: Vertex shader UBO */
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.ubo_vs),
      },
    },
    /* Binding 1: Fragment color map texture */
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    /* Binding 2: Fragment normal+height map texture */
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    /* Binding 3: Fragment shader UBO */
    [3] = {
      .binding    = 3,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.ubo_fs),
      },
    },
    /* Binding 4: Texture sampler (shared) */
    [4] = {
      .binding    = 4,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Parallax Mapping - Bind Group Layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(entries),
      .entries    = entries,
    });
}

static void init_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry entries[5] = {
    /* Binding 0: VS UBO */
    [0] = {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs,
      .size    = sizeof(state.ubo_vs),
    },
    /* Binding 1: Color map texture view */
    [1] = {
      .binding     = 1,
      .textureView = state.textures.color_map.view,
    },
    /* Binding 2: Normal+height map texture view */
    [2] = {
      .binding     = 2,
      .textureView = state.textures.normal_height_map.view,
    },
    /* Binding 3: FS UBO */
    [3] = {
      .binding = 3,
      .buffer  = state.uniform_buffer_fs,
      .size    = sizeof(state.ubo_fs),
    },
    /* Binding 4: Texture sampler (shared) */
    [4] = {
      .binding = 4,
      .sampler = state.textures.color_map.sampler,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = STRVIEW("Parallax Mapping - Bind Group"),
                            .layout = state.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Render pipeline
 * -------------------------------------------------------------------------- */

static void init_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Pipeline Layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layout,
            });

  /* Shader module */
  WGPUShaderModule shader_module
    = wgpu_create_shader_module(device, parallax_shader_wgsl);

  /* Vertex buffer layout matching gltf_vertex_t:
   * Position (vec3f)  @ location 0
   * UV0      (vec2f)  @ location 1
   * Normal   (vec3f)  @ location 2
   * Tangent  (vec4f)  @ location 3
   */
  WGPUVertexAttribute vertex_attrs[] = {
    {.shaderLocation = 0,
     .offset         = offsetof(gltf_vertex_t, position),
     .format         = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 1,
     .offset         = offsetof(gltf_vertex_t, uv0),
     .format         = WGPUVertexFormat_Float32x2},
    {.shaderLocation = 2,
     .offset         = offsetof(gltf_vertex_t, normal),
     .format         = WGPUVertexFormat_Float32x3},
    {.shaderLocation = 3,
     .offset         = offsetof(gltf_vertex_t, tangent),
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

  /* Render pipeline (cull none, matching Vulkan original) */
  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Parallax Mapping - Render Pipeline"),
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
      .cullMode  = WGPUCullMode_None,
    },
    .depthStencil = &depth_stencil_state,
    .multisample  = {
      .count = 1,
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

  if (imgui_overlay_header("Settings")) {
    imgui_overlay_combo_box("Mode", &state.settings.mapping_mode,
                            state.mapping_modes_str, 5);
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

  /* Initialize sokol_fetch for async texture loading */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 4,
    .num_channels = 1,
    .num_lanes    = 2,
    .logger.func  = slog_func,
  });

  /* Camera: first-person type */
  camera_init(&state.camera);
  state.camera.type = CameraType_FirstPerson;
  camera_set_position(&state.camera, (vec3){0.0f, 1.25f, -1.5f});
  camera_set_rotation(&state.camera, (vec3){45.0f, 0.0f, 0.0f});
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Load model synchronously */
  load_model();

  /* Create GPU buffers from model data */
  create_model_buffers(wgpu_context);

  /* Create uniform buffers */
  init_uniform_buffers(wgpu_context);

  /* Create bind group layout */
  init_bind_group_layout(wgpu_context);

  /* Initialize textures (async loading) */
  init_textures(wgpu_context);

  /* Create bind group with placeholder textures */
  init_bind_group(wgpu_context);

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

  /* Process async file loading */
  sfetch_dowork();

  /* Check for newly loaded textures */
  update_textures(wgpu_context);

  /* Timing */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Update animation timer */
  if (!state.paused) {
    state.timer += state.timer_speed * delta_time;
    if (state.timer > 1.0f) {
      state.timer -= 1.0f;
    }
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

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder pass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;
  wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)w, (float)h, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, w, h);

  wgpuRenderPassEncoderSetPipeline(pass, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(pass, 0, state.bind_group, 0, NULL);
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
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_fs)

  /* Textures */
  wgpu_destroy_texture(&state.textures.color_map);
  wgpu_destroy_texture(&state.textures.normal_height_map);

  /* Bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)

  /* Bind group layout */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)

  /* Pipeline */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)

  sfetch_shutdown();
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Parallax Mapping",
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
static const char* parallax_shader_wgsl = CODE(
  /* Vertex shader UBO */
  struct UBO_VS {
    projection : mat4x4f,
    view       : mat4x4f,
    model      : mat4x4f,
    lightPos   : vec4f,
    cameraPos  : vec4f,
  };

  /* Fragment shader UBO */
  struct UBO_FS {
    heightScale  : f32,
    parallaxBias : f32,
    numLayers    : f32,
    mappingMode  : i32,
  };

  @group(0) @binding(0) var<uniform> ubo_vs : UBO_VS;
  @group(0) @binding(1) var colorMap : texture_2d<f32>;
  @group(0) @binding(2) var normalHeightMap : texture_2d<f32>;
  @group(0) @binding(3) var<uniform> ubo_fs : UBO_FS;
  @group(0) @binding(4) var texSampler : sampler;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) uv       : vec2f,
    @location(2) normal   : vec3f,
    @location(3) tangent  : vec4f,
  };

  struct VertexOutput {
    @builtin(position) position        : vec4f,
    @location(0)       uv              : vec2f,
    @location(1)       tangentLightPos : vec3f,
    @location(2)       tangentViewPos  : vec3f,
    @location(3)       tangentFragPos  : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = ubo_vs.projection * ubo_vs.view * ubo_vs.model * vec4f(in.position, 1.0);

    let fragPos = (ubo_vs.model * vec4f(in.position, 1.0)).xyz;
    out.uv = in.uv;

    let N = normalize((ubo_vs.model * vec4f(in.normal, 0.0)).xyz);
    let T = normalize((ubo_vs.model * vec4f(in.tangent.xyz, 0.0)).xyz);
    let B = normalize(cross(N, T));
    let TBN = transpose(mat3x3f(T, B, N));

    out.tangentLightPos = TBN * ubo_vs.lightPos.xyz;
    out.tangentViewPos  = TBN * ubo_vs.cameraPos.xyz;
    out.tangentFragPos  = TBN * fragPos;
    return out;
  }

  /* Parallax mapping */
  fn parallaxMapping(uv : vec2f, viewDir : vec3f) -> vec2f {
    let height = 1.0 - textureSampleLevel(normalHeightMap, texSampler, uv, 0.0).a;
    let p = viewDir.xy * (height * (ubo_fs.heightScale * 0.5) + ubo_fs.parallaxBias) / viewDir.z;
    return uv - p;
  }

  /* Steep parallax mapping */
  fn steepParallaxMapping(uv : vec2f, viewDir : vec3f) -> vec2f {
    let layerDepth = 1.0 / ubo_fs.numLayers;
    var currLayerDepth = 0.0;
    let deltaUV = viewDir.xy * ubo_fs.heightScale / (viewDir.z * ubo_fs.numLayers);
    var currUV = uv;
    var height = 1.0 - textureSampleLevel(normalHeightMap, texSampler, currUV, 0.0).a;
    for (var i = 0; i < i32(ubo_fs.numLayers); i = i + 1) {
      currLayerDepth = currLayerDepth + layerDepth;
      currUV = currUV - deltaUV;
      height = 1.0 - textureSampleLevel(normalHeightMap, texSampler, currUV, 0.0).a;
      if (height < currLayerDepth) {
        break;
      }
    }
    return currUV;
  }

  /* Parallax occlusion mapping */
  fn parallaxOcclusionMapping(uv : vec2f, viewDir : vec3f) -> vec2f {
    let layerDepth = 1.0 / ubo_fs.numLayers;
    var currLayerDepth = 0.0;
    let deltaUV = viewDir.xy * ubo_fs.heightScale / (viewDir.z * ubo_fs.numLayers);
    var currUV = uv;
    var height = 1.0 - textureSampleLevel(normalHeightMap, texSampler, currUV, 0.0).a;
    for (var i = 0; i < i32(ubo_fs.numLayers); i = i + 1) {
      currLayerDepth = currLayerDepth + layerDepth;
      currUV = currUV - deltaUV;
      height = 1.0 - textureSampleLevel(normalHeightMap, texSampler, currUV, 0.0).a;
      if (height < currLayerDepth) {
        break;
      }
    }
    let prevUV = currUV + deltaUV;
    let nextDepth = height - currLayerDepth;
    let prevDepth = 1.0 - textureSampleLevel(normalHeightMap, texSampler, prevUV, 0.0).a - currLayerDepth + layerDepth;
    return mix(currUV, prevUV, nextDepth / (nextDepth - prevDepth));
  }

  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    let V = normalize(in.tangentViewPos - in.tangentFragPos);
    var uv = in.uv;

    /* Mode 0: Color only */
    if (ubo_fs.mappingMode == 0) {
      return textureSample(colorMap, texSampler, in.uv);
    }

    /* Compute displaced UVs for parallax modes */
    if (ubo_fs.mappingMode == 2) {
      uv = parallaxMapping(in.uv, V);
    } else if (ubo_fs.mappingMode == 3) {
      uv = steepParallaxMapping(in.uv, V);
    } else if (ubo_fs.mappingMode == 4) {
      uv = parallaxOcclusionMapping(in.uv, V);
    }

    /* Sample textures before discard decision */
    let normalHeightSample = textureSampleLevel(normalHeightMap, texSampler, uv, 0.0).rgb;
    let color = textureSample(colorMap, texSampler, uv).rgb;

    /* Discard fragments at texture border */
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      discard;
    }

    let N = normalize(normalHeightSample * 2.0 - 1.0);
    let L = normalize(in.tangentLightPos - in.tangentFragPos);
    let R = reflect(-L, N);
    let H = normalize(L + V);

    let ambient  = 0.2 * color;
    let diffuse  = max(dot(L, N), 0.0) * color;
    let specular = vec3f(0.15) * pow(max(dot(N, H), 0.0), 32.0);

    return vec4f(ambient + diffuse + specular, 1.0);
  }
);
// clang-format on
