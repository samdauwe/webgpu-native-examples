#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
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

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Texture Arrays and Instanced Rendering
 *
 * Demonstrates loading and rendering a 2D texture array. Each layer of the
 * texture array contains different image data. The layers are displayed on
 * cubes using GPU instancing: each instance selects a different layer from
 * the texture array via the fragment shader.
 *
 * Ported from Sascha Willems' Vulkan example "texturearray"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/texturearray
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* texture_array_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

static const char* texture_path = "assets/textures/texturearray_rgba.png";

/* Texture atlas: 7 layers stacked vertically, each 256x256 RGBA */
#define TEXTURE_LAYER_SIZE 256
#define TEXTURE_LAYER_COUNT 7

/* Maximum number of array layers supported by the UBO */
#define MAX_LAYERS 8

/* File buffer: generous headroom beyond the actual PNG size (~445 KB) */
#define FILE_BUFFER_SIZE (640 * 1024)

/* -------------------------------------------------------------------------- *
 * Vertex layout: position (xyz) + texture coordinate (uv)
 * -------------------------------------------------------------------------- */

typedef struct {
  float pos[3];
  float uv[2];
} vertex_t;

/* -------------------------------------------------------------------------- *
 * Uniform buffer layout – must match WGSL struct exactly
 *
 * Per-instance data (80 bytes, aligned to 16):
 *   mat4  model       – 64 bytes at offset 0
 *   float array_index –  4 bytes at offset 64
 *   float _pad[3]     – 12 bytes at offset 68 (explicit padding to 80)
 *
 * Full UBO (768 bytes):
 *   mat4             projection   – 64 bytes at offset 0
 *   mat4             view         – 64 bytes at offset 64
 *   instance_data_t  instances[8] – 640 bytes at offset 128
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 model;
  float array_index;
  float _pad[3];
} instance_data_t;

typedef struct {
  mat4 projection;
  mat4 view;
  instance_data_t instances[MAX_LAYERS];
} uniform_data_t;

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Cube mesh geometry */
  struct {
    WGPUBuffer vertex;
    WGPUBuffer index;
    uint32_t index_count;
  } cube;

  /* Number of loaded texture layers (= instance count) */
  uint32_t layer_count;

  /* Texture array (loaded asynchronously from PNG atlas) */
  wgpu_texture_t texture_array;
  uint8_t file_buffer[FILE_BUFFER_SIZE];

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo;

  /* Depth texture (recreated on resize) */
  struct {
    WGPUTexture texture;
    WGPUTextureView view;
  } depth;

  /* Bind group & layout */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;

  /* Render pass descriptors */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
  WGPUBool texture_loaded;
} state = {
  .layer_count = TEXTURE_LAYER_COUNT,
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
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = &state.depth_stencil_attachment,
  },
};

/* -------------------------------------------------------------------------- *
 * Depth texture management
 * -------------------------------------------------------------------------- */

static void init_depth_texture(struct wgpu_context_t* wgpu_context)
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
  state.depth.view = wgpuTextureCreateView(state.depth.texture, NULL);
}

/* -------------------------------------------------------------------------- *
 * Cube geometry
 *
 * 6 faces × 4 vertices = 24 vertices, 6 faces × 2 triangles × 3 = 36 indices
 * UV coordinates map 0..1 across each face.
 * -------------------------------------------------------------------------- */

static void init_cube_buffers(struct wgpu_context_t* wgpu_context)
{
  /* clang-format off */
  static const vertex_t vertices[24] = {
    /* Front face */
    {{ -1.0f, -1.0f,  1.0f }, { 0.0f, 0.0f }},
    {{  1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f }},
    {{  1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f }},
    {{ -1.0f,  1.0f,  1.0f }, { 0.0f, 1.0f }},
    /* Right face */
    {{  1.0f,  1.0f,  1.0f }, { 0.0f, 0.0f }},
    {{  1.0f,  1.0f, -1.0f }, { 1.0f, 0.0f }},
    {{  1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f }},
    {{  1.0f, -1.0f,  1.0f }, { 0.0f, 1.0f }},
    /* Back face */
    {{ -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f }},
    {{  1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f }},
    {{  1.0f,  1.0f, -1.0f }, { 1.0f, 1.0f }},
    {{ -1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f }},
    /* Left face */
    {{ -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f }},
    {{ -1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f }},
    {{ -1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f }},
    {{ -1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f }},
    /* Top face */
    {{  1.0f,  1.0f,  1.0f }, { 0.0f, 0.0f }},
    {{ -1.0f,  1.0f,  1.0f }, { 1.0f, 0.0f }},
    {{ -1.0f,  1.0f, -1.0f }, { 1.0f, 1.0f }},
    {{  1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f }},
    /* Bottom face */
    {{ -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f }},
    {{  1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f }},
    {{  1.0f, -1.0f,  1.0f }, { 1.0f, 1.0f }},
    {{ -1.0f, -1.0f,  1.0f }, { 0.0f, 1.0f }},
  };

  static const uint32_t indices[36] = {
     0,  1,  2,   0,  2,  3,   /* front  */
     4,  5,  6,   4,  6,  7,   /* right  */
     8,  9, 10,   8, 10, 11,   /* back   */
    12, 13, 14,  12, 14, 15,   /* left   */
    16, 17, 18,  16, 18, 19,   /* top    */
    20, 21, 22,  20, 22, 23,   /* bottom */
  };
  /* clang-format on */

  state.cube.index_count = 36;

  /* Vertex buffer */
  state.cube.vertex = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Cube vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = sizeof(vertices),
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, state.cube.vertex, 0, vertices,
                       sizeof(vertices));

  /* Index buffer */
  state.cube.index = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Cube index buffer"),
      .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
      .size             = sizeof(indices),
      .mappedAtCreation = false,
    });
  wgpuQueueWriteBuffer(wgpu_context->queue, state.cube.index, 0, indices,
                       sizeof(indices));
}

/* -------------------------------------------------------------------------- *
 * Texture array (PNG atlas loaded via sokol_fetch)
 *
 * The atlas stores TEXTURE_LAYER_COUNT layers stacked vertically:
 *   atlas width  = TEXTURE_LAYER_SIZE
 *   atlas height = TEXTURE_LAYER_SIZE * TEXTURE_LAYER_COUNT
 * -------------------------------------------------------------------------- */

/* Forward declaration */
static void init_bind_group(struct wgpu_context_t* wgpu_context);

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("[texture_array] Fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  uint8_t* pixels = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, 4 /* desired RGBA channels */);

  if (!pixels) {
    printf("[texture_array] Failed to decode image\n");
    return;
  }

  const int expected_w = TEXTURE_LAYER_SIZE;
  const int expected_h = TEXTURE_LAYER_SIZE * TEXTURE_LAYER_COUNT;
  if (img_width != expected_w || img_height != expected_h) {
    printf("[texture_array] Unexpected atlas size: %dx%d (expected %dx%d)\n",
           img_width, img_height, expected_w, expected_h);
    image_free(pixels);
    return;
  }

  /* Mark dirty – actual GPU upload happens in frame() */
  state.texture_array.desc = (wgpu_texture_desc_t){
    .extent = (WGPUExtent3D){
      .width              = (uint32_t)TEXTURE_LAYER_SIZE,
      .height             = (uint32_t)TEXTURE_LAYER_SIZE,
      .depthOrArrayLayers = (uint32_t)TEXTURE_LAYER_COUNT,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .pixels = {
      .ptr  = pixels,
      .size = (size_t)TEXTURE_LAYER_SIZE * TEXTURE_LAYER_SIZE
              * TEXTURE_LAYER_COUNT * 4,
    },
    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
  };
  state.texture_array.desc.is_dirty = true;
}

static void init_texture_array(wgpu_context_t* wgpu_context)
{
  /* Create a 1×1 placeholder array until the real data loads */
  uint8_t placeholder[4 * TEXTURE_LAYER_COUNT];
  memset(placeholder, 64, sizeof(placeholder));
  state.texture_array = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent = {1, 1, TEXTURE_LAYER_COUNT},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = placeholder,
        .size = sizeof(placeholder),
      },
      .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
    });

  sfetch_send(&(sfetch_request_t){
    .path     = texture_path,
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

static void update_texture_array(wgpu_context_t* wgpu_context)
{
  if (!state.texture_array.desc.is_dirty) {
    return;
  }

  wgpu_recreate_texture(wgpu_context, &state.texture_array);

  if (state.texture_array.desc.pixels.ptr) {
    image_free((void*)state.texture_array.desc.pixels.ptr);
    state.texture_array.desc.pixels.ptr  = NULL;
    state.texture_array.desc.pixels.size = 0;
  }

  /* Rebind with the new texture view */
  if (state.bind_group) {
    wgpuBindGroupRelease(state.bind_group);
    state.bind_group = NULL;
  }
  init_bind_group(wgpu_context);

  state.texture_loaded = true;
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 *
 * The static per-instance model matrices are computed once in init() and
 * never change. Only projection and view are updated every frame.
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  /* Compute static per-instance model matrices.
   * Instances are arranged in a row along the X-axis, centred at the origin.
   * offset = -1.5  →  spacing between instances
   * center = (layerCount * offset) / 2 - (offset * 0.5)
   */
  const float offset = -1.5f;
  const float center
    = ((float)state.layer_count * offset) / 2.0f - (offset * 0.5f);

  for (uint32_t i = 0; i < state.layer_count; ++i) {
    mat4 t, s;
    glm_translate_make(t, (vec3){(float)i * offset - center, 0.0f, 0.0f});
    glm_scale_make(s, (vec3){0.5f, 0.5f, 0.5f});

    glm_mat4_mul(t, s, state.ubo.instances[i].model);
    state.ubo.instances[i].array_index = (float)i;
    memset(state.ubo.instances[i]._pad, 0, sizeof(state.ubo.instances[i]._pad));
  }

  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Uniform buffer"),
      .usage            = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size             = sizeof(uniform_data_t),
      .mappedAtCreation = false,
    });
}

static void update_uniform_buffer(struct wgpu_context_t* wgpu_context,
                                  float delta_time)
{
  camera_update(&state.camera, delta_time);

  glm_mat4_copy(state.camera.matrices.perspective, state.ubo.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo.view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, &state.ubo,
                       sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout + bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniform_data_t),
      },
    },
    [1] = {
      /* Binding 1: Fragment shader sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = {
      /* Binding 2: Fragment shader texture array */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
        .multisampled  = false,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label = STRVIEW("Texture array bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
}

static void init_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry entries[3] = {
    [0] = {
      .binding = 0,
      .buffer  = state.uniform_buffer,
      .offset  = 0,
      .size    = sizeof(uniform_data_t),
    },
    [1] = {
      .binding = 1,
      .sampler = state.texture_array.sampler,
    },
    [2] = {
      .binding     = 2,
      .textureView = state.texture_array.view,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Texture array bind group"),
                            .layout     = state.bind_group_layout,
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

  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("Texture array pipeline layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layout,
            });

  WGPUShaderModule shader
    = wgpu_create_shader_module(device, texture_array_shader_wgsl);

  WGPUVertexAttribute attrs[2] = {
    [0] = {
      .shaderLocation = 0,
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(vertex_t, pos),
    },
    [1] = {
      .shaderLocation = 1,
      .format         = WGPUVertexFormat_Float32x2,
      .offset         = offsetof(vertex_t, uv),
    },
  };

  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(attrs),
    .attributes     = attrs,
  };

  WGPUBlendState blend = wgpu_create_blend_state(false);

  WGPUColorTargetState color_target = {
    .format    = wgpu_context->render_format,
    .blend     = &blend,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUDepthStencilState depth_stencil = {
    .format            = WGPUTextureFormat_Depth24Plus,
    .depthWriteEnabled = WGPUOptionalBool_True,
    .depthCompare      = WGPUCompareFunction_LessEqual,
    .stencilFront      = {.compare = WGPUCompareFunction_Always},
    .stencilBack       = {.compare = WGPUCompareFunction_Always},
  };

  state.pipeline = wgpuDeviceCreateRenderPipeline(
    device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Texture array render pipeline"),
      .layout = state.pipeline_layout,
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

  WGPU_RELEASE_RESOURCE(ShaderModule, shader);
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

  igBegin("Texture Arrays", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Info", NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_text("GPU: %s", wgpu_context->platform_info.device[0] ?
                                    wgpu_context->platform_info.device :
                                    "Unknown");
    imgui_overlay_text("Layers: %u  (MAX_LAYERS=%d)", state.layer_count,
                       MAX_LAYERS);
    imgui_overlay_text("Layer size: %d x %d", TEXTURE_LAYER_SIZE,
                       TEXTURE_LAYER_SIZE);
    imgui_overlay_text("Instances: %u (one per layer)", state.layer_count);
    if (!state.texture_loaded) {
      igTextColored((ImVec4){1.0f, 1.0f, 0.0f, 1.0f}, "Loading texture...");
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

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    init_depth_texture(wgpu_context);
    camera_set_perspective(
      &state.camera, 45.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
    return;
  }

  if (!imgui_overlay_want_capture_mouse()) {
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  stm_setup();

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 1,
    .num_channels = 1,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  /* Camera – matches Vulkan example:
   *   position:  (0, 0, -7.5)
   *   rotation:  (-35, 0, 0)  pitch down
   *   VKY_TO_WGPU_VEC3:   (x, y, z)       → {x, -y, z}
   *   VKY_TO_WGPU_CAM_ROT:(pitch, yaw, roll) → {-pitch, yaw, roll}
   */
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  state.camera.rotation_speed = 0.5f;
  camera_set_position(&state.camera, (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -7.5f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-35.0f, 0.0f, 0.0f));
  camera_set_perspective(
    &state.camera, 45.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* GPU resources */
  init_depth_texture(wgpu_context);
  init_cube_buffers(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_texture_array(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_bind_group(wgpu_context);
  init_pipeline(wgpu_context);

  imgui_overlay_init(wgpu_context);

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_SUCCESS;
  }

  sfetch_dowork();
  update_texture_array(wgpu_context);

  const uint64_t now    = stm_now();
  const float dt        = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  update_uniform_buffer(wgpu_context, dt);

  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* Render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth.view;

  WGPUDevice device          = wgpu_context->device;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  wgpuRenderPassEncoderSetPipeline(rpass, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bind_group, 0, 0);
  wgpuRenderPassEncoderSetVertexBuffer(rpass, 0, state.cube.vertex, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass, state.cube.index, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  /* Draw all instances in a single call */
  wgpuRenderPassEncoderDrawIndexed(rpass, state.cube.index_count,
                                   state.layer_count, 0, 0, 0);

  wgpuRenderPassEncoderEnd(rpass);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);

  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  /* Depth texture */
  if (state.depth.view) {
    wgpuTextureViewRelease(state.depth.view);
  }
  if (state.depth.texture) {
    wgpuTextureDestroy(state.depth.texture);
    wgpuTextureRelease(state.depth.texture);
  }

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.cube.index);

  wgpu_destroy_texture(&state.texture_array);
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Texture Arrays and Instanced Rendering",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 *
 * Vertex shader:
 *   - Reads per-instance model matrix and arrayIndex from the UBO using the
 *     built-in instance_index.
 *   - Passes a vec3 UV (xy = surface UV, z = texture array index) to the
 *     fragment stage.
 *
 * Fragment shader:
 *   - Samples the 2D texture array at the integer layer given by uv.z.
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* texture_array_shader_wgsl = CODE(
  /* Per-instance data: model matrix + texture array index */
  struct InstanceData {
    model:       mat4x4f,
    array_index: f32,
    _pad0:       f32,
    _pad1:       f32,
    _pad2:       f32,
  }

  /* Uniform buffer – matches C struct uniform_data_t */
  struct Uniforms {
    projection: mat4x4f,
    view:       mat4x4f,
    instances:  array<InstanceData, 8>,
  }

  @group(0) @binding(0) var<uniform> ubo          : Uniforms;
  @group(0) @binding(1) var          tex_sampler  : sampler;
  @group(0) @binding(2) var          tex_array    : texture_2d_array<f32>;

  /* Vertex shader */
  struct VertexInput {
    @location(0)         position     : vec3f,
    @location(1)         uv           : vec2f,
    @builtin(instance_index) inst_idx : u32,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0)       tex_uv   : vec3f, /* xy = UV, z = array layer index */
  }

  @vertex
  fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let inst       = ubo.instances[in.inst_idx];
    let model_view = ubo.view * inst.model;
    out.tex_uv     = vec3f(in.uv, inst.array_index);
    out.position   = ubo.projection * model_view * vec4f(in.position, 1.0);
    return out;
  }

  /* Fragment shader */
  @fragment
  fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex_array, tex_sampler, in.tex_uv.xy,
                         u32(in.tex_uv.z));
  }
);
// clang-format on
