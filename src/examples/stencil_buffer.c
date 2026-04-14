#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include "core/camera.h"
#include "core/gltf_model.h"

#include <cglm/cglm.h>
#include <string.h>

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
 * WebGPU Example - Stencil Buffer Outlines
 *
 * This example demonstrates rendering outlines using the stencil buffer.
 * A 3D model is rendered with toon shading and the stencil buffer is filled.
 * Then the model is rendered again, slightly extruded along its normals,
 * only where the stencil was not set by the first pass, creating an outline
 * effect.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/stencilbuffer
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* toon_shader_wgsl;
static const char* outline_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define DEPTH_FORMAT WGPUTextureFormat_Depth24PlusStencil8

/* -------------------------------------------------------------------------- *
 * Uniform data - matches shader layout with proper alignment
 *
 * Layout (std140):
 *   offset  0: mat4x4f projection  (64 bytes)
 *   offset 64: mat4x4f model       (64 bytes)
 *   offset 128: vec4f  lightPos    (16 bytes)
 *   offset 144: f32    outlineWidth (4 bytes)
 *   offset 148: padding            (12 bytes) -> total 160 bytes
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 projection;     /* offset 0   */
  mat4 model;          /* offset 64  */
  float light_pos[4];  /* offset 128 */
  float outline_width; /* offset 144 */
  float _padding[3];   /* offset 148 - pad to 16-byte alignment */
} uniform_data_t;

_Static_assert(sizeof(uniform_data_t) == 160,
               "uniform_data_t must be 160 bytes");

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* glTF model */
  gltf_model_t model;
  bool model_loaded;

  /* GPU buffers for model geometry */
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  uint32_t index_count;

  /* Uniform data and buffer */
  uniform_data_t uniform_data;
  WGPUBuffer uniform_buffer;

  /* Bind group and layout */
  WGPUBindGroup bind_group;
  WGPUBindGroupLayout bind_group_layout;

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout;

  /* Render pipelines */
  struct {
    WGPURenderPipeline stencil;
    WGPURenderPipeline outline;
  } pipelines;

  /* Depth/stencil texture */
  WGPUTexture depth_texture;
  WGPUTextureView depth_texture_view;
  uint32_t depth_texture_width;
  uint32_t depth_texture_height;

  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* GUI settings */
  struct {
    float outline_width;
  } settings;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
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
    .outline_width = 0.025f,
  },
};

/* -------------------------------------------------------------------------- *
 * Depth/stencil texture management
 * -------------------------------------------------------------------------- */

static void update_depth_texture(wgpu_context_t* wgpu_context)
{
  if (state.depth_texture != NULL
      && state.depth_texture_width == (uint32_t)wgpu_context->width
      && state.depth_texture_height == (uint32_t)wgpu_context->height) {
    return;
  }

  /* Release old resources */
  if (state.depth_texture_view) {
    wgpuTextureViewRelease(state.depth_texture_view);
    state.depth_texture_view = NULL;
  }
  if (state.depth_texture) {
    wgpuTextureDestroy(state.depth_texture);
    wgpuTextureRelease(state.depth_texture);
    state.depth_texture = NULL;
  }

  state.depth_texture_width  = (uint32_t)wgpu_context->width;
  state.depth_texture_height = (uint32_t)wgpu_context->height;

  state.depth_texture = wgpuDeviceCreateTexture(
    wgpu_context->device,
    &(WGPUTextureDescriptor){
      .label = STRVIEW("Depth stencil texture"),
      .size  = {state.depth_texture_width, state.depth_texture_height, 1},
      .mipLevelCount = 1,
      .sampleCount   = 1,
      .dimension     = WGPUTextureDimension_2D,
      .format        = DEPTH_FORMAT,
      .usage         = WGPUTextureUsage_RenderAttachment,
    });

  state.depth_texture_view = wgpuTextureCreateView(
    state.depth_texture, &(WGPUTextureViewDescriptor){
                           .label           = STRVIEW("Depth stencil view"),
                           .format          = DEPTH_FORMAT,
                           .dimension       = WGPUTextureViewDimension_2D,
                           .mipLevelCount   = 1,
                           .arrayLayerCount = 1,
                         });
}

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

static void load_model(wgpu_context_t* wgpu_context)
{
  /* Load venus model with pre-transformed vertices (no FlipY for WebGPU) */
  bool ok = gltf_model_load_from_file_ext(
    &state.model, "assets/models/venus.gltf", 1.0f,
    &(gltf_model_desc_t){
      .loading_flags = GltfLoadingFlag_PreTransformVertices,
    });

  if (!ok) {
    fprintf(stderr, "Failed to load venus.gltf\n");
    return;
  }
  state.model_loaded = true;

  gltf_model_t* m = &state.model;

  /* Create vertex buffer */
  size_t vb_size      = m->vertex_count * sizeof(gltf_vertex_t);
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label            = STRVIEW("Model vertex buffer"),
      .usage            = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
      .size             = vb_size,
      .mappedAtCreation = true,
    });
  void* vdata = wgpuBufferGetMappedRange(state.vertex_buffer, 0, vb_size);
  memcpy(vdata, m->vertices, vb_size);
  wgpuBufferUnmap(state.vertex_buffer);

  /* Create index buffer */
  if (m->index_count > 0) {
    size_t ib_size     = m->index_count * sizeof(uint32_t);
    state.index_buffer = wgpuDeviceCreateBuffer(
      wgpu_context->device,
      &(WGPUBufferDescriptor){
        .label            = STRVIEW("Model index buffer"),
        .usage            = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
        .size             = ib_size,
        .mappedAtCreation = true,
      });
    void* idata = wgpuBufferGetMappedRange(state.index_buffer, 0, ib_size);
    memcpy(idata, m->indices, ib_size);
    wgpuBufferUnmap(state.index_buffer);
    state.index_count = m->index_count;
  }
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  /* Initialize uniform data */
  state.uniform_data.light_pos[0] = 0.0f;
  state.uniform_data.light_pos[1]
    = 2.0f; /* Vulkan (0,-2,1,0) -> WebGPU negate Y */
  state.uniform_data.light_pos[2]  = 1.0f;
  state.uniform_data.light_pos[3]  = 0.0f;
  state.uniform_data.outline_width = state.settings.outline_width;

  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(uniform_data_t),
    });
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update projection and view matrices from camera */
  glm_mat4_copy(state.camera.matrices.perspective,
                state.uniform_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.uniform_data.model);
  state.uniform_data.outline_width = state.settings.outline_width;

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.uniform_data, sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group and layout
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniform_data_t),
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupEntry bg_entries[1] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.uniform_buffer,
      .size    = sizeof(uniform_data_t),
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Uniform bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Render pipelines
 * -------------------------------------------------------------------------- */

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });

  /* Vertex buffer layout matching gltf_vertex_t:
   * position (vec3f), normal (vec3f), uv0, uv1, tangent, color, joint0, weight0
   * We only need position, color, and normal from the vertex.
   */

  /* gltf_vertex_t layout:
   * vec3 position  @ offset 0   (12 bytes)
   * vec3 normal    @ offset 12  (12 bytes)
   * vec2 uv0       @ offset 24  (8 bytes)
   * vec2 uv1       @ offset 32  (8 bytes)
   * vec4 tangent   @ offset 40  (16 bytes)
   * vec4 color     @ offset 56  (16 bytes)
   * uint32[4] joint0 @ offset 72 (16 bytes)
   * vec4 weight0   @ offset 88  (16 bytes)
   * Total: 104 bytes
   */
  WGPUVertexAttribute vertex_attributes[3] = {
    /* location 0 : position (vec3f) */
    {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, position),
      .shaderLocation = 0,
    },
    /* location 1 : color (vec4f -> we read vec3f in shader) */
    {
      .format         = WGPUVertexFormat_Float32x4,
      .offset         = offsetof(gltf_vertex_t, color),
      .shaderLocation = 1,
    },
    /* location 2 : normal (vec3f) */
    {
      .format         = WGPUVertexFormat_Float32x3,
      .offset         = offsetof(gltf_vertex_t, normal),
      .shaderLocation = 2,
    },
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = (uint32_t)ARRAY_SIZE(vertex_attributes),
    .attributes     = vertex_attributes,
  };

  /* Create shader modules */
  WGPUShaderModule toon_shader
    = wgpu_create_shader_module(wgpu_context->device, toon_shader_wgsl);
  WGPUShaderModule outline_shader
    = wgpu_create_shader_module(wgpu_context->device, outline_shader_wgsl);

  /* Color target state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* ------------------------------------------------------------------ *
   * Pipeline 1: Toon shading + stencil fill
   *
   * Vulkan original:
   *   cullMode = NONE
   *   stencilTestEnable = TRUE
   *   both front/back: compare=ALWAYS, fail=REPLACE, depthFail=REPLACE,
   *                    pass=REPLACE, compareMask=0xFF, writeMask=0xFF, ref=1
   *   depthTest = TRUE, depthWrite = TRUE, depthCompare = LESS_OR_EQUAL
   * ------------------------------------------------------------------ */
  state.pipelines.stencil = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Stencil fill pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = toon_shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = toon_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = DEPTH_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare      = WGPUCompareFunction_LessEqual,
        .stencilFront = {
          .compare     = WGPUCompareFunction_Always,
          .failOp      = WGPUStencilOperation_Replace,
          .depthFailOp = WGPUStencilOperation_Replace,
          .passOp      = WGPUStencilOperation_Replace,
        },
        .stencilBack = {
          .compare     = WGPUCompareFunction_Always,
          .failOp      = WGPUStencilOperation_Replace,
          .depthFailOp = WGPUStencilOperation_Replace,
          .passOp      = WGPUStencilOperation_Replace,
        },
        .stencilReadMask  = 0xFF,
        .stencilWriteMask = 0xFF,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  /* ------------------------------------------------------------------ *
   * Pipeline 2: Outline rendering
   *
   * Vulkan original:
   *   cullMode = NONE (inherited from first pipeline description)
   *   stencilTestEnable = TRUE
   *   both front/back: compare=NOT_EQUAL, fail=KEEP, depthFail=KEEP,
   *                    pass=REPLACE, compareMask=0xFF, writeMask=0xFF, ref=1
   *   depthTest = FALSE, depthWrite = TRUE (inherited but depth test off)
   * ------------------------------------------------------------------ */
  state.pipelines.outline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device,
    &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Outline pipeline"),
      .layout = state.pipeline_layout,
      .vertex = {
        .module      = outline_shader,
        .entryPoint  = STRVIEW("vs_main"),
        .bufferCount = 1,
        .buffers     = &vertex_buffer_layout,
      },
      .fragment = &(WGPUFragmentState){
        .module      = outline_shader,
        .entryPoint  = STRVIEW("fs_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &blend_state,
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .cullMode  = WGPUCullMode_None,
        .frontFace = WGPUFrontFace_CCW,
      },
      .depthStencil = &(WGPUDepthStencilState){
        .format            = DEPTH_FORMAT,
        .depthWriteEnabled = false,
        .depthCompare      = WGPUCompareFunction_Always,
        .stencilFront = {
          .compare     = WGPUCompareFunction_NotEqual,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Replace,
        },
        .stencilBack = {
          .compare     = WGPUCompareFunction_NotEqual,
          .failOp      = WGPUStencilOperation_Keep,
          .depthFailOp = WGPUStencilOperation_Keep,
          .passOp      = WGPUStencilOperation_Replace,
        },
        .stencilReadMask  = 0xFF,
        .stencilWriteMask = 0xFF,
      },
      .multisample = {
        .count = 1,
        .mask  = 0xFFFFFFFF,
      },
    });

  /* Release shader modules */
  wgpuShaderModuleRelease(toon_shader);
  wgpuShaderModuleRelease(outline_shader);
}

/* -------------------------------------------------------------------------- *
 * Camera setup
 * -------------------------------------------------------------------------- */

static void init_camera(wgpu_context_t* wgpu_context)
{
  camera_init(&state.camera);
  state.camera.type = CameraType_LookAt;

  /* Vulkan rotation: (2.5, -35, 0) -> WebGPU: negate pitch */
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(2.5f, -35.0f, 0.0f));

  /* Vulkan translation: (0, 0, -2) */
  camera_set_translation(&state.camera, (vec3){0.0f, 0.0f, -2.0f});

  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 512.0f);

  state.camera.rotation_speed = 0.5f;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;

  camera_update_view_matrix(&state.camera);
}

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder rpass_enc)
{
  if (!state.model_loaded) {
    return;
  }

  gltf_model_t* m = &state.model;
  for (uint32_t n = 0; n < m->linear_node_count; n++) {
    gltf_node_t* node = m->linear_nodes[n];
    if (node->mesh == NULL) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];
      if (prim->has_indices && prim->index_count > 0) {
        wgpuRenderPassEncoderDrawIndexed(rpass_enc, prim->index_count, 1,
                                         prim->first_index, 0, 0);
      }
      else if (prim->vertex_count > 0) {
        wgpuRenderPassEncoderDraw(rpass_enc, prim->vertex_count, 1, 0, 0);
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
  igSetNextWindowSize((ImVec2){250.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  imgui_overlay_input_float("Outline width", &state.settings.outline_width,
                            0.01f, "%.3f");

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    update_depth_texture(wgpu_context);
    camera_update_aspect_ratio(&state.camera, (float)wgpu_context->width
                                                / (float)wgpu_context->height);
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
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  init_camera(wgpu_context);
  load_model(wgpu_context);
  update_depth_texture(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_bind_group(wgpu_context);
  init_pipelines(wgpu_context);
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Frame
 * -------------------------------------------------------------------------- */

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized || !state.model_loaded) {
    return EXIT_FAILURE;
  }

  /* Update uniform buffers */
  update_uniform_buffers(wgpu_context);

  /* Calculate delta time for ImGui */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;

  /* Start ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  /* Setup render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth_texture_view;

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Bind shared vertex/index buffers */
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetIndexBuffer(
    rpass_enc, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group, 0, NULL);

  /* Set stencil reference value to 1 (matching Vulkan reference=1) */
  wgpuRenderPassEncoderSetStencilReference(rpass_enc, 1);

  /* Pass 1: Toon shading + fill stencil buffer */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.stencil);
  draw_model(rpass_enc);

  /* Pass 2: Outline rendering (only where stencil != 1) */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.outline);
  draw_model(rpass_enc);

  wgpuRenderPassEncoderEnd(rpass_enc);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * Shutdown
 * -------------------------------------------------------------------------- */

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  /* Release pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.stencil)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.outline)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)

  /* Release bind group */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)

  /* Release uniform buffer */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Release model buffers */
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Release depth/stencil texture */
  if (state.depth_texture_view) {
    wgpuTextureViewRelease(state.depth_texture_view);
    state.depth_texture_view = NULL;
  }
  if (state.depth_texture) {
    wgpuTextureDestroy(state.depth_texture);
    wgpuTextureRelease(state.depth_texture);
    state.depth_texture = NULL;
  }

  /* Release model data */
  if (state.model_loaded) {
    gltf_model_destroy(&state.model);
  }
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Stencil Buffer Outlines",
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

/* Toon shader - renders the model with cel/toon shading and fills stencil */
// clang-format off
static const char* toon_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model : mat4x4f,
    lightPos : vec4f,
    outlineWidth : f32,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) color : vec4f,
    @location(2) normal : vec3f,
  }

  struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) normal : vec3f,
    @location(1) color : vec3f,
    @location(2) lightVec : vec3f,
  }

  @vertex
  fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.color = vec3f(1.0, 0.0, 0.0);
    output.position = ubo.projection * ubo.model * vec4f(input.position, 1.0);
    output.normal = (mat3x3f(
      ubo.model[0].xyz,
      ubo.model[1].xyz,
      ubo.model[2].xyz
    ) * input.normal);
    let pos = ubo.model * vec4f(input.position, 1.0);
    let lPos = mat3x3f(
      ubo.model[0].xyz,
      ubo.model[1].xyz,
      ubo.model[2].xyz
    ) * ubo.lightPos.xyz;
    output.lightVec = lPos - pos.xyz;
    return output;
  }

  @fragment
  fn fs_main(input : VertexOutput) -> @location(0) vec4f {
    let N = normalize(input.normal);
    let L = normalize(input.lightVec);
    let intensity = dot(N, L);
    var color : vec3f;
    if (intensity > 0.98) {
      color = input.color * 1.5;
    } else if (intensity > 0.9) {
      color = input.color * 1.0;
    } else if (intensity > 0.5) {
      color = input.color * 0.6;
    } else if (intensity > 0.25) {
      color = input.color * 0.4;
    } else {
      color = input.color * 0.2;
    }
    // Desaturate a bit
    let gray = dot(vec3f(0.2126, 0.7152, 0.0722), color);
    color = mix(color, vec3f(gray), 0.1);
    return vec4f(color, 1.0);
  }
);
// clang-format on

/* Outline shader - extrudes the model along normals and renders white */
// clang-format off
static const char* outline_shader_wgsl = CODE(
  struct UBO {
    projection : mat4x4f,
    model : mat4x4f,
    lightPos : vec4f,
    outlineWidth : f32,
  }

  @group(0) @binding(0) var<uniform> ubo : UBO;

  struct VertexInput {
    @location(0) position : vec3f,
    @location(1) color : vec4f,
    @location(2) normal : vec3f,
  }

  @vertex
  fn vs_main(input : VertexInput) -> @builtin(position) vec4f {
    // Extrude along normal
    let pos = vec4f(input.position + input.normal * ubo.outlineWidth, 1.0);
    return ubo.projection * ubo.model * pos;
  }

  @fragment
  fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
  }
);
// clang-format on
