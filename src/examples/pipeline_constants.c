/* -------------------------------------------------------------------------- *
 * WebGPU Example - Pipeline Constants (Specialization Constants)
 *
 * Demonstrates using WebGPU pipeline override constants (WGSL `override`) to
 * create multiple render pipelines from a single "uber" shader. Each pipeline
 * is compiled with different constant values that control the lighting model
 * used in the fragment shader.
 *
 * The scene consists of a glTF model (color_teapot_spheres) rendered three
 * times side-by-side in separate viewports, each using a different lighting
 * model selected at pipeline creation time via WGPUConstantEntry:
 *   - Left:   Phong shading (model 0)
 *   - Center: Toon shading  (model 1) with configurable desaturation
 *   - Right:  Textured      (model 2) using a metalplate colormap
 *
 * This is the WebGPU equivalent of Vulkan's VkSpecializationInfo, mapped to
 * WGSL `override` declarations and WGPUConstantEntry at pipeline creation.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/specializationconstants
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
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* uber_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Pipeline Constants example
 * -------------------------------------------------------------------------- */

/* Texture file buffer for async loading (512*512*4 = 1 MB) */
#define TEX_FILE_BUFFER_SIZE (1024 * 1024)

/* Uniform data matching the shader UBO layout */
typedef struct {
  mat4 projection;
  mat4 model_view;
  vec4 light_pos;
} uniform_data_t;

/* State struct */
static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t model;
  WGPUBuffer vertex_buffer;
  WGPUBuffer index_buffer;
  bool model_loaded;

  /* Texture (colormap for textured lighting mode) */
  wgpu_texture_t colormap;
  uint8_t tex_file_buffer[TEX_FILE_BUFFER_SIZE];

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo_data;

  /* Bind group layout + bind group */
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;

  /* Pipeline layout */
  WGPUPipelineLayout pipeline_layout;

  /* Three pipelines for three lighting modes */
  struct {
    WGPURenderPipeline phong;
    WGPURenderPipeline toon;
    WGPURenderPipeline textured;
  } pipelines;

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

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
};

/* -------------------------------------------------------------------------- *
 * Model loading
 * -------------------------------------------------------------------------- */

/* Loading descriptor: pre-transform vertices, pre-multiply vertex colors.
 * No FlipY — WebGPU uses Y-up like OpenGL. */
static const gltf_model_desc_t model_load_desc = {
  .loading_flags = GltfLoadingFlag_PreTransformVertices
                   | GltfLoadingFlag_PreMultiplyVertexColors,
};

static void load_model(void)
{
  bool ok = gltf_model_load_from_file(
    &state.model, "assets/models/color_teapot_spheres.gltf", 1.0f);
  if (!ok) {
    printf("Failed to load color_teapot_spheres.gltf\n");
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
  gltf_model_t* m   = &state.model;
  size_t vb_size    = m->vertex_count * sizeof(gltf_vertex_t);

  /* Bake node transforms into vertex data */
  gltf_vertex_t* xformed = (gltf_vertex_t*)malloc(vb_size);
  memcpy(xformed, m->vertices, vb_size);
  gltf_model_bake_node_transforms(m, xformed, &model_load_desc);

  /* Upload vertices */
  state.vertex_buffer = wgpuDeviceCreateBuffer(
    device, &(WGPUBufferDescriptor){
              .label = STRVIEW("Model vertex buffer"),
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
                .label = STRVIEW("Model index buffer"),
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
 * Texture loading (async via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void texture_fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  uint8_t* pixels            = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    state.colormap.desc = (wgpu_texture_desc_t){
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
    state.colormap.desc.is_dirty = true;
  }
}

static void init_texture(struct wgpu_context_t* wgpu_context)
{
  /* Create a placeholder texture (will be replaced when fetch completes) */
  state.colormap = wgpu_create_color_bars_texture(wgpu_context, NULL);

  /* Start async fetch */
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/metalplate_nomips_rgba.png",
    .callback = texture_fetch_callback,
    .buffer   = SFETCH_RANGE(state.tex_file_buffer),
  });
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
  /* Initialize light position */
  /* Vulkan: lightPos = (0, -2, 1, 0) → WebGPU: negate Y → (0, 2, 1, 0) */
  glm_vec4_copy((vec4){0.0f, 2.0f, 1.0f, 0.0f}, state.ubo_data.light_pos);

  state.uniform_buffer = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
      .label = STRVIEW("Uniform buffer"),
      .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
      .size  = sizeof(uniform_data_t),
    });
}

static void update_uniform_buffers(struct wgpu_context_t* wgpu_context)
{
  float aspect
    = ((float)wgpu_context->width / 3.0f) / (float)wgpu_context->height;

  camera_set_perspective(&state.camera, 60.0f, aspect, 0.1f, 512.0f);

  glm_mat4_copy(state.camera.matrices.perspective, state.ubo_data.projection);
  glm_mat4_copy(state.camera.matrices.view, state.ubo_data.model_view);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0,
                       &state.ubo_data, sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group layout & bind group
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
      .buffer = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniform_data_t),
      },
    },
    [1] = {
      /* Binding 1: Fragment shader sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = {
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
    [2] = {
      /* Binding 2: Fragment shader texture */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .texture = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Pipeline constants BGL"),
                            .entryCount = ARRAY_SIZE(entries),
                            .entries    = entries,
                          });

  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Pipeline constants layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
}

static void init_bind_group(struct wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)

  WGPUBindGroupEntry entries[3] = {
    [0] = {
      .binding = 0,
      .buffer  = state.uniform_buffer,
      .offset  = 0,
      .size    = sizeof(uniform_data_t),
    },
    [1] = {
      .binding = 1,
      .sampler = state.colormap.sampler,
    },
    [2] = {
      .binding     = 2,
      .textureView = state.colormap.view,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label  = STRVIEW("Pipeline constants bind group"),
                            .layout = state.bind_group_layout,
                            .entryCount = ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
}

/* -------------------------------------------------------------------------- *
 * Pipelines
 * -------------------------------------------------------------------------- */

/**
 * @brief Creates a render pipeline with the given override constants.
 *
 * In WebGPU, Vulkan specialization constants map to WGSL `override`
 * declarations. The constants are provided via WGPUConstantEntry at
 * pipeline creation time, allowing the shader compiler to optimize
 * dead branches away.
 */
static WGPURenderPipeline
create_pipeline_with_constants(struct wgpu_context_t* wgpu_context,
                               uint32_t lighting_model,
                               float toon_desaturation_factor)
{
  /* Override constants for the fragment shader */
  WGPUConstantEntry frag_constants[2] = {
    [0] = {
      .key   = STRVIEW("LIGHTING_MODEL"),
      .value = (double)lighting_model,
    },
    [1] = {
      .key   = STRVIEW("PARAM_TOON_DESATURATION"),
      .value = (double)toon_desaturation_factor,
    },
  };

  WGPUShaderModule shader_module
    = wgpu_create_shader_module(wgpu_context->device, uber_shader_wgsl);

  /* Vertex buffer layout matching gltf_vertex_t */
  WGPUVertexAttribute vertex_attrs[] = {
    /* location 0: position (vec3f) */
    {.shaderLocation = 0,
     .offset         = offsetof(gltf_vertex_t, position),
     .format         = WGPUVertexFormat_Float32x3},
    /* location 1: normal (vec3f) */
    {.shaderLocation = 1,
     .offset         = offsetof(gltf_vertex_t, normal),
     .format         = WGPUVertexFormat_Float32x3},
    /* location 2: uv (vec2f) */
    {.shaderLocation = 2,
     .offset         = offsetof(gltf_vertex_t, uv0),
     .format         = WGPUVertexFormat_Float32x2},
    /* location 3: color (vec4f) */
    {.shaderLocation = 3,
     .offset         = offsetof(gltf_vertex_t, color),
     .format         = WGPUVertexFormat_Float32x4},
  };

  WGPUVertexBufferLayout vertex_buffer_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(vertex_attrs),
    .attributes     = vertex_attrs,
  };

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(true);

  /* Depth stencil state */
  WGPUDepthStencilState depth_stencil_state
    = wgpu_create_depth_stencil_state(&(create_depth_stencil_state_desc_t){
      .format              = wgpu_context->depth_stencil_format,
      .depth_write_enabled = true,
    });

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Pipeline constants - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module        = shader_module,
      .entryPoint    = STRVIEW("fs_main"),
      .constantCount = ARRAY_SIZE(frag_constants),
      .constants     = frag_constants,
      .targetCount   = 1,
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
    .depthStencil = &depth_stencil_state,
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(pipeline != NULL);

  wgpuShaderModuleRelease(shader_module);
  return pipeline;
}

static void init_pipelines(struct wgpu_context_t* wgpu_context)
{
  /* Phong: lighting model 0, desaturation unused (0.0) */
  state.pipelines.phong = create_pipeline_with_constants(wgpu_context, 0, 0.0f);

  /* Toon: lighting model 1, desaturation factor 0.5 */
  state.pipelines.toon = create_pipeline_with_constants(wgpu_context, 1, 0.5f);

  /* Textured: lighting model 2, desaturation unused (0.0) */
  state.pipelines.textured
    = create_pipeline_with_constants(wgpu_context, 2, 0.0f);
}

/* -------------------------------------------------------------------------- *
 * Draw model helper
 * -------------------------------------------------------------------------- */

static void draw_model(WGPURenderPassEncoder pass)
{
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, state.vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  if (state.index_buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(
      pass, state.index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
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

  igBegin("Pipeline Constants", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Device", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igText("GPU: %s", wgpu_context->platform_info.device);
    igText("Backend: %s", wgpu_context->platform_info.backend);
  }

  if (igCollapsingHeader_BoolPtr("Info", NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
    igText("Three viewports, one shader");
    igSeparator();
    igText("Left:   Phong shading");
    igText("Center: Toon shading");
    igText("Right:  Textured");
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

  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 2,
    .num_channels = 1,
    .num_lanes    = 2,
    .logger.func  = slog_func,
  });

  /* Camera setup */
  /* Vulkan: type = lookat, rotation = (-40, -90, 0), translation = (0, 0, -2)
   * WebGPU: negate Y for cam position, negate pitch for cam rotation */
  camera_init(&state.camera);
  state.camera.type      = CameraType_LookAt;
  state.camera.invert_dx = true;
  state.camera.invert_dy = true;
  camera_set_position(&state.camera, (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -2.0f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-40.0f, -90.0f, 0.0f));
  camera_set_perspective(&state.camera, 60.0f,
                         ((float)wgpu_context->width / 3.0f)
                           / (float)wgpu_context->height,
                         0.1f, 512.0f);

  /* Load model synchronously */
  load_model();
  create_model_buffers(wgpu_context);

  /* Initialize texture (async fetch) */
  init_texture(wgpu_context);

  /* Uniform buffer */
  init_uniform_buffer(wgpu_context);

  /* Bind group layout + pipeline layout */
  init_bind_group_layout(wgpu_context);

  /* Bind group */
  init_bind_group(wgpu_context);

  /* Create the three pipelines with different override constants */
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

  /* Recreate texture when pixel data loaded */
  if (state.colormap.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.colormap);
    FREE_TEXTURE_PIXELS(state.colormap);
    init_bind_group(wgpu_context);
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

  /* Render */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = wgpu_context->depth_stencil_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Full scissor rect */
  uint32_t w = (uint32_t)wgpu_context->width;
  uint32_t h = (uint32_t)wgpu_context->height;
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0, 0, w, h);

  /* Shared bind group for all three pipelines */
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group, 0, 0);

  float vp_width = (float)w / 3.0f;

  if (state.model_loaded) {
    /* Left viewport: Phong shading */
    wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f, vp_width, (float)h,
                                     0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.phong);
    draw_model(rpass_enc);

    /* Center viewport: Toon shading */
    wgpuRenderPassEncoderSetViewport(rpass_enc, vp_width, 0.0f, vp_width,
                                     (float)h, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.toon);
    draw_model(rpass_enc);

    /* Right viewport: Textured */
    wgpuRenderPassEncoderSetViewport(rpass_enc, vp_width * 2.0f, 0.0f, vp_width,
                                     (float)h, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipelines.textured);
    draw_model(rpass_enc);
  }

  wgpuRenderPassEncoderEnd(rpass_enc);

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  wgpuRenderPassEncoderRelease(rpass_enc);
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
  gltf_model_destroy(&state.model);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertex_buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.index_buffer)

  /* Destroy texture */
  wgpu_destroy_texture(&state.colormap);

  /* Uniform buffer */
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)

  /* Bind group + layout */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)

  /* Pipeline layout */
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)

  /* Pipelines */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.phong)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.toon)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipelines.textured)
}

/* -------------------------------------------------------------------------- *
 * Entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Pipeline Constants",
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
 * "Uber" shader that uses override constants to select the lighting model
 * at pipeline creation time. This is the WebGPU equivalent of Vulkan
 * specialization constants.
 *
 * Override constants:
 *   LIGHTING_MODEL          : u32 — 0=Phong, 1=Toon, 2=Textured
 *   PARAM_TOON_DESATURATION : f32 — desaturation factor for toon mode
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* uber_shader_wgsl = CODE(
  /* ---- Uniform buffer ---- */
  struct UBO {
    projection : mat4x4f,
    modelView  : mat4x4f,
    lightPos   : vec4f,
  };

  @group(0) @binding(0) var<uniform> ubo : UBO;
  @group(0) @binding(1) var colorSampler : sampler;
  @group(0) @binding(2) var colorTexture : texture_2d<f32>;

  /* ---- Override constants (pipeline constants) ---- */
  override LIGHTING_MODEL : u32 = 0u;
  override PARAM_TOON_DESATURATION : f32 = 0.0;

  /* ---- Vertex shader ---- */
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
    @location(2) uv       : vec2f,
    @location(3) viewVec  : vec3f,
    @location(4) lightVec : vec3f,
  };

  @vertex
  fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.color = in.color.rgb;
    out.uv = in.uv;

    out.position = ubo.projection * ubo.modelView * vec4f(in.position, 1.0);

    let pos = ubo.modelView * vec4f(in.position, 1.0);
    out.normal = (ubo.modelView * vec4f(in.normal, 0.0)).xyz;
    let lPos = (ubo.modelView * vec4f(ubo.lightPos.xyz, 0.0)).xyz;
    out.lightVec = lPos - pos.xyz;
    out.viewVec = -pos.xyz;

    return out;
  }

  /* ---- Fragment shader ---- */
  @fragment
  fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    if (LIGHTING_MODEL == 0u) {
      /* Phong shading */
      let ambient = in.color * vec3f(0.25);
      let N = normalize(in.normal);
      let L = normalize(in.lightVec);
      let V = normalize(in.viewVec);
      let R = reflect(-L, N);
      let diffuse = max(dot(N, L), 0.0) * in.color;
      let specular = pow(max(dot(R, V), 0.0), 32.0) * vec3f(0.75);
      return vec4f(ambient + diffuse * 1.75 + specular, 1.0);
    } else if (LIGHTING_MODEL == 1u) {
      /* Toon shading */
      let N = normalize(in.normal);
      let L = normalize(in.lightVec);
      let intensity = dot(N, L);
      var color : vec3f;
      if (intensity > 0.98) {
        color = in.color * 1.5;
      } else if (intensity > 0.9) {
        color = in.color * 1.0;
      } else if (intensity > 0.5) {
        color = in.color * 0.6;
      } else if (intensity > 0.25) {
        color = in.color * 0.4;
      } else {
        color = in.color * 0.2;
      }
      /* Desaturate */
      let gray = dot(vec3f(0.2126, 0.7152, 0.0722), color);
      color = mix(color, vec3f(gray), PARAM_TOON_DESATURATION);
      return vec4f(color, 1.0);
    } else {
      /* Textured with lighting */
      let texColor = textureSample(colorTexture, colorSampler, in.uv).rrra;
      let ambient = texColor.rgb * vec3f(0.25) * in.color;
      let N = normalize(in.normal);
      let L = normalize(in.lightVec);
      let V = normalize(in.viewVec);
      let R = reflect(-L, N);
      let diffuse = max(dot(N, L), 0.0) * texColor.rgb;
      let specular = pow(max(dot(R, V), 0.0), 32.0) * texColor.a;
      return vec4f(ambient + diffuse + vec3f(specular), 1.0);
    }
  }
);
// clang-format on
