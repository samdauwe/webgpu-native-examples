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
 * WebGPU Example - Spherical Environment Mapping
 *
 * Demonstrates spherical environment mapping using different mat cap textures
 * stored in an array texture. Uses a glTF Chinese Dragon model with
 * per-vertex coloring as the mesh, and allows toggling between 6 different
 * material captures via a GUI slider.
 *
 * Mat cap lookup: the eye-space reflection vector is projected into a sphere
 * to index into the 2D matcap texture. The array layer selects the material.
 *
 * Ported from Sascha Willems' Vulkan example "sphericalenvmapping"
 * https://github.com/SaschaWillems/Vulkan/tree/master/examples/sphericalenvmapping
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders (declared here, defined at bottom of file)
 * -------------------------------------------------------------------------- */

static const char* sem_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

static const char* model_path   = "assets/models/chinesedragon.gltf";
static const char* texture_path = "assets/textures/matcap_array_rgba.png";

/* Matcap atlas: 6 layers stacked vertically, each 256x256 RGBA */
#define MATCAP_LAYER_SIZE 256
#define MATCAP_LAYER_COUNT 6

/* -------------------------------------------------------------------------- *
 * Uniform data (must match WGSL layout)
 * -------------------------------------------------------------------------- */

typedef struct {
  mat4 projection;
  mat4 model;
  mat4 normal;
  mat4 view;
  int32_t tex_index;
  float _pad[3]; /* Align to 16 bytes */
} uniform_data_t;

/* -------------------------------------------------------------------------- *
 * Global state
 * -------------------------------------------------------------------------- */

static struct {
  /* Camera */
  camera_t camera;

  /* Model */
  gltf_model_t model;
  bool model_loaded;

  struct {
    WGPUBuffer vertex;
    WGPUBuffer index;
  } model_buffers;

  /* Uniform buffer */
  WGPUBuffer uniform_buffer;
  uniform_data_t ubo;

  /* Matcap array texture */
  wgpu_texture_t matcap_texture;
  uint8_t
    file_buffer[MATCAP_LAYER_SIZE * MATCAP_LAYER_SIZE * MATCAP_LAYER_COUNT * 4];

  /* Depth texture */
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

  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDepthStencilAttachment depth_stencil_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;

  /* Timing */
  uint64_t last_frame_time;

  WGPUBool initialized;
} state = {
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
 * Model loading and GPU buffer creation
 * -------------------------------------------------------------------------- */

static void load_model(void)
{
  gltf_model_desc_t desc = {
    .loading_flags = GltfLoadingFlag_PreTransformVertices
                     | GltfLoadingFlag_PreMultiplyVertexColors,
  };

  state.model_loaded
    = gltf_model_load_from_file_ext(&state.model, model_path, 1.0f, &desc);
  if (!state.model_loaded) {
    printf("Failed to load model: %s\n", model_path);
  }
}

static void create_model_buffers(struct wgpu_context_t* wgpu_context)
{
  if (!state.model_loaded) {
    return;
  }

  WGPUDevice device = wgpu_context->device;

  /* Vertex buffer */
  {
    uint32_t vb_size
      = state.model.vertex_count * (uint32_t)sizeof(gltf_vertex_t);
    state.model_buffers.vertex = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model vertex buffer"),
                .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
                .size  = vb_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.vertex, 0,
                         state.model.vertices, vb_size);
  }

  /* Index buffer */
  {
    uint32_t ib_size = state.model.index_count * (uint32_t)sizeof(uint32_t);
    state.model_buffers.index = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .label = STRVIEW("Model index buffer"),
                .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
                .size  = ib_size,
                .mappedAtCreation = false,
              });
    wgpuQueueWriteBuffer(wgpu_context->queue, state.model_buffers.index, 0,
                         state.model.indices, ib_size);
  }
}

/* Forward declaration */
static void init_bind_group(struct wgpu_context_t* wgpu_context);

/* -------------------------------------------------------------------------- *
 * Matcap array texture (PNG atlas loaded via sokol_fetch)
 * -------------------------------------------------------------------------- */

static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("Matcap texture fetch failed, error: %d\n", response->error_code);
    return;
  }

  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  uint8_t* pixels            = image_pixels_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);

  if (!pixels) {
    printf("Failed to decode matcap atlas image\n");
    return;
  }

  /* Validate atlas dimensions: W x (W * MATCAP_LAYER_COUNT) */
  if (img_width != MATCAP_LAYER_SIZE
      || img_height != MATCAP_LAYER_SIZE * MATCAP_LAYER_COUNT) {
    printf("Unexpected atlas size: %dx%d (expected %dx%d)\n", img_width,
           img_height, MATCAP_LAYER_SIZE,
           MATCAP_LAYER_SIZE * MATCAP_LAYER_COUNT);
    image_free(pixels);
    return;
  }

  /* Mark the texture as dirty so it will be created/recreated on next frame */
  state.matcap_texture.desc = (wgpu_texture_desc_t){
    .extent = (WGPUExtent3D){
      .width              = MATCAP_LAYER_SIZE,
      .height             = MATCAP_LAYER_SIZE,
      .depthOrArrayLayers = MATCAP_LAYER_COUNT,
    },
    .format = WGPUTextureFormat_RGBA8Unorm,
    .pixels = {
      .ptr  = pixels,
      .size = (size_t)MATCAP_LAYER_SIZE * MATCAP_LAYER_SIZE * MATCAP_LAYER_COUNT
              * 4,
    },
    /* Force 2DArray view (not Cube, even though layer_count == 6) */
    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
  };
  state.matcap_texture.desc.is_dirty = true;
}

static void init_matcap_texture(wgpu_context_t* wgpu_context)
{
  /* Create a 1x1 placeholder texture array until the real one loads */
  uint8_t placeholder[4 * MATCAP_LAYER_COUNT];
  memset(placeholder, 128, sizeof(placeholder));
  state.matcap_texture = wgpu_create_texture(
    wgpu_context, &(wgpu_texture_desc_t){
                    .extent = {1, 1, MATCAP_LAYER_COUNT},
                    .format = WGPUTextureFormat_RGBA8Unorm,
                    .pixels = {
                      .ptr  = placeholder,
                      .size = sizeof(placeholder),
                    },
                    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_2D_ARRAY,
                  });

  /* Kick off async file load */
  sfetch_send(&(sfetch_request_t){
    .path     = texture_path,
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
  });
}

static void update_matcap_texture(wgpu_context_t* wgpu_context)
{
  if (!state.matcap_texture.desc.is_dirty) {
    return;
  }

  wgpu_recreate_texture(wgpu_context, &state.matcap_texture);

  /* Free the stbi pixel data */
  if (state.matcap_texture.desc.pixels.ptr) {
    image_free((void*)state.matcap_texture.desc.pixels.ptr);
    state.matcap_texture.desc.pixels.ptr  = NULL;
    state.matcap_texture.desc.pixels.size = 0;
  }

  /* Recreate bind group with new texture view */
  if (state.bind_group) {
    wgpuBindGroupRelease(state.bind_group);
    state.bind_group = NULL;
  }
  init_bind_group(wgpu_context);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer
 * -------------------------------------------------------------------------- */

static void init_uniform_buffer(struct wgpu_context_t* wgpu_context)
{
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
  glm_mat4_identity(state.ubo.model);

  /* Normal matrix = transpose(inverse(view * model)) */
  mat4 view_model;
  glm_mat4_mul(state.ubo.view, state.ubo.model, view_model);
  mat4 inv;
  glm_mat4_inv(view_model, inv);
  glm_mat4_transpose_to(inv, state.ubo.normal);

  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer, 0, &state.ubo,
                       sizeof(uniform_data_t));
}

/* -------------------------------------------------------------------------- *
 * Bind group & pipeline
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(struct wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
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
                            .label      = STRVIEW("SEM bind group layout"),
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
      .sampler = state.matcap_texture.sampler,
    },
    [2] = {
      .binding     = 2,
      .textureView = state.matcap_texture.view,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("SEM bind group"),
                            .layout     = state.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(entries),
                            .entries    = entries,
                          });
}

static void init_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;

  /* Pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    device, &(WGPUPipelineLayoutDescriptor){
              .label                = STRVIEW("SEM pipeline layout"),
              .bindGroupLayoutCount = 1,
              .bindGroupLayouts     = &state.bind_group_layout,
            });

  /* Shader module */
  WGPUShaderModule shader = wgpu_create_shader_module(device, sem_shader_wgsl);

  /* Vertex attributes: position, normal, color */
  WGPUVertexAttribute attrs[3] = {
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
    [2] = {
      .shaderLocation = 2,
      .format         = WGPUVertexFormat_Float32x4,
      .offset         = offsetof(gltf_vertex_t, color),
    },
  };

  WGPUVertexBufferLayout vb_layout = {
    .arrayStride    = sizeof(gltf_vertex_t),
    .stepMode       = WGPUVertexStepMode_Vertex,
    .attributeCount = ARRAY_SIZE(attrs),
    .attributes     = attrs,
  };

  WGPUBlendState blend        = wgpu_create_blend_state(false);
  WGPUColorTargetState target = {
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
    device, &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("SEM pipeline"),
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
        .cullMode  = WGPUCullMode_Back,
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
        .targets     = &target,
      },
    });

  WGPU_RELEASE_RESOURCE(ShaderModule, shader);
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

  for (uint32_t n = 0; n < mdl->linear_node_count; n++) {
    gltf_node_t* node = mdl->linear_nodes[n];
    if (!node->mesh) {
      continue;
    }
    gltf_mesh_t* mesh = node->mesh;
    for (uint32_t p = 0; p < mesh->primitive_count; p++) {
      gltf_primitive_t* prim = &mesh->primitives[p];
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

static void render_gui(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  if (igCollapsingHeader_BoolPtr("Material", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    imgui_overlay_slider_int("Material cap", &state.ubo.tex_index, 0,
                             MATCAP_LAYER_COUNT - 1);
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
      &state.camera, 60.0f,
      (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);
  }
  else if (!imgui_overlay_want_capture_mouse()) {
    if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
        && input_event->mouse_btn_pressed
        && input_event->mouse_button == BUTTON_LEFT) {
      camera_rotate(&state.camera,
                    (vec3){-input_event->mouse_dy * state.camera.rotation_speed,
                           input_event->mouse_dx * state.camera.rotation_speed,
                           0.0f});
      return;
    }
    camera_on_input_event(&state.camera, input_event);
  }
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  stm_setup();

  /* Sokol_fetch for async texture loading */
  sfetch_setup(&(sfetch_desc_t){
    .max_requests = 1,
    .num_channels = 1,
    .num_lanes    = 1,
    .logger.func  = slog_func,
  });

  /* Camera */
  camera_init(&state.camera);
  state.camera.type           = CameraType_LookAt;
  state.camera.invert_dx      = true;
  state.camera.invert_dy      = true;
  state.camera.rotation_speed = 0.75f;
  camera_set_position(&state.camera, (vec3)VKY_TO_WGPU_VEC3(0.0f, 0.0f, -3.5f));
  camera_set_rotation(&state.camera,
                      (vec3)VKY_TO_WGPU_CAM_ROT(-25.0f, 23.75f, 0.0f));
  camera_set_perspective(
    &state.camera, 60.0f,
    (float)wgpu_context->width / (float)wgpu_context->height, 0.1f, 256.0f);

  /* Default texture index */
  state.ubo.tex_index = 0;

  /* Load model synchronously */
  load_model();
  create_model_buffers(wgpu_context);

  /* Init GPU resources */
  init_depth_texture(wgpu_context);
  init_matcap_texture(wgpu_context);
  init_uniform_buffer(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_bind_group(wgpu_context);
  init_pipeline(wgpu_context);

  /* ImGui overlay */
  imgui_overlay_init(wgpu_context);

  state.last_frame_time = stm_now();
  state.initialized     = true;

  return EXIT_SUCCESS;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized || !state.model_loaded) {
    return EXIT_SUCCESS;
  }

  /* Process sokol_fetch requests */
  sfetch_dowork();

  /* Update texture when pixel data loaded */
  update_matcap_texture(wgpu_context);

  /* Timing */
  uint64_t now          = stm_now();
  float dt              = (float)stm_sec(stm_diff(now, state.last_frame_time));
  state.last_frame_time = now;

  /* Update uniforms */
  update_uniform_buffer(wgpu_context, dt);

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, dt);
  render_gui(wgpu_context);

  /* Begin render pass */
  state.color_attachment.view         = wgpu_context->swapchain_view;
  state.depth_stencil_attachment.view = state.depth.view;

  WGPUDevice device          = wgpu_context->device;
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Draw */
  wgpuRenderPassEncoderSetPipeline(rpass, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass, 0, state.bind_group, 0, 0);
  draw_model(rpass, &state.model, state.model_buffers.vertex,
             state.model_buffers.index);

  wgpuRenderPassEncoderEnd(rpass);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass);
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd_buffer);
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_enc);

  /* ImGui overlay */
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

  /* GPU resources */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.vertex);
  WGPU_RELEASE_RESOURCE(Buffer, state.model_buffers.index);

  wgpu_destroy_texture(&state.matcap_texture);
  gltf_model_destroy(&state.model);
}

/* -------------------------------------------------------------------------- *
 * Main entry point
 * -------------------------------------------------------------------------- */

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Spherical Environment Mapping",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* sem_shader_wgsl = CODE(
  struct Uniforms {
    projection: mat4x4f,
    model: mat4x4f,
    normal: mat4x4f,
    view: mat4x4f,
    texIndex: i32,
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var matCapSampler: sampler;
  @group(0) @binding(2) var matCapArray: texture_2d_array<f32>;

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
  }

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) eyePos: vec3f,
    @location(2) normal: vec3f,
  }

  @vertex
  fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let modelView = uniforms.view * uniforms.model;
    out.color = in.color.rgb;
    out.eyePos = normalize((modelView * vec4f(in.position, 1.0)).xyz);
    out.normal = normalize((uniforms.normal * vec4f(in.normal, 0.0)).xyz);
    out.position = uniforms.projection * modelView * vec4f(in.position, 1.0);
    return out;
  }

  @fragment
  fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = reflect(in.eyePos, in.normal);
    let r2 = vec3f(r.x, r.y, r.z + 1.0);
    let m = 2.0 * length(r2);
    let vN = r.xy / m + 0.5;
    let matcap = textureSample(matCapArray, matCapSampler, vN, uniforms.texIndex);
    return vec4f(matcap.rgb * clamp(in.color.r * 2.0, 0.0, 1.0), 1.0);
  }
);
// clang-format on
