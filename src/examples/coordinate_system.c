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

#define STB_IMAGE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include <stb_image.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef STB_IMAGE_IMPLEMENTATION

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Coordinate System
 *
 * Illustrates the coordinate systems used in WebGPU. WebGPU's coordinate
 * systems match DirectX and Metal's coordinate systems in a graphics pipeline.
 * Y-axis is up in normalized device coordinate (NDC): point(-1.0, -1.0) in NDC
 * is located at the bottom-left corner of NDC. This example has several options
 * for changing relevant pipeline state, and displaying meshes with WebGPU or
 * Vulkan style coordinates.
 *
 * Ref:
 * https://gpuweb.github.io/gpuweb/
 * https://github.com/gpuweb/gpuweb/issues/416
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/negativeviewportheight/negativeviewportheight.cpp
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* quad_vertex_shader_wgsl;
static const char* quad_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Coordinate System example
 * -------------------------------------------------------------------------- */

#define TEXTURE_COUNT (2u)

/* Winding order options */
typedef enum winding_order_t {
  WINDING_ORDER_CW  = 0,
  WINDING_ORDER_CCW = 1,
} winding_order_t;

/* Quad type options */
typedef enum quad_type_t {
  QUAD_TYPE_VK_Y_DOWN   = 0, /* Vulkan style (y points downwards) */
  QUAD_TYPE_WEBGPU_Y_UP = 1, /* WebGPU style (y points upwards) */
} quad_type_t;

/* Vertex structure */
typedef struct vertex_t {
  vec3 pos;
  vec2 uv;
} vertex_t;

/* State struct */
static struct {
  /* Textures */
  struct {
    wgpu_texture_t cw;
    wgpu_texture_t ccw;
  } textures;
  struct {
    const char* file;
    wgpu_texture_t* texture;
  } texture_mappings[TEXTURE_COUNT];
  uint8_t file_buffer[512 * 512 * 4];
  /* Quad buffers */
  struct {
    wgpu_buffer_t vertices_y_up;
    wgpu_buffer_t vertices_y_down;
    wgpu_buffer_t indices_ccw;
    wgpu_buffer_t indices_cw;
  } quad;
  /* Bind groups */
  struct {
    WGPUBindGroup cw;
    WGPUBindGroup ccw;
  } bind_groups;
  WGPUBindGroupLayout bind_group_layout;
  /* Pipeline */
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline pipeline;
  /* Render pass descriptor */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* GUI settings */
  struct {
    int32_t winding_order;
    int32_t cull_mode;
    int32_t quad_type;
  } settings;
  const char* winding_order_str[2];
  const char* cull_mode_str[3];
  const char* quad_type_str[2];
  /* State flags */
  bool pipeline_needs_rebuild;
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .texture_mappings = {
    { .file = "assets/textures/texture_orientation_cw_rgba.png",  .texture = &state.textures.cw  },
    { .file = "assets/textures/texture_orientation_ccw_rgba.png", .texture = &state.textures.ccw },
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .settings = {
    .winding_order = WINDING_ORDER_CCW,
    .cull_mode     = 2, /* Index into cull_mode_str: 0=None, 1=Front, 2=Back */
    .quad_type     = QUAD_TYPE_WEBGPU_Y_UP,
  },
  .winding_order_str = {
    "Clock Wise",
    "Counter Clock Wise",
  },
  .cull_mode_str = {
    "None",
    "Front Face",
    "Back Face",
  },
  .quad_type_str = {
    "VK (Y Negative)",
    "WebGPU (Y Positive)",
  },
  .pipeline_needs_rebuild = false,
};

/**
 * @brief The fetch-callback is called by sokol_fetch.h when the data is loaded,
 * or when an error has occurred.
 */
static void fetch_callback(const sfetch_response_t* response)
{
  if (!response->fetched) {
    printf("File fetch failed, error: %d\n", response->error_code);
    return;
  }

  /* The file data has been fetched, since we provided a big-enough buffer we
   * can be sure that all data has been loaded here */
  int img_width, img_height, num_channels;
  const int desired_channels = 4;
  stbi_uc* pixels            = stbi_load_from_memory(
    response->data.ptr, (int)response->data.size, &img_width, &img_height,
    &num_channels, desired_channels);
  if (pixels) {
    wgpu_texture_t* texture = *(wgpu_texture_t**)response->user_data;
    texture->desc = (wgpu_texture_desc_t){
      .extent = (WGPUExtent3D) {
        .width              = img_width,
        .height             = img_height,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .pixels = {
        .ptr  = pixels,
        .size = img_width * img_height * 4,
      },
    };
    texture->desc.is_dirty = true;
  }
}

static void init_textures(wgpu_context_t* wgpu_context)
{
  /* Fetch the images and upload them into GPUTextures */
  for (uint8_t i = 0; i < (uint8_t)ARRAY_SIZE(state.texture_mappings); ++i) {
    wgpu_texture_t* texture = state.texture_mappings[i].texture;
    /* Create dummy texture */
    *(texture) = wgpu_create_color_bars_texture(
      wgpu_context,
      &(wgpu_texture_desc_t){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_RenderAttachment,
      });
    /* Start loading the image file */
    sfetch_send(&(sfetch_request_t){
      .path      = state.texture_mappings[i].file,
      .callback  = fetch_callback,
      .buffer    = SFETCH_RANGE(state.file_buffer),
      .user_data = {
        .ptr  = &texture,
        .size = sizeof(wgpu_texture_t*),
      },
    });
  }
}

static void init_quad_buffers(wgpu_context_t* wgpu_context)
{
  const float ar = (float)wgpu_context->height / (float)wgpu_context->width;

  /* WebGPU style (y points upwards) */
  // clang-format off
  vertex_t vertices_y_pos[4] = {
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f * ar,  1.0f, 1.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f * ar,  1.0f, 1.0f}, .uv = {1.0f, 0.0f}},
    {.pos = { 1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 1.0f}},
  };
  // clang-format on

  /* Vulkan style (y points downwards) */
  // clang-format off
  vertex_t vertices_y_neg[4] = {
    {.pos = {-1.0f * ar,  1.0f, 1.0f}, .uv = {0.0f, 1.0f}},
    {.pos = {-1.0f * ar, -1.0f, 1.0f}, .uv = {0.0f, 0.0f}},
    {.pos = { 1.0f * ar, -1.0f, 1.0f}, .uv = {1.0f, 0.0f}},
    {.pos = { 1.0f * ar,  1.0f, 1.0f}, .uv = {1.0f, 1.0f}},
  };
  // clang-format on

  state.quad.vertices_y_up = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad vertices buffer - Y up",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_t) * 4,
                    .initial.data = vertices_y_pos,
                  });

  state.quad.vertices_y_down = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad vertices buffer - Y down",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(vertex_t) * 4,
                    .initial.data = vertices_y_neg,
                  });

  /* Counter clock wise indices */
  static uint32_t indices_ccw[6] = {
    2, 1, 0, //
    0, 3, 2, //
  };
  state.quad.indices_ccw = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad indices buffer - CCW",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(uint32_t) * 6,
                    .count = (uint32_t)ARRAY_SIZE(indices_ccw),
                    .initial.data = indices_ccw,
                  });

  /* Clock wise indices */
  static uint32_t indices_cw[6] = {
    0, 1, 2, //
    2, 3, 0, //
  };
  state.quad.indices_cw = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Quad indices buffer - CW",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index,
                    .size  = sizeof(uint32_t) * 6,
                    .count = (uint32_t)ARRAY_SIZE(indices_cw),
                    .initial.data = indices_cw,
                  });
}

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 0 : Fragment shader texture view */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader image sampler */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    }
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Coordinate system - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    });
  ASSERT(state.bind_group_layout != NULL);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Coordinate system - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layout,
    });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Cleanup existing bind groups */
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.cw)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.ccw)

  /* Bind group CW */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = state.textures.cw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = state.textures.cw.sampler,
      },
    };
    state.bind_groups.cw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Bind group - CW"),
                              .layout     = state.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.cw != NULL);
  }

  /* Bind group CCW */
  {
    WGPUBindGroupEntry bg_entries[2] = {
      [0] = (WGPUBindGroupEntry) {
        .binding     = 0,
        .textureView = state.textures.ccw.view,
      },
      [1] = (WGPUBindGroupEntry) {
        .binding = 1,
        .sampler = state.textures.ccw.sampler,
      },
    };
    state.bind_groups.ccw = wgpuDeviceCreateBindGroup(
      wgpu_context->device, &(WGPUBindGroupDescriptor){
                              .label      = STRVIEW("Bind group - CCW"),
                              .layout     = state.bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            });
    ASSERT(state.bind_groups.ccw != NULL);
  }
}

static WGPUCullMode get_cull_mode(int32_t index)
{
  switch (index) {
    case 0:
      return WGPUCullMode_None;
    case 1:
      return WGPUCullMode_Front;
    case 2:
      return WGPUCullMode_Back;
    default:
      return WGPUCullMode_None;
  }
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Cleanup existing pipeline */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)

  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, quad_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, quad_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    quad, sizeof(vertex_t),
    /* Attribute location 0: Position */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x3, offsetof(vertex_t, pos)),
    /* Attribute location 1: UV */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x2, offsetof(vertex_t, uv)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Coordinate system - Render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = 1,
      .buffers     = &quad_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_TriangleList,
      .frontFace = state.settings.winding_order == 0 ?
                     WGPUFrontFace_CW : WGPUFrontFace_CCW,
      .cullMode  = get_cull_mode(state.settings.cull_mode),
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff
    },
  };

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);

  state.pipeline_needs_rebuild = false;
}

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Set window position closer to upper left corner */
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});

  /* Set initial window size with content-aware padding */
  igSetNextWindowSize((ImVec2){280.0f, 0.0f}, ImGuiCond_FirstUseEver);

  /* Build GUI */
  igBegin("Coordinate System", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Scene section */
  if (igCollapsingHeaderBoolPtr("Scene", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    int quad_type = state.settings.quad_type;
    if (imgui_overlay_combo_box("Quad Type", &quad_type, state.quad_type_str,
                                2)) {
      state.settings.quad_type = quad_type;
    }
  }

  /* Pipeline section */
  if (igCollapsingHeaderBoolPtr("Pipeline", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    int winding_order = state.settings.winding_order;
    if (imgui_overlay_combo_box("Winding Order", &winding_order,
                                state.winding_order_str, 2)) {
      state.settings.winding_order = winding_order;
      state.pipeline_needs_rebuild = true;
    }

    int cull_mode = state.settings.cull_mode;
    if (imgui_overlay_combo_box("Cull Mode", &cull_mode, state.cull_mode_str,
                                3)) {
      state.settings.cull_mode     = cull_mode;
      state.pipeline_needs_rebuild = true;
    }
  }

  igEnd();
}

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
}

static void update_textures(wgpu_context_t* wgpu_context)
{
  bool all_dirty        = true;
  uint8_t texture_count = (uint8_t)ARRAY_SIZE(state.texture_mappings);
  for (uint8_t i = 0; i < texture_count; ++i) {
    all_dirty = all_dirty && state.texture_mappings[i].texture->desc.is_dirty;
  }

  if (all_dirty) {
    /* Recreate textures */
    for (uint8_t i = 0; i < texture_count; ++i) {
      wgpu_recreate_texture(wgpu_context, state.texture_mappings[i].texture);
      FREE_TEXTURE_PIXELS(*state.texture_mappings[i].texture);
    }
    /* Update bind groups with new texture views */
    init_bind_groups(wgpu_context);
  }
}

static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = TEXTURE_COUNT,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    init_textures(wgpu_context);
    init_quad_buffers(wgpu_context);
    init_bind_group_layout(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_groups(wgpu_context);
    init_pipeline(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Update textures when pixel data loaded */
  update_textures(wgpu_context);

  /* Rebuild pipeline if settings changed */
  if (state.pipeline_needs_rebuild) {
    init_pipeline(wgpu_context);
  }

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

  /* Render GUI controls */
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Bind the rendering pipeline */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);

  /* Set scissor rectangle */
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u, wgpu_context->width,
                                      wgpu_context->height);

  /* Select vertex buffer based on quad type */
  WGPUBuffer vertex_buffer = state.settings.quad_type == QUAD_TYPE_VK_Y_DOWN ?
                               state.quad.vertices_y_down.buffer :
                               state.quad.vertices_y_up.buffer;

  /* Render CW quad */
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.cw, 0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.quad.indices_cw.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 0, vertex_buffer, 0,
                                       WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, 6, 1, 0, 0, 0);

  /* Render CCW quad */
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_groups.ccw, 0, 0);
  wgpuRenderPassEncoderSetIndexBuffer(rpass_enc, state.quad.indices_ccw.buffer,
                                      WGPUIndexFormat_Uint32, 0,
                                      WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDrawIndexed(rpass_enc, 6, 1, 0, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render ImGui overlay on top */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();
  sfetch_shutdown();

  wgpu_destroy_texture(&state.textures.cw);
  wgpu_destroy_texture(&state.textures.ccw);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.cw)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups.ccw)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad.vertices_y_up.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad.vertices_y_down.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad.indices_ccw.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.quad.indices_cw.buffer)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Coordinate System",
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
static const char* quad_vertex_shader_wgsl = CODE(
  struct Output {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>
  };

  @vertex
  fn main(
    @location(0) inPos: vec3<f32>,
    @location(1) inUV: vec2<f32>
  ) -> Output {
    var output: Output;
    output.uv = inUV;
    output.position = vec4<f32>(inPos.xyz, 1.0);
    return output;
  }
);

static const char* quad_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var textureColor: texture_2d<f32>;
  @group(0) @binding(1) var samplerColor: sampler;

  @fragment
  fn main(
    @location(0) inUV : vec2<f32>
  ) -> @location(0) vec4<f32> {
    return textureSample(textureColor, samplerColor, inUV);
  }
);
// clang-format on
