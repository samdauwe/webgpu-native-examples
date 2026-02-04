#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

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
 * WebGPU Example - Out-of-bounds Viewport
 *
 * WebGPU doesn't let you set the viewport's values to be out-of-bounds.
 * Therefore, the viewport's values need to be clamped to the screen-size, which
 * means the viewport values can't be defined in a way that makes the viewport
 * go off the screen. This example shows how to render a viewport out-of-bounds.
 *
 * Ref:
 * https://babylonjs.medium.com/how-to-simulate-out-of-bounds-viewports-when-using-webgpu-or-babylonnative-2280637c0660
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* out_of_bounds_viewport_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Out-of-bounds Viewport example
 * -------------------------------------------------------------------------- */

/* Viewport uniform buffer data */
typedef struct viewport_params_t {
  float x;
  float y;
  float width;
  float height;
} viewport_params_t;

/* State struct */
static struct {
  wgpu_texture_t texture;
  uint8_t file_buffer[512 * 512 * 4];
  struct {
    WGPUBindGroup handle;
    bool is_dirty;
  } bind_group;
  WGPUBindGroupLayout bind_group_layout;
  wgpu_buffer_t uniform_buffer;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  viewport_params_t viewport_params;
  uint64_t last_frame_time;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.1f, 0.2f, 0.3f, 1.0f},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .viewport_params = {
    .x      = 0.0f,
    .y      = 0.0f,
    .width  = 1.0f,
    .height = 1.0f,
  },
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
      .extent = (WGPUExtent3D){
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

/* Initialize texture - use color bars as placeholder while loading */
static void init_texture(wgpu_context_t* wgpu_context)
{
  state.texture           = wgpu_create_color_bars_texture(wgpu_context, NULL);
  wgpu_texture_t* texture = &state.texture;
  sfetch_send(&(sfetch_request_t){
    .path     = "assets/textures/Di-3d.png",
    .callback = fetch_callback,
    .buffer   = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

/* Initialize uniform buffer */
static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Out-of-bounds Viewport - Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(viewport_params_t),
                    .initial.data = &state.viewport_params,
                  });
  ASSERT(state.uniform_buffer.buffer != NULL);
}

/* Update uniform buffer with current viewport parameters */
static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer.buffer, 0,
                       &state.viewport_params, sizeof(viewport_params_t));
}

/* Initialize bind group layout */
static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry){
      /* Binding 0: Vertex shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout){
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(viewport_params_t),
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry){
      /* Binding 1: Fragment shader texture view */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry){
      /* Binding 2: Fragment shader texture sampler */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device,
    &(WGPUBindGroupLayoutDescriptor){
      .label      = STRVIEW("Out-of-bounds Viewport - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    });
  ASSERT(state.bind_group_layout != NULL);
}

/* Initialize bind group */
static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group.handle)

  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer.size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding     = 1,
      .textureView = state.texture.view,
    },
    [2] = (WGPUBindGroupEntry){
      .binding = 2,
      .sampler = state.texture.sampler,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Out-of-bounds Viewport - Bind group"),
    .layout     = state.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_group.handle
    = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_group.handle != NULL);
  state.bind_group.is_dirty = false;
}

/* Initialize render pipeline */
static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Create pipeline layout */
  WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label = STRVIEW("Out-of-bounds Viewport - Pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.bind_group_layout,
    });
  ASSERT(pipeline_layout != NULL);

  /* Create shader module */
  WGPUShaderModule shader_module = wgpu_create_shader_module(
    wgpu_context->device, out_of_bounds_viewport_shader_wgsl);
  ASSERT(shader_module != NULL);

  /* Create render pipeline */
  state.pipeline = wgpuDeviceCreateRenderPipeline(
    wgpu_context->device, &(WGPURenderPipelineDescriptor){
      .label  = STRVIEW("Out-of-bounds Viewport - Render pipeline"),
      .layout = pipeline_layout,
      .vertex = {
        .module     = shader_module,
        .entryPoint = STRVIEW("vertex_main"),
      },
      .primitive = {
        .topology  = WGPUPrimitiveTopology_TriangleList,
        .frontFace = WGPUFrontFace_CCW,
        .cullMode  = WGPUCullMode_None,
      },
      .fragment = &(WGPUFragmentState){
        .module     = shader_module,
        .entryPoint = STRVIEW("fragment_main"),
        .targetCount = 1,
        .targets = &(WGPUColorTargetState){
          .format    = wgpu_context->render_format,
          .blend     = &(WGPUBlendState){
            .color = {
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
            },
            .alpha = {
              .operation = WGPUBlendOperation_Add,
              .srcFactor = WGPUBlendFactor_One,
              .dstFactor = WGPUBlendFactor_Zero,
            },
          },
          .writeMask = WGPUColorWriteMask_All,
        },
      },
      .multisample = {
        .count = 1,
        .mask  = ~0u,
      },
    });
  ASSERT(state.pipeline != NULL);

  /* Cleanup */
  WGPU_RELEASE_RESOURCE(ShaderModule, shader_module);
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout);
}

/* Initialize example */
static int init(wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    /* Initialize timing and fetch */
    stm_setup();
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
    });

    /* Initialize resources */
    init_texture(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_bind_group_layout(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipeline(wgpu_context);

    /* Initialize ImGui overlay */
    imgui_overlay_init(wgpu_context);

    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

/* Render GUI panel */
static void render_gui(void)
{
  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){300.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Viewport Parameters", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  imgui_overlay_slider_float("X", &state.viewport_params.x, -0.5f, 0.5f,
                             "%.2f");
  imgui_overlay_slider_float("Y", &state.viewport_params.y, -0.5f, 0.5f,
                             "%.2f");
  imgui_overlay_slider_float("Width", &state.viewport_params.width, 0.0f, 2.0f,
                             "%.2f");
  imgui_overlay_slider_float("Height", &state.viewport_params.height, 0.0f,
                             2.0f, "%.2f");
  igEnd();
}

/* Frame render function */
static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Process async file loading */
  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    /* Update the bind group */
    init_bind_group(wgpu_context);
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

  /* Update uniform buffer */
  update_uniform_buffer(wgpu_context);

  /* Get device and queue */
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  /* Update render pass color attachment */
  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder */
  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Begin render pass */
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group.handle, 0,
                                    0);
  wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u,
                                      (uint32_t)wgpu_context->width,
                                      (uint32_t)wgpu_context->height);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass_enc);

  /* Submit command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  /* Render GUI */
  render_gui();

  /* Render ImGui overlay */
  imgui_overlay_render(wgpu_context);

  return EXIT_SUCCESS;
}

/* Cleanup resources */
static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  /* Shutdown fetch and ImGui */
  sfetch_shutdown();
  imgui_overlay_shutdown();

  /* Release resources */
  wgpu_destroy_texture(&state.texture);
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer.buffer);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group.handle);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
}

/* Input event callback for ImGui interaction */
static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* event)
{
  imgui_overlay_handle_input(wgpu_context, event);
}

/* Example entry point */
int main(int argc, char* argv[])
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  wgpu_start(&(wgpu_desc_t){
    .title          = "Out-of-bounds Viewport",
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
static const char* out_of_bounds_viewport_shader_wgsl = CODE(
  struct Viewport {
    x : f32,
    y : f32,
    w : f32,
    h : f32,
  };

  @group(0) @binding(0)
  var<uniform> viewport : Viewport;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
  }

  @vertex
  fn vertex_main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
      vec2( 1.0,  1.0),  vec2( 1.0, -1.0), vec2(-1.0, -1.0),
      vec2( 1.0,  1.0),  vec2(-1.0, -1.0), vec2(-1.0,  1.0)
    );

    let uv = array(
      vec2( 1.0,  0.0),  vec2( 1.0,  1.0), vec2( 0.0,  1.0),
      vec2( 1.0,  0.0),  vec2( 0.0,  1.0), vec2( 0.0,  0.0)
    );

    var position : vec4<f32> = vec4(pos[VertexIndex], 0.0, 1.0);
    position.x = position.x * viewport.w
                  + (viewport.x + viewport.w - 1.0 + viewport.x) * position.w;
    position.y = position.y * viewport.h
                  + (viewport.y + viewport.h - 1.0 + viewport.y) * position.w;

    var output : VertexOutput;
    output.Position = position;
    output.fragUV = uv[VertexIndex];

    return output;
  }

  @group(0) @binding(1) var myTexture : texture_2d<f32>;
  @group(0) @binding(2) var mySampler : sampler;

  @fragment
  fn fragment_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    let dx: f32 = 8.0 / 640.0;
    let dy: f32 = 8.0 / 640.0;
    let uv: vec2<f32> = vec2(dx * floor(fragUV.x / dx), dy * floor(fragUV.y / dy));

    let color: vec3<f32> = textureSample(myTexture, mySampler, uv).rgb;

    return vec4<f32>(color, 1.0);
  }
);
// clang-format on
