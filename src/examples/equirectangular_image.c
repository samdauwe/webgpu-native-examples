#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_FETCH_IMPL
#include <sokol_fetch.h>

#define SOKOL_LOG_IMPL
#include <sokol_log.h>

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
 * WebGPU Example - Equirectangular Image
 *
 * This example shows how to render an equirectangular panorama consisting of a
 * single rectangular image. The equirectangular input can be used for a 360
 * degrees viewing experience to achieve more realistic surroundings and
 * convincing real-time effects.
 *
 * Ref:
 * https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers
 * https://onix-systems.com/blog/how-to-use-360-equirectangular-panoramas-for-greater-realism-in-games
 * https://threejs.org/examples/webgl_panorama_equirectangular.html
 * https://www.shadertoy.com/view/4lK3DK
 * http://www.hdrlabs.com/sibl/archive.html
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* equirectangular_image_vertex_shader_wgsl;
static const char* equirectangular_image_fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Equirectangular Image example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    vec2 iResolution; // viewport resolution (in pixels)
    vec4 iMouse; // mouse pixel coords. xy: current (if MLB down), zw: click
    float iHFovDegrees;       // Horizontal field of view in degrees
    float iVFovDegrees;       // Vertical field of view in degrees
    uint32_t iVisualizeInput; // Show the unprocessed input image
    vec4 padding; // Padding to reach the minimum binding size of 64 bytes
  } shader_inputs_ubo;
  wgpu_texture_t texture;
  uint8_t file_buffer[1024 * 1024 * 5];
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  bool initialized;
} state = {
  .shader_inputs_ubo = {
    .iMouse       = {535, 415},
    .iHFovDegrees = 80.0f,
    .iVFovDegrees = 50.0f,
  },
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
  },
};

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
          .depthOrArrayLayers = 4,
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

static void init_texture(wgpu_context_t* wgpu_context)
{
  /* Dummy texture */
  state.texture = wgpu_create_color_bars_texture(wgpu_context, 16, 16);

  /* Start loading the image file */
  const char* particle_texture_path = "assets/textures/Circus_Backstage_8k.jpg";
  wgpu_texture_t* texture           = &state.texture;
  sfetch_send(&(sfetch_request_t){
    .path      = particle_texture_path,
    .callback  = fetch_callback,
    .buffer    = SFETCH_RANGE(state.file_buffer),
    .user_data = {
      .ptr  = &texture,
      .size = sizeof(wgpu_texture_t*),
    },
  });
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.shader_inputs_ubo,
                       sizeof(state.shader_inputs_ubo));
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.shader_inputs_ubo),
                    .initial.data = &state.shader_inputs_ubo,
                  });

  // iResolution: viewport resolution (in pixels)
  state.shader_inputs_ubo.iResolution[0] = (float)wgpu_context->width;
  state.shader_inputs_ubo.iResolution[1] = (float)wgpu_context->height;

  update_uniform_buffer(wgpu_context);
}

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[3] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader uniform buffer */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout) {
        .type             = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = false,
        .minBindingSize   = sizeof(mat4), /* 4x4 matrix */
      },
      .sampler = {0},
    },
    [1] = (WGPUBindGroupLayoutEntry) {
      /* Binding 1: Fragment shader texture view */
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout) {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
      .storageTexture = {0},
    },
    [2] = (WGPUBindGroupLayoutEntry) {
      /* Binding 2: Fragment shader texture sampler */
      .binding    = 2,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type  = WGPUSamplerBindingType_Filtering,
      },
      .texture = {0},
    },
  };
  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Render - Bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.bind_group_layout != NULL);

  /* Create the pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Render - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_group(wgpu_context_t* wgpu_context)
{
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPUBindGroupEntry bg_entries[3] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
    [1] = (WGPUBindGroupEntry) {
      .binding     = 1,
      .textureView = state.texture.view,
    },
    [2] = (WGPUBindGroupEntry) {
      .binding = 2,
      .sampler = state.texture.sampler,
    },
  };
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Bind group"),
    .layout     = state.bind_group_layout,
    .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
    .entries    = bg_entries,
  };
  state.bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_group != NULL);
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Shader modules */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, equirectangular_image_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, equirectangular_image_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Equirectangular image - render pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
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
      .cullMode  = WGPUCullMode_Back,
      .frontFace = WGPUFrontFace_CCW
    },
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  state.render_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.render_pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    sfetch_setup(&(sfetch_desc_t){
      .max_requests = 1,
      .num_channels = 1,
      .num_lanes    = 1,
      .logger.func  = slog_func,
    });
    init_texture(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_group(wgpu_context);
    init_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    state.shader_inputs_ubo.iResolution[0] = (float)input_event->window_width;
    state.shader_inputs_ubo.iResolution[1] = (float)input_event->window_height;
    update_uniform_buffer(wgpu_context);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && input_event->mouse_btn_pressed
           && input_event->mouse_button == BUTTON_LEFT) {
    state.shader_inputs_ubo.iMouse[0] += input_event->mouse_dx;
    state.shader_inputs_ubo.iMouse[1] += input_event->mouse_dy;
    update_uniform_buffer(wgpu_context);
  }
  else if (input_event->type == INPUT_EVENT_TYPE_CHAR
           && input_event->char_code == (uint32_t)'t') {
    state.shader_inputs_ubo.iVisualizeInput
      = state.shader_inputs_ubo.iVisualizeInput == 0 ? 1 : 0;
    update_uniform_buffer(wgpu_context);
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  sfetch_dowork();

  /* Recreate texture when pixel data loaded */
  if (state.texture.desc.is_dirty) {
    wgpu_recreate_texture(wgpu_context, &state.texture);
    FREE_TEXTURE_PIXELS(state.texture);
    /* Upddate the bindgroup */
    init_bind_group(wgpu_context);
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_descriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.render_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.bind_group, 0, 0);
  wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u, wgpu_context->width,
                                      wgpu_context->height);
  wgpuRenderPassEncoderDraw(rpass_enc, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpass_enc);
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  sfetch_shutdown();

  wgpu_destroy_texture(&state.texture);
  wgpu_destroy_buffer(&state.uniform_buffer_vs);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Equirectangular Image",
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
static const char* equirectangular_image_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) frag_pos : vec2<f32>,
  };

  @vertex
  fn main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    output.frag_pos = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    output.position = vec4<f32>(output.frag_pos * 2.0 + vec2<f32>(-1.0, -1.0), 0.0, 1.0);
    return output;
  }
);

static const char* equirectangular_image_fragment_shader_wgsl = CODE(
  // Constants
  const PI: f32 = 3.14159265;
  const DEG2RAD: f32 = 0.01745329251994329576923690768489;

  // Uniforms
  struct ShaderInputs {
    u_Resolution: vec2<f32>,
    u_Mouse: vec4<f32>,
    u_HFovDegrees: f32,
    u_VFovDegrees: f32,
    u_VisualizeInput: u32,
  };

  @group(0) @binding(0)
  var<uniform> shader_inputs: ShaderInputs;

  @group(0) @binding(1)
  var iChannel0Texture: texture_2d<f32>;
  @group(0) @binding(2)
  var iChannel0TextureSampler: sampler;

  // Input from vertex shader
  struct FragmentInput {
    @location(0) frag_pos: vec2<f32>,
  };

  struct FragmentOutput {
    @location(0) color: vec4<f32>,
  };

  // Helper function
  fn rotateXY(p: vec3<f32>, angle: vec2<f32>) -> vec3<f32> {
    let c = vec2<f32>(cos(angle.x), cos(angle.y));
    let s = vec2<f32>(sin(angle.x), sin(angle.y));
    let p1 = vec3<f32>(p.x, c.x * p.y + s.x * p.z, -s.x * p.y + c.x * p.z);
    return vec3<f32>(c.y * p1.x + s.y * p1.z, p1.y, -s.y * p1.x + c.y * p1.z);
  }

  fn mainImage(fragCoord: vec2<f32>) -> vec4<f32> {
    let iResolution = vec3<f32>(shader_inputs.u_Resolution, 1.0);
    let iMouseOrig = shader_inputs.u_Mouse;
    let iHFovDegrees = shader_inputs.u_HFovDegrees;
    let iVFovDegrees = shader_inputs.u_VFovDegrees;
    let iVisualizeInput = shader_inputs.u_VisualizeInput;

    // place 0,0 in center from -1 to 1 ndc
    let uv = (fragCoord * 2.0 / iResolution.xy) - vec2<f32>(1.0, 1.0);

    // Flip x and y
    let uv_flipped = uv * vec2<f32>(-1.0, -1.0);

    // Compensate for flipped axises
    let iMouse = vec4<f32>(iMouseOrig.x, iResolution.y - iMouseOrig.y, iMouseOrig.z, iMouseOrig.w);

    // to spherical
    let camDir = normalize(vec3<f32>(
      uv_flipped * vec2<f32>(tan(0.5 * iHFovDegrees * DEG2RAD), tan(0.5 * iVFovDegrees * DEG2RAD)),
      1.0
    ));

    // camRot is angle vec in rad
    let camRot = vec3<f32>(
      ((iMouse.xy / iResolution.xy) - vec2<f32>(0.5, 0.5)) * vec2<f32>(2.0 * PI, PI),
      0.0
    );

    // rotate
    let rd = normalize(rotateXY(camDir, camRot.yx));

    // radial azimuth polar
    var texCoord: vec2<f32> = vec2<f32>(atan2(rd.z, rd.x) + PI, acos(-rd.y)) / vec2<f32>(2.0 * PI, PI);

    // Input visualization
    var fragCoordY: f32 = fragCoord.y;
    if (iVisualizeInput == 1u) {
      fragCoordY = iResolution.y - fragCoord.y;
      texCoord = vec2<f32>(fragCoord.x, fragCoordY) / iResolution.xy;
    }

    return textureSample(iChannel0Texture, iChannel0TextureSampler, texCoord);
  }

  @fragment
  fn main(input: FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    var fragCoord = input.frag_pos;
    fragCoord = floor(shader_inputs.u_Resolution * fragCoord);
    output.color = mainImage(fragCoord);
    return output;
  }
);
// clang-format on
