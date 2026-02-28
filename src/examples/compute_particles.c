#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

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
 * WebGPU Example - Compute Shader Particle System
 *
 * Attraction based 2D GPU particle system using compute shaders. Particle data
 * is stored in a shader storage buffer. A compute shader updates particle
 * positions based on attraction/repulsion forces, and the particles are
 * rendered as points with gradient coloring.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/computeparticles
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* particle_vertex_shader_wgsl;
static const char* particle_fragment_shader_wgsl;
static const char* particle_compute_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Compute Shader Particle System example
 * -------------------------------------------------------------------------- */

#define PARTICLE_COUNT (256u * 1024u)

/* Gradient ramp size */
#define GRADIENT_WIDTH 256
#define GRADIENT_HEIGHT 1

/* SSBO particle declaration */
typedef struct particle_t {
  float pos[2];          /* Particle position */
  float vel[2];          /* Particle velocity */
  float gradient_pos[4]; /* Texture coord for gradient ramp map */
} particle_t;

/* State struct */
static struct {
  /* Gradient ramp texture */
  struct {
    wgpu_texture_t texture;
    uint8_t pixels[GRADIENT_WIDTH * GRADIENT_HEIGHT * 4];
  } gradient;
  /* Graphics resources */
  struct {
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
  } graphics;
  /* Compute resources */
  struct {
    wgpu_buffer_t storage_buffer;
    wgpu_buffer_t uniform_buffer;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroup bind_group;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    struct {
      float delta_t;
      float dest_x;
      float dest_y;
      int32_t particle_count;
    } ubo;
  } compute;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Animation */
  float timer;
  float anim_start;
  /* Settings */
  struct {
    bool attach_to_cursor;
  } settings;
  /* Mouse state (normalized -1..1) */
  float cursor_x;
  float cursor_y;
  /* GUI timing */
  uint64_t last_frame_time;
  float frame_timer;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.025, 0.025, 0.025, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
  .anim_start = 20.0f,
  .settings = {
    .attach_to_cursor = false,
  },
};

/* -- Procedural gradient ramp ---------------------------------------------- */

static uint8_t clamp_u8(float v)
{
  if (v < 0.0f)
    return 0;
  if (v > 255.0f)
    return 255;
  return (uint8_t)v;
}

static void generate_gradient_ramp(void)
{
  /* Generate a smooth gradient ramp:
   * 0.00 - 0.25: blue   -> cyan
   * 0.25 - 0.50: cyan   -> green
   * 0.50 - 0.75: green  -> yellow
   * 0.75 - 1.00: yellow -> red
   */
  for (int i = 0; i < GRADIENT_WIDTH; ++i) {
    float t = (float)i / (float)(GRADIENT_WIDTH - 1);
    float r, g, b;

    if (t < 0.25f) {
      float s = t / 0.25f;
      r       = 0.0f;
      g       = s;
      b       = 1.0f;
    }
    else if (t < 0.5f) {
      float s = (t - 0.25f) / 0.25f;
      r       = 0.0f;
      g       = 1.0f;
      b       = 1.0f - s;
    }
    else if (t < 0.75f) {
      float s = (t - 0.5f) / 0.25f;
      r       = s;
      g       = 1.0f;
      b       = 0.0f;
    }
    else {
      float s = (t - 0.75f) / 0.25f;
      r       = 1.0f;
      g       = 1.0f - s;
      b       = 0.0f;
    }

    state.gradient.pixels[i * 4 + 0] = clamp_u8(r * 255.0f);
    state.gradient.pixels[i * 4 + 1] = clamp_u8(g * 255.0f);
    state.gradient.pixels[i * 4 + 2] = clamp_u8(b * 255.0f);
    state.gradient.pixels[i * 4 + 3] = 255;
  }
}

/* -- Texture init ---------------------------------------------------------- */

static void init_gradient_texture(wgpu_context_t* wgpu_context)
{
  generate_gradient_ramp();

  state.gradient.texture = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent = (WGPUExtent3D){
        .width              = GRADIENT_WIDTH,
        .height             = GRADIENT_HEIGHT,
        .depthOrArrayLayers = 1,
      },
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {
        .ptr  = state.gradient.pixels,
        .size = sizeof(state.gradient.pixels),
      },
    });
}

/* -- Storage buffer -------------------------------------------------------- */

static void init_storage_buffer(wgpu_context_t* wgpu_context)
{
  static particle_t particle_buffer[PARTICLE_COUNT];
  for (uint32_t i = 0; i < PARTICLE_COUNT; ++i) {
    particle_buffer[i] = (particle_t){
      .pos = {
        random_float_min_max(-1.0f, 1.0f),
        random_float_min_max(-1.0f, 1.0f),
      },
      .vel = {0.0f, 0.0f},
      .gradient_pos = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    particle_buffer[i].gradient_pos[0] = particle_buffer[i].pos[0] / 2.0f;
  }

  state.compute.storage_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Particle storage buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex
                             | WGPUBufferUsage_Storage,
                    .size         = PARTICLE_COUNT * sizeof(particle_t),
                    .initial.data = particle_buffer,
                  });
}

/* -- Uniform buffer -------------------------------------------------------- */

static void init_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.compute.ubo.particle_count = PARTICLE_COUNT;

  state.compute.uniform_buffer = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Compute uniform buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.compute.ubo),
                  });
}

static void update_uniform_buffer(wgpu_context_t* wgpu_context)
{
  state.compute.ubo.delta_t = state.frame_timer * 2.5f;

  if (state.settings.attach_to_cursor) {
    state.compute.ubo.dest_x = state.cursor_x;
    state.compute.ubo.dest_y = state.cursor_y;
  }
  else {
    state.compute.ubo.dest_x = sinf(state.timer * PI2) * 0.75f;
    state.compute.ubo.dest_y = 0.0f;
  }

  wgpuQueueWriteBuffer(wgpu_context->queue, state.compute.uniform_buffer.buffer,
                       0, &state.compute.ubo, sizeof(state.compute.ubo));
}

/* -- Graphics pipeline ----------------------------------------------------- */

static void init_graphics_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .texture = (WGPUTextureBindingLayout){
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .sampler = (WGPUSamplerBindingLayout){
        .type = WGPUSamplerBindingType_Filtering,
      },
    },
  };

  state.graphics.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Graphics bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.graphics.bind_group_layout != NULL);
}

static void init_graphics_bind_group(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding     = 0,
      .textureView = state.gradient.texture.view,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .sampler = state.gradient.texture.sampler,
    },
  };

  state.graphics.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Graphics bind group"),
                            .layout     = state.graphics.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.graphics.bind_group != NULL);
}

static void init_graphics_pipeline(wgpu_context_t* wgpu_context)
{
  /* Pipeline layout */
  state.graphics.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Graphics pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.graphics.bind_group_layout,
    });
  ASSERT(state.graphics.pipeline_layout != NULL);

  /* Shader modules */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_fragment_shader_wgsl);

  /* Blend state: additive blending */
  WGPUBlendState blend_state  = wgpu_create_blend_state(true);
  blend_state.color.srcFactor = WGPUBlendFactor_One;
  blend_state.color.dstFactor = WGPUBlendFactor_One;
  blend_state.alpha.srcFactor = WGPUBlendFactor_SrcAlpha;
  blend_state.alpha.dstFactor = WGPUBlendFactor_DstAlpha;

  /* Vertex buffer layout */
  WGPU_VERTEX_BUFFER_LAYOUT(
    particle, sizeof(particle_t),
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x2,
                       offsetof(particle_t, pos)),
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4,
                       offsetof(particle_t, gradient_pos)))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Particle render pipeline"),
    .layout = state.graphics.pipeline_layout,
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("vs_main"),
      .bufferCount = 1,
      .buffers     = &particle_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState){
      .module      = frag_shader_module,
      .entryPoint  = STRVIEW("fs_main"),
      .targetCount = 1,
      .targets = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology  = WGPUPrimitiveTopology_PointList,
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xffffffff,
    },
  };

  state.graphics.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.graphics.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

/* -- Compute pipeline ------------------------------------------------------ */

static void init_compute_pipeline(wgpu_context_t* wgpu_context)
{
  /* Bind group layout */
  WGPUBindGroupLayoutEntry bgl_entries[2] = {
    [0] = (WGPUBindGroupLayoutEntry){
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Storage,
        .minBindingSize = PARTICLE_COUNT * sizeof(particle_t),
      },
    },
    [1] = (WGPUBindGroupLayoutEntry){
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.compute.ubo),
      },
    },
  };

  state.compute.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Compute bind group layout"),
                            .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
                            .entries    = bgl_entries,
                          });
  ASSERT(state.compute.bind_group_layout != NULL);

  /* Pipeline layout */
  state.compute.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device,
    &(WGPUPipelineLayoutDescriptor){
      .label                = STRVIEW("Compute pipeline layout"),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts     = &state.compute.bind_group_layout,
    });
  ASSERT(state.compute.pipeline_layout != NULL);

  /* Bind group */
  WGPUBindGroupEntry bg_entries[2] = {
    [0] = (WGPUBindGroupEntry){
      .binding = 0,
      .buffer  = state.compute.storage_buffer.buffer,
      .size    = state.compute.storage_buffer.size,
    },
    [1] = (WGPUBindGroupEntry){
      .binding = 1,
      .buffer  = state.compute.uniform_buffer.buffer,
      .offset  = 0,
      .size    = state.compute.uniform_buffer.size,
    },
  };

  state.compute.bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, &(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Compute bind group"),
                            .layout     = state.compute.bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                            .entries    = bg_entries,
                          });
  ASSERT(state.compute.bind_group != NULL);

  /* Compute shader module */
  WGPUShaderModule comp_shader_module = wgpu_create_shader_module(
    wgpu_context->device, particle_compute_shader_wgsl);

  /* Compute pipeline */
  state.compute.pipeline = wgpuDeviceCreateComputePipeline(
    wgpu_context->device,
    &(WGPUComputePipelineDescriptor){
      .label   = STRVIEW("Particle compute pipeline"),
      .layout  = state.compute.pipeline_layout,
      .compute = {
        .module     = comp_shader_module,
        .entryPoint = STRVIEW("main"),
      },
    });
  ASSERT(state.compute.pipeline != NULL);

  wgpuShaderModuleRelease(comp_shader_module);
}

/* -- GUI ------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){10.0f, 10.0f}, ImGuiCond_FirstUseEver,
                     (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){260.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);
  igCheckbox("Attach Attractor to Cursor", &state.settings.attach_to_cursor);
  igEnd();
}

/* -- Main callbacks -------------------------------------------------------- */

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_storage_buffer(wgpu_context);
    init_uniform_buffer(wgpu_context);
    init_gradient_texture(wgpu_context);
    init_graphics_bind_group_layout(wgpu_context);
    init_graphics_bind_group(wgpu_context);
    init_graphics_pipeline(wgpu_context);
    init_compute_pipeline(wgpu_context);
    imgui_overlay_init(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Calculate delta time */
  uint64_t current_time = stm_now();
  if (state.last_frame_time == 0) {
    state.last_frame_time = current_time;
  }
  float delta_time
    = (float)stm_sec(stm_diff(current_time, state.last_frame_time));
  state.last_frame_time = current_time;
  state.frame_timer     = delta_time;

  /* Update animation */
  if (!state.settings.attach_to_cursor) {
    if (state.anim_start > 0.0f) {
      state.anim_start -= delta_time * 5.0f;
    }
    else {
      state.timer += delta_time * 0.04f;
      if (state.timer > 1.0f) {
        state.timer = 0.0f;
      }
    }
  }

  /* Update uniform buffer */
  update_uniform_buffer(wgpu_context);

  /* ImGui frame */
  imgui_overlay_new_frame(wgpu_context, delta_time);
  render_gui(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  /* Compute pass: update particle positions */
  {
    WGPUComputePassEncoder cpass_enc
      = wgpuCommandEncoderBeginComputePass(cmd_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cpass_enc, state.compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(cpass_enc, 0, state.compute.bind_group,
                                       0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cpass_enc, PARTICLE_COUNT / 256, 1,
                                             1);
    wgpuComputePassEncoderEnd(cpass_enc);
    wgpuComputePassEncoderRelease(cpass_enc);
  }

  /* Render pass: draw particles */
  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(rpass_enc, state.graphics.pipeline);
    wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, state.graphics.bind_group,
                                      0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(
      rpass_enc, 0, state.compute.storage_buffer.buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(rpass_enc, PARTICLE_COUNT, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpass_enc);
    wgpuRenderPassEncoderRelease(rpass_enc);
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Render imgui overlay */
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  imgui_overlay_shutdown();

  /* Gradient texture */
  wgpu_destroy_texture(&state.gradient.texture);

  /* Graphics pipeline */
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.graphics.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.graphics.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.graphics.pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.graphics.pipeline)

  /* Compute pipeline */
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.storage_buffer.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.compute.uniform_buffer.buffer)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.compute.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.compute.bind_group)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.compute.pipeline_layout)
  WGPU_RELEASE_RESOURCE(ComputePipeline, state.compute.pipeline)
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);

  /* Track cursor position for attractor */
  if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE) {
    if (input_event->window_width > 0 && input_event->window_height > 0) {
      state.cursor_x
        = (input_event->mouse_x / (float)input_event->window_width) * 2.0f
          - 1.0f;
      state.cursor_y
        = -((input_event->mouse_y / (float)input_event->window_height) * 2.0f
            - 1.0f);
    }
  }
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Compute Shader Particle System",
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
static const char* particle_vertex_shader_wgsl = CODE(
  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) gradient_pos : f32,
  };

  @vertex
  fn vs_main(
    @location(0) pos : vec2<f32>,
    @location(1) gradient_pos_in : vec4<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.position = vec4<f32>(pos, 1.0, 1.0);
    output.gradient_pos = gradient_pos_in.x;
    return output;
  }
);

static const char* particle_fragment_shader_wgsl = CODE(
  @group(0) @binding(0) var gradient_texture : texture_2d<f32>;
  @group(0) @binding(1) var gradient_sampler : sampler;

  @fragment
  fn fs_main(
    @location(0) gradient_pos : f32
  ) -> @location(0) vec4<f32> {
    let gradient_color = textureSample(
      gradient_texture, gradient_sampler, vec2<f32>(gradient_pos, 0.0)
    );
    return vec4<f32>(gradient_color.rgb, 1.0);
  }
);

static const char* particle_compute_shader_wgsl = CODE(
  struct Particle {
    pos : vec2<f32>,
    vel : vec2<f32>,
    gradient_pos : vec4<f32>,
  };

  struct Params {
    delta_t : f32,
    dest_x : f32,
    dest_y : f32,
    particle_count : i32,
  };

  @group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
  @group(0) @binding(1) var<uniform> params : Params;

  fn attraction(pos : vec2<f32>, attract_pos : vec2<f32>) -> vec2<f32> {
    let delta = attract_pos - pos;
    let damp = 0.5;
    let d_damped_dot = dot(delta, delta) + damp;
    let inv_dist = 1.0 / sqrt(d_damped_dot);
    let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
    return delta * inv_dist_cubed * 0.0035;
  }

  fn repulsion(pos : vec2<f32>, attract_pos : vec2<f32>) -> vec2<f32> {
    let delta = attract_pos - pos;
    let target_distance = sqrt(dot(delta, delta));
    return delta * (1.0 / (target_distance * target_distance * target_distance))
           * -0.000035;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(params.particle_count)) {
      return;
    }

    var v_vel = particles[index].vel;
    var v_pos = particles[index].pos;

    let dest_pos = vec2<f32>(params.dest_x, params.dest_y);

    v_vel = v_vel + attraction(v_pos, dest_pos) * 0.5;
    v_vel = v_vel + repulsion(v_pos, dest_pos) * 0.05;

    /* Move by velocity */
    v_pos = v_pos + v_vel * params.delta_t;

    /* Collide with boundary */
    if (v_pos.x < -1.0 || v_pos.x > 1.0 || v_pos.y < -1.0 || v_pos.y > 1.0) {
      v_vel = (-v_vel * 0.1) + attraction(v_pos, dest_pos) * 12.0;
    } else {
      particles[index].pos = v_pos;
    }

    /* Write back velocity */
    particles[index].vel = v_vel;
    particles[index].gradient_pos.x = particles[index].gradient_pos.x
                                      + 0.02 * params.delta_t;
    if (particles[index].gradient_pos.x > 1.0) {
      particles[index].gradient_pos.x =
        particles[index].gradient_pos.x - 1.0;
    }
  }
);
// clang-format on
