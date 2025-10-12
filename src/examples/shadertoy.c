#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdbool.h>
#include <sys/time.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Shadertoy
 *
 * Minimal "shadertoy launcher" using WebGPU, demonstrating how to load an
 * example Shadertoy shader 'Seascape'.
 *
 * Ref:
 * https://www.shadertoy.com/view/Ms2SD1
 * https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* shadertoy_vertex_shader_wgsl;
static const char* shadertoy_fragment_shader_p1_wgsl;
static const char* shadertoy_fragment_shader_p2_wgsl;

/* -------------------------------------------------------------------------- *
 * Shadertoy example
 * -------------------------------------------------------------------------- */

/* Date class */
typedef struct date_t {
  int msec;
  int sec;
  float day_sec;
  int min;
  int hour;
  int day;
  int month;
  int year;
} date_t;

/* State struct */
static struct {
  wgpu_buffer_t uniform_buffer_vs;
  struct {
    vec2 iResolution; // viewport resolution (in pixels)
    float iTime;      // shader playback time (in seconds)
    float iTimeDelta; // render time (in seconds)
    int iFrame;       // shader playback frame
    vec4 iMouse; // mouse pixel coords. xy: current (if MLB down), zw: click
    vec4 iDate;  // (year, month, day, time in seconds)
    float iSampleRate; // sound sample rate (i.e., 44100)
  } shader_inputs_ubo;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroup bind_group;
  WGPUPipelineLayout pipeline_layout;
  WGPURenderPipeline render_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  float prev_time;
  uint64_t frame_index;
  bool initialized;
} state = {
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

static void init_pipeline_layout(wgpu_context_t* wgpu_context)
{
  /* Create the bind group layout */
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = STRVIEW("Shadertoy - Bind group layout"),
    .entryCount = 1,
    .entries = &(WGPUBindGroupLayoutEntry) {
      /* Binding 0: Uniform buffer (Fragment shader) */
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .buffer = (WGPUBufferBindingLayout){
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(state.shader_inputs_ubo),
      },
      .sampler = {0},
    }
  };
  state.bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(state.bind_group_layout != NULL);

  /* Create the pipeline layout */
  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label = STRVIEW("Shadertoy - Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
  ASSERT(state.pipeline_layout != NULL);
}

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Bind Group */
  WGPUBindGroupDescriptor bg_desc = {
    .label      = STRVIEW("Shadertoy - Bind group"),
    .layout     = state.bind_group_layout,
    .entryCount = 1,
    .entries    = &(WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  = state.uniform_buffer_vs.buffer,
      .offset  = 0,
      .size    = state.uniform_buffer_vs.size,
    },
  };

  state.bind_group = wgpuDeviceCreateBindGroup(wgpu_context->device, &bg_desc);
  ASSERT(state.bind_group != NULL);
}

static void get_local_time(date_t* current_date)
{
  struct timeval te;
  gettimeofday(&te, NULL);
  time_t T               = time(NULL);
  struct tm tm           = *localtime(&T);
  long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
  current_date->msec     = (int)(milliseconds % (1000));
  current_date->sec      = tm.tm_sec;
  current_date->min      = tm.tm_min;
  current_date->hour     = tm.tm_hour;
  current_date->day      = tm.tm_mday;
  current_date->month    = tm.tm_mon + 1;
  current_date->year     = tm.tm_year + 1900;
  current_date->day_sec  = ((float)current_date->msec) / 1000.0
                          + current_date->sec + current_date->min * 60
                          + current_date->hour * 3600;
  return;
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* iResolution: viewport resolution (in pixels) */
  state.shader_inputs_ubo.iResolution[0] = (float)wgpu_context->width;
  state.shader_inputs_ubo.iResolution[1] = (float)wgpu_context->height;

  /* iTime: Time since the shader started (in seconds) */
  const float now        = stm_sec(stm_now());
  const float frame_time = now - state.prev_time;
  state.prev_time        = now;
  state.shader_inputs_ubo.iTime += frame_time;

  /* iTimeDelta: time between each frame (duration since the previous frame) */
  state.shader_inputs_ubo.iTimeDelta = frame_time;

  /* iFrame: shader playback frame */
  state.shader_inputs_ubo.iFrame = (int)state.frame_index;

  /* iDate: year, month, day, time in seconds */
  struct date_t current_date;
  get_local_time(&current_date);
  state.shader_inputs_ubo.iDate[0] = current_date.year,
  state.shader_inputs_ubo.iDate[1] = current_date.month,
  state.shader_inputs_ubo.iDate[2] = current_date.day,
  state.shader_inputs_ubo.iDate[3] = current_date.day_sec;

  /* iSampleRate: iSampleRate */
  state.shader_inputs_ubo.iSampleRate = 44100.0f;

  /* Map uniform buffer and update it */
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer_vs.buffer, 0,
                       &state.shader_inputs_ubo, state.uniform_buffer_vs.size);
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.uniform_buffer_vs = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
                    .size  = sizeof(state.shader_inputs_ubo),
                    .initial.data = &state.shader_inputs_ubo,
                  });

  /* Update uniform buffer data and uniform buffer */
  update_uniform_buffers(wgpu_context);
}

static char* get_full_fragment_shader(void)
{
  size_t len1      = strlen(shadertoy_fragment_shader_p1_wgsl);
  size_t len2      = strlen(shadertoy_fragment_shader_p2_wgsl);
  size_t total_len = len1 + len2 + 1; /* +1 for null terminator */

  char* full_shader = malloc(total_len);
  if (full_shader == NULL) {
    return NULL; /* Handle allocation failure */
  }

  snprintf(full_shader, total_len, "%s%s", shadertoy_fragment_shader_p1_wgsl,
           shadertoy_fragment_shader_p2_wgsl);

  return full_shader;
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  /* Shader modules */
  WGPUShaderModule vert_shader_module = wgpu_create_shader_module(
    wgpu_context->device, shadertoy_vertex_shader_wgsl);
  char* shadertoy_fragment_shader_wgsl = get_full_fragment_shader();
  WGPUShaderModule frag_shader_module  = wgpu_create_shader_module(
    wgpu_context->device, shadertoy_fragment_shader_wgsl);
  free(shadertoy_fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Shadertoy - render pipeline"),
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
    stm_setup();
    prepare_uniform_buffers(wgpu_context);
    init_pipeline_layout(wgpu_context);
    init_bind_groups(wgpu_context);
    init_pipeline(wgpu_context);
    state.initialized = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void input_event_cb(struct wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  UNUSED_VAR(wgpu_context);

  if (input_event->type == INPUT_EVENT_TYPE_RESIZED) {
    state.shader_inputs_ubo.iResolution[0] = (float)input_event->window_width;
    state.shader_inputs_ubo.iResolution[1] = (float)input_event->window_height;
  }
  else if (input_event->type == INPUT_EVENT_TYPE_MOUSE_MOVE
           && input_event->mouse_btn_pressed
           && input_event->mouse_button == BUTTON_LEFT) {
    state.shader_inputs_ubo.iMouse[0] += input_event->mouse_dx;
    state.shader_inputs_ubo.iMouse[1] += input_event->mouse_dy;
  }
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update the uniform buffers */
  update_uniform_buffers(wgpu_context);

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
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer_vs.buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_group)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.render_pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Shadertoy",
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
static const char* shadertoy_vertex_shader_wgsl = CODE(
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

static const char* shadertoy_fragment_shader_p1_wgsl = CODE(
  struct ShaderInputs {
    u_Resolution: vec2<f32>,
    u_Time: f32,
    u_TimeDelta: f32,
    u_Frame: i32,
    u_Mouse: vec4<f32>,
    u_Date: vec4<f32>,
    u_SampleRate: f32,
  };

  @group(0) @binding(0)
  var<uniform> shader_inputs: ShaderInputs;

  struct FragmentInput {
    @location(0) frag_pos: vec2<f32>,
  };

  struct FragmentOutput {
    @location(0) out_color: vec4<f32>,
  };

  // Optimization: Lower iteration counts for raymarch and geometry
  const NUM_STEPS: i32 = 6;
  const PI: f32 = 3.141592;
  const EPSILON: f32 = 1e-3;
  const ITER_GEOMETRY: i32 = 3;
  const ITER_FRAGMENT: i32 = 4;
  const SEA_HEIGHT: f32 = 0.6;
  const SEA_CHOPPY: f32 = 4.0;
  const SEA_SPEED: f32 = 0.8;
  const SEA_FREQ: f32 = 0.16;
  const EPSILON_NRM: f32 = 0.1 / 1920.0; // You may want to use shader_inputs.u_Resolution.x
  const SEA_BASE: vec3<f32> = vec3<f32>(0.0, 0.09, 0.18);
  const SEA_WATER_COLOR: vec3<f32> = vec3<f32>(0.8, 0.9, 0.6) * 0.6;
  const octave_m: mat2x2<f32> = mat2x2<f32>(1.6, 1.2, -1.2, 1.6);

  fn fromEuler(ang: vec3<f32>) -> mat3x3<f32> {
    let a1 = vec2<f32>(sin(ang.x), cos(ang.x));
    let a2 = vec2<f32>(sin(ang.y), cos(ang.y));
    let a3 = vec2<f32>(sin(ang.z), cos(ang.z));
    return mat3x3<f32>(
      vec3<f32>(a1.y*a3.y+a1.x*a2.x*a3.x, a1.y*a2.x*a3.x+a3.y*a1.x, -a2.y*a3.x),
      vec3<f32>(-a2.y*a1.x, a1.y*a2.y, a2.x),
      vec3<f32>(a3.y*a1.x*a2.x+a1.y*a3.x, a1.x*a3.x-a1.y*a3.y*a2.x, a2.y*a3.y)
    );
  }

  fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
  }

  fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return -1.0 + 2.0 * mix(
      mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
      mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
      u.y
    );
  }

  fn diffuse(n: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    return pow(dot(n, l) * 0.4 + 0.6, p);
  }

  fn specular(n: vec3<f32>, l: vec3<f32>, e: vec3<f32>, s: f32) -> f32 {
    let nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
  }

  fn getSkyColor(e: vec3<f32>) -> vec3<f32> {
    let ey = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return vec3<f32>(pow(1.0 - ey, 2.0), 1.0 - ey, 0.6 + (1.0 - ey) * 0.4) * 1.1;
  }

  fn sea_octave(uv: vec2<f32>, choppy: f32) -> f32 {
    let uv2 = uv + noise(uv);
    var wv = 1.0 - abs(sin(uv2));
    let swv = abs(cos(uv2));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
  }

  fn map(p: vec3<f32>, sea_time: f32) -> f32 {
    var freq = SEA_FREQ;
    var amp = SEA_HEIGHT;
    var choppy = SEA_CHOPPY;
    var uv = p.xz;
    uv.x = uv.x * 0.75;
    var d: f32;
    var h: f32 = 0.0;
    for (var i: i32 = 0; i < ITER_GEOMETRY; i = i + 1) {
      d = sea_octave((uv + sea_time) * freq, choppy);
      d = d + sea_octave((uv - sea_time) * freq, choppy);
      h = h + d * amp;
      uv = octave_m * uv;
      freq = freq * 1.9;
      amp = amp * 0.22;
      choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
  }
);

static const char* shadertoy_fragment_shader_p2_wgsl = CODE(
  fn map_detailed(p: vec3<f32>, sea_time: f32) -> f32 {
    var freq = SEA_FREQ;
    var amp = SEA_HEIGHT;
    var choppy = SEA_CHOPPY;
    var uv = p.xz;
    uv.x = uv.x * 0.75;
    var d: f32;
    var h: f32 = 0.0;
    for (var i: i32 = 0; i < ITER_FRAGMENT; i = i + 1) {
      d = sea_octave((uv + sea_time) * freq, choppy);
      d = d + sea_octave((uv - sea_time) * freq, choppy);
      h = h + d * amp;
      uv = octave_m * uv;
      freq = freq * 1.9;
      amp = amp * 0.22;
      choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
  }

  fn getSeaColor(p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, eye: vec3<f32>, dist: vec3<f32>, sea_time: f32) -> vec3<f32> {
    var fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = pow(fresnel, 3.0) * 0.5;
    let reflected = getSkyColor(reflect(eye, n));
    let refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;
    var color = mix(refracted, reflected, fresnel);
    let atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
    color = color + SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;
    color = color + vec3<f32>(specular(n, l, eye, 60.0));
    return color;
  }

  fn getNormal(p: vec3<f32>, eps: f32, sea_time: f32) -> vec3<f32> {
    var n: vec3<f32>;
    n.y = map_detailed(p, sea_time);
    n.x = map_detailed(vec3<f32>(p.x + eps, p.y, p.z), sea_time) - n.y;
    n.z = map_detailed(vec3<f32>(p.x, p.y, p.z + eps), sea_time) - n.y;
    n.y = eps;
    return normalize(n);
  }

  fn heightMapTracing(ori: vec3<f32>, dir: vec3<f32>, sea_time: f32) -> vec3<f32> {
    var tm: f32 = 0.0;
    var tx: f32 = 1000.0;
    var p: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var hx: f32 = map(ori + dir * tx, sea_time);
    if (hx > 0.0) {
        p = ori + dir * tx;
        return p;
    }
    var hm: f32 = map(ori + dir * tm, sea_time);
    var tmid: f32 = 0.0;
    for (var i: i32 = 0; i < NUM_STEPS; i = i + 1) {
      tmid = mix(tm, tx, hm / (hm - hx));
      p = ori + dir * tmid;
      let hmid = map(p, sea_time);
      if (hmid < 0.0) {
        tx = tmid;
        hx = hmid;
      } else {
        tm = tmid;
        hm = hmid;
      }
    }
    return p;
  }

  fn getPixel(coord: vec2<f32>, time: f32, sea_time: f32) -> vec3<f32> {
    var iResolution = vec3<f32>(shader_inputs.u_Resolution, 1.0);
    var uv = coord / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x = uv.x * iResolution.x / iResolution.y;
    let ang = vec3<f32>(sin(time * 3.0) * 0.1, sin(time) * 0.2 + 0.3, time);
    let ori = vec3<f32>(0.0, 3.5, time * 5.0);
    var dir = normalize(vec3<f32>(uv.xy, -2.0));
    dir.z = dir.z + length(uv) * 0.14;
    dir = normalize(dir) * fromEuler(ang);
    let p = heightMapTracing(ori, dir, sea_time);
    let dist = p - ori;
    let n = getNormal(p, dot(dist, dist) * EPSILON_NRM, sea_time);
    let light = normalize(vec3<f32>(0.0, 1.0, 0.8));
    return mix(
      getSkyColor(dir),
      getSeaColor(p, n, light, dir, dist, sea_time),
      pow(smoothstep(0.0, -0.02, dir.y), 0.2)
    );
  }

  fn mainImage(fragCoord: vec2<f32>) -> vec4<f32> {
    let iTime = shader_inputs.u_Time;
    let iMouse = shader_inputs.u_Mouse;
    let time = iTime * 0.3 + iMouse.x * 0.01;
    let sea_time = 1.0 + iTime * SEA_SPEED;
    var color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: i32 = 0; i < 2; i = i + 1) {
      for (var j: i32 = 0; j < 2; j = j + 1) {
        let uv = fragCoord + vec2<f32>(f32(i), f32(j)) / 2.0;
        color = color + getPixel(uv, time, sea_time);
      }
    }
    color = color / 4.0;
    return vec4<f32>(pow(color, vec3<f32>(0.65)), 1.0);
  }

  @fragment
  fn main(input: FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    var fragCoord: vec2<f32> = input.frag_pos;
    let iResolution = vec3<f32>(shader_inputs.u_Resolution, 1.0);
    fragCoord = floor(iResolution.xy * fragCoord);
    output.out_color = mainImage(fragCoord);
    return output;
  }
);
// clang-format on
