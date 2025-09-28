#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Animometer
 *
 * A WebGPU port of the Animometer MotionMark benchmark.
 *
 * Ref:
 * https://github.com/austinEng/webgpu-samples/blob/main/src/pages/samples/animometer.ts
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Animometer example
 * -------------------------------------------------------------------------- */

/* State struct */
static struct {
  struct {
    WGPUBuffer buffer;
    uint32_t count;
  } vertices;
  WGPUBuffer uniform_buffer;
  uint64_t time_offset;
  float uniform_time[1];
  uint64_t uniform_bytes;
  uint64_t aligned_uniform_bytes;
  uint64_t aligned_uniform_floats;
  WGPUPipelineLayout pipeline_layout;
  WGPUPipelineLayout dynamic_pipeline_layout;
  WGPURenderPipeline pipeline;
  WGPURenderPipeline dynamic_pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  WGPURenderBundle render_bundle;
  WGPUBindGroupLayout time_bind_group_layout;
  WGPUBindGroupLayout bind_group_layout;
  WGPUBindGroupLayout dynamic_bind_group_layout;
  WGPUBindGroup* bind_groups;
  WGPUBindGroup dynamic_bind_group;
  WGPUBindGroup time_bind_group;
  struct {
    uint64_t num_triangles;
    WGPUBool render_bundles;
    WGPUBool dynamic_offsets;
  } settings;
  WGPUBool initialized;
} state = {
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
    .num_triangles   = 20000,
    .render_bundles  = 1,
    .dynamic_offsets = 0,
  },
};

/* Initialize vertex buffers */
static void init_vertex_buffer(wgpu_context_t* wgpu_context)
{
  // clang-format off
  static const float vertices_data[(4 + 4) * 3] = {
    // position data          /**/ color data              /**/
     0.0f,  0.1f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, 1.0f, /**/
    -0.1f, -0.1f, 0.0f, 1.0f, /**/ 0.0f, 1.0f, 0.0f, 1.0f, /**/
     0.1f, -0.1f, 0.0f, 1.0f, /**/ 0.0f, 0.0f, 1.0f, 1.0f, /**/
  };
  // clang-format on

  state.vertices.count        = 3u;
  uint64_t vertex_buffer_size = (uint64_t)sizeof(vertices_data);

  /* Vertex buffer */
  state.vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertices_data, vertex_buffer_size, WGPUBufferUsage_Vertex);
}

static void init_pipeline_layouts(wgpu_context_t* wgpu_context)
{
  /* Time bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 4,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Time - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.time_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.time_bind_group_layout != NULL);
  }

  /* Bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type           = WGPUBufferBindingType_Uniform,
          .minBindingSize = 20,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.bind_group_layout != NULL);
  }

  /* Dynamic bind group layout */
  {
    WGPUBindGroupLayoutEntry bgl_entries[1] = {
      [0] = (WGPUBindGroupLayoutEntry) {
        .binding    = 0,
        .visibility = WGPUShaderStage_Vertex,
        .buffer = (WGPUBufferBindingLayout) {
          .type = WGPUBufferBindingType_Uniform,
          .hasDynamicOffset = true,
          .minBindingSize   = 20,
        },
      },
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {
      .label      = STRVIEW("Dynamic - Bind group layout"),
      .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
      .entries    = bgl_entries,
    };
    state.dynamic_bind_group_layout
      = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
    ASSERT(state.dynamic_bind_group_layout != NULL);
  }

  /* Pipeline layout */
  {
    WGPUBindGroupLayout bgl_pipeline[2]
      = {state.time_bind_group_layout, state.bind_group_layout};
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Render - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bgl_pipeline),
      .bindGroupLayouts     = bgl_pipeline,
    };
    state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(state.pipeline_layout != NULL);
  }

  /* Dynamic pipeline layout */
  {
    WGPUBindGroupLayout bgl_pipeline[2]
      = {state.time_bind_group_layout, state.dynamic_bind_group_layout};
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
      .label                = STRVIEW("Dynamic - Pipeline layout"),
      .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bgl_pipeline),
      .bindGroupLayouts     = bgl_pipeline,
    };
    state.dynamic_pipeline_layout = wgpuDeviceCreatePipelineLayout(
      wgpu_context->device, &pipeline_layout_desc);
    ASSERT(state.dynamic_pipeline_layout != NULL);
  }
}

static void update_uniform_buffers(wgpu_context_t* wgpu_context)
{
  /* Update uniforms */
  state.uniform_time[0] = stm_sec(stm_now());
  wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer,
                       state.time_offset, &state.uniform_time,
                       sizeof(state.uniform_time));
}

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.uniform_bytes          = 5 * sizeof(float);
  state.aligned_uniform_bytes  = ceil(state.uniform_bytes / 256.0f) * 256;
  state.aligned_uniform_floats = state.aligned_uniform_bytes / sizeof(float);
  state.uniform_buffer         = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
              .size = MAX(state.settings.num_triangles, 1) * state.aligned_uniform_bytes
              + sizeof(float),
    });
  float uniform_buffer_data[state.settings.num_triangles
                            * state.aligned_uniform_floats];
  state.bind_groups
    = malloc(state.settings.num_triangles * sizeof(WGPUBindGroup));
  for (uint64_t i = 0; i < state.settings.num_triangles; ++i) {
    uniform_buffer_data[state.aligned_uniform_floats * i + 0]
      = random_float_min_max(0.0f, 1.0f) * 0.2f + 0.2f; /* scale */
    uniform_buffer_data[state.aligned_uniform_floats * i + 1]
      = 0.9f * 2.0f * (random_float_min_max(0.0f, 1.0f) - 0.5f); /* offsetX */
    uniform_buffer_data[state.aligned_uniform_floats * i + 2]
      = 0.9f * 2.0f * (random_float_min_max(0.0f, 1.0f) - 0.5f); /* offsetY */
    uniform_buffer_data[state.aligned_uniform_floats * i + 3]
      = random_float_min_max(0.0f, 1.0f) * 1.5f + 0.5f; /* scalar */
    uniform_buffer_data[state.aligned_uniform_floats * i + 4]
      = random_float_min_max(0.0f, 1.0f) * 10.0f; /* scalarOffset */

    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer  = state.uniform_buffer,
        .offset  = i * state.aligned_uniform_bytes,
        .size    = 6 * sizeof(float),
      },
    };

    state.bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      (&(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Uniform buffer - Bind group layout"),
        .layout     = state.bind_group_layout,
        .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
        .entries    = bg_entries,
      }));
  }

  WGPUBindGroupEntry dynamic_bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer  =  state.uniform_buffer,
      .offset  = 0,
      .size    = 6 * sizeof(float),
    },
  };
  state.dynamic_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    (&(WGPUBindGroupDescriptor){
      .label      = STRVIEW("Dynamic - Bind group"),
      .layout     = state.dynamic_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(dynamic_bg_entries),
      .entries    = dynamic_bg_entries,
    }));

  state.time_offset
    = state.settings.num_triangles * state.aligned_uniform_bytes;
  WGPUBindGroupEntry time_bg_entries[1] = {
    [0] = (WGPUBindGroupEntry) {
      .binding = 0,
      .buffer = state.uniform_buffer,
      .offset = state.time_offset,
      .size = sizeof(float),
    },
  };
  state.time_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, (&(WGPUBindGroupDescriptor){
                            .label      = STRVIEW("Time - Bind group layout"),
                            .layout     = state.time_bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(time_bg_entries),
                            .entries    = time_bg_entries,
                          }));

  const uint64_t max_mapping_length = (14 * 1024 * 1024) / sizeof(float);
  for (uint64_t offset = 0; offset < ARRAY_SIZE(uniform_buffer_data);
       offset += max_mapping_length) {
    const uint64_t upload_count
      = MIN(ARRAY_SIZE(uniform_buffer_data) - offset, max_mapping_length);

    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffer,
                         offset * sizeof(float), &uniform_buffer_data[offset],
                         upload_count * sizeof(float));
  }
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    if (state.settings.dynamic_offsets) {                                      \
      wgpu##Type##SetPipeline(rpass_enc, state.dynamic_pipeline);              \
    }                                                                          \
    else {                                                                     \
      wgpu##Type##SetPipeline(rpass_enc, state.pipeline);                      \
    }                                                                          \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, state.vertices.buffer, 0,        \
                                WGPU_WHOLE_SIZE);                              \
    wgpu##Type##SetBindGroup(rpass_enc, 0, state.time_bind_group, 0, 0);       \
    uint32_t dynamic_offsets[1] = {0};                                         \
    for (uint64_t i = 0; i < state.settings.num_triangles; ++i) {              \
      if (state.settings.dynamic_offsets) {                                    \
        dynamic_offsets[0] = i * state.aligned_uniform_bytes;                  \
        wgpu##Type##SetBindGroup(rpass_enc, 1, state.dynamic_bind_group, 1,    \
                                 dynamic_offsets);                             \
      }                                                                        \
      else {                                                                   \
        wgpu##Type##SetBindGroup(rpass_enc, 1, state.bind_groups[i], 0, 0);    \
      }                                                                        \
      wgpu##Type##Draw(rpass_enc, 3, 1, 0, 0);                                 \
    }                                                                          \
  }

static void init_pipelines(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Color blend state */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  /* Vertex buffer layout */
  const size_t vec4_size = sizeof(vec4);
  WGPU_VERTEX_BUFFER_LAYOUT(
    animometer, 2 * vec4_size,
    /* Attribute location 0: Vertex positions */
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0),
    /* Attribute location 1: Vertex colors */
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, vec4_size))

  WGPURenderPipelineDescriptor rp_desc = {
    .label  = STRVIEW("Animometer - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("vert_main"),
      .bufferCount = 1,
      .buffers     = &animometer_vertex_buffer_layout,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("frag_main"),
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
      .frontFace = WGPUFrontFace_CCW,
      .cullMode  = WGPUCullMode_None,
    },
    .multisample = {
       .count = 1,
       .mask  = 0xffffffff
    },
  };

  /* Create render pipelines */
  rp_desc.layout = state.pipeline_layout;
  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  rp_desc.layout = state.dynamic_pipeline_layout;
  state.dynamic_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.dynamic_pipeline != NULL);

  /* Partial cleanup */
  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static void init_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->render_format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(
      wgpu_context->device, &(WGPURenderBundleEncoderDescriptor){
                              .label = STRVIEW("Render bundle encoder"),
                              .colorFormatCount = 1,
                              .colorFormats     = color_formats,
                              .sampleCount      = 1,
                            });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  state.render_bundle
    = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    stm_setup();
    init_vertex_buffer(wgpu_context);
    init_pipeline_layouts(wgpu_context);
    init_uniform_buffers(wgpu_context);
    init_pipelines(wgpu_context);
    init_render_bundle_encoder(wgpu_context);
    state.initialized = 1;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Update matrix data */
  update_uniform_buffers(wgpu_context);

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);

  {
    WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
      cmd_enc, &state.render_pass_descriptor);

    if (state.settings.render_bundles) {
      const WGPURenderBundle render_bundles[1] = {state.render_bundle};
      wgpuRenderPassEncoderExecuteBundles(rpass_enc, 1, render_bundles);
    }
    else {
      RECORD_RENDER_PASS(RenderPassEncoder, rpass_enc)
    }

    wgpuRenderPassEncoderEnd(rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass_enc)
  }

  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_enc, NULL);

  /* Submit and present. */
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

  /* Cleanup */
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_enc);

  return EXIT_SUCCESS;
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(Buffer, state.vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.uniform_buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.dynamic_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.dynamic_pipeline)
  WGPU_RELEASE_RESOURCE(RenderBundle, state.render_bundle)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.time_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.dynamic_bind_group_layout);
  for (uint64_t i = 0; i < state.settings.num_triangles; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups[i])
  }
  free(state.bind_groups);
  WGPU_RELEASE_RESOURCE(BindGroup, state.dynamic_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, state.time_bind_group)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Animometer",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* vertex_shader_wgsl = CODE(
  struct Time {
    value : f32,
  };

  struct Uniforms {
    scale : f32,
    offsetX : f32,
    offsetY : f32,
    scalar : f32,
    scalarOffset : f32,
  };

  @binding(0) @group(0) var<uniform> time : Time;
  @binding(0) @group(1) var<uniform> uniforms : Uniforms;

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) v_color : vec4<f32>,
  };

  @vertex
  fn vert_main(
    @location(0) position : vec4<f32>,
    @location(1) color : vec4<f32>
  ) -> VertexOutput {
    var fade = (uniforms.scalarOffset + time.value * uniforms.scalar / 10.0) % 1.0;
    if (fade < 0.5) {
      fade = fade * 2.0;
    } else {
      fade = (1.0 - fade) * 2.0;
    }
    var xpos = position.x * uniforms.scale;
    var ypos = position.y * uniforms.scale;
    var angle = 3.14159 * 2.0 * fade;
    var xrot = xpos * cos(angle) - ypos * sin(angle);
    var yrot = xpos * sin(angle) + ypos * cos(angle);
    xpos = xrot + uniforms.offsetX;
    ypos = yrot + uniforms.offsetY;

    var output : VertexOutput;
    output.v_color = vec4(fade, 1.0 - fade, 0.0, 1.0) + color;
    output.Position = vec4(xpos, ypos, 0.0, 1.0);
    return output;
  }
);

static const char* fragment_shader_wgsl = CODE(
  @fragment
  fn frag_main(@location(0) v_color : vec4<f32>) -> @location(0) vec4<f32> {
    return v_color;
  }
);
// clang-format on
