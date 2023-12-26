#include "example_base.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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

// Settings
static struct settings_t {
  uint64_t num_triangles;
  bool render_bundles;
  bool dynamic_offsets;
} settings = {
  .num_triangles   = 20000,
  .render_bundles  = true,
  .dynamic_offsets = false,
};
static uint64_t uniform_bytes          = 0;
static uint64_t aligned_uniform_bytes  = 0;
static uint64_t aligned_uniform_floats = 0;

// Vertex buffer
static struct {
  WGPUBuffer buffer;
  uint32_t count;
} vertices = {0};

//  Uniform buffer
static WGPUBuffer uniform_buffer = NULL;
static uint64_t time_offset      = 0;
static float uniform_time[1]     = {0};

// The pipeline layouts
static WGPUPipelineLayout pipeline_layout         = NULL;
static WGPUPipelineLayout dynamic_pipeline_layout = NULL;

// Pipelines
static WGPURenderPipeline pipeline         = NULL;
static WGPURenderPipeline dynamic_pipeline = NULL;

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Render bundles execute the commands previously recorded into the given
// GPURenderBundles as part of this render pass.
static WGPURenderBundle render_bundle = NULL;

// Bind groups stores the resources bound to the binding points in a shader
static WGPUBindGroupLayout time_bind_group_layout    = NULL;
static WGPUBindGroupLayout bind_group_layout         = NULL;
static WGPUBindGroupLayout dynamic_bind_group_layout = NULL;

static WGPUBindGroup* bind_groups       = NULL;
static WGPUBindGroup dynamic_bind_group = NULL;
static WGPUBindGroup time_bind_group    = NULL;

// Other variables
static const char* example_title = "Animometer";
static bool prepared             = false;
static float start_time          = -1.0f;

// Prepare vertex buffers
static void prepare_vertex_buffer(wgpu_context_t* wgpu_context)
{
  // clang-format off
  static const float vertices_data[(4 + 4) * 3] = {
    // position data          /**/ color data              //
     0.0f,  0.1f, 0.0f, 1.0f, /**/ 1.0f, 0.0f, 0.0f, 1.0f, //
    -0.1f, -0.1f, 0.0f, 1.0f, /**/ 0.0f, 1.0f, 0.0f, 1.0f, //
     0.1f, -0.1f, 0.0f, 1.0f, /**/ 0.0f, 0.0f, 1.0f, 1.0f, //
  };
  // clang-format on

  vertices.count              = 3u;
  uint64_t vertex_buffer_size = (uint64_t)sizeof(vertices_data);

  // Vertex buffer
  vertices.buffer = wgpu_create_buffer_from_data(
    wgpu_context, vertices_data, vertex_buffer_size, WGPUBufferUsage_Vertex);
}

static void setup_pipeline_layout(wgpu_context_t* wgpu_context)
{
  // Time bind group layout
  WGPUBindGroupLayoutEntry time_bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Time
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = 4,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor time_bgl_desc = {
    .label      = "Time bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(time_bgl_entries),
    .entries    = time_bgl_entries,
  };
  time_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &time_bgl_desc);
  ASSERT(time_bind_group_layout != NULL);

  // Bind group layout
  WGPUBindGroupLayoutEntry bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      // Binding 0: Time
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .minBindingSize = 20,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor bgl_desc = {
    .label      = "Time bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(bgl_entries),
    .entries    = bgl_entries,
  };
  bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &bgl_desc);
  ASSERT(bind_group_layout != NULL);

  // Dynamic bind group layout
  WGPUBindGroupLayoutEntry dynamic_bgl_entries[1] = {
    [0] = (WGPUBindGroupLayoutEntry) {
      .binding = 0,
      .visibility = WGPUShaderStage_Vertex,
      .buffer = (WGPUBufferBindingLayout) {
        .type = WGPUBufferBindingType_Uniform,
        .hasDynamicOffset = true,
        .minBindingSize   = 20,
      },
      .sampler = {0},
    },
  };
  WGPUBindGroupLayoutDescriptor dynamic_bgl_desc = {
    .label      = "Dynamic bind group layout",
    .entryCount = (uint32_t)ARRAY_SIZE(dynamic_bgl_entries),
    .entries    = dynamic_bgl_entries,
  };
  dynamic_bind_group_layout
    = wgpuDeviceCreateBindGroupLayout(wgpu_context->device, &dynamic_bgl_desc);
  ASSERT(dynamic_bind_group_layout != NULL);

  // Create the pipeline layouts that are used to generate the rendering
  // pipelines that are based on this bind group layouts
  WGPUBindGroupLayout bgl_pipeline[2]
    = {time_bind_group_layout, bind_group_layout};
  WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
    .label                = "Render pipeline layout",
    .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bgl_pipeline),
    .bindGroupLayouts     = bgl_pipeline,
  };
  pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_context->device,
                                                   &pipeline_layout_desc);
  ASSERT(pipeline_layout != NULL);

  WGPUBindGroupLayout bgl_dynamic_pipeline[2]
    = {time_bind_group_layout, dynamic_bind_group_layout};
  WGPUPipelineLayoutDescriptor _dynamic_pipeline_layout_desc = {
    .label                = "Dynamic pipeline layout",
    .bindGroupLayoutCount = (uint32_t)ARRAY_SIZE(bgl_dynamic_pipeline),
    .bindGroupLayouts     = bgl_dynamic_pipeline,
  };
  dynamic_pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &_dynamic_pipeline_layout_desc);
  ASSERT(dynamic_pipeline_layout != NULL);
}

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  render_pass.color_attachments[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL, /* Assigned later */
      .depthSlice = ~0,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearValue = (WGPUColor) {
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                = "Render pass descriptor",
    .colorAttachmentCount = 1,
    .colorAttachments     = render_pass.color_attachments,
  };
}

static float float_random(float min, float max)
{
  const float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
  return min + scale * (max - min);             /* [min, max] */
}

static void update_uniform_buffers(wgpu_example_context_t* context)
{
  float frame_timestamp_millis = context->frame.timestamp_millis;

  // Update uniforms
  if (start_time < 0.0f) {
    start_time = frame_timestamp_millis;
  }
  uniform_time[0] = (frame_timestamp_millis - start_time) / 1000.0f;
  wgpu_queue_write_buffer(context->wgpu_context, uniform_buffer, time_offset,
                          &uniform_time, sizeof(uniform_time));
}

static void prepare_uniform_buffers(wgpu_context_t* wgpu_context)
{
  uniform_bytes          = 5 * sizeof(float);
  aligned_uniform_bytes  = ceil(uniform_bytes / 256.0f) * 256;
  aligned_uniform_floats = aligned_uniform_bytes / sizeof(float);
  uniform_buffer         = wgpuDeviceCreateBuffer(
    wgpu_context->device,
    &(WGPUBufferDescriptor){
              .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
              .size  = settings.num_triangles * aligned_uniform_bytes + sizeof(float),
    });
  float uniform_buffer_data[settings.num_triangles * aligned_uniform_floats];
  bind_groups = malloc(settings.num_triangles * sizeof(WGPUBindGroup));
  for (uint64_t i = 0; i < settings.num_triangles; ++i) {
    uniform_buffer_data[aligned_uniform_floats * i + 0]
      = float_random(0.0f, 1.0f) * 0.2f + 0.2f; // scale
    uniform_buffer_data[aligned_uniform_floats * i + 1]
      = 0.9f * 2.0f * (float_random(0.0f, 1.0f) - 0.5f); // offsetX
    uniform_buffer_data[aligned_uniform_floats * i + 2]
      = 0.9f * 2.0f * (float_random(0.0f, 1.0f) - 0.5f); // offsetY
    uniform_buffer_data[aligned_uniform_floats * i + 3]
      = float_random(0.0f, 1.0f) * 1.5f + 0.5f; // scalar
    uniform_buffer_data[aligned_uniform_floats * i + 4]
      = float_random(0.0f, 1.0f) * 10.0f; // scalarOffset

    WGPUBindGroupEntry bg_entries[1] = {
      [0] = (WGPUBindGroupEntry) {
        .binding = 0,
        .buffer = uniform_buffer,
        .offset = i * aligned_uniform_bytes,
        .size = 6 * sizeof(float),
      },
    };

    bind_groups[i] = wgpuDeviceCreateBindGroup(
      wgpu_context->device, (&(WGPUBindGroupDescriptor){
                              .label      = "Uniform buffer bind group layout",
                              .layout     = bind_group_layout,
                              .entryCount = (uint32_t)ARRAY_SIZE(bg_entries),
                              .entries    = bg_entries,
                            }));
  }

  WGPUBindGroupEntry dynamic_bg_entries[1] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer = uniform_buffer,
          .offset = 0,
          .size = 6 * sizeof(float),
        },
      };
  dynamic_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device,
    (&(WGPUBindGroupDescriptor){
      .label      = "Dynamic bind group",
      .layout     = dynamic_bind_group_layout,
      .entryCount = (uint32_t)ARRAY_SIZE(dynamic_bg_entries),
      .entries    = dynamic_bg_entries,
    }));

  time_offset = settings.num_triangles * aligned_uniform_bytes;
  WGPUBindGroupEntry time_bg_entries[1] = {
        [0] = (WGPUBindGroupEntry) {
          .binding = 0,
          .buffer = uniform_buffer,
          .offset = time_offset,
          .size = sizeof(float),
        },
      };
  time_bind_group = wgpuDeviceCreateBindGroup(
    wgpu_context->device, (&(WGPUBindGroupDescriptor){
                            .label      = "Time bind group layout",
                            .layout     = time_bind_group_layout,
                            .entryCount = (uint32_t)ARRAY_SIZE(time_bg_entries),
                            .entries    = time_bg_entries,
                          }));

  const uint64_t max_mapping_length = (14 * 1024 * 1024) / sizeof(float);
  for (uint64_t offset = 0; offset < ARRAY_SIZE(uniform_buffer_data);
       offset += max_mapping_length) {
    const uint64_t upload_count
      = MIN(ARRAY_SIZE(uniform_buffer_data) - offset, max_mapping_length);

    wgpuQueueWriteBuffer(wgpu_context->queue, uniform_buffer,
                         offset * sizeof(float), &uniform_buffer_data[offset],
                         upload_count * sizeof(float));
  }
}

#define RECORD_RENDER_PASS(Type, rpass_enc)                                    \
  if (rpass_enc) {                                                             \
    if (settings.dynamic_offsets) {                                            \
      wgpu##Type##SetPipeline(rpass_enc, dynamic_pipeline);                    \
    }                                                                          \
    else {                                                                     \
      wgpu##Type##SetPipeline(rpass_enc, pipeline);                            \
    }                                                                          \
    wgpu##Type##SetVertexBuffer(rpass_enc, 0, vertices.buffer, 0,              \
                                WGPU_WHOLE_SIZE);                              \
    wgpu##Type##SetBindGroup(rpass_enc, 0, time_bind_group, 0, 0);             \
    uint32_t dynamic_offsets[1] = {0};                                         \
    for (uint64_t i = 0; i < settings.num_triangles; ++i) {                    \
      if (settings.dynamic_offsets) {                                          \
        dynamic_offsets[0] = i * aligned_uniform_bytes;                        \
        wgpu##Type##SetBindGroup(rpass_enc, 1, dynamic_bind_group, 1,          \
                                 dynamic_offsets);                             \
      }                                                                        \
      else {                                                                   \
        wgpu##Type##SetBindGroup(rpass_enc, 1, bind_groups[i], 0, 0);          \
      }                                                                        \
      wgpu##Type##Draw(rpass_enc, 3, 1, 0, 0);                                 \
    }                                                                          \
  }

static void prepare_pipelines(wgpu_context_t* wgpu_context)
{
  // Primitive state
  WGPUPrimitiveState primitive_state_desc = {
    .topology  = WGPUPrimitiveTopology_TriangleList,
    .frontFace = WGPUFrontFace_CCW,
    .cullMode  = WGPUCullMode_None,
  };

  // Color target state
  WGPUBlendState blend_state                   = wgpu_create_blend_state(false);
  WGPUColorTargetState color_target_state_desc = (WGPUColorTargetState){
    .format    = wgpu_context->swap_chain.format,
    .blend     = &blend_state,
    .writeMask = WGPUColorWriteMask_All,
  };

  // Vertex buffer layout
  const size_t vec4_size = sizeof(vec4);
  WGPU_VERTEX_BUFFER_LAYOUT(
    animometer, 2 * vec4_size,
    // Attribute location 0: Vertex positions
    WGPU_VERTATTR_DESC(0, WGPUVertexFormat_Float32x4, 0),
    // Attribute location 1: Vertex colors
    WGPU_VERTATTR_DESC(1, WGPUVertexFormat_Float32x4, vec4_size))

  // Vertex state
  WGPUVertexState vertex_state_desc = wgpu_create_vertex_state(
                wgpu_context, &(wgpu_vertex_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Vertex shader WGSL
                  .label            = "Vertex shader WGSL",
                  .wgsl_code.source = vertex_shader_wgsl,
                  .entry            = "vert_main",
                },
                .buffer_count = 1,
                .buffers      = &animometer_vertex_buffer_layout,
              });

  // Fragment state
  WGPUFragmentState fragment_state_desc = wgpu_create_fragment_state(
                wgpu_context, &(wgpu_fragment_state_t){
                .shader_desc = (wgpu_shader_desc_t){
                  // Fragment shader WGSL
                  .label            = "Fragment shader WGSL",
                  .wgsl_code.source = fragment_shader_wgsl,
                  .entry            = "frag_main",
                },
                .target_count = 1,
                .targets      = &color_target_state_desc,
              });

  // Multisample state
  WGPUMultisampleState multisample_state_desc
    = wgpu_create_multisample_state_descriptor(
      &(create_multisample_state_desc_t){
        .sample_count = 1,
      });

  // Create rendering pipeline layout using the specified states
  WGPURenderPipelineDescriptor pipeline_desc = {
    .label       = "Animometer render pipeline",
    .primitive   = primitive_state_desc,
    .vertex      = vertex_state_desc,
    .fragment    = &fragment_state_desc,
    .multisample = multisample_state_desc,
  };

  // Create render pipelines
  pipeline_desc.layout = pipeline_layout;
  pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);

  pipeline_desc.layout = dynamic_pipeline_layout;
  dynamic_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);

  // Partial cleanup
  WGPU_RELEASE_RESOURCE(ShaderModule, vertex_state_desc.module);
  WGPU_RELEASE_RESOURCE(ShaderModule, fragment_state_desc.module);
}

static void prepare_render_bundle_encoder(wgpu_context_t* wgpu_context)
{
  WGPUTextureFormat color_formats[1] = {wgpu_context->swap_chain.format};
  WGPURenderBundleEncoder render_bundle_encoder
    = wgpuDeviceCreateRenderBundleEncoder(wgpu_context->device,
                                          &(WGPURenderBundleEncoderDescriptor){
                                            .label = "Render bundle encoder",
                                            .colorFormatCount = 1,
                                            .colorFormats      = color_formats,
                                            .sampleCount       = 1,
                                          });
  RECORD_RENDER_PASS(RenderBundleEncoder, render_bundle_encoder)
  render_bundle = wgpuRenderBundleEncoderFinish(render_bundle_encoder, NULL);

  WGPU_RELEASE_RESOURCE(RenderBundleEncoder, render_bundle_encoder)
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    prepare_vertex_buffer(context->wgpu_context);
    setup_pipeline_layout(context->wgpu_context);
    prepare_uniform_buffers(context->wgpu_context);
    prepare_pipelines(context->wgpu_context);
    setup_render_pass(context->wgpu_context);
    prepare_render_bundle_encoder(context->wgpu_context);
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static void example_on_update_ui_overlay(wgpu_example_context_t* context)
{
  if (imgui_overlay_header("Settings")) {
    imgui_overlay_checkBox(context->imgui_overlay, "Paused", &context->paused);
    imgui_overlay_checkBox(context->imgui_overlay, "Render Bundles",
                           &settings.render_bundles);
    imgui_overlay_checkBox(context->imgui_overlay, "Dynamic Offsets",
                           &settings.dynamic_offsets);
  }
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  render_pass.color_attachments[0].view = wgpu_context->swap_chain.frame_buffer;

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  {
    // Render pass
    wgpu_context->rpass_enc = wgpuCommandEncoderBeginRenderPass(
      wgpu_context->cmd_enc, &render_pass.descriptor);

    if (settings.render_bundles) {
      const WGPURenderBundle render_bundles[1] = {render_bundle};
      wgpuRenderPassEncoderExecuteBundles(wgpu_context->rpass_enc, 1,
                                          render_bundles);
    }
    else {
      RECORD_RENDER_PASS(RenderPassEncoder, wgpu_context->rpass_enc)
    }

    wgpuRenderPassEncoderEnd(wgpu_context->rpass_enc);
    WGPU_RELEASE_RESOURCE(RenderPassEncoder, wgpu_context->rpass_enc)
  }

  // Draw ui overlay
  draw_ui(wgpu_context->context, example_on_update_ui_overlay);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
  ASSERT(command_buffer != NULL);
  WGPU_RELEASE_RESOURCE(CommandEncoder, wgpu_context->cmd_enc)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0]
    = build_command_buffer(context->wgpu_context);

  // Submit command buffer to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return EXIT_SUCCESS;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return EXIT_FAILURE;
  }
  const int draw_result = example_draw(context);
  if (!context->paused) {
    update_uniform_buffers(context);
  }
  return draw_result;
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  WGPU_RELEASE_RESOURCE(Buffer, vertices.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, uniform_buffer)
  WGPU_RELEASE_RESOURCE(PipelineLayout, pipeline_layout)
  WGPU_RELEASE_RESOURCE(PipelineLayout, dynamic_pipeline_layout)
  WGPU_RELEASE_RESOURCE(RenderPipeline, pipeline)
  WGPU_RELEASE_RESOURCE(RenderPipeline, dynamic_pipeline)
  WGPU_RELEASE_RESOURCE(RenderBundle, render_bundle)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, time_bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, bind_group_layout)
  WGPU_RELEASE_RESOURCE(BindGroupLayout, dynamic_bind_group_layout);
  for (uint64_t i = 0; i < settings.num_triangles; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, bind_groups[i])
  }
  free(bind_groups);
  WGPU_RELEASE_RESOURCE(BindGroup, dynamic_bind_group)
  WGPU_RELEASE_RESOURCE(BindGroup, time_bind_group)
}

void example_animometer(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
      .title   = example_title,
      .overlay = true,
      .vsync   = false,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
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
