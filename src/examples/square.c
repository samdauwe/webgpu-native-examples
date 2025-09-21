#include "webgpu/wgpu_common.h"

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Square
 *
 * This example shows how to render a static colored square in WebGPU with only
 * using vertex buffers.
 *
 * Ref:
 * https://github.com/cx20/webgpu-test/tree/master/examples/webgpu_wgsl/square
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* vertex_shader_wgsl;
static const char* fragment_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Square example
 * -------------------------------------------------------------------------- */

static struct {
  struct {
    wgpu_buffer_t positions; /* Positions */
    wgpu_buffer_t colors;    /* Colors */
  } square;
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  WGPUBool initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
};

static void init_vertex_buffers(wgpu_context_t* wgpu_context)
{
  // Square data
  //             1.0 y
  //              ^  -1.0
  //              | / z
  //              |/       x
  // -1.0 -----------------> +1.0
  //            / |
  //      +1.0 /  |
  //           -1.0
  //
  //        [0]------[1]
  //         |        |
  //         |        |
  //         |        |
  //        [2]------[3]
  //
  // clang-format off
  static const float positions[12] = {
    -0.5f,  0.5f, 0.0f, // v0
     0.5f,  0.5f, 0.0f, // v1
    -0.5f, -0.5f, 0.0f, // v2
     0.5f, -0.5f, 0.0f, // v3
  };
  // clang-format on
  state.square.positions = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(positions),
                    .count = 4,
                    .initial.data = positions,
                  });

  static const float colors[16] = {
    1.0f, 0.0f, 0.0f, 1.0f, /* v0 */
    0.0f, 1.0f, 0.0f, 1.0f, /* v1 */
    0.0f, 0.0f, 1.0f, 1.0f, /* v2 */
    1.0f, 1.0f, 0.0f, 1.0f  /* v3 */
  };
  state.square.colors = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Colored square - Vertex buffer",
                    .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex,
                    .size  = sizeof(colors),
                    .count = 4,
                    .initial.data = colors,
                  });
}

static void init_pipeline(wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, vertex_shader_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, fragment_shader_wgsl);

  /* Vertex buffer layout */
  WGPUVertexBufferLayout vertex_buffer_layouts[2] = {
    [0] = {
      .arrayStride    = sizeof(float) * 3,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &(WGPUVertexAttribute) {
        /* Shader location 0 : vertex position */
        .shaderLocation = 0,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x3,
      },
    },
    [1] = {
      .arrayStride    = sizeof(float) * 4,
      .stepMode       = WGPUVertexStepMode_Vertex,
      .attributeCount = 1,
      .attributes     = &(WGPUVertexAttribute) {
        /* Shader location 1 : vertex color */
        .shaderLocation = 1,
        .offset         = 0,
        .format         = WGPUVertexFormat_Float32x4,
      },
    }
  };

  WGPURenderPipelineDescriptor rp_desc = {
    .label = STRVIEW("Square - Render pipeline"),
    .vertex = {
      .module      = vert_shader_module,
      .entryPoint  = STRVIEW("main"),
      .bufferCount = (uint32_t)ARRAY_SIZE(vertex_buffer_layouts),
      .buffers     = vertex_buffer_layouts,
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState) {
        .format    = wgpu_context->render_format,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology         = WGPUPrimitiveTopology_TriangleStrip,
      .stripIndexFormat = WGPUIndexFormat_Uint32,
      .frontFace        = WGPUFrontFace_CCW,
      .cullMode         = WGPUCullMode_Front,
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
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_vertex_buffers(wgpu_context);
    init_pipeline(wgpu_context);
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

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
  wgpuRenderPassEncoderSetViewport(rpass_enc, 0.0f, 0.0f,
                                   (float)wgpu_context->width,
                                   (float)wgpu_context->height, 0.0f, 1.0f);
  wgpuRenderPassEncoderSetScissorRect(rpass_enc, 0u, 0u, wgpu_context->width,
                                      wgpu_context->height);

  wgpuRenderPassEncoderSetVertexBuffer(
    rpass_enc, 0, state.square.positions.buffer, 0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderSetVertexBuffer(rpass_enc, 1, state.square.colors.buffer,
                                       0, WGPU_WHOLE_SIZE);
  wgpuRenderPassEncoderDraw(rpass_enc, state.square.positions.count, 1, 0, 0);
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
  WGPU_RELEASE_RESOURCE(Buffer, state.square.positions.buffer)
  WGPU_RELEASE_RESOURCE(Buffer, state.square.colors.buffer)
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline)
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Square",
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
  struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color : vec4<f32>
  };

  struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragColor : vec4<f32>
  };

  @vertex
  fn main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.fragColor = input.color;
    output.Position = vec4<f32>(input.position, 1.0);
    return output;
  };
);

static const char* fragment_shader_wgsl = CODE(
  struct FragmentInput {
    @location(0) fragColor : vec4<f32>
  };

  struct FragmentOutput {
    @location(0) outColor : vec4<f32>
  };

  @fragment
  fn main(input : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.outColor = input.fragColor;
    return output;
  };
);
// clang-format on
