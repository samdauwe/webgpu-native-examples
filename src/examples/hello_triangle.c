#include "webgpu/wgpu_common.h"

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Hello Triangle
 *
 * This example shows rendering a basic triangle.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/sample/helloTriangle
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shaders
 * -------------------------------------------------------------------------- */

static const char* triangle_vert_wgsl;
static const char* red_frag_wgsl;

/* -------------------------------------------------------------------------- *
 * Hello Triangle example
 * -------------------------------------------------------------------------- */

static struct {
  WGPURenderPipeline pipeline;
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  int8_t initialized;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 0.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
};

static void init_pipeline(struct wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = wgpu_create_shader_module(wgpu_context->device, triangle_vert_wgsl);
  WGPUShaderModule frag_shader_module
    = wgpu_create_shader_module(wgpu_context->device, red_frag_wgsl);

  WGPURenderPipelineDescriptor rp_desc = {
    .label = STRVIEW("Hello Triangle - Render pipeline"),
    .vertex = {
      .module     = vert_shader_module,
      .entryPoint = STRVIEW("main"),
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &(WGPUColorTargetState)  {
        .format    = wgpu_context->render_format,
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

  state.pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rp_desc);
  ASSERT(state.pipeline != NULL);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    init_pipeline(wgpu_context);
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

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.pipeline);
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
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .width       = 800,
    .height      = 800,
    .title       = "Hello Triangle",
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
static const char* triangle_vert_wgsl = CODE(
  @vertex
  fn main(
    @builtin(vertex_index) VertexIndex : u32
  ) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(
      vec2(0.0, 0.5),
      vec2(-0.5, -0.5),
      vec2(0.5, -0.5)
    );

    return vec4f(pos[VertexIndex], 0.0, 1.0);
  }
);

static const char* red_frag_wgsl = CODE(
  @fragment
  fn main() -> @location(0) vec4f {
    return vec4(1.0, 0.0, 0.0, 1.0);
  }
);
// clang-format on
