#include "webgpu/wgpu_common.h"

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

static WGPUShaderModule create_shader_module(WGPUDevice device,
                                             const char* wgsl_source)
{
  WGPUShaderSourceWGSL shaderCodeDesc
    = {.chain = {.sType = WGPUSType_ShaderSourceWGSL},
       .code  = {
          .data   = wgsl_source,
          .length = WGPU_STRLEN,
       }};
  WGPUShaderModuleDescriptor shaderDesc
    = {.nextInChain = &shaderCodeDesc.chain};
  return wgpuDeviceCreateShaderModule(device, &shaderDesc);
}

static void prepare_pipelines(struct wgpu_context_t* wgpu_context)
{
  WGPUShaderModule vert_shader_module
    = create_shader_module(wgpu_context->device, triangle_vert_wgsl);
  WGPUShaderModule frag_shader_module
    = create_shader_module(wgpu_context->device, red_frag_wgsl);

  WGPURenderPipelineDescriptor rpdesc = {
    .vertex = {
      .module     = vert_shader_module,
      .entryPoint = STRVIEW("main"),
    },
    .fragment = &(WGPUFragmentState) {
      .entryPoint  = STRVIEW("main"),
      .module      = frag_shader_module,
      .targetCount = 1,
      .targets = &  (WGPUColorTargetState)  {
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
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &rpdesc);

  wgpuShaderModuleRelease(vert_shader_module);
  wgpuShaderModuleRelease(frag_shader_module);
}

static void init(struct wgpu_context_t* wgpu_context)
{
  prepare_pipelines(wgpu_context);
}

static void frame(struct wgpu_context_t* wgpu_context)
{
  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  WGPUCommandEncoder cenc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpenc
    = wgpuCommandEncoderBeginRenderPass(cenc, &state.render_pass_dscriptor);

  // Record render commands.
  wgpuRenderPassEncoderSetPipeline(rpenc, state.pipeline);
  wgpuRenderPassEncoderDraw(rpenc, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(rpenc);
  WGPUCommandBuffer cbuffer = wgpuCommandEncoderFinish(cenc, NULL);

  // Submit and present.
  wgpuQueueSubmit(queue, 1, &cbuffer);

  wgpuRenderPassEncoderRelease(rpenc);
  wgpuCommandBufferRelease(cbuffer);
  wgpuCommandEncoderRelease(cenc);
}

static void shutdown(struct wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.pipeline);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .width       = 900,
    .height      = 900,
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
