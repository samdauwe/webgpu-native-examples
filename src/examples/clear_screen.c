#include "webgpu/wgpu_common.h"

#include <stdio.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Clear Screen
 *
 * This example shows how to set up a swap chain and clearing the screen.
 *
 * Ref:
 * https://tsherif.github.io/webgpu-examples
 * https://github.com/tsherif/webgpu-examples/blob/gh-pages/blank.html
 * -------------------------------------------------------------------------- */

static struct {
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_dscriptor;
  bool prepared;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {1.0, 1.0, 1.0, 1.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_dscriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments     = &state.color_attachment,
  },
};

static int init(struct wgpu_context_t* wgpu_context)
{
  if (wgpu_context) {
    state.prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUColor lerp(WGPUColor* a, WGPUColor* b, float t)
{
  return (WGPUColor){
    .r = (1 - t) * a->r + t * b->r,
    .g = (1 - t) * a->g + t * b->g,
    .b = (1 - t) * a->b + t * b->b,
  };
}

static int frame(struct wgpu_context_t* wgpu_context)
{
  if (!state.prepared) {
    return EXIT_FAILURE;
  }

  WGPUDevice device = wgpu_context->device;
  WGPUQueue queue   = wgpu_context->queue;

  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Figure out how far along duration we are, between 0.0 and 1.0 */
  const float t = cos(nano_time() * powf(10, -9)) * 0.5f + 0.5f;

  /* Interpolate between two colors */
  state.color_attachment.clearValue = lerp(
    &(WGPUColor){
      .r = 0.0,
      .g = 0.0,
      .b = 0.0,
    },
    &(WGPUColor){
      .r = 1.0,
      .g = 1.0,
      .b = 1.0,
    },
    t);

  WGPUCommandEncoder cmd_enc = wgpuDeviceCreateCommandEncoder(device, NULL);
  WGPURenderPassEncoder rpass_enc
    = wgpuCommandEncoderBeginRenderPass(cmd_enc, &state.render_pass_dscriptor);

  /* Record render commands. */
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
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title       = "Clear Screen",
    .init_cb     = init,
    .frame_cb    = frame,
    .shutdown_cb = shutdown,
  });

  return EXIT_SUCCESS;
}
