#include "example_base.h"

#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Clear Screen
 *
 * This example shows how to set up a swap chain and clearing the screen.
 *
 * Ref:
 * https://tsherif.github.io/webgpu-examples
 * https://github.com/tsherif/webgpu-examples/blob/gh-pages/blank.html
 * -------------------------------------------------------------------------- */

// Render pass descriptor for frame buffer writes
static struct {
  WGPURenderPassColorAttachment color_attachments[1];
  WGPURenderPassDescriptor descriptor;
} render_pass = {0};

// Other variables
static const char* example_title = "Clear Screen";
static bool prepared             = false;

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
        .r = 1.0f,
        .g = 1.0f,
        .b = 1.0f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass.descriptor = (WGPURenderPassDescriptor){
    .label                  = "Render pass descriptor",
    .colorAttachmentCount   = 1,
    .colorAttachments       = render_pass.color_attachments,
    .depthStencilAttachment = NULL,
  };
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    // Setup render pass
    setup_render_pass(context->wgpu_context);
    prepared = true;
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

static WGPUCommandBuffer build_command_buffer(wgpu_example_context_t* context)
{
  render_pass.color_attachments[0].view
    = context->wgpu_context->swap_chain.frame_buffer;

  /* Figure out how far along duration we are, between 0.0 and 1.0 */
  const float t = cos(context->frame.timestamp_millis * 0.001f) * 0.5f + 0.5f;

  /* Interpolate between two colors */
  render_pass.color_attachments[0].clearValue = lerp(
    &(WGPUColor){
      .r = 0.0f,
      .g = 0.0f,
      .b = 0.0f,
    },
    &(WGPUColor){
      .r = 1.0f,
      .g = 1.0f,
      .b = 1.0f,
    },
    t);

  /* Create command encoder */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(context->wgpu_context->device, NULL);

  /* Create render pass */
  WGPURenderPassEncoder rpass
    = wgpuCommandEncoderBeginRenderPass(cmd_encoder, &render_pass.descriptor);

  /* End render pass */
  wgpuRenderPassEncoderEnd(rpass);
  WGPU_RELEASE_RESOURCE(RenderPassEncoder, rpass)

  /* Get command buffer */
  WGPUCommandBuffer command_buffer = wgpu_get_command_buffer(cmd_encoder);
  ASSERT(command_buffer != NULL)
  WGPU_RELEASE_RESOURCE(CommandEncoder, cmd_encoder)

  return command_buffer;
}

static int example_draw(wgpu_example_context_t* context)
{
  // Prepare frame
  prepare_frame(context);

  // Command buffer to be submitted to the queue
  wgpu_context_t* wgpu_context                   = context->wgpu_context;
  wgpu_context->submit_info.command_buffer_count = 1;
  wgpu_context->submit_info.command_buffers[0] = build_command_buffer(context);

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
  return example_draw(context);
}

// Clean up used resources
static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
}

void example_clear_screen(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title = example_title,
     .vsync = true,
    },
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
