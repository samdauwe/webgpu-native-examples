#include "example_base.h"
#include "examples.h"

#include <string.h>

#include "../webgpu/imgui_overlay.h"

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
 * WebGPU Example - ImGui Overlay
 *
 * Generates and renders a complex user interface with multiple windows,
 * controls and user interaction on top of a 3D scene. The UI is generated using
 * Dear ImGUI and updated each frame.
 *
 * Ref:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/imgui
 * -------------------------------------------------------------------------- */

// Render pass descriptor for frame buffer writes
static WGPURenderPassColorAttachment rp_color_att_descriptors[1];
static WGPURenderPassDescriptor render_pass_desc;

// Other variables
static const char* example_title = "ImGui Overlay";
static bool prepared             = false;

static imgui_overlay_t* imgui_overlay = NULL;

static void setup_render_pass(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  // Color attachment
  rp_color_att_descriptors[0] = (WGPURenderPassColorAttachment) {
      .view       = NULL,
      .attachment = NULL,
      .loadOp     = WGPULoadOp_Clear,
      .storeOp    = WGPUStoreOp_Store,
      .clearColor = (WGPUColor) {
        .r = 0.1f,
        .g = 0.2f,
        .b = 0.3f,
        .a = 1.0f,
      },
  };

  // Render pass descriptor
  render_pass_desc = (WGPURenderPassDescriptor){
    .colorAttachmentCount   = 1,
    .colorAttachments       = rp_color_att_descriptors,
    .depthStencilAttachment = NULL,
  };
}

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    // Setup render pass
    setup_render_pass(context->wgpu_context);
    prepared = true;
    return 0;
  }

  return 1;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  // Set target frame buffer
  rp_color_att_descriptors[0].view = wgpu_context->swap_chain.frame_buffer;

  if (imgui_overlay == NULL) {
    // Create and intialize ImGui ovelay
    imgui_overlay = imgui_overlay_create(wgpu_context);
  }

  // Create command encoder
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  // Start the Dear ImGui frame
  imgui_overlay_new_frame(imgui_overlay, wgpu_context->context);

  static bool show_demo_window    = true;
  static bool show_another_window = true;
  static ImVec4 clearColor        = {0.45f, 0.55f, 0.60f, 1.00f};

  {
    if (show_demo_window) {
      igShowDemoWindow(&show_demo_window);
    }
  }

  // show a simple window that we created ourselves.
  {
    static float f     = 0.0f;
    static int counter = 0;

    igBegin("Hello, world!", NULL, 0);
    igText("This is some useful text");
    igCheckbox("Demo window", &show_demo_window);
    igCheckbox("Another window", &show_another_window);

    igSliderFloat("Float", &f, 0.0f, 1.0f, "%.3f", 0);
    igColorEdit3("clear color", (float*)&clearColor, 0);

    ImVec2 buttonSize;
    buttonSize.x = 0;
    buttonSize.y = 0;
    if (igButton("Button", buttonSize)) {
      counter++;
    }
    igSameLine(0.0f, -1.0f);
    igText("counter = %d", counter);

    igText("Application average %.3f ms/frame (%.1f FPS)",
           1000.0f / igGetIO()->Framerate, igGetIO()->Framerate);
    igEnd();
  }

  if (show_another_window) {
    igBegin("imgui Another Window", &show_another_window, 0);
    igText("Hello from imgui");
    ImVec2 buttonSize;
    buttonSize.x = 0;
    buttonSize.y = 0;
    if (igButton("Close me", buttonSize)) {
      show_another_window = false;
    }
    igEnd();
  }

  imgui_overlay_render(imgui_overlay);
  imgui_overlay_draw_frame(imgui_overlay,
                           wgpu_context->swap_chain.frame_buffer);

  // Get command buffer
  WGPUCommandBuffer command_buffer
    = wgpu_get_command_buffer(wgpu_context->cmd_enc);
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

  // Submit to queue
  submit_command_buffers(context);

  // Submit frame
  submit_frame(context);

  return 0;
}

static int example_render(wgpu_example_context_t* context)
{
  if (!prepared) {
    return 1;
  }
  return example_draw(context);
}

static void example_destroy(wgpu_example_context_t* context)
{
  UNUSED_VAR(context);
  imgui_overlay_release(imgui_overlay);
}

void example_imgui_overlay(int argc, char* argv[])
{
  // clang-format off
  example_run(argc, argv, &(refexport_t){
    .example_settings = (wgpu_example_settings_t){
     .title  = example_title,
    },
    .example_initialize_func      = &example_initialize,
    .example_render_func          = &example_render,
    .example_destroy_func         = &example_destroy,
  });
  // clang-format on
}
