#include "example_base.h"

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

static const char* example_title = "ImGui Overlay";
static bool prepared             = false;

static imgui_overlay_t* imgui_overlay = NULL;

static int example_initialize(wgpu_example_context_t* context)
{
  if (context) {
    // Setup render pass
    prepared = true;
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}

static WGPUCommandBuffer build_command_buffer(wgpu_context_t* wgpu_context)
{
  if (imgui_overlay == NULL) {
    /* Create and intialize ImGui ovelay */
    imgui_overlay = imgui_overlay_create(wgpu_context,
                                         WGPUTextureFormat_Depth24PlusStencil8);
  }

  // Create command encoder */
  wgpu_context->cmd_enc
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);

  /* Start the Dear ImGui frame */
  imgui_overlay_new_frame(imgui_overlay, wgpu_context->context);

  static bool show_demo_window    = true;
  static bool show_another_window = true;
  static ImVec4 clearColor        = {0.45f, 0.55f, 0.60f, 1.00f};

  /* Show the ImGui demo window */
  {
    if (show_demo_window) {
      igShowDemoWindow(&show_demo_window);
    }
  }

  /* Show a simple window that we created ourselves. */
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

  /* Get command buffer */
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
    .example_initialize_func = &example_initialize,
    .example_render_func     = &example_render,
    .example_destroy_func    = &example_destroy,
  });
  // clang-format on
}
