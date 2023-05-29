#ifndef EXAMPLE_BASE_H
#define EXAMPLE_BASE_H

#include "../core/api.h"
#include "../webgpu/api.h"

typedef struct {
  window_t* window;
  struct {
    uint32_t width;
    uint32_t height;
    float aspect_ratio;
  } window_size;
  callbacks_t callbacks;
  wgpu_context_t* wgpu_context;
  bool vsync;
  struct {
    size_t index;
    float timestamp_millis;
  } frame;
  // Title of the example
  char example_title[STRMAX];
  // Backend info:
  //  0: adapter name
  //  1: adapter type name
  //  2: backend name
  char adapter_info[3][256];
  // Frame counter to display fps
  uint32_t frame_counter;
  // Used to display fps
  uint32_t last_fps;
  // ImGui overlay
  bool show_imgui_overlay;
  void* imgui_overlay;
  // Time the example has been running (in seconds)
  float run_time;
  // Last frame time measured using a high performance timer (if available)
  float frame_timer;
  // Defines a frame rate independent timer value clamped from -1.0...1.0
  // For use in animations, rotations, etc.
  float timer;
  // Multiplier for speeding up (or slowing down) the global timer
  float timer_speed;
  bool paused;
  camera_t* camera;
  // Input
  vec2 mouse_position;
  struct {
    bool left;
    bool right;
    bool middle;
  } mouse_buttons, mouse_dragging;
} wgpu_example_context_t;

typedef struct {
  /** @brief Example title */
  const char* title;
  /** @brief Set to true if v-sync will be forced for the swapchain */
  bool vsync;
  /** @brief Enable UI overlay */
  bool overlay;
  /** @brief Depth stencil format of UI overlay */
  WGPUTextureFormat overlay_deph_stencil_format;
  /** @brief Create texture client */
  bool create_texture_client;
} wgpu_example_settings_t;

typedef void* surface_t;
typedef int initializefunc_t(wgpu_example_context_t* context);
typedef int renderfunc_t(wgpu_example_context_t* context);
typedef void destroyfunc_t(wgpu_example_context_t* context);
typedef void onviewchangedfunc_t(wgpu_example_context_t* context);
typedef void onupdateuioverlayfunc_t(wgpu_example_context_t* context);
typedef void onkeypressedfunc_t(keycode_t key);
typedef void onpointerdownfunc_t(button_t button);
typedef void onpointerupfunc_t(button_t button);

typedef struct {
  onkeypressedfunc_t* example_on_key_pressed_func;
  onpointerdownfunc_t* example_on_pointer_down_func;
  onpointerupfunc_t* example_on_pointer_up_func;
} user_input_event_listener_export_t;

typedef struct {
  wgpu_example_settings_t example_settings;
  window_config_t example_window_config;
  initializefunc_t* example_initialize_func;
  renderfunc_t* example_render_func;
  destroyfunc_t* example_destroy_func;
  onviewchangedfunc_t* example_on_view_changed_func;
  onkeypressedfunc_t* example_on_key_pressed_func;
} refexport_t;

/* Helper functions */
void draw_ui(wgpu_example_context_t* context,
             onupdateuioverlayfunc_t* example_on_update_ui_overlay_func);
void prepare_frame(wgpu_example_context_t* context);
void submit_command_buffers(wgpu_example_context_t* context);
void submit_frame(wgpu_example_context_t* context);

void example_run(int argc, char* argv[], refexport_t* ref_export);

#endif
