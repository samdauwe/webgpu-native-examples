#include "example_base.h"

#include <string.h>

#include "../core/argparse.h"
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

/* mainloop related functions */

static const char* const WINDOW_TITLE = "WebGPU Example";
static const uint32_t WINDOW_WIDTH    = 1280;
static const uint32_t WINDOW_HEIGHT   = 720;

typedef struct {
  bool window_resized;
  bool view_updated;
  /* zoom */
  bool mouse_scrolled;
  float wheel_delta;
  /* click */
  bool buttons[BUTTON_NUM];
  vec2 cursor_pos;
  /* key press */
  bool keys[KEY_NUM];
  bool keys_changed;
  int32_t last_key_pressed;
  /* timings */
  uint32_t frame_counter;
  float last_timestamp;
  float frame_timer;
  float last_fps;
} record_t;

static void get_pos_delta(vec2 old_pos, vec2 new_pos, vec2* result)
{
  glm_vec2_sub(new_pos, old_pos, *result);
}

static void get_cursor_pos(window_t* window, vec2* result)
{
  input_query_cursor(window, &(*result)[0], &(*result)[1]);
}

static void mouse_button_callback(window_t* window, int ctrl_key, int shift_key,
                                  float mouse_x, float mouse_y, button_t button,
                                  button_action_t button_action)
{
  UNUSED_VAR(ctrl_key);
  UNUSED_VAR(shift_key);

  record_t* record = (record_t*)window_get_userdata(window);

  record->cursor_pos[0] = mouse_x;
  record->cursor_pos[1] = mouse_y;
  const bool pressed    = button_action == BUTTON_ACTION_PRESS;

  record->buttons[BUTTON_LEFT]   = (button == BUTTON_LEFT) && pressed;
  record->buttons[BUTTON_MIDDLE] = (button == BUTTON_MIDDLE) && pressed;
  record->buttons[BUTTON_RIGHT]  = (button == BUTTON_RIGHT) && pressed;
}

static void key_callback(window_t* window, int ctrl_key, int shift_key,
                         keycode_t key_code, button_action_t button_action)
{
  UNUSED_VAR(ctrl_key);
  UNUSED_VAR(shift_key);

  record_t* record   = (record_t*)window_get_userdata(window);
  const bool pressed = button_action == BUTTON_ACTION_PRESS;

  record->keys[key_code] = pressed ? true : false;
  record->keys_changed   = true;
  if (!pressed) {
    record->last_key_pressed = key_code;
  }
}

static void scroll_callback(window_t* window, int ctrl_key, int shift_key,
                            float mouse_x, float mouse_y, float wheel_delta_y)
{
  UNUSED_VAR(ctrl_key);
  UNUSED_VAR(shift_key);

  record_t* record = (record_t*)window_get_userdata(window);

  record->cursor_pos[0]  = mouse_x;
  record->cursor_pos[1]  = mouse_y;
  record->mouse_scrolled = true;
  record->wheel_delta += wheel_delta_y;
}

static void resize_callback(window_t* window, int width, int height)
{
  UNUSED_VAR(width);
  UNUSED_VAR(height);

  record_t* record       = (record_t*)window_get_userdata(window);
  record->window_resized = true;
}

#if 0
static void update_window_size(wgpu_example_context_t* context,
                               record_t* record)
{
  if (record->window_resized) {
    wgpu_context_t* wgpu_context = context->wgpu_context;
    // Update window size and aspect ratio
    window_get_size(context->window, &context->window_size.width,
                    &context->window_size.height);
    window_get_aspect_ratio(context->window,
                            &context->window_size.aspect_ratio);
    // Update surface size
    window_get_size(context->window, &wgpu_context->surface.width,
                    &wgpu_context->surface.height);
    // Recreate swap chain
    wgpu_setup_swap_chain(context->wgpu_context);
    record->window_resized = false;
  }
}
#endif

static void update_camera(wgpu_example_context_t* context, record_t* record)
{
  window_t* window = context->window;
  camera_t* camera = context->camera;

  if (camera == NULL) {
    return;
  }

  camera->updated = false;

  vec2 cursor_pos;
  get_cursor_pos(window, &cursor_pos);

  vec2 pos_delta;
  get_pos_delta(record->cursor_pos, cursor_pos, &pos_delta);
  glm_vec2_copy(cursor_pos, record->cursor_pos);

  bool handled = false;

  if (context->show_imgui_overlay) {
    handled = imgui_overlay_want_capture_mouse();
  }

  if (handled) {
    glm_vec2_copy(cursor_pos, context->mouse_position);
    return;
  }

  /* Mouse clicks */
  if (record->buttons[BUTTON_LEFT]) {
    camera_rotate(camera, (vec3){pos_delta[1] * camera->rotation_speed,
                                 -pos_delta[0] * camera->rotation_speed, 0.0f});
    record->view_updated = true;
  }

  if (record->buttons[BUTTON_MIDDLE]) {
    camera_translate(
      camera, (vec3){-pos_delta[0] * 0.01f, -pos_delta[1] * 0.01f, 0.0f});
    record->view_updated = true;
  }

  if (record->buttons[BUTTON_RIGHT]) {
    camera_translate(camera, (vec3){-0.0f, 0.0f, pos_delta[1] * 0.005f});
    record->view_updated = true;
  }

  if (record->mouse_scrolled) {
    camera_translate(camera,
                     (vec3){0.0f, 0.0f, -(float)record->wheel_delta * 0.05f});
    record->view_updated = true;
  }
  glm_vec2_copy(cursor_pos, context->mouse_position);

  /* Key events */
  if (record->keys_changed) {
    camera->keys.up    = record->keys[KEY_W];
    camera->keys.down  = record->keys[KEY_S];
    camera->keys.left  = record->keys[KEY_A];
    camera->keys.right = record->keys[KEY_D];
    camera_update(camera, context->frame_timer);
    if (camera_moving(camera)) {
      record->view_updated = true;
    }
    record->keys_changed = false;
  }
}

static void update_input_state(wgpu_example_context_t* context,
                               record_t* record)
{
  // Mouse position
  get_cursor_pos(context->window, &context->mouse_position);

  // Mouse buttons press state
  context->mouse_buttons.left   = record->buttons[BUTTON_LEFT];
  context->mouse_buttons.right  = record->buttons[BUTTON_RIGHT];
  context->mouse_buttons.middle = record->buttons[BUTTON_MIDDLE];

  // Mouse buttons dragging state
  if (!context->mouse_dragging.left && context->mouse_buttons.left) {
    context->mouse_dragging.left = true;
  }
  else if (context->mouse_dragging.left && !context->mouse_buttons.left) {
    context->mouse_dragging.left = false;
  }
}

static void
notify_key_input_state(record_t* record,
                       onkeypressedfunc_t* example_on_key_pressed_func)
{
  if (record->last_key_pressed != KEY_UNKNOWN) {
    example_on_key_pressed_func(record->last_key_pressed);
    record->last_key_pressed = KEY_UNKNOWN;
  }
}

static void parse_example_arguments(int argc, char* argv[],
                                    refexport_t* ref_export)
{
  char* filters_short[2]           = {"-w", "-h"};
  char* filters_eq[2]              = {"--width=", "--height="};
  char* filtered_argv[1 + (2 * 2)] = {0};
  char** argvc                     = (char**)argv;
  int fargc                        = 1;
  for (int32_t i = 0; i < argc; ++i) {
    for (uint32_t j = 0; j < (uint32_t)ARRAY_SIZE(filters_short); ++j) {
      if (strcmp(argvc[i], filters_short[j]) == 0) {
        filtered_argv[fargc++] = filters_short[j];
        filtered_argv[fargc++] = argvc[++i];
      }
    }
    for (uint32_t j = 0; j < (uint32_t)ARRAY_SIZE(filters_eq); ++j) {
      if (has_prefix(argvc[i], filters_eq[j])) {
        filtered_argv[fargc++] = argvc[i];
      }
    }
  }

  int window_width = 0, window_height = 0;
  struct argparse_option options[] = {
    OPT_INTEGER('w', "width", &window_width, "window width", NULL, 0, 0),
    OPT_INTEGER('h', "height", &window_height, "window height", NULL, 0, 0),
    OPT_END(),
  };
  struct argparse argparse;
  const char* const usages[] = {NULL};
  argparse_init(&argparse, options, usages, 0);
  argparse_parse(&argparse, fargc, (const char**)filtered_argv);

  // Override default example window dimensions
  if (window_width > 100) {
    ref_export->example_window_config.width = window_width;
  }
  if (window_height > 100) {
    ref_export->example_window_config.height = window_height;
  }
}

static void
intialize_wgpu_example_context(wgpu_example_context_t* context,
                               wgpu_example_settings_t* example_settings)
{
  memset(context, 0, sizeof(wgpu_example_context_t));

  // Example settings
  snprintf(context->example_title, strlen(example_settings->title) + 1, "%s",
           example_settings->title);

  // V-Sync setting for the swapchain
  context->vsync = example_settings->vsync;

  // FPS
  context->frame_counter = 0;
  context->last_fps      = 0;

  // Timers
  context->run_time    = 0.0f;
  context->frame_timer = 1.0f;
  context->timer       = 0.0f;
  context->timer_speed = 0.25f;
  context->paused      = false;

  // Input
  glm_vec2_zero(context->mouse_position);
  context->mouse_buttons.left    = false;
  context->mouse_buttons.right   = false;
  context->mouse_buttons.middle  = false;
  context->mouse_dragging.left   = false;
  context->mouse_dragging.right  = false;
  context->mouse_dragging.middle = false;
}

static void setup_window(wgpu_example_context_t* context,
                         window_config_t* windows_config)
{
  char window_title[STRMAX];
  snprintf(window_title,
           strlen("WebGPU Example - ") + strlen(context->example_title) + 1,
           "WebGPU Example - %s", context->example_title);
  window_config_t config = {
    .title     = GET_DEFAULT_IF_ZERO((const char*)window_title, WINDOW_TITLE),
    .width     = GET_DEFAULT_IF_ZERO(windows_config->width, WINDOW_WIDTH),
    .height    = GET_DEFAULT_IF_ZERO(windows_config->height, WINDOW_HEIGHT),
    .resizable = windows_config->resizable,
  };
  context->window = window_create(&config);
  window_get_size(context->window, &context->window_size.width,
                  &context->window_size.height);
  window_get_aspect_ratio(context->window, &context->window_size.aspect_ratio);

  memset(&context->callbacks, 0, sizeof(callbacks_t));
  context->callbacks.mouse_button_callback = mouse_button_callback;
  context->callbacks.key_callback          = key_callback;
  context->callbacks.scroll_callback       = scroll_callback;
  context->callbacks.resize_callback       = resize_callback;

  input_set_callbacks(context->window, context->callbacks);
}

static void intialize_webgpu(wgpu_example_context_t* context)
{
  context->wgpu_context = wgpu_context_create(&(wgpu_context_create_options_t){
    .vsync = context->vsync,
  });
  context->wgpu_context->context = context;

  wgpu_create_device_and_queue(context->wgpu_context);
  wgpu_setup_window_surface(context->wgpu_context, context->window);
  wgpu_setup_swap_chain(context->wgpu_context);
  wgpu_get_context_info(context->adapter_info);
}

static void intialize_imgui(wgpu_example_context_t* context,
                            wgpu_example_settings_t* example_settings)
{
  context->show_imgui_overlay = example_settings->overlay;
  if (context->show_imgui_overlay) {
    // Create and intialize ImGui ovelay
    WGPUTextureFormat format
      = (example_settings->overlay_deph_stencil_format
             != WGPUTextureFormat_Undefined ?
           example_settings->overlay_deph_stencil_format :
           WGPUTextureFormat_Depth24PlusStencil8);
    context->imgui_overlay
      = imgui_overlay_create(context->wgpu_context, format);
  }
}

static void release_imgui(wgpu_example_context_t* context)
{
  if (context->imgui_overlay != NULL) {
    imgui_overlay_release(context->imgui_overlay);
  }
}

static void release_webgpu(wgpu_example_context_t* context)
{
  wgpu_context_release(context->wgpu_context);
}

static void
update_overlay(wgpu_example_context_t* context,
               onupdateuioverlayfunc_t* example_on_update_ui_overlay_func)
{
  if (!context->show_imgui_overlay) {
    return;
  }

  imgui_overlay_new_frame(context->imgui_overlay, context);

  igSetNextWindowPos((ImVec2){10, 10}, ImGuiCond_None, (ImVec2){0, 0});
  igSetNextWindowSize((ImVec2){0, 0}, ImGuiCond_FirstUseEver);
  static bool show_window = true;
  igBegin(WINDOW_TITLE, &show_window,
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize
            | ImGuiWindowFlags_NoMove);
  igTextUnformatted(context->example_title, NULL);
  igTextUnformatted(context->adapter_info[0], NULL);
  igText("%s backend - %s", context->adapter_info[2], context->adapter_info[1]);
  igText("%.2f ms/frame (%.1d fps)", (1000.0f / context->last_fps),
         context->last_fps);
  if (example_on_update_ui_overlay_func) {
    igPushItemWidth(110.0f * imgui_overlay_get_scale(context->imgui_overlay));
    example_on_update_ui_overlay_func(context);
    igPopItemWidth();
  }
  igEnd();

  imgui_overlay_render(context->imgui_overlay);
}

static void render_loop(wgpu_example_context_t* context,
                        renderfunc_t* render_func,
                        onviewchangedfunc_t* view_changed_func,
                        onkeypressedfunc_t* example_on_key_pressed_func)
{
  record_t record;
  memset(&record, 0, sizeof(record_t));
  window_set_userdata(context->window, &record);

  float time_start, time_end, time_diff, fps_timer;
  record.last_timestamp = platform_get_time();
  while (!window_should_close(context->window)) {
    time_start                      = platform_get_time();
    context->frame.timestamp_millis = time_start * 1000.0f;
    if (record.view_updated) {
      record.mouse_scrolled = 0;
      record.wheel_delta    = 0;
      record.view_updated   = false;
    }
    input_poll_events();
    // update_window_size(context, &record);
    render_func(context);
    ++record.frame_counter;
    ++context->frame.index;
    time_end             = platform_get_time();
    time_diff            = (time_end - time_start) * 1000.0f;
    record.frame_timer   = time_diff / 1000.0f;
    context->frame_timer = record.frame_timer;
    context->run_time += context->frame_timer;
    update_camera(context, &record);
    update_input_state(context, &record);
    if (example_on_key_pressed_func) {
      notify_key_input_state(&record, example_on_key_pressed_func);
    }
    // Convert to clamped timer value
    if (!context->paused) {
      context->timer += context->timer_speed * record.frame_timer;
      if (context->timer >= 1.0) {
        context->timer -= 1.0f;
      }
    }
    if (record.view_updated && view_changed_func) {
      view_changed_func(context);
    }
    fps_timer = (time_end - record.last_timestamp) * 1000.0f;
    if (fps_timer > 1000.0f) {
      record.last_fps   = (float)record.frame_counter * (1000.0f / fps_timer);
      context->last_fps = (int)(record.last_fps + 0.5f);
      record.frame_counter  = 0;
      record.last_timestamp = time_end;
    }
    context->frame_counter = record.frame_counter;
  }
}

void draw_ui(wgpu_example_context_t* context,
             onupdateuioverlayfunc_t* example_on_update_ui_overlay_func)
{
  if (context->show_imgui_overlay) {
    update_overlay(context, example_on_update_ui_overlay_func);
    imgui_overlay_draw_frame(context->imgui_overlay,
                             context->wgpu_context->swap_chain.frame_buffer);
  }
}

void prepare_frame(wgpu_example_context_t* context)
{
  // Acquire the current image from the swap chain
  wgpu_swap_chain_get_current_image(context->wgpu_context);

  ASSERT(context->wgpu_context->swap_chain.frame_buffer != NULL);
}

void submit_command_buffers(wgpu_example_context_t* context)
{
  wgpu_context_t* wgpu_context = context->wgpu_context;

  // Submit command buffer(s) to the queue
  wgpu_flush_command_buffers(wgpu_context,
                             wgpu_context->submit_info.command_buffers,
                             wgpu_context->submit_info.command_buffer_count);
}

void submit_frame(wgpu_example_context_t* context)
{
  // Present the current buffer to the swap chain
  wgpu_swap_chain_present(context->wgpu_context);
}

void example_run(int argc, char* argv[], refexport_t* ref_export)
{
  // Parse the example arguments
  parse_example_arguments(argc, argv, ref_export);
  // Initialize WebGPU example context
  wgpu_example_context_t context;
  intialize_wgpu_example_context(&context, &ref_export->example_settings);
  // Setup Window
  setup_window(&context, &ref_export->example_window_config);
  // Intialize WebGPU
  intialize_webgpu(&context);
  // Intialize ImGui
  intialize_imgui(&context, &ref_export->example_settings);
  // Intialize example
  ref_export->example_initialize_func(&context);
  // Render loop
  render_loop(&context, ref_export->example_render_func,
              ref_export->example_on_view_changed_func,
              ref_export->example_on_key_pressed_func);
  // Cleanup
  ref_export->example_destroy_func(&context);
  release_imgui(&context);
  release_webgpu(&context);
  window_destroy(context.window);
}
