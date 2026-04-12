#include "wgpu_common.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Context
 * -------------------------------------------------------------------------- */

static wgpu_context_t wgpu_context;
static struct {
  input_event_type_t event_type;
  button_t button_type;
  double cursor_pos[2];
  double cursor_pos_delta[2];
  double mouse_offset[2];
  WGPUBool mouse_btn_pressed;
  WGPUBool alt_key;
  WGPUBool ctrl_key;
  WGPUBool shift_key;
  keycode_t key_code;
  uint32_t char_code;
  int window_width;
  int window_height;
  uint8_t
    keys_down[KEY_NUM]; /* persistent key state, survives event overwrites */
  /* File drop state */
  char drop_paths[MAX_DROP_PATHS][MAX_DROP_PATH_LEN];
  int drop_count;
} input_state = {0};

/* Forward declarations */
static void wgpu_platform_start(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_init(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_discard(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_resized(wgpu_context_t* wgpu_context);
static WGPUTextureView wgpu_swapchain_next(wgpu_context_t* wgpu_context);

static keycode_t remap_glfw_key_code(int key)
{
  keycode_t key_code = KEY_UNKNOWN;
  switch (key) {
      // clang-format off
    case GLFW_KEY_ESCAPE:        key_code = KEY_ESCAPE;        break;
    case GLFW_KEY_TAB:           key_code = KEY_TAB;           break;
    case GLFW_KEY_LEFT_SHIFT:    key_code = KEY_LEFT_SHIFT;    break;
    case GLFW_KEY_RIGHT_SHIFT:   key_code = KEY_RIGHT_SHIFT;   break;
    case GLFW_KEY_LEFT_CONTROL:  key_code = KEY_LEFT_CONTROL;  break;
    case GLFW_KEY_RIGHT_CONTROL: key_code = KEY_RIGHT_CONTROL; break;
    case GLFW_KEY_LEFT_ALT:      key_code = KEY_LEFT_ALT;      break;
    case GLFW_KEY_RIGHT_ALT:     key_code = KEY_RIGHT_ALT;     break;
    case GLFW_KEY_LEFT_SUPER:    key_code = KEY_LEFT_SUPER;    break;
    case GLFW_KEY_RIGHT_SUPER:   key_code = KEY_RIGHT_SUPER;   break;
    case GLFW_KEY_MENU:          key_code = KEY_MENU;          break;
    case GLFW_KEY_NUM_LOCK:      key_code = KEY_NUM_LOCK;      break;
    case GLFW_KEY_CAPS_LOCK:     key_code = KEY_CAPS_LOCK;     break;
    case GLFW_KEY_PRINT_SCREEN:  key_code = KEY_PRINT_SCREEN;  break;
    case GLFW_KEY_SCROLL_LOCK:   key_code = KEY_SCROLL_LOCK;   break;
    case GLFW_KEY_PAUSE:         key_code = KEY_PAUSE;         break;
    case GLFW_KEY_DELETE:        key_code = KEY_DELETE;        break;
    case GLFW_KEY_BACKSPACE:     key_code = KEY_BACKSPACE;     break;
    case GLFW_KEY_ENTER:         key_code = KEY_ENTER;         break;
    case GLFW_KEY_HOME:          key_code = KEY_HOME;          break;
    case GLFW_KEY_END:           key_code = KEY_END;           break;
    case GLFW_KEY_PAGE_UP:       key_code = KEY_PAGE_UP;       break;
    case GLFW_KEY_PAGE_DOWN:     key_code = KEY_PAGE_DOWN;     break;
    case GLFW_KEY_INSERT:        key_code = KEY_INSERT;        break;
    case GLFW_KEY_LEFT:          key_code = KEY_LEFT;          break;
    case GLFW_KEY_RIGHT:         key_code = KEY_RIGHT;         break;
    case GLFW_KEY_DOWN:          key_code = KEY_DOWN;          break;
    case GLFW_KEY_UP:            key_code = KEY_UP;            break;
    case GLFW_KEY_F1:            key_code = KEY_F1;            break;
    case GLFW_KEY_F2:            key_code = KEY_F2;            break;
    case GLFW_KEY_F3:            key_code = KEY_F3;            break;
    case GLFW_KEY_F4:            key_code = KEY_F4;            break;
    case GLFW_KEY_F5:            key_code = KEY_F5;            break;
    case GLFW_KEY_F6:            key_code = KEY_F6;            break;
    case GLFW_KEY_F7:            key_code = KEY_F7;            break;
    case GLFW_KEY_F8:            key_code = KEY_F8;            break;
    case GLFW_KEY_F9:            key_code = KEY_F9;            break;
    case GLFW_KEY_F10:           key_code = KEY_F10;           break;
    case GLFW_KEY_F11:           key_code = KEY_F11;           break;
    case GLFW_KEY_F12:           key_code = KEY_F12;           break;
    case GLFW_KEY_F13:           key_code = KEY_F13;           break;
    case GLFW_KEY_F14:           key_code = KEY_F14;           break;
    case GLFW_KEY_F15:           key_code = KEY_F15;           break;
    case GLFW_KEY_F16:           key_code = KEY_F16;           break;
    case GLFW_KEY_F17:           key_code = KEY_F17;           break;
    case GLFW_KEY_F18:           key_code = KEY_F18;           break;
    case GLFW_KEY_F19:           key_code = KEY_F19;           break;
    case GLFW_KEY_F20:           key_code = KEY_F20;           break;
    case GLFW_KEY_F21:           key_code = KEY_F21;           break;
    case GLFW_KEY_F22:           key_code = KEY_F22;           break;
    case GLFW_KEY_F23:           key_code = KEY_F23;           break;
    case GLFW_KEY_F24:           key_code = KEY_F24;           break;
    case GLFW_KEY_F25:           key_code = KEY_F25;           break;

    /* Numeric keypad */
    case GLFW_KEY_KP_DIVIDE:     key_code = KEY_KP_DIVIDE;     break;
    case GLFW_KEY_KP_MULTIPLY:   key_code = KEY_KP_MULTIPLY;   break;
    case GLFW_KEY_KP_SUBTRACT:   key_code = KEY_KP_SUBTRACT;   break;
    case GLFW_KEY_KP_ADD:        key_code = KEY_KP_ADD;        break;

    /* These should have been detected in secondary keysym test above! */
    case GLFW_KEY_KP_0:          key_code = KEY_KP_0;          break;
    case GLFW_KEY_KP_1:          key_code = KEY_KP_1;          break;
    case GLFW_KEY_KP_2:          key_code = KEY_KP_2;          break;
    case GLFW_KEY_KP_3:          key_code = KEY_KP_3;          break;
    case GLFW_KEY_KP_4:          key_code = KEY_KP_4;          break;
    case GLFW_KEY_KP_5:          key_code = KEY_KP_5;          break;
    case GLFW_KEY_KP_6:          key_code = KEY_KP_6;          break;
    case GLFW_KEY_KP_7:          key_code = KEY_KP_7;          break;
    case GLFW_KEY_KP_8:          key_code = KEY_KP_8;          break;
    case GLFW_KEY_KP_9:          key_code = KEY_KP_9;          break;
    case GLFW_KEY_KP_DECIMAL:    key_code = KEY_KP_DECIMAL;    break;
    case GLFW_KEY_KP_EQUAL:      key_code = KEY_KP_EQUAL;      break;
    case GLFW_KEY_KP_ENTER:      key_code = KEY_KP_ENTER;      break;

    /*
     * Last resort: Check for printable keys (should not happen if the XKB
     * extension is available). This will give a layout dependent mapping
     * (which is wrong, and we may miss some keys, especially on non-US
     * keyboards), but it's better than nothing...
     */
    case GLFW_KEY_A:             key_code = KEY_A;             break;
    case GLFW_KEY_B:             key_code = KEY_B;             break;
    case GLFW_KEY_C:             key_code = KEY_C;             break;
    case GLFW_KEY_D:             key_code = KEY_D;             break;
    case GLFW_KEY_E:             key_code = KEY_E;             break;
    case GLFW_KEY_F:             key_code = KEY_F;             break;
    case GLFW_KEY_G:             key_code = KEY_G;             break;
    case GLFW_KEY_H:             key_code = KEY_H;             break;
    case GLFW_KEY_I:             key_code = KEY_I;             break;
    case GLFW_KEY_J:             key_code = KEY_J;             break;
    case GLFW_KEY_K:             key_code = KEY_K;             break;
    case GLFW_KEY_L:             key_code = KEY_L;             break;
    case GLFW_KEY_M:             key_code = KEY_M;             break;
    case GLFW_KEY_N:             key_code = KEY_N;             break;
    case GLFW_KEY_O:             key_code = KEY_O;             break;
    case GLFW_KEY_P:             key_code = KEY_P;             break;
    case GLFW_KEY_Q:             key_code = KEY_Q;             break;
    case GLFW_KEY_R:             key_code = KEY_R;             break;
    case GLFW_KEY_S:             key_code = KEY_S;             break;
    case GLFW_KEY_T:             key_code = KEY_T;             break;
    case GLFW_KEY_U:             key_code = KEY_U;             break;
    case GLFW_KEY_V:             key_code = KEY_V;             break;
    case GLFW_KEY_W:             key_code = KEY_W;             break;
    case GLFW_KEY_X:             key_code = KEY_X;             break;
    case GLFW_KEY_Y:             key_code = KEY_Y;             break;
    case GLFW_KEY_Z:             key_code = KEY_Z;             break;
    case GLFW_KEY_1:             key_code = KEY_1;             break;
    case GLFW_KEY_2:             key_code = KEY_2;             break;
    case GLFW_KEY_3:             key_code = KEY_3;             break;
    case GLFW_KEY_4:             key_code = KEY_4;             break;
    case GLFW_KEY_5:             key_code = KEY_5;             break;
    case GLFW_KEY_6:             key_code = KEY_6;             break;
    case GLFW_KEY_7:             key_code = KEY_7;             break;
    case GLFW_KEY_8:             key_code = KEY_8;             break;
    case GLFW_KEY_9:             key_code = KEY_9;             break;
    case GLFW_KEY_0:             key_code = KEY_0;             break;
    case GLFW_KEY_SPACE:         key_code = KEY_SPACE;         break;
    case GLFW_KEY_MINUS:         key_code = KEY_MINUS;         break;
    case GLFW_KEY_EQUAL:         key_code = KEY_EQUAL;         break;
    case GLFW_KEY_LEFT_BRACKET:  key_code = KEY_LEFT_BRACKET;  break;
    case GLFW_KEY_RIGHT_BRACKET: key_code = KEY_RIGHT_BRACKET; break;
    case GLFW_KEY_BACKSLASH:     key_code = KEY_BACKSLASH;     break;
    case GLFW_KEY_SEMICOLON:     key_code = KEY_SEMICOLON;     break;
    case GLFW_KEY_APOSTROPHE:    key_code = KEY_APOSTROPHE;    break;
    case GLFW_KEY_GRAVE_ACCENT:  key_code = KEY_GRAVE_ACCENT;  break;
    case GLFW_KEY_COMMA:         key_code = KEY_COMMA;         break;
    case GLFW_KEY_PERIOD:        key_code = KEY_PERIOD;        break;
    case GLFW_KEY_SLASH:         key_code = KEY_SLASH;         break;
    case GLFW_KEY_WORLD_1:       key_code = KEY_WORLD_1;       break;
    default:                     key_code = KEY_UNKNOWN;       break;
      // clang-format on
  }

  return key_code;
}

void wgpu_start(const wgpu_desc_t* desc)
{
  assert(desc);
  assert(desc->title);
  assert((desc->width >= 0) && (desc->height >= 0));
  assert(desc->init_cb && desc->frame_cb && desc->shutdown_cb);

  memset(&wgpu_context, 0, sizeof(wgpu_context));

  wgpu_context.desc  = *desc;
  wgpu_context.width = VALUE_OR(wgpu_context.desc.width, DEFAULT_WINDOW_WIDTH);
  wgpu_context.height
    = VALUE_OR(wgpu_context.desc.height, DEFAULT_WINDOW_HEIGHT);
  wgpu_context.desc.sample_count = VALUE_OR(wgpu_context.desc.sample_count, 1);
  wgpu_context.input_event_cb    = desc->input_event_cb;

  wgpu_platform_start(&wgpu_context);
}

static void glfw_key_cb(GLFWwindow* window, int key, int scancode, int action,
                        int mods)
{
  UNUSED_VAR(window);
  UNUSED_VAR(scancode);

  /* Determine event type */
  input_state.event_type = INPUT_EVENT_TYPE_INVALID;
  if (action == GLFW_PRESS) {
    input_state.event_type = INPUT_EVENT_TYPE_KEY_DOWN;
  }
  else if (action == GLFW_RELEASE) {
    input_state.event_type = INPUT_EVENT_TYPE_KEY_UP;
  }
  /* Determine modifier */
  input_state.alt_key   = (mods & GLFW_MOD_ALT);
  input_state.ctrl_key  = (mods & GLFW_MOD_CONTROL);
  input_state.shift_key = (mods & GLFW_MOD_SHIFT);
  /* Remap GLFW keycode to internal code */
  input_state.key_code = remap_glfw_key_code(key);

  /* Maintain persistent key state (not lost when other events overwrite
   * event_type, e.g. continuous MOUSE_MOVE on some platforms). */
  if (input_state.key_code < KEY_NUM) {
    input_state.keys_down[input_state.key_code]
      = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1 : 0;
  }
}

static void glfw_char_cb(GLFWwindow* window, unsigned int chr)
{
  UNUSED_VAR(window);

  input_state.event_type = INPUT_EVENT_TYPE_CHAR;
  input_state.char_code  = chr;
}

static void glfw_mousebutton_cb(GLFWwindow* window, int button, int action,
                                int mods)
{
  /* Determine event type */
  input_state.event_type = INPUT_EVENT_TYPE_INVALID;
  if (action == GLFW_PRESS) {
    input_state.event_type        = INPUT_EVENT_TYPE_MOUSE_DOWN;
    input_state.mouse_btn_pressed = 1;
  }
  else if (action == GLFW_RELEASE) {
    input_state.event_type        = INPUT_EVENT_TYPE_MOUSE_UP;
    input_state.mouse_btn_pressed = 0;
  }
  /* Determine mouse button type */
  input_state.button_type = BUTTON_UNDEFINED;
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    input_state.button_type = BUTTON_LEFT;
  }
  else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    input_state.button_type = BUTTON_MIDDLE;
  }
  else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    input_state.button_type = BUTTON_RIGHT;
  }
  /* Determine modifier */
  input_state.alt_key   = (mods & GLFW_MOD_ALT);
  input_state.ctrl_key  = (mods & GLFW_MOD_CONTROL);
  input_state.shift_key = (mods & GLFW_MOD_SHIFT);
  /* Get cursor position */
  glfwGetCursorPos(window, &input_state.cursor_pos[0],
                   &input_state.cursor_pos[1]);
}

static void glfw_cursorpos_cb(GLFWwindow* window, double xpos, double ypos)
{
  /* Determine event type */
  input_state.event_type = INPUT_EVENT_TYPE_MOUSE_MOVE;
  /* Determine modifier */
  input_state.alt_key = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == 1)
                        || (glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == 1);
  input_state.ctrl_key = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == 1)
                         || (glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == 1);
  input_state.shift_key = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == 1)
                          || (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == 1);
  /* Determine delta */
  input_state.cursor_pos_delta[0] = input_state.cursor_pos[0] - xpos;
  input_state.cursor_pos_delta[1] = input_state.cursor_pos[1] - ypos;
  /* Set new cursor position */
  input_state.cursor_pos[0] = xpos;
  input_state.cursor_pos[1] = ypos;
}

static void glfw_scroll_cb(GLFWwindow* window, double xoffset, double yoffset)
{
  /* Determine event type */
  input_state.event_type = INPUT_EVENT_TYPE_MOUSE_SCROLL;
  /* Determine modifier */
  input_state.alt_key = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == 1)
                        || (glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == 1);
  input_state.ctrl_key = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == 1)
                         || (glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == 1);
  input_state.shift_key = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == 1)
                          || (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == 1);
  /* Set mouse wheel offset */
  input_state.mouse_offset[0] = xoffset;
  input_state.mouse_offset[1] = yoffset;
  /* Get cursor position */
  glfwGetCursorPos(window, &input_state.cursor_pos[0],
                   &input_state.cursor_pos[1]);
}

static void glfw_resize_cb(GLFWwindow* window, int width, int height)
{
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  wgpu_context->width  = width;
  wgpu_context->height = height;
  wgpu_swapchain_resized(wgpu_context);

  /* Determine event type */
  input_state.event_type = INPUT_EVENT_TYPE_RESIZED;
  /* Window width */
  input_state.window_width  = wgpu_context->width;
  input_state.window_height = wgpu_context->height;
}

static void glfw_drop_cb(GLFWwindow* window, int count, const char** paths)
{
  UNUSED_VAR(window);

  input_state.event_type = INPUT_EVENT_TYPE_FILE_DROP;
  input_state.drop_count = (count > MAX_DROP_PATHS) ? MAX_DROP_PATHS : count;
  for (int i = 0; i < input_state.drop_count; i++) {
    strncpy(input_state.drop_paths[i], paths[i], MAX_DROP_PATH_LEN - 1);
    input_state.drop_paths[i][MAX_DROP_PATH_LEN - 1] = '\0';
  }
}

static void uncaptured_error_cb(const WGPUDevice* dev, WGPUErrorType type,
                                WGPUStringView message, void* userdata1,
                                void* userdata2)
{
  UNUSED_VAR(dev);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);
  if (type != WGPUErrorType_NoError) {
    printf("UNCAPTURED ERROR: %s\n", message.data);
  }
}

static void device_lost_cb(const WGPUDevice* dev, WGPUDeviceLostReason reason,
                           WGPUStringView message, void* userdata1,
                           void* userdata2)
{
  UNUSED_VAR(dev);
  UNUSED_VAR(reason);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);
  printf("DEVICE LOST: %s\n", message.data);
}

static void error_scope_cb(WGPUPopErrorScopeStatus status, WGPUErrorType type,
                           WGPUStringView message, void* userdata1,
                           void* userdata2)
{
  UNUSED_VAR(status);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);
  if (type != WGPUErrorType_NoError) {
    printf("ERROR: %s\n", message.data);
  }
}

static void logging_cb(WGPULoggingType type, WGPUStringView message,
                       void* userdata1, void* userdata2)
{
  UNUSED_VAR(type);
  UNUSED_VAR(userdata1);
  UNUSED_VAR(userdata2);
  printf("LOG: %s\n", message.data);
}

static void request_device_cb(WGPURequestDeviceStatus status, WGPUDevice device,
                              WGPUStringView message, void* userdata1,
                              void* userdata2)
{
  UNUSED_VAR(status);
  UNUSED_VAR(message);
  UNUSED_VAR(userdata2);
  wgpu_context_t* wgpu_context   = (wgpu_context_t*)userdata1;
  wgpu_context->device           = device;
  wgpu_context->async_setup_done = 1;
}

static void request_adapter_cb(WGPURequestAdapterStatus status,
                               WGPUAdapter adapter, WGPUStringView message,
                               void* userdata1, void* userdata2)
{
  UNUSED_VAR(message);
  UNUSED_VAR(userdata2);
  wgpu_context_t* wgpu_context = (wgpu_context_t*)userdata1;
  if (status != WGPURequestAdapterStatus_Success) {
    printf("wgpuInstanceRequestAdapter failed!\n");
    exit(10);
  }
  wgpu_context->adapter = adapter;
}

static void request_adapter(wgpu_context_t* wgpu_context)
{
  WGPUFuture future
    = wgpuInstanceRequestAdapter(wgpu_context->instance, 0,
                                 (WGPURequestAdapterCallbackInfo){
                                   .mode      = WGPUCallbackMode_WaitAnyOnly,
                                   .callback  = request_adapter_cb,
                                   .userdata1 = wgpu_context,
                                 });
  WGPUFutureWaitInfo future_info = {.future = future};
  WGPUWaitStatus res
    = wgpuInstanceWaitAny(wgpu_context->instance, 1, &future_info, UINT64_MAX);
  assert(res == WGPUWaitStatus_Success);
}

static void request_device(wgpu_context_t* wgpu_context)
{
  /* Always include Depth32FloatStencil8 feature */
  WGPUFeatureName default_features[] = {WGPUFeatureName_Depth32FloatStencil8};
  uint32_t default_feature_count     = 1;

  /* Combine default features with user-requested features */
  uint32_t total_feature_count
    = default_feature_count + wgpu_context->desc.required_feature_count;
  WGPUFeatureName* all_features
    = (WGPUFeatureName*)malloc(total_feature_count * sizeof(WGPUFeatureName));

  /* Copy default features */
  for (uint32_t i = 0; i < default_feature_count; ++i) {
    all_features[i] = default_features[i];
  }

  /* Copy user-requested features (avoid duplicates) */
  uint32_t actual_count = default_feature_count;
  for (uint32_t i = 0; i < wgpu_context->desc.required_feature_count; ++i) {
    WGPUFeatureName feature = wgpu_context->desc.required_features[i];
    /* Check if feature is not already in the list */
    WGPUBool is_duplicate = 0;
    for (uint32_t j = 0; j < actual_count; ++j) {
      if (all_features[j] == feature) {
        is_duplicate = 1;
        break;
      }
    }
    if (!is_duplicate) {
      all_features[actual_count++] = feature;
    }
  }

  /* Query adapter limits so we can request the maximum buffer sizes */
  WGPULimits adapter_limits = WGPU_LIMITS_INIT;
  WGPUStatus limits_status
    = wgpuAdapterGetLimits(wgpu_context->adapter, &adapter_limits);

  WGPULimits required_limits            = WGPU_LIMITS_INIT;
  WGPULimits const* required_limits_ptr = NULL;
  if (limits_status == WGPUStatus_Success) {
    required_limits.maxBufferSize = adapter_limits.maxBufferSize;
    required_limits.maxStorageBufferBindingSize
      = adapter_limits.maxStorageBufferBindingSize;
    required_limits_ptr = &required_limits;
  }

  WGPUDeviceDescriptor dev_desc = {
    .requiredFeatureCount = actual_count,
    .requiredFeatures     = all_features,
    .requiredLimits       = required_limits_ptr,
    .deviceLostCallbackInfo
    = {
      .mode     = WGPUCallbackMode_AllowProcessEvents,
      .callback = device_lost_cb,
    },
    .uncapturedErrorCallbackInfo
    = {
      .callback = uncaptured_error_cb,
    },
  };
  WGPUFuture future
    = wgpuAdapterRequestDevice(wgpu_context->adapter, &dev_desc,
                               (WGPURequestDeviceCallbackInfo){
                                 .mode      = WGPUCallbackMode_WaitAnyOnly,
                                 .callback  = request_device_cb,
                                 .userdata1 = wgpu_context,
                               });
  WGPUFutureWaitInfo future_info = {.future = future};
  WGPUWaitStatus res
    = wgpuInstanceWaitAny(wgpu_context->instance, 1, &future_info, UINT64_MAX);
  assert(res == WGPUWaitStatus_Success);
  assert(wgpu_context->device);

  /* Free the combined features array */
  free(all_features);
}

static void update_input_event(input_event_t* input_event, uint64_t frame_count)
{
  (*input_event) = (input_event_t){
    .frame_count       = frame_count,
    .type              = input_state.event_type,
    .key_code          = input_state.key_code,
    .char_code         = input_state.char_code,
    .mouse_button      = input_state.button_type,
    .mouse_btn_pressed = input_state.mouse_btn_pressed,
    .mouse_x           = input_state.cursor_pos[0],
    .mouse_y           = input_state.cursor_pos[1],
    .mouse_dx          = input_state.cursor_pos_delta[0],
    .mouse_dy          = input_state.cursor_pos_delta[1],
    .scroll_x          = input_state.mouse_offset[0],
    .scroll_y          = input_state.mouse_offset[1],
    .window_width      = input_state.window_width,
    .window_height     = input_state.window_height,
  };
  memcpy(input_event->keys_down, input_state.keys_down,
         sizeof(input_state.keys_down));
  /* Copy file drop data */
  input_event->drop_count = input_state.drop_count;
  if (input_state.event_type == INPUT_EVENT_TYPE_FILE_DROP) {
    memcpy(input_event->drop_paths, input_state.drop_paths,
           sizeof(input_state.drop_paths));
  }
}

static const char* backend_type_str(WGPUBackendType type)
{
  switch (type) {
    case WGPUBackendType_Null:
      return "Null";
    case WGPUBackendType_WebGPU:
      return "WebGPU";
    case WGPUBackendType_D3D11:
      return "D3D11";
    case WGPUBackendType_D3D12:
      return "D3D12";
    case WGPUBackendType_Metal:
      return "Metal";
    case WGPUBackendType_Vulkan:
      return "Vulkan";
    case WGPUBackendType_OpenGL:
      return "OpenGL";
    case WGPUBackendType_OpenGLES:
      return "OpenGL ES";
    default:
      return "Unknown";
  }
}

static const char* adapter_type_str(WGPUAdapterType type)
{
  switch (type) {
    case WGPUAdapterType_DiscreteGPU:
      return "Discrete GPU";
    case WGPUAdapterType_IntegratedGPU:
      return "Integrated GPU";
    case WGPUAdapterType_CPU:
      return "CPU";
    default:
      return "Unknown";
  }
}

static void copy_string_view(char* dst, size_t dst_size, WGPUStringView sv)
{
  if (sv.data && sv.length > 0) {
    size_t n = sv.length < dst_size - 1 ? sv.length : dst_size - 1;
    memcpy(dst, sv.data, n);
    dst[n] = '\0';
  }
  else {
    dst[0] = '\0';
  }
}

static void query_platform_info(wgpu_context_t* wgpu_context)
{
  wgpu_platform_info_t* info = &wgpu_context->platform_info;
  memset(info, 0, sizeof(*info));

  WGPUAdapterInfo ai = {0};
  if (wgpuAdapterGetInfo(wgpu_context->adapter, &ai) == WGPUStatus_Success) {
    copy_string_view(info->vendor, sizeof(info->vendor), ai.vendor);
    copy_string_view(info->architecture, sizeof(info->architecture),
                     ai.architecture);
    copy_string_view(info->device, sizeof(info->device), ai.device);
    copy_string_view(info->description, sizeof(info->description),
                     ai.description);
    snprintf(info->backend, sizeof(info->backend), "%s",
             backend_type_str(ai.backendType));
    snprintf(info->adapter_type, sizeof(info->adapter_type), "%s",
             adapter_type_str(ai.adapterType));
    info->vendor_id = ai.vendorID;
    info->device_id = ai.deviceID;
    wgpuAdapterInfoFreeMembers(ai);
  }

  printf("WebGPU Platform Info:\n");
  printf("  Backend:      %s\n", info->backend);
  printf("  Adapter:      %s\n", info->adapter_type);
  printf("  Device:       %s\n", info->device);
  printf("  Vendor:       %s (0x%04X)\n", info->vendor, info->vendor_id);
  printf("  Architecture: %s\n", info->architecture);
  printf("  Description:  %s\n", info->description);
}

static void wgpu_platform_start(wgpu_context_t* wgpu_context)
{
#define wgpu_context_struct ((struct wgpu_context_t*)wgpu_context)

  assert(wgpu_context->instance == 0);

  WGPUInstanceFeatureName requiredFeatures[1]
    = {WGPUInstanceFeatureName_TimedWaitAny};
  wgpu_context->instance = wgpuCreateInstance(&(WGPUInstanceDescriptor){
    .requiredFeatureCount = 1,
    .requiredFeatures     = requiredFeatures,
  });
  assert(wgpu_context->instance);
  request_adapter(wgpu_context);
  request_device(wgpu_context);
  query_platform_info(wgpu_context);

  wgpuDeviceSetLoggingCallback(
    wgpu_context->device, (WGPULoggingCallbackInfo){.callback = logging_cb});
  wgpuDevicePushErrorScope(wgpu_context->device, WGPUErrorFilter_Validation);
  wgpu_context->queue = wgpuDeviceGetQueue(wgpu_context->device);

  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(
    wgpu_context->width, wgpu_context->height, wgpu_context->desc.title, 0, 0);
  /* On HiDPI displays the framebuffer (physical pixels) may be larger than the
   * logical window size.  Always use the framebuffer dimensions so that the
   * WebGPU surface, depth textures, and viewports are consistent. */
  glfwGetFramebufferSize(window, &wgpu_context->width, &wgpu_context->height);
  glfwSetWindowUserPointer(window, wgpu_context);
  glfwSetKeyCallback(window, glfw_key_cb);
  glfwSetCharCallback(window, glfw_char_cb);
  glfwSetMouseButtonCallback(window, glfw_mousebutton_cb);
  glfwSetCursorPosCallback(window, glfw_cursorpos_cb);
  glfwSetScrollCallback(window, glfw_scroll_cb);
  /* Use framebuffer-size callback so width/height are always in physical
   * (device) pixels, matching what Dawn uses for the swapchain texture. */
  glfwSetFramebufferSizeCallback(window, glfw_resize_cb);
  glfwSetDropCallback(window, glfw_drop_cb);

  wgpu_context->surface
    = glfw_create_surface_for_window(wgpu_context->instance, window);
  assert(wgpu_context->surface);
  WGPUSurfaceCapabilities surf_caps;
  wgpuSurfaceGetCapabilities(wgpu_context->surface, wgpu_context->adapter,
                             &surf_caps);
  wgpu_context->render_format = surf_caps.formats[0];
  for (uint32_t f = 0; f < surf_caps.formatCount; ++f) {
    if (surf_caps.formats[f] == WGPUTextureFormat_BGRA8Unorm) {
      wgpu_context->render_format = surf_caps.formats[f];
    }
  }

  wgpu_swapchain_init(wgpu_context);
  wgpu_context->desc.init_cb(wgpu_context_struct);
  wgpuDevicePopErrorScope(
    wgpu_context->device,
    (WGPUPopErrorScopeCallbackInfo){.mode = WGPUCallbackMode_AllowProcessEvents,
                                    .callback = error_scope_cb});
  wgpuInstanceProcessEvents(wgpu_context->instance);

  uint64_t frame_count      = 1;
  input_event_t input_event = {0};

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (wgpu_context->input_event_cb
        && input_state.event_type != INPUT_EVENT_TYPE_INVALID) {
      update_input_event(&input_event, frame_count);
      wgpu_context->input_event_cb(wgpu_context, &input_event);
      input_state.event_type = INPUT_EVENT_TYPE_INVALID;
    }
    wgpuDevicePushErrorScope(wgpu_context->device, WGPUErrorFilter_Validation);
    wgpu_context->swapchain_view = wgpu_swapchain_next(wgpu_context);
    if (wgpu_context->swapchain_view) {
      wgpu_context->desc.frame_cb(wgpu_context_struct);
      wgpuTextureViewRelease(wgpu_context->swapchain_view);
      wgpu_context->swapchain_view = 0;
      wgpuSurfacePresent(wgpu_context->surface);
    }
    wgpuDevicePopErrorScope(wgpu_context->device,
                            (WGPUPopErrorScopeCallbackInfo){
                              .mode     = WGPUCallbackMode_AllowProcessEvents,
                              .callback = error_scope_cb});
    wgpuInstanceProcessEvents(wgpu_context->instance);
    frame_count++;
  }
  wgpu_context->desc.shutdown_cb(wgpu_context_struct);
  /* Clean up mipmap generator if it was created */
  if (wgpu_context->mipmap_generator) {
    wgpu_mipmap_generator_destroy(wgpu_context->mipmap_generator);
    wgpu_context->mipmap_generator = NULL;
  }
  wgpu_swapchain_discard(wgpu_context);
  wgpuDeviceRelease(wgpu_context->device);
  wgpuAdapterRelease(wgpu_context->adapter);
}

/* -------------------------------------------------------------------------- *
 * WebGPU SwapChain
 * Ref:
 * https://github.com/floooh/sokol-samples/blob/master/wgpu/wgpu_entry_swapchain.c
 * -------------------------------------------------------------------------- */

static void wgpu_swapchain_init(wgpu_context_t* wgpu_context)
{
  assert(wgpu_context->adapter);
  assert(wgpu_context->device);
  assert(wgpu_context->surface);
  assert(wgpu_context->render_format != WGPUTextureFormat_Undefined);
  assert(0 == wgpu_context->depth_stencil_tex);
  assert(0 == wgpu_context->depth_stencil_view);
  assert(0 == wgpu_context->msaa_tex);
  assert(0 == wgpu_context->msaa_view);

  wgpuSurfaceConfigure(wgpu_context->surface,
                       &(WGPUSurfaceConfiguration){
                         .device      = wgpu_context->device,
                         .format      = wgpu_context->render_format,
                         .usage       = WGPUTextureUsage_RenderAttachment,
                         .alphaMode   = WGPUCompositeAlphaMode_Auto,
                         .width       = (uint32_t)wgpu_context->width,
                         .height      = (uint32_t)wgpu_context->height,
                         .presentMode = WGPUPresentMode_Fifo,
                       });

  if (!wgpu_context->desc.no_depth_buffer) {
    wgpu_context->depth_stencil_format = WGPUTextureFormat_Depth32FloatStencil8;
    wgpu_context->depth_stencil_tex = wgpuDeviceCreateTexture(wgpu_context->device, &(WGPUTextureDescriptor){
                                                                        .usage = WGPUTextureUsage_RenderAttachment,
                                                                        .dimension = WGPUTextureDimension_2D,
                                                                        .size = {
                                                                               .width = (uint32_t) wgpu_context->width,
                                                                               .height = (uint32_t) wgpu_context->height,
                                                                               .depthOrArrayLayers = 1,
                                                                        },
                                                                        .format = wgpu_context->depth_stencil_format,
                                                                        .mipLevelCount = 1,
                                                                        .sampleCount = (uint32_t)wgpu_context->desc.sample_count
                                                                      });
    assert(wgpu_context->depth_stencil_tex);
    wgpu_context->depth_stencil_view
      = wgpuTextureCreateView(wgpu_context->depth_stencil_tex, 0);
    assert(wgpu_context->depth_stencil_view);
  }

  if (wgpu_context->desc.sample_count > 1) {
    wgpu_context->msaa_tex = wgpuDeviceCreateTexture(wgpu_context->device, &(WGPUTextureDescriptor){
                                                               .usage = WGPUTextureUsage_RenderAttachment,
                                                               .dimension = WGPUTextureDimension_2D,
                                                               .size = {
                                                                      .width = (uint32_t) wgpu_context->width,
                                                                      .height = (uint32_t) wgpu_context->height,
                                                                      .depthOrArrayLayers = 1,
                                                               },
                                                               .format = wgpu_context->render_format,
                                                               .mipLevelCount = 1,
                                                               .sampleCount = (uint32_t)wgpu_context->desc.sample_count,
                                                             });
    assert(wgpu_context->msaa_tex);
    wgpu_context->msaa_view = wgpuTextureCreateView(wgpu_context->msaa_tex, 0);
    assert(wgpu_context->msaa_view);
  }
}

static void wgpu_swapchain_discard(wgpu_context_t* wgpu_context)
{
  if (wgpu_context->msaa_view) {
    wgpuTextureViewRelease(wgpu_context->msaa_view);
    wgpu_context->msaa_view = 0;
  }
  if (wgpu_context->msaa_tex) {
    wgpuTextureRelease(wgpu_context->msaa_tex);
    wgpu_context->msaa_tex = 0;
  }
  if (wgpu_context->depth_stencil_view) {
    wgpuTextureViewRelease(wgpu_context->depth_stencil_view);
    wgpu_context->depth_stencil_view = 0;
  }
  if (wgpu_context->depth_stencil_tex) {
    wgpuTextureRelease(wgpu_context->depth_stencil_tex);
    wgpu_context->depth_stencil_tex = 0;
  }
}

static void wgpu_swapchain_resized(wgpu_context_t* wgpu_context)
{
  if (wgpu_context->async_setup_done) {
    wgpu_swapchain_discard(wgpu_context);
    wgpu_swapchain_init(wgpu_context);
  }
}

// may return 0, in that case: skip this frame
static WGPUTextureView wgpu_swapchain_next(wgpu_context_t* wgpu_context)
{
  WGPUSurfaceTexture surface_texture = {0};
  wgpuSurfaceGetCurrentTexture(wgpu_context->surface, &surface_texture);
  switch (surface_texture.status) {
    case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
    case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
      // all ok
      break;
    case WGPUSurfaceGetCurrentTextureStatus_Timeout:
    case WGPUSurfaceGetCurrentTextureStatus_Outdated:
    case WGPUSurfaceGetCurrentTextureStatus_Lost:
      // skip this frame and reconfigure surface
      if (surface_texture.texture) {
        wgpuTextureRelease(surface_texture.texture);
      }
      wgpu_swapchain_discard(wgpu_context);
      wgpu_swapchain_init(wgpu_context);
      return 0;
    case WGPUSurfaceGetCurrentTextureStatus_Error:
    default:
      printf("wgpuSurfaceGetCurrentTexture() failed with: %#.8x\n",
             surface_texture.status);
      abort();
  }
  WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, 0);
  wgpuTextureRelease(surface_texture.texture);
  return view;
}

/* -------------------------------------------------------------------------- *
 * GLFW WebGPU Extension
 * Ref: https://github.com/eliemichel/glfw3webgpu/
 * -------------------------------------------------------------------------- */

#ifdef __EMSCRIPTEN__
#define GLFW_EXPOSE_NATIVE_EMSCRIPTEN
#ifndef GLFW_PLATFORM_EMSCRIPTEN // not defined in older versions of emscripten
#define GLFW_PLATFORM_EMSCRIPTEN 0
#endif
#else // __EMSCRIPTEN__
#ifdef _GLFW_X11
#define GLFW_EXPOSE_NATIVE_X11
#endif
#ifdef _GLFW_WAYLAND
#define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#ifdef _GLFW_COCOA
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#ifdef _GLFW_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#endif // __EMSCRIPTEN__

#ifdef GLFW_EXPOSE_NATIVE_COCOA
#include <Foundation/Foundation.h>
#include <QuartzCore/CAMetalLayer.h>
#endif

#ifndef __EMSCRIPTEN__
#include <GLFW/glfw3native.h>
#endif

WGPUSurface glfw_create_surface_for_window(WGPUInstance instance,
                                           GLFWwindow* window)
{
#ifndef __EMSCRIPTEN__
  switch (glfwGetPlatform()) {
#else
  // glfwGetPlatform is not available in older versions of emscripten
  switch (GLFW_PLATFORM_EMSCRIPTEN) {
#endif

#ifdef GLFW_EXPOSE_NATIVE_X11
    case GLFW_PLATFORM_X11: {
      Display* x11_display = glfwGetX11Display();
      Window x11_window    = glfwGetX11Window(window);

      WGPUSurfaceSourceXlibWindow fromXlibWindow;
      fromXlibWindow.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
      fromXlibWindow.chain.next  = NULL;
      fromXlibWindow.display     = x11_display;
      fromXlibWindow.window      = x11_window;

      WGPUSurfaceDescriptor surfaceDescriptor;
      surfaceDescriptor.nextInChain = &fromXlibWindow.chain;
      surfaceDescriptor.label       = (WGPUStringView){NULL, WGPU_STRLEN};

      return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_X11

#ifdef GLFW_EXPOSE_NATIVE_WAYLAND
    case GLFW_PLATFORM_WAYLAND: {
      struct wl_display* wayland_display = glfwGetWaylandDisplay();
      struct wl_surface* wayland_surface = glfwGetWaylandWindow(window);

      WGPUSurfaceSourceWaylandSurface fromWaylandSurface;
      fromWaylandSurface.chain.sType = WGPUSType_SurfaceSourceWaylandSurface;
      fromWaylandSurface.chain.next  = NULL;
      fromWaylandSurface.display     = wayland_display;
      fromWaylandSurface.surface     = wayland_surface;

      WGPUSurfaceDescriptor surfaceDescriptor;
      surfaceDescriptor.nextInChain = &fromWaylandSurface.chain;
      surfaceDescriptor.label       = (WGPUStringView){NULL, WGPU_STRLEN};

      return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_WAYLAND

#ifdef GLFW_EXPOSE_NATIVE_COCOA
    case GLFW_PLATFORM_COCOA: {
      id metal_layer      = [CAMetalLayer layer];
      NSWindow* ns_window = glfwGetCocoaWindow(window);
      [ns_window.contentView setWantsLayer:YES];
      [ns_window.contentView setLayer:metal_layer];

      WGPUSurfaceSourceMetalLayer fromMetalLayer;
      fromMetalLayer.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
      fromMetalLayer.chain.next  = NULL;
      fromMetalLayer.layer       = metal_layer;

      WGPUSurfaceDescriptor surfaceDescriptor;
      surfaceDescriptor.nextInChain = &fromMetalLayer.chain;
      surfaceDescriptor.label       = (WGPUStringView){NULL, WGPU_STRLEN};

      return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_COCOA

#ifdef GLFW_EXPOSE_NATIVE_WIN32
    case GLFW_PLATFORM_WIN32: {
      HWND hwnd           = glfwGetWin32Window(window);
      HINSTANCE hinstance = GetModuleHandle(NULL);

      WGPUSurfaceSourceWindowsHWND fromWindowsHWND;
      fromWindowsHWND.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
      fromWindowsHWND.chain.next  = NULL;
      fromWindowsHWND.hinstance   = hinstance;
      fromWindowsHWND.hwnd        = hwnd;

      WGPUSurfaceDescriptor surfaceDescriptor;
      surfaceDescriptor.nextInChain = &fromWindowsHWND.chain;
      surfaceDescriptor.label       = (WGPUStringView){NULL, WGPU_STRLEN};

      return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_WIN32

#ifdef GLFW_EXPOSE_NATIVE_EMSCRIPTEN
    case GLFW_PLATFORM_EMSCRIPTEN: {
#ifdef WEBGPU_BACKEND_EMDAWNWEBGPU
      WGPUEmscriptenSurfaceSourceCanvasHTMLSelector fromCanvasHTMLSelector;
      fromCanvasHTMLSelector.chain.sType
        = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
      fromCanvasHTMLSelector.selector = (WGPUStringView){"canvas", WGPU_STRLEN};
#else
      WGPUSurfaceDescriptorFromCanvasHTMLSelector fromCanvasHTMLSelector;
      fromCanvasHTMLSelector.chain.sType
        = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
      fromCanvasHTMLSelector.selector = "canvas";
#endif
      fromCanvasHTMLSelector.chain.next = NULL;

      WGPUSurfaceDescriptor surfaceDescriptor;
      surfaceDescriptor.nextInChain = &fromCanvasHTMLSelector.chain;
#ifdef WEBGPU_BACKEND_EMDAWNWEBGPU
      surfaceDescriptor.label = (WGPUStringView){NULL, WGPU_STRLEN};
#else
      surfaceDescriptor.label = NULL;
#endif
      return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_EMSCRIPTEN

    default:
      // Unsupported platform
      return NULL;
  }
}

/* -------------------------------------------------------------------------- *
 * WebGPU buffer helper functions
 * -------------------------------------------------------------------------- */

WGPUBuffer wgpu_create_buffer_from_data(wgpu_context_t* wgpu_context,
                                        const void* data, size_t size,
                                        WGPUBufferUsage usage)
{
  WGPUBufferDescriptor buffer_desc = {
    .usage = WGPUBufferUsage_CopyDst | usage,
    .size  = size,
  };
  WGPUBuffer buffer
    = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
  wgpuQueueWriteBuffer(wgpu_context->queue, buffer, 0, data, size);
  return buffer;
}

wgpu_buffer_t wgpu_create_buffer(struct wgpu_context_t* wgpu_context,
                                 const wgpu_buffer_desc_t* desc)
{
  /* Ensure that buffer size is a multiple of 4 */
  const uint32_t size = (desc->size + 3) & ~3;

  wgpu_buffer_t wgpu_buffer = {
    .usage = desc->usage,
    .size  = size,
    .count = desc->count,
  };

  WGPUBufferDescriptor buffer_desc = {
    .label            = STRVIEW(VALUE_OR(desc->label, "WebGPU buffer")),
    .usage            = desc->usage,
    .size             = size,
    .mappedAtCreation = desc->mapped_at_creation,
  };

  const uint32_t initial_size
    = (desc->initial.size == 0) ? desc->size : desc->initial.size;

  if (desc->initial.data && initial_size > 0 && initial_size <= desc->size) {
    buffer_desc.mappedAtCreation = 1;
    WGPUBuffer buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(buffer != NULL);
    void* mapping = wgpuBufferGetMappedRange(buffer, 0, size);
    ASSERT(mapping != NULL);
    memcpy(mapping, desc->initial.data, initial_size);
    wgpuBufferUnmap(buffer);
    wgpu_buffer.buffer = buffer;
  }
  else {
    wgpu_buffer.buffer
      = wgpuDeviceCreateBuffer(wgpu_context->device, &buffer_desc);
    ASSERT(wgpu_buffer.buffer != NULL);
  }
  return wgpu_buffer;
}

void wgpu_destroy_buffer(wgpu_buffer_t* buffer)
{
  WGPU_RELEASE_RESOURCE(Buffer, buffer->buffer);
}

/* -------------------------------------------------------------------------- *
 * WebGPU texture helper functions
 * -------------------------------------------------------------------------- */

/* Forward declaration for mipmap_view_to_wgpu (defined in mipmap generator) */
static WGPUTextureViewDimension
mipmap_view_to_wgpu(wgpu_mipmap_view_dimension_t dim);

wgpu_texture_t wgpu_create_texture(struct wgpu_context_t* wgpu_context,
                                   const wgpu_texture_desc_t* desc)
{
  wgpu_texture_t texture = {0};
  if (desc) {
    memcpy(&texture.desc, desc, sizeof(wgpu_texture_desc_t));
  }

  const uint32_t width  = VALUE_OR_DEFAULT(desc, extent.width, 16);
  const uint32_t height = VALUE_OR_DEFAULT(desc, extent.height, 16);
  const uint32_t depth_or_array_layers
    = MAX(1, VALUE_OR_DEFAULT(desc, extent.depthOrArrayLayers, 1));
  const WGPUTextureFormat format
    = VALUE_OR_DEFAULT(desc, format, WGPUTextureFormat_RGBA8Unorm);
  const int8_t gen_mipmaps = desc ? desc->generate_mipmaps : 0;

  /* Calculate mip level count: auto-compute if generate_mipmaps is set and
   * mip_level_count is 0 or 1 */
  uint32_t mip_level_count;
  if (gen_mipmaps && (!desc || desc->mip_level_count <= 1)) {
    mip_level_count = wgpu_texture_mip_level_count(width, height);
  }
  else {
    mip_level_count = MAX(1, VALUE_OR_DEFAULT(desc, mip_level_count, 1));
  }

  /* Store computed mip level count back */
  texture.desc.mip_level_count = mip_level_count;

  /* Texture */
  {
    WGPUTextureUsage usage
      = (desc && desc->usage) ?
          desc->usage :
          WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;

    /* Mipmap generation requires RenderAttachment usage for the destination
     * views and TextureBinding for the source views */
    if (gen_mipmaps && mip_level_count > 1) {
      usage
        |= WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
    }

    /* Determine texture dimension: 0 (default / zero-initialized) = 2D */
    WGPUTextureDimension tex_dim = WGPUTextureDimension_2D;
    if (desc && desc->dimension == WGPUTextureDimension_3D) {
      tex_dim = WGPUTextureDimension_3D;
    }
    else if (desc && desc->dimension == WGPUTextureDimension_1D) {
      tex_dim = WGPUTextureDimension_1D;
    }

    WGPUTextureDescriptor tdesc = {
      .usage         = usage,
      .dimension     = tex_dim,
      .size          = {width, height, depth_or_array_layers},
      .format        = format,
      .mipLevelCount = mip_level_count,
      .sampleCount   = 1,
    };
    texture.handle = wgpuDeviceCreateTexture(wgpu_context->device, &tdesc);
  }

  /* Texture data */
  if (desc && desc->pixels.ptr) {
    /* Calculate bytes per pixel based on format */
    uint32_t bytes_per_pixel;
    switch (format) {
      case WGPUTextureFormat_R8Unorm:
      case WGPUTextureFormat_R8Snorm:
      case WGPUTextureFormat_R8Uint:
      case WGPUTextureFormat_R8Sint:
        bytes_per_pixel = 1;
        break;
      case WGPUTextureFormat_RG8Unorm:
      case WGPUTextureFormat_RG8Snorm:
      case WGPUTextureFormat_RG8Uint:
      case WGPUTextureFormat_RG8Sint:
      case WGPUTextureFormat_R16Float:
      case WGPUTextureFormat_R16Uint:
      case WGPUTextureFormat_R16Sint:
        bytes_per_pixel = 2;
        break;
      case WGPUTextureFormat_RGBA16Float:
      case WGPUTextureFormat_RGBA16Uint:
      case WGPUTextureFormat_RGBA16Sint:
      case WGPUTextureFormat_RG32Float:
      case WGPUTextureFormat_RG32Uint:
      case WGPUTextureFormat_RG32Sint:
        bytes_per_pixel = 8;
        break;
      case WGPUTextureFormat_RGBA32Float:
      case WGPUTextureFormat_RGBA32Uint:
      case WGPUTextureFormat_RGBA32Sint:
        bytes_per_pixel = 16;
        break;
      default:
        bytes_per_pixel = 4; /* RGBA8, BGRA8, RGB10A2, R32Float, etc. */
        break;
    }

    wgpuQueueWriteTexture(
      wgpu_context->queue,
      &(WGPUTexelCopyTextureInfo){
        .texture = texture.handle,
        .aspect  = WGPUTextureAspect_All,
      },
      desc->pixels.ptr, desc->pixels.size,
      &(WGPUTexelCopyBufferLayout){
        .bytesPerRow  = width * bytes_per_pixel,
        .rowsPerImage = height,
      },
      &(WGPUExtent3D){width, height, depth_or_array_layers});
  }

  /* Generate mipmaps if requested */
  if (gen_mipmaps && mip_level_count > 1 && desc && desc->pixels.ptr) {
    wgpu_mipmap_view_dimension_t mip_view
      = desc ? desc->mipmap_view_dimension : WGPU_MIPMAP_VIEW_UNDEFINED;
    wgpu_generate_mipmaps(wgpu_context, texture.handle, mip_view);
  }

  /* Texture view */
  {
    WGPUTextureUsage usage
      = (desc && desc->usage) ?
          desc->usage :
          WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;

    /* Determine the view dimension for the final texture view */
    WGPUTextureViewDimension view_dim = WGPUTextureViewDimension_2D;
    if (desc && desc->dimension == WGPUTextureDimension_3D) {
      view_dim = WGPUTextureViewDimension_3D;
    }
    else if (desc
             && desc->mipmap_view_dimension != WGPU_MIPMAP_VIEW_UNDEFINED) {
      view_dim = mipmap_view_to_wgpu(desc->mipmap_view_dimension);
    }
    else if (depth_or_array_layers == 6) {
      view_dim = WGPUTextureViewDimension_Cube;
    }
    else if (depth_or_array_layers > 1) {
      view_dim = WGPUTextureViewDimension_2DArray;
    }

    /* For 3D textures, arrayLayerCount must be 1 */
    uint32_t array_layer_count
      = (view_dim == WGPUTextureViewDimension_3D) ? 1 : depth_or_array_layers;

    WGPUTextureViewDescriptor view_desc = {
      .format          = format,
      .dimension       = view_dim,
      .baseMipLevel    = 0,
      .mipLevelCount   = mip_level_count,
      .baseArrayLayer  = 0,
      .arrayLayerCount = array_layer_count,
      .aspect          = WGPUTextureAspect_All,
      .usage           = usage,
    };
    texture.view = wgpuTextureCreateView(texture.handle, &view_desc);
  }

  /* Texture sampler */
  {
    /* Use specified address mode, default to Repeat (0 maps to Repeat) */
    WGPUAddressMode addr_mode = (desc && desc->address_mode) ?
                                  desc->address_mode :
                                  WGPUAddressMode_Repeat;

    WGPUSamplerDescriptor sampler_desc = {
      .addressModeU = addr_mode,
      .addressModeV = addr_mode,
      .addressModeW = addr_mode,
      .magFilter    = WGPUFilterMode_Linear,
      .minFilter
      = (mip_level_count > 1) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
      .mipmapFilter  = WGPUMipmapFilterMode_Linear,
      .lodMinClamp   = 0.0f,
      .lodMaxClamp   = (float)mip_level_count,
      .compare       = WGPUCompareFunction_Undefined,
      .maxAnisotropy = 1,
    };
    texture.sampler
      = wgpuDeviceCreateSampler(wgpu_context->device, &sampler_desc);
  }

  texture.initialized   = 1;
  texture.desc.is_dirty = 0;

  return texture;
}

/**
 * @brief Generates a texture containing the 100% EBU Color Bars (named after
 * the standards body, the European Broadcasting Union).
 *
 * The EBU Color Bars consist of 8 vertical bars of equal width. They are
 * defined in the same way for both SD and HD formats. In the RGB color space,
 * they alternate each of the red, green and blue channels at different rates
 * (much like counting in binary) from 0 to 100% intensity.
 *
 * The blue channel alternates every column, the red channel after two columns,
 * and the green after four columns. This arrangement has the useful property
 * that the luminance (Y in YCb'Cr' colour space) results in a downward stepping
 * plot.
 *
 * @ref https://en.wikipedia.org/wiki/Color_bars
 */
wgpu_texture_t
wgpu_create_color_bars_texture(struct wgpu_context_t* wgpu_context,
                               const wgpu_texture_desc_t* desc)
{
  typedef struct rgba_t {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
  } rgba_t;

  static const rgba_t BAR_COLOUR[8] = {
    // clang-format off
    {255, 255, 255, 255}, /* 100% White */
    {255, 255,   0, 255}, /* Yellow     */
    {  0, 255, 255, 255}, /* Cyan       */
    {  0, 255,   0, 255}, /* Green      */
    {255,   0, 255, 255}, /* Magenta    */
    {255,   0,   0, 255}, /* Red        */
    {  0,   0, 255, 255}, /* Blue       */
    {  0,   0,   0, 255}, /* Black      */
    // clang-format on
  };

  /* Check with and height parameters */
  uint32_t width  = MAX(ARRAY_SIZE(BAR_COLOUR), desc ? desc->extent.width : 16);
  uint32_t height = MAX(1, desc ? desc->extent.height : 16);

  /* Allocate frame buffer */
  size_t frame_bytes    = width * height * sizeof(rgba_t);
  rgba_t* frame         = malloc(frame_bytes);
  uint32_t column_width = width / ARRAY_SIZE(BAR_COLOUR);

  /* Generate complete frame */
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      uint32_t col_idx     = x / column_width;
      frame[y * width + x] = BAR_COLOUR[col_idx];
    }
  }

  /* Usage */
  WGPUTextureUsage usage
    = (desc && desc->usage) ?
        desc->usage :
        WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;

  /* Texture */
  wgpu_texture_t texture = wgpu_create_texture(wgpu_context, &(wgpu_texture_desc_t){
                                                 .extent = (WGPUExtent3D) {
                                                   .width              = width,
                                                   .height             = height,
                                                   .depthOrArrayLayers = 1,
                                                 },
                                                 .format = WGPUTextureFormat_RGBA8Unorm,
                                                 .mip_level_count = 1,
                                                 .usage = usage,
                                                 .pixels = {
                                                   .ptr  = frame,
                                                   .size = frame_bytes,
                                                 },
                                               });
  if (texture.desc.pixels.ptr) {
    free((void*)texture.desc.pixels.ptr);
    texture.desc.pixels.size = 0;
  }

  return texture;
}

void wgpu_recreate_texture(struct wgpu_context_t* wgpu_context,
                           wgpu_texture_t* texture)
{
  wgpu_destroy_texture(texture);
  *texture = wgpu_create_texture(wgpu_context, &texture->desc);
}

void wgpu_image_to_texure(wgpu_context_t* wgpu_context, WGPUTexture texture,
                          void* pixels, WGPUExtent3D size, uint32_t channels)
{
  const uint64_t data_size = size.width * size.height * size.depthOrArrayLayers
                             * channels * sizeof(uint8_t);
  wgpuQueueWriteTexture(wgpu_context->queue,
                        &(WGPUTexelCopyTextureInfo) {
                          .texture = texture,
                          .mipLevel = 0,
                          .origin = (WGPUOrigin3D) {
                              .x = 0,
                              .y = 0,
                              .z = 0,
                          },
                          .aspect = WGPUTextureAspect_All,
                        },
                        pixels, data_size,
                        &(WGPUTexelCopyBufferLayout){
                          .offset       = 0,
                          .bytesPerRow  = size.width * channels * sizeof(uint8_t),
                          .rowsPerImage = size.height,
                        },
                        &(WGPUExtent3D){
                          .width              = size.width,
                          .height             = size.height,
                          .depthOrArrayLayers = size.depthOrArrayLayers,
                        });
}

void wgpu_destroy_texture(wgpu_texture_t* texture)
{
  WGPU_RELEASE_RESOURCE(Sampler, texture->sampler);
  WGPU_RELEASE_RESOURCE(TextureView, texture->view);
  WGPU_RELEASE_RESOURCE(Texture, texture->handle);
}

/* -------------------------------------------------------------------------- *
 * WebGPU mipmap generator
 *
 * Generates mipmaps using render-based downsampling with a fullscreen triangle
 * and hardware bilinear filtering. Supports 2D, 2D-array, cube, and cube-array
 * textures. Pipelines are cached per format+view dimension combination.
 *
 * Based on:
 * - webgpu-samples generateMipmap (render approach with textureSample)
 * - webgpu-gltf-viewer mipmap_generator (compute approach, used for reference)
 * -------------------------------------------------------------------------- */

/* -- Mipmap generation WGSL shader code ----------------------------------- */

// clang-format off

/**
 * Fullscreen triangle vertex shader + fragment shaders for 2D, 2D-array,
 * cube, and cube-array textures.
 *
 * The vertex shader generates a fullscreen triangle using vertex_index.
 * The fragment shaders sample from the previous mip level using bilinear
 * filtering. For cube textures, UV coordinates are converted to 3D cube
 * directions using face matrices.
 */
static const char* mipmap_generator_shader_wgsl_part1 = CODE(
  const faceMat = array(
    mat3x3f( 0,  0, -2,  0, -2,  0,  1,  1,  1),
    mat3x3f( 0,  0,  2,  0, -2,  0, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0,  2, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0, -2, -1, -1,  1),
    mat3x3f( 2,  0,  0,  0, -2,  0, -1,  1,  1),
    mat3x3f(-2,  0,  0,  0, -2,  0,  1,  1, -1)
  );

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat, either) baseArrayLayer: u32,
  };

  @vertex fn vs(
    @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) baseArrayLayer: u32,
  ) -> VSOutput {
    var pos = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    var vsOutput: VSOutput;
    let xy = pos[vertexIndex];
    vsOutput.position = vec4f(xy, 0.0, 1.0);
    vsOutput.texcoord = xy * vec2f(0.5, -0.5) + vec2f(0.5);
    vsOutput.baseArrayLayer = baseArrayLayer;
    return vsOutput;
  }

  @group(0) @binding(0) var ourSampler: sampler;

  @group(0) @binding(1) var ourTexture2d: texture_2d<f32>;
  @fragment fn fs2d(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(ourTexture2d, ourSampler, fsInput.texcoord);
  }
);

static const char* mipmap_generator_shader_wgsl_part2 = CODE(
  const faceMat = array(
    mat3x3f( 0,  0, -2,  0, -2,  0,  1,  1,  1),
    mat3x3f( 0,  0,  2,  0, -2,  0, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0,  2, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0, -2, -1, -1,  1),
    mat3x3f( 2,  0,  0,  0, -2,  0, -1,  1,  1),
    mat3x3f(-2,  0,  0,  0, -2,  0,  1,  1, -1)
  );

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat, either) baseArrayLayer: u32,
  };

  @vertex fn vs(
    @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) baseArrayLayer: u32,
  ) -> VSOutput {
    var pos = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    var vsOutput: VSOutput;
    let xy = pos[vertexIndex];
    vsOutput.position = vec4f(xy, 0.0, 1.0);
    vsOutput.texcoord = xy * vec2f(0.5, -0.5) + vec2f(0.5);
    vsOutput.baseArrayLayer = baseArrayLayer;
    return vsOutput;
  }

  @group(0) @binding(0) var ourSampler: sampler;

  @group(0) @binding(1) var ourTexture2dArray: texture_2d_array<f32>;
  @fragment fn fs2darray(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(
      ourTexture2dArray,
      ourSampler,
      fsInput.texcoord,
      fsInput.baseArrayLayer);
  }
);

static const char* mipmap_generator_shader_wgsl_part3 = CODE(
  const faceMat = array(
    mat3x3f( 0,  0, -2,  0, -2,  0,  1,  1,  1),
    mat3x3f( 0,  0,  2,  0, -2,  0, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0,  2, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0, -2, -1, -1,  1),
    mat3x3f( 2,  0,  0,  0, -2,  0, -1,  1,  1),
    mat3x3f(-2,  0,  0,  0, -2,  0,  1,  1, -1)
  );

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat, either) baseArrayLayer: u32,
  };

  @vertex fn vs(
    @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) baseArrayLayer: u32,
  ) -> VSOutput {
    var pos = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    var vsOutput: VSOutput;
    let xy = pos[vertexIndex];
    vsOutput.position = vec4f(xy, 0.0, 1.0);
    vsOutput.texcoord = xy * vec2f(0.5, -0.5) + vec2f(0.5);
    vsOutput.baseArrayLayer = baseArrayLayer;
    return vsOutput;
  }

  @group(0) @binding(0) var ourSampler: sampler;

  @group(0) @binding(1) var ourTextureCube: texture_cube<f32>;
  @fragment fn fscube(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(
      ourTextureCube,
      ourSampler,
      faceMat[fsInput.baseArrayLayer] * vec3f(fract(fsInput.texcoord), 1));
  }
);

static const char* mipmap_generator_shader_wgsl_part4 = CODE(
  const faceMat = array(
    mat3x3f( 0,  0, -2,  0, -2,  0,  1,  1,  1),
    mat3x3f( 0,  0,  2,  0, -2,  0, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0,  2, -1,  1, -1),
    mat3x3f( 2,  0,  0,  0,  0, -2, -1, -1,  1),
    mat3x3f( 2,  0,  0,  0, -2,  0, -1,  1,  1),
    mat3x3f(-2,  0,  0,  0, -2,  0,  1,  1, -1)
  );

  struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat, either) baseArrayLayer: u32,
  };

  @vertex fn vs(
    @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) baseArrayLayer: u32,
  ) -> VSOutput {
    var pos = array<vec2f, 3>(
      vec2f(-1.0, -1.0),
      vec2f(-1.0,  3.0),
      vec2f( 3.0, -1.0),
    );
    var vsOutput: VSOutput;
    let xy = pos[vertexIndex];
    vsOutput.position = vec4f(xy, 0.0, 1.0);
    vsOutput.texcoord = xy * vec2f(0.5, -0.5) + vec2f(0.5);
    vsOutput.baseArrayLayer = baseArrayLayer;
    return vsOutput;
  }

  @group(0) @binding(0) var ourSampler: sampler;

  @group(0) @binding(1) var ourTextureCubeArray: texture_cube_array<f32>;
  @fragment fn fscubearray(fsInput: VSOutput) -> @location(0) vec4f {
    return textureSample(
      ourTextureCubeArray,
      ourSampler,
      faceMat[fsInput.baseArrayLayer] * vec3f(fract(fsInput.texcoord), 1),
      fsInput.baseArrayLayer);
  }
);

// clang-format on

/* -- Mipmap generator pipeline cache -------------------------------------- */

/**
 * Maximum number of cached pipelines per generator.
 * Each unique (format, view_dimension) pair requires a separate pipeline.
 */
#define MIPMAP_MAX_CACHED_PIPELINES 16

typedef struct mipmap_pipeline_cache_entry_t {
  WGPUTextureFormat format;
  WGPUTextureViewDimension view_dimension;
  WGPURenderPipeline pipeline;
} mipmap_pipeline_cache_entry_t;

struct wgpu_mipmap_generator_t {
  WGPUDevice device;
  WGPUSampler sampler;
  WGPUShaderModule module_2d;
  WGPUShaderModule module_2d_array;
  WGPUShaderModule module_cube;
  WGPUShaderModule module_cube_array;
  mipmap_pipeline_cache_entry_t cache[MIPMAP_MAX_CACHED_PIPELINES];
  uint32_t cache_count;
};

/* -- Internal helpers ----------------------------------------------------- */

uint32_t wgpu_texture_mip_level_count(uint32_t width, uint32_t height)
{
  const uint32_t max_dim = (width > height) ? width : height;
  if (max_dim == 0) {
    return 1;
  }
  uint32_t levels = 1;
  uint32_t dim    = max_dim;
  while (dim > 1) {
    dim >>= 1;
    levels++;
  }
  return levels;
}

/**
 * @brief Map mipmap_view_dimension enum to WGPUTextureViewDimension.
 */
static WGPUTextureViewDimension
mipmap_view_to_wgpu(wgpu_mipmap_view_dimension_t dim)
{
  switch (dim) {
    case WGPU_MIPMAP_VIEW_2D:
      return WGPUTextureViewDimension_2D;
    case WGPU_MIPMAP_VIEW_2D_ARRAY:
      return WGPUTextureViewDimension_2DArray;
    case WGPU_MIPMAP_VIEW_3D:
      return WGPUTextureViewDimension_3D;
    case WGPU_MIPMAP_VIEW_CUBE:
      return WGPUTextureViewDimension_Cube;
    case WGPU_MIPMAP_VIEW_CUBE_ARRAY:
      return WGPUTextureViewDimension_CubeArray;
    default:
      return WGPUTextureViewDimension_2D;
  }
}

/**
 * @brief Auto-detect the default view dimension from texture properties.
 */
static WGPUTextureViewDimension
mipmap_detect_view_dimension(uint32_t depth_or_array_layers)
{
  if (depth_or_array_layers == 6) {
    return WGPUTextureViewDimension_Cube;
  }
  else if (depth_or_array_layers > 1) {
    return WGPUTextureViewDimension_2DArray;
  }
  return WGPUTextureViewDimension_2D;
}

/**
 * @brief Get the shader module for the given view dimension.
 */
static WGPUShaderModule
mipmap_get_shader_module(wgpu_mipmap_generator_t* gen,
                         WGPUTextureViewDimension view_dim)
{
  switch (view_dim) {
    case WGPUTextureViewDimension_2D:
      return gen->module_2d;
    case WGPUTextureViewDimension_2DArray:
      return gen->module_2d_array;
    case WGPUTextureViewDimension_Cube:
      return gen->module_cube;
    case WGPUTextureViewDimension_CubeArray:
      return gen->module_cube_array;
    default:
      return gen->module_2d;
  }
}

/**
 * @brief Get the fragment shader entry point for the given view dimension.
 */
static const char* mipmap_get_fragment_entry(WGPUTextureViewDimension view_dim)
{
  switch (view_dim) {
    case WGPUTextureViewDimension_2D:
      return "fs2d";
    case WGPUTextureViewDimension_2DArray:
      return "fs2darray";
    case WGPUTextureViewDimension_Cube:
      return "fscube";
    case WGPUTextureViewDimension_CubeArray:
      return "fscubearray";
    default:
      return "fs2d";
  }
}

/**
 * @brief Create the mipmap generator, allocating shader modules and sampler.
 */
static wgpu_mipmap_generator_t* mipmap_generator_create(WGPUDevice device)
{
  wgpu_mipmap_generator_t* gen = calloc(1, sizeof(wgpu_mipmap_generator_t));
  if (!gen) {
    return NULL;
  }

  gen->device = device;

  /* Create shader modules for each view dimension */
  gen->module_2d
    = wgpu_create_shader_module(device, mipmap_generator_shader_wgsl_part1);
  gen->module_2d_array
    = wgpu_create_shader_module(device, mipmap_generator_shader_wgsl_part2);
  gen->module_cube
    = wgpu_create_shader_module(device, mipmap_generator_shader_wgsl_part3);
  gen->module_cube_array
    = wgpu_create_shader_module(device, mipmap_generator_shader_wgsl_part4);

  /* Create linear sampler for bilinear downsampling */
  WGPUSamplerDescriptor sampler_desc = {
    .label         = STRVIEW("Mipmap generator - Sampler"),
    .minFilter     = WGPUFilterMode_Linear,
    .magFilter     = WGPUFilterMode_Linear,
    .maxAnisotropy = 1,
  };
  gen->sampler = wgpuDeviceCreateSampler(device, &sampler_desc);

  gen->cache_count = 0;

  return gen;
}

/**
 * @brief Find or create a render pipeline for the given format + view dim.
 */
static WGPURenderPipeline mipmap_get_pipeline(wgpu_mipmap_generator_t* gen,
                                              WGPUTextureFormat format,
                                              WGPUTextureViewDimension view_dim)
{
  /* Search cache */
  for (uint32_t i = 0; i < gen->cache_count; i++) {
    if (gen->cache[i].format == format
        && gen->cache[i].view_dimension == view_dim) {
      return gen->cache[i].pipeline;
    }
  }

  /* Cache miss - create new pipeline */
  ASSERT(gen->cache_count < MIPMAP_MAX_CACHED_PIPELINES);
  if (gen->cache_count >= MIPMAP_MAX_CACHED_PIPELINES) {
    fprintf(stderr, "wgpu_mipmap: pipeline cache full (max %d)\n",
            MIPMAP_MAX_CACHED_PIPELINES);
    return NULL;
  }

  WGPUShaderModule module    = mipmap_get_shader_module(gen, view_dim);
  const char* fragment_entry = mipmap_get_fragment_entry(view_dim);

  WGPUColorTargetState color_target = {
    .format    = format,
    .writeMask = WGPUColorWriteMask_All,
  };

  WGPUFragmentState fragment_state = {
    .module      = module,
    .entryPoint  = {.data = fragment_entry, .length = WGPU_STRLEN},
    .targetCount = 1,
    .targets     = &color_target,
  };

  WGPURenderPipelineDescriptor pipeline_desc = {
    .label = STRVIEW("mipmap generator pipeline"),
    .layout = NULL, /* auto layout */
    .vertex = {
      .module     = module,
      .entryPoint = STRVIEW("vs"),
    },
    .fragment   = &fragment_state,
    .primitive  = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  WGPURenderPipeline pipeline
    = wgpuDeviceCreateRenderPipeline(gen->device, &pipeline_desc);
  ASSERT(pipeline != NULL);

  /* Store in cache */
  gen->cache[gen->cache_count] = (mipmap_pipeline_cache_entry_t){
    .format         = format,
    .view_dimension = view_dim,
    .pipeline       = pipeline,
  };
  gen->cache_count++;

  return pipeline;
}

/* -- Public API ----------------------------------------------------------- */

void wgpu_generate_mipmaps(wgpu_context_t* wgpu_context, WGPUTexture texture,
                           wgpu_mipmap_view_dimension_t view_dim)
{
  if (!texture) {
    return;
  }

  const uint32_t mip_count = wgpuTextureGetMipLevelCount(texture);
  if (mip_count <= 1) {
    return; /* Nothing to generate */
  }

  const uint32_t depth_or_array_layers
    = wgpuTextureGetDepthOrArrayLayers(texture);
  const WGPUTextureFormat format = wgpuTextureGetFormat(texture);

  /* Determine view dimension */
  WGPUTextureViewDimension wgpu_view_dim;
  if (view_dim != WGPU_MIPMAP_VIEW_UNDEFINED) {
    wgpu_view_dim = mipmap_view_to_wgpu(view_dim);
  }
  else {
    wgpu_view_dim = mipmap_detect_view_dimension(depth_or_array_layers);
  }

  /* Lazily create mipmap generator */
  if (!wgpu_context->mipmap_generator) {
    wgpu_context->mipmap_generator
      = mipmap_generator_create(wgpu_context->device);
    if (!wgpu_context->mipmap_generator) {
      fprintf(stderr, "wgpu_mipmap: failed to create generator\n");
      return;
    }
  }

  wgpu_mipmap_generator_t* gen = wgpu_context->mipmap_generator;

  /* Get or create cached pipeline */
  WGPURenderPipeline pipeline = mipmap_get_pipeline(gen, format, wgpu_view_dim);
  if (!pipeline) {
    return;
  }

  /* Create command encoder */
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    wgpu_context->device, &(WGPUCommandEncoderDescriptor){
                            .label = STRVIEW("mipmap generation encoder"),
                          });

  /* For each mip level > 0, sample from previous level */
  for (uint32_t mip = 1; mip < mip_count; ++mip) {
    for (uint32_t layer = 0; layer < depth_or_array_layers; ++layer) {
      /* Source: previous mip level, all layers visible via view_dim */
      WGPUTextureView src_view = wgpuTextureCreateView(
        texture, &(WGPUTextureViewDescriptor){
                   .label           = STRVIEW("mipmap src view"),
                   .format          = format,
                   .dimension       = wgpu_view_dim,
                   .baseMipLevel    = mip - 1,
                   .mipLevelCount   = 1,
                   .baseArrayLayer  = 0,
                   .arrayLayerCount = depth_or_array_layers,
                   .aspect          = WGPUTextureAspect_All,
                 });

      /* Create bind group for this mip level */
      WGPUBindGroupLayout bgl
        = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);

      WGPUBindGroupEntry entries[2] = {
        {
          .binding = 0,
          .sampler = gen->sampler,
        },
        {
          .binding     = 1,
          .textureView = src_view,
        },
      };

      WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
        wgpu_context->device, &(WGPUBindGroupDescriptor){
                                .label      = STRVIEW("mipmap bind group"),
                                .layout     = bgl,
                                .entryCount = 2,
                                .entries    = entries,
                              });

      /* Destination: this mip level, single layer */
      WGPUTextureView dst_view = wgpuTextureCreateView(
        texture, &(WGPUTextureViewDescriptor){
                   .label           = STRVIEW("mipmap dst view"),
                   .format          = format,
                   .dimension       = WGPUTextureViewDimension_2D,
                   .baseMipLevel    = mip,
                   .mipLevelCount   = 1,
                   .baseArrayLayer  = layer,
                   .arrayLayerCount = 1,
                   .aspect          = WGPUTextureAspect_All,
                 });

      WGPURenderPassColorAttachment color_attachment = {
        .view       = dst_view,
        .loadOp     = WGPULoadOp_Clear,
        .storeOp    = WGPUStoreOp_Store,
        .clearValue = {0},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      };

      WGPURenderPassDescriptor rp_desc = {
        .label                = STRVIEW("mipmap render pass"),
        .colorAttachmentCount = 1,
        .colorAttachments     = &color_attachment,
      };

      WGPURenderPassEncoder pass
        = wgpuCommandEncoderBeginRenderPass(encoder, &rp_desc);
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
      /* Draw 3 vertices, 1 instance; first instance = layer index */
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, layer);
      wgpuRenderPassEncoderEnd(pass);

      /* Release per-iteration resources */
      WGPU_RELEASE_RESOURCE(RenderPassEncoder, pass);
      WGPU_RELEASE_RESOURCE(BindGroup, bind_group);
      WGPU_RELEASE_RESOURCE(BindGroupLayout, bgl);
      WGPU_RELEASE_RESOURCE(TextureView, src_view);
      WGPU_RELEASE_RESOURCE(TextureView, dst_view);
    }
  }

  /* Submit command buffer */
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(
    encoder, &(WGPUCommandBufferDescriptor){
               .label = STRVIEW("mipmap generation commands"),
             });
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd);

  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder);
}

void wgpu_mipmap_generator_destroy(wgpu_mipmap_generator_t* generator)
{
  if (!generator) {
    return;
  }

  /* Release cached pipelines */
  for (uint32_t i = 0; i < generator->cache_count; i++) {
    WGPU_RELEASE_RESOURCE(RenderPipeline, generator->cache[i].pipeline);
  }
  generator->cache_count = 0;

  /* Release shader modules */
  WGPU_RELEASE_RESOURCE(ShaderModule, generator->module_2d);
  WGPU_RELEASE_RESOURCE(ShaderModule, generator->module_2d_array);
  WGPU_RELEASE_RESOURCE(ShaderModule, generator->module_cube);
  WGPU_RELEASE_RESOURCE(ShaderModule, generator->module_cube_array);

  /* Release sampler */
  WGPU_RELEASE_RESOURCE(Sampler, generator->sampler);

  free(generator);
}

/* -------------------------------------------------------------------------- *
 * WebGPU environment map (IBL) helper functions
 *
 * Complete Image-Based Lighting pipeline:
 *   HDR panorama (.hdr) → equirectangular cubemap → mipmapped cubemap
 *   → irradiance cubemap + prefiltered specular cubemap + BRDF LUT
 * -------------------------------------------------------------------------- */

/* -- Panorama to cubemap WGSL shader -------------------------------------- */

// clang-format off

static const char* panorama_to_cubemap_shader_wgsl = CODE(
  const PI: f32 = 3.14159265359;

  @group(0) @binding(0) var inputSampler: sampler;
  @group(0) @binding(1) var inputTexture: texture_2d<f32>;
  @group(0) @binding(2) var outputTexture: texture_storage_2d_array<rgba16float, write>;

  @group(1) @binding(0) var<uniform> faceIndex: u32;

  fn dirToUV(dir: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(
      0.5 + 0.5 * atan2(dir.z, dir.x) / PI,
      acos(clamp(dir.y, -1.0, 1.0)) / PI
    );
  }

  fn uvToDirection(uv: vec2<f32>, face: u32) -> vec3<f32> {
    const faceDirs = array<vec3<f32>, 6>(
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>(-1.0,  0.0,  0.0),
      vec3<f32>( 0.0,  1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 0.0,  0.0, -1.0)
    );
    const upVectors = array<vec3<f32>, 6>(
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 0.0,  0.0, -1.0),
      vec3<f32>( 0.0, -1.0,  0.0),
      vec3<f32>( 0.0, -1.0,  0.0)
    );
    const rightVectors = array<vec3<f32>, 6>(
      vec3<f32>( 0.0,  0.0, -1.0),
      vec3<f32>( 0.0,  0.0,  1.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>( 1.0,  0.0,  0.0),
      vec3<f32>(-1.0,  0.0,  0.0)
    );
    let u = (uv.x * 2.0) - 1.0;
    let v = (uv.y * 2.0) - 1.0;
    return normalize(faceDirs[face] + (u * rightVectors[face]) + (v * upVectors[face]));
  }

  @compute @workgroup_size(8, 8)
  fn panoramaToCubemap(@builtin(global_invocation_id) id: vec3<u32>) {
    let outputSize = textureDimensions(outputTexture).xy;
    if (id.x >= outputSize.x || id.y >= outputSize.y) { return; }

    let uvDst = vec2<f32>(f32(id.x) / f32(outputSize.x), f32(id.y) / f32(outputSize.y));
    let dir = uvToDirection(uvDst, faceIndex);
    let uvSrc = clamp(dirToUV(dir), vec2<f32>(0.0), vec2<f32>(1.0));

    let dims = vec2<i32>(textureDimensions(inputTexture));
    let width = f32(dims.x);
    let height = f32(dims.y);

    let srcXF = uvSrc.x * (width - 1.0);
    let srcYF = uvSrc.y * (height - 1.0);
    let x0 = i32(floor(srcXF));
    let y0 = i32(floor(srcYF));
    let x1 = (x0 + 1) % dims.x;
    let y1 = min(y0 + 1, dims.y - 1);
    let fx = srcXF - floor(srcXF);
    let fy = srcYF - floor(srcYF);

    let c00 = textureLoad(inputTexture, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(inputTexture, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(inputTexture, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(inputTexture, vec2<i32>(x1, y1), 0);

    let top = mix(c00, c10, fx);
    let bottom = mix(c01, c11, fx);
    let color = mix(top, bottom, fy);

    textureStore(outputTexture, id.xy, faceIndex, color);
  }
);

// clang-format on

/* -------------------------------------------------------------------------- *
 * WebGPU panorama-to-cubemap converter
 * -------------------------------------------------------------------------- */

struct wgpu_panorama_to_cubemap_converter_t {
  WGPUDevice device;
  WGPUSampler sampler;
  WGPUBindGroupLayout bind_group_layouts[2]; /* 0: common, 1: per-face */
  WGPUComputePipeline pipeline;
  WGPUBuffer per_face_uniform_buffers[6];
  WGPUBindGroup per_face_bind_groups[6];
};

wgpu_panorama_to_cubemap_converter_t*
wgpu_panorama_to_cubemap_converter_create(WGPUDevice device)
{
  if (!device) {
    return NULL;
  }

  wgpu_panorama_to_cubemap_converter_t* c
    = (wgpu_panorama_to_cubemap_converter_t*)calloc(
      1, sizeof(wgpu_panorama_to_cubemap_converter_t));
  if (!c) {
    return NULL;
  }
  c->device = device;

  /* Sampler (nearest, repeat-U, clamp-V — matches C++ original) */
  c->sampler = wgpuDeviceCreateSampler(
    device, &(WGPUSamplerDescriptor){
              .addressModeU  = WGPUAddressMode_Repeat,
              .addressModeV  = WGPUAddressMode_ClampToEdge,
              .addressModeW  = WGPUAddressMode_Repeat,
              .minFilter     = WGPUFilterMode_Nearest,
              .magFilter     = WGPUFilterMode_Nearest,
              .mipmapFilter  = WGPUMipmapFilterMode_Nearest,
              .maxAnisotropy = 1,
            });

  /* Bind group layout 0: sampler + input 2D + output 2D-array */
  WGPUBindGroupLayoutEntry bg0_entries[3] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .sampler    = {.type = WGPUSamplerBindingType_NonFiltering},
    },
    {
      .binding    = 1,
      .visibility = WGPUShaderStage_Compute,
      .texture    = {
        .sampleType    = WGPUTextureSampleType_UnfilterableFloat,
        .viewDimension = WGPUTextureViewDimension_2D,
      },
    },
    {
      .binding        = 2,
      .visibility     = WGPUShaderStage_Compute,
      .storageTexture = {
        .access        = WGPUStorageTextureAccess_WriteOnly,
        .format        = WGPUTextureFormat_RGBA16Float,
        .viewDimension = WGPUTextureViewDimension_2DArray,
      },
    },
  };
  c->bind_group_layouts[0]
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 3,
                                                .entries    = bg0_entries,
                                              });

  /* Bind group layout 1: face index uniform */
  WGPUBindGroupLayoutEntry bg1_entries[1] = {
    {
      .binding    = 0,
      .visibility = WGPUShaderStage_Compute,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uint32_t),
      },
    },
  };
  c->bind_group_layouts[1]
    = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor){
                                                .entryCount = 1,
                                                .entries    = bg1_entries,
                                              });

  /* Per-face uniform buffers + bind groups */
  WGPUQueue queue = wgpuDeviceGetQueue(device);
  for (uint32_t face = 0; face < 6; ++face) {
    c->per_face_uniform_buffers[face] = wgpuDeviceCreateBuffer(
      device, &(WGPUBufferDescriptor){
                .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                .size  = sizeof(uint32_t),
              });
    uint32_t face_val = face;
    wgpuQueueWriteBuffer(queue, c->per_face_uniform_buffers[face], 0, &face_val,
                         sizeof(uint32_t));

    WGPUBindGroupEntry bg1_e[1] = {
      {.binding = 0,
       .buffer  = c->per_face_uniform_buffers[face],
       .size    = sizeof(uint32_t)},
    };
    c->per_face_bind_groups[face]
      = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor){
                                            .layout = c->bind_group_layouts[1],
                                            .entryCount = 1,
                                            .entries    = bg1_e,
                                          });
  }
  WGPU_RELEASE_RESOURCE(Queue, queue);

  /* Compute pipeline */
  WGPUShaderModule shader
    = wgpu_create_shader_module(device, panorama_to_cubemap_shader_wgsl);

  WGPUBindGroupLayout bgls[]
    = {c->bind_group_layouts[0], c->bind_group_layouts[1]};
  WGPUPipelineLayout pipe_layout
    = wgpuDeviceCreatePipelineLayout(device, &(WGPUPipelineLayoutDescriptor){
                                               .bindGroupLayoutCount = 2,
                                               .bindGroupLayouts     = bgls,
                                             });

  c->pipeline = wgpuDeviceCreateComputePipeline(
    device, &(WGPUComputePipelineDescriptor){
              .layout  = pipe_layout,
              .compute = {
                .module     = shader,
                .entryPoint = STRVIEW("panoramaToCubemap"),
              },
            });

  WGPU_RELEASE_RESOURCE(PipelineLayout, pipe_layout);
  WGPU_RELEASE_RESOURCE(ShaderModule, shader);

  return c;
}

bool wgpu_panorama_to_cubemap_converter_convert(
  wgpu_panorama_to_cubemap_converter_t* converter, const float* panorama_data,
  uint32_t panorama_width, uint32_t panorama_height,
  WGPUTexture environment_cubemap)
{
  if (!converter || !panorama_data || !environment_cubemap) {
    return false;
  }
  if (panorama_width == 0 || panorama_height == 0) {
    return false;
  }

  WGPUDevice device = converter->device;
  WGPUQueue queue   = wgpuDeviceGetQueue(device);

  /* Upload panorama as RGBA32Float 2D texture */
  WGPUTexture panorama_tex = wgpuDeviceCreateTexture(
    device,
    &(WGPUTextureDescriptor){
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {panorama_width, panorama_height, 1},
      .format    = WGPUTextureFormat_RGBA32Float,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    });
  if (!panorama_tex) {
    fprintf(stderr, "panorama_to_cubemap: failed to create panorama texture\n");
    WGPU_RELEASE_RESOURCE(Queue, queue);
    return false;
  }

  const size_t data_size
    = (size_t)4 * panorama_width * panorama_height * sizeof(float);
  wgpuQueueWriteTexture(
    queue,
    &(WGPUTexelCopyTextureInfo){
      .texture = panorama_tex,
      .aspect  = WGPUTextureAspect_All,
    },
    panorama_data, data_size,
    &(WGPUTexelCopyBufferLayout){
      .bytesPerRow  = 4 * panorama_width * (uint32_t)sizeof(float),
      .rowsPerImage = panorama_height,
    },
    &(WGPUExtent3D){panorama_width, panorama_height, 1});

  /* Create views for input panorama and output cubemap */
  WGPUTextureView input_view = wgpuTextureCreateView(
    panorama_tex, &(WGPUTextureViewDescriptor){
                    .format          = WGPUTextureFormat_RGBA32Float,
                    .dimension       = WGPUTextureViewDimension_2D,
                    .mipLevelCount   = 1,
                    .arrayLayerCount = 1,
                  });

  WGPUTextureView output_view = wgpuTextureCreateView(
    environment_cubemap, &(WGPUTextureViewDescriptor){
                           .format          = WGPUTextureFormat_RGBA16Float,
                           .dimension       = WGPUTextureViewDimension_2DArray,
                           .baseMipLevel    = 0,
                           .mipLevelCount   = 1,
                           .baseArrayLayer  = 0,
                           .arrayLayerCount = 6,
                           .aspect          = WGPUTextureAspect_All,
                         });

  /* Bind group 0: common for all faces */
  WGPUBindGroupEntry bg0_e[3] = {
    {.binding = 0, .sampler = converter->sampler},
    {.binding = 1, .textureView = input_view},
    {.binding = 2, .textureView = output_view},
  };
  WGPUBindGroup bg0 = wgpuDeviceCreateBindGroup(
    device, &(WGPUBindGroupDescriptor){
              .layout     = converter->bind_group_layouts[0],
              .entryCount = 3,
              .entries    = bg0_e,
            });

  /* Encode compute pass: dispatch per face */
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
    device, &(WGPUCommandEncoderDescriptor){
              .label = STRVIEW("panorama to cubemap encoder"),
            });
  WGPUComputePassEncoder pass
    = wgpuCommandEncoderBeginComputePass(encoder, NULL);
  wgpuComputePassEncoderSetPipeline(pass, converter->pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg0, 0, NULL);

  const uint32_t cubemap_size = wgpuTextureGetWidth(environment_cubemap);
  const uint32_t wg           = 8;
  const uint32_t wg_x         = (cubemap_size + wg - 1) / wg;
  const uint32_t wg_y         = (cubemap_size + wg - 1) / wg;
  for (uint32_t face = 0; face < 6; ++face) {
    wgpuComputePassEncoderSetBindGroup(
      pass, 1, converter->per_face_bind_groups[face], 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, 1);
  }

  wgpuComputePassEncoderEnd(pass);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
  wgpuQueueSubmit(queue, 1, &cmd);

  /* Cleanup per-conversion resources */
  WGPU_RELEASE_RESOURCE(CommandBuffer, cmd);
  WGPU_RELEASE_RESOURCE(ComputePassEncoder, pass);
  WGPU_RELEASE_RESOURCE(CommandEncoder, encoder);
  WGPU_RELEASE_RESOURCE(BindGroup, bg0);
  WGPU_RELEASE_RESOURCE(TextureView, input_view);
  WGPU_RELEASE_RESOURCE(TextureView, output_view);
  WGPU_RELEASE_RESOURCE(Texture, panorama_tex);
  WGPU_RELEASE_RESOURCE(Queue, queue);

  return true;
}

void wgpu_panorama_to_cubemap_converter_destroy(
  wgpu_panorama_to_cubemap_converter_t* converter)
{
  if (!converter) {
    return;
  }
  for (uint32_t i = 0; i < 6; ++i) {
    WGPU_RELEASE_RESOURCE(BindGroup, converter->per_face_bind_groups[i]);
    WGPU_RELEASE_RESOURCE(Buffer, converter->per_face_uniform_buffers[i]);
  }
  WGPU_RELEASE_RESOURCE(ComputePipeline, converter->pipeline);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, converter->bind_group_layouts[0]);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, converter->bind_group_layouts[1]);
  WGPU_RELEASE_RESOURCE(Sampler, converter->sampler);
  free(converter);
}

/* -------------------------------------------------------------------------- *
 * WebGPU shader helper functions
 * -------------------------------------------------------------------------- */

WGPUShaderModule wgpu_create_shader_module(WGPUDevice device,
                                           const char* wgsl_source_code)
{
  WGPUShaderSourceWGSL shader_code_desc
    = {.chain = {.sType = WGPUSType_ShaderSourceWGSL},
       .code  = {
          .data   = wgsl_source_code,
          .length = WGPU_STRLEN,
       }};
  WGPUShaderModuleDescriptor shader_desc
    = {.nextInChain = &shader_code_desc.chain};
  return wgpuDeviceCreateShaderModule(device, &shader_desc);
}

/* -------------------------------------------------------------------------- *
 * WebGPU pipeline helper functions
 * -------------------------------------------------------------------------- */

WGPUBlendState wgpu_create_blend_state(WGPUBool enable_blend)
{
  WGPUBlendComponent blend_component_descriptor = {
    .operation = WGPUBlendOperation_Add,
  };

  if (enable_blend) {
    blend_component_descriptor.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend_component_descriptor.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
  }
  else {
    blend_component_descriptor.srcFactor = WGPUBlendFactor_One;
    blend_component_descriptor.dstFactor = WGPUBlendFactor_Zero;
  }

  return (WGPUBlendState){
    .color = blend_component_descriptor,
    .alpha = blend_component_descriptor,
  };
}

WGPUDepthStencilState
wgpu_create_depth_stencil_state(create_depth_stencil_state_desc_t* desc)
{
  WGPUStencilFaceState stencil_state_face_descriptor = {
    .compare     = WGPUCompareFunction_Always,
    .failOp      = WGPUStencilOperation_Keep,
    .depthFailOp = WGPUStencilOperation_Keep,
    .passOp      = WGPUStencilOperation_Keep,
  };

  return (WGPUDepthStencilState){
    .depthWriteEnabled   = desc->depth_write_enabled,
    .format              = desc->format,
    .depthCompare        = WGPUCompareFunction_LessEqual,
    .stencilFront        = stencil_state_face_descriptor,
    .stencilBack         = stencil_state_face_descriptor,
    .stencilReadMask     = 0xFFFFFFFF,
    .stencilWriteMask    = 0xFFFFFFFF,
    .depthBias           = 0,
    .depthBiasSlopeScale = 0.0f,
    .depthBiasClamp      = 0.0f,
  };
}

/* -------------------------------------------------------------------------- *
 * Math
 * -------------------------------------------------------------------------- */

float random_float_min_max(float min, float max)
{
  /* [min, max] */
  return ((max - min) * ((float)rand() / (float)RAND_MAX)) + min;
}

float random_float(void)
{
  return random_float_min_max(0.0f, 1.0f); /* [0, 1.0] */
}
