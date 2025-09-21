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
  WGPUFeatureName required_features[1] = {WGPUFeatureName_Depth32FloatStencil8};
  WGPUDeviceDescriptor dev_desc = {
    .requiredFeatureCount = 1,
    .requiredFeatures = required_features,
    .deviceLostCallbackInfo = {
      .mode = WGPUCallbackMode_AllowProcessEvents,
      .callback = device_lost_cb,
    },
    .uncapturedErrorCallbackInfo = {
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
  };
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

  wgpuDeviceSetLoggingCallback(
    wgpu_context->device, (WGPULoggingCallbackInfo){.callback = logging_cb});
  wgpuDevicePushErrorScope(wgpu_context->device, WGPUErrorFilter_Validation);
  wgpu_context->queue = wgpuDeviceGetQueue(wgpu_context->device);

  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(
    wgpu_context->width, wgpu_context->height, wgpu_context->desc.title, 0, 0);
  glfwSetWindowUserPointer(window, wgpu_context);
  glfwSetKeyCallback(window, glfw_key_cb);
  glfwSetCharCallback(window, glfw_char_cb);
  glfwSetMouseButtonCallback(window, glfw_mousebutton_cb);
  glfwSetCursorPosCallback(window, glfw_cursorpos_cb);
  glfwSetScrollCallback(window, glfw_scroll_cb);
  glfwSetWindowSizeCallback(window, glfw_resize_cb);

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

wgpu_texture_t wgpu_create_texture(struct wgpu_context_t* wgpu_context,
                                   const wgpu_texture_desc_t* desc)
{
  wgpu_texture_t texture = {0};
  memcpy(&texture.desc, desc, sizeof(wgpu_texture_desc_t));

  /* Texture */
  {
    WGPUTextureDescriptor tdesc = {
      .usage     = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .dimension = WGPUTextureDimension_2D,
      .size      = {desc->extent.width, desc->extent.height, 1},
      .format    = desc->format,
      .mipLevelCount = 1,
      .sampleCount   = 1,
    };
    texture.handle = wgpuDeviceCreateTexture(wgpu_context->device, &tdesc);
  }

  /* Texture data */
  {
    wgpuQueueWriteTexture(
      wgpu_context->queue,
      &(WGPUTexelCopyTextureInfo){
        .texture = texture.handle,
        .aspect  = WGPUTextureAspect_All,
      },
      desc->pixels.ptr,
      desc->extent.width * desc->extent.height
        * desc->extent.depthOrArrayLayers,
      &(WGPUTexelCopyBufferLayout){
        .bytesPerRow  = desc->extent.width * desc->extent.depthOrArrayLayers,
        .rowsPerImage = desc->extent.height,
      },
      &(WGPUExtent3D){desc->extent.width, desc->extent.height, 1});
  }

  /* Texture view */
  {
    WGPUTextureViewDescriptor view_desc = {
      .format          = desc->format,
      .dimension       = WGPUTextureViewDimension_2D,
      .baseMipLevel    = 0,
      .mipLevelCount   = 1,
      .baseArrayLayer  = 0,
      .arrayLayerCount = 1,
      .aspect          = WGPUTextureAspect_All,
      .usage           = WGPUTextureUsage_TextureBinding,
    };
    texture.view = wgpuTextureCreateView(texture.handle, &view_desc);
  }

  /* Texture sampler */
  {
    WGPUSamplerDescriptor sampler_desc = {
      .addressModeU  = WGPUAddressMode_Repeat,
      .addressModeV  = WGPUAddressMode_Repeat,
      .addressModeW  = WGPUAddressMode_Repeat,
      .magFilter     = WGPUFilterMode_Linear,
      .minFilter     = WGPUFilterMode_Nearest,
      .mipmapFilter  = WGPUMipmapFilterMode_Linear,
      .lodMinClamp   = 0,
      .lodMaxClamp   = 1,
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
                               uint32_t width, uint32_t height)
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
  width  = MAX(ARRAY_SIZE(BAR_COLOUR), width);
  height = MAX(1, height);

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

  wgpu_texture_t texture = wgpu_create_texture(wgpu_context, &(wgpu_texture_desc_t){
                                                 .extent = (WGPUExtent3D) {
                                                   .width              = width,
                                                   .height             = height,
                                                   .depthOrArrayLayers = 4,
                                                 },
                                                 .format = WGPUTextureFormat_RGBA8Unorm,
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
