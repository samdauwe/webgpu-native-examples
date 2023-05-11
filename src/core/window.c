#include "window.h"

#if defined(__linux__)
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#if defined(WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include "GLFW/glfw3native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "macro.h"

#include "../../lib/wgpu_native/wgpu_native.h"

struct window {
  GLFWwindow* handle;
  struct {
    WGPUSurface handle;
    uint32_t width, height;
    float dpscale;
  } surface;
  callbacks_t callbacks;
  int intialized;
  /* common data */
  float mouse_scroll_scale_factor;
  void* userdata;
};

/* Function prototypes */
static void surface_update_framebuffer_size(window_t* window);
static void glfw_window_error_callback(int error, const char* description);
static void glfw_window_key_callback(GLFWwindow* src_window, int key,
                                     int scancode, int action, int mods);
static void glfw_window_cursor_position_callback(GLFWwindow* src_window,
                                                 double xpos, double ypos);
static void glfw_window_mouse_button_callback(GLFWwindow* src_window,
                                              int button, int action, int mods);
static void glfw_window_scroll_callback(GLFWwindow* src_window, double xoffset,
                                        double yoffset);
static void glfw_window_size_callback(GLFWwindow* src_window, int width,
                                      int height);

window_t* window_create(window_config_t* config)
{
  if (!config) {
    return NULL;
  }

  window_t* window = (window_t*)malloc(sizeof(window_t));
  memset(window, 0, sizeof(window_t));
  window->mouse_scroll_scale_factor = 1.0f;

  /* Initialize error handling */
  glfwSetErrorCallback(glfw_window_error_callback);

  /* Initialize the library */
  if (!glfwInit()) {
    /* Handle initialization failure */
    fprintf(stderr, "Failed to initialize GLFW\n");
    fflush(stderr);
    return window;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, config->resizable ? GLFW_TRUE : GLFW_FALSE);

  /* Create GLFW window */
  window->handle = glfwCreateWindow(config->width, config->height,
                                    config->title, NULL, NULL);

  /* Confirm that GLFW window was created successfully */
  if (!window->handle) {
    glfwTerminate();
    fprintf(stderr, "Failed to create window\n");
    fflush(stderr);
    return window;
  }

  surface_update_framebuffer_size(window);

  /* Set user pointer to window class */
  glfwSetWindowUserPointer(window->handle, (void*)window);

  /* -- Setup callbacks -- */
  /* clang-format off */
  /* Key input events */
  glfwSetKeyCallback(window->handle, glfw_window_key_callback);
  /* Cursor position */
  glfwSetCursorPosCallback(window->handle, glfw_window_cursor_position_callback);
  /* Mouse button input */
  glfwSetMouseButtonCallback(window->handle, glfw_window_mouse_button_callback);
  /* Scroll input */
  glfwSetScrollCallback(window->handle, glfw_window_scroll_callback);
  /* Window resize events */
  glfwSetWindowSizeCallback(window->handle, glfw_window_size_callback);
  /* clang-format on */

  /* Change the state of the window to intialized */
  window->intialized = 1;

  return window;
}

void window_destroy(window_t* window)
{
  /* Cleanup window(s) */
  if (window) {
    if (window->handle) {
      glfwDestroyWindow(window->handle);
      window->handle = NULL;

      /* Terminate GLFW */
      glfwTerminate();
    }

    /* Free allocated memory */
    free(window);
    window = NULL;
  }
}

int window_should_close(window_t* window)
{
  return glfwWindowShouldClose(window->handle);
}

void window_set_title(window_t* window, const char* title)
{
  glfwSetWindowTitle(window->handle, title);
}

void window_set_userdata(window_t* window, void* userdata)
{
  window->userdata = userdata;
}

void* window_get_userdata(window_t* window)
{
  return window->userdata;
}

void* window_get_surface(window_t* window)
{
#if defined(WIN32)
  void* display         = NULL;
  uint32_t windowHandle = glfwGetWin32Window(window->handle);
#elif defined(__linux__) /* X11 */
  void* display         = glfwGetX11Display();
  uint32_t windowHandle = glfwGetX11Window(window->handle);
#endif
  window->surface.handle = wgpu_create_surface(display, &windowHandle);

  return window->surface.handle;
}

void window_get_size(window_t* window, uint32_t* width, uint32_t* height)
{
  *width  = window->surface.width;
  *height = window->surface.height;
}

void window_get_aspect_ratio(window_t* window, float* aspect_ratio)
{
  *aspect_ratio = (float)window->surface.width / (float)window->surface.height;
}

/* Input related functions */

void input_poll_events(void)
{
  glfwPollEvents();
}

void input_query_cursor(window_t* window, float* xpos, float* ypos)
{
  double cursor_xpos, cursor_ypos;
  glfwGetCursorPos(window->handle, &cursor_xpos, &cursor_ypos);

  *xpos = (float)cursor_xpos;
  *ypos = (float)cursor_ypos;
}

void input_set_callbacks(window_t* window, callbacks_t callbacks)
{
  window->callbacks = callbacks;
}

/* -------------------------------------------------------------------------- *
 * GLFW events handlers
 * -------------------------------------------------------------------------- */

static keycode_t remap_glfw_key_code(int key);

static void glfw_window_error_callback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error occured, Error id: %i, Description: %s\n", error,
          description);
}

static void glfw_window_key_callback(GLFWwindow* src_window, int key,
                                     int scancode, int action, int mods)
{
  UNUSED_VAR(scancode);

  window_t* window = (window_t*)glfwGetWindowUserPointer(src_window);
  if (window && window->handle) {
    GLFWwindow* glfw_window = window->handle;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(glfw_window, GLFW_TRUE);
      return;
    }

    if (window->callbacks.key_callback) {
      /* Determine modifier */
      const int ctrl_key = (mods & GLFW_MOD_CONTROL);
      const int alt_key  = (mods & GLFW_MOD_ALT);
      /* Remap GLFW keycode to internal code */
      keycode_t key_code = remap_glfw_key_code(key);
      /* Determine button action */
      button_action_t button_action = BUTTON_ACTION_UNDEFINED;
      if (action == GLFW_PRESS) {
        button_action = BUTTON_ACTION_PRESS;
      }
      else if (action == GLFW_RELEASE) {
        button_action = BUTTON_ACTION_RELEASE;
      }
      /* Raise event */
      window->callbacks.key_callback(window, ctrl_key, alt_key, key_code,
                                     button_action);
    }
  }
}

static void glfw_window_cursor_position_callback(GLFWwindow* src_window,
                                                 double xpos, double ypos)
{
  window_t* window = (window_t*)glfwGetWindowUserPointer(src_window);
  if (window && window->handle && window->callbacks.cursor_position_callback) {
    GLFWwindow* glfw_window = window->handle;
    /* Determine modifier */
    int ctrl_key = (glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) == 1)
                   || (glfwGetKey(glfw_window, GLFW_KEY_RIGHT_CONTROL) == 1);
    int shift_key = (glfwGetKey(glfw_window, GLFW_KEY_LEFT_SHIFT) == 1)
                    || (glfwGetKey(glfw_window, GLFW_KEY_RIGHT_SHIFT) == 1);
    /* Raise event */
    window->callbacks.cursor_position_callback(window, ctrl_key, shift_key,
                                               (float)xpos, (float)ypos);
  }
}

static void glfw_window_mouse_button_callback(GLFWwindow* src_window,
                                              int button, int action, int mods)
{
  window_t* window = (window_t*)glfwGetWindowUserPointer(src_window);
  if (window && window->handle && window->callbacks.mouse_button_callback) {
    GLFWwindow* glfw_window = window->handle;
    /* Determine mouse button type */
    button_t button_type = BUTTON_UNDEFINED;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      button_type = BUTTON_LEFT;
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
      button_type = BUTTON_MIDDLE;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
      button_type = BUTTON_RIGHT;
    }
    /* Get cursor position */
    double xpos, ypos;
    glfwGetCursorPos(glfw_window, &xpos, &ypos);
    /* Determine modifier */
    int ctrl_key  = (mods & GLFW_MOD_CONTROL);
    int shift_key = (mods & GLFW_MOD_SHIFT);
    /* Determine button action */
    button_action_t button_action = BUTTON_ACTION_UNDEFINED;
    if (action == GLFW_PRESS) {
      button_action = BUTTON_ACTION_PRESS;
    }
    else if (action == GLFW_RELEASE) {
      button_action = BUTTON_ACTION_RELEASE;
    }
    /* Raise event */
    window->callbacks.mouse_button_callback(window, ctrl_key, shift_key,
                                            (float)xpos, (float)ypos,
                                            button_type, button_action);
  }
}

static float rescale_mouse_scroll(double offset, float scale_factor)
{
  return (float)offset * scale_factor;
}

static void glfw_window_scroll_callback(GLFWwindow* src_window, double xoffset,
                                        double yoffset)
{
  UNUSED_VAR(xoffset);

  window_t* window = (window_t*)glfwGetWindowUserPointer(src_window);
  if (window && window->handle && window->callbacks.scroll_callback) {
    GLFWwindow* glfw_window = window->handle;
    /* Get cursor position */
    double xpos, ypos;
    glfwGetCursorPos(glfw_window, &xpos, &ypos);
    /* Determine modifier */
    int ctrl_key = (glfwGetKey(glfw_window, GLFW_KEY_LEFT_CONTROL) == 1)
                   || (glfwGetKey(glfw_window, GLFW_KEY_RIGHT_CONTROL) == 1);
    int shift_key = (glfwGetKey(glfw_window, GLFW_KEY_LEFT_SHIFT) == 1)
                    || (glfwGetKey(glfw_window, GLFW_KEY_RIGHT_SHIFT) == 1);
    /* Raise event */
    window->callbacks.scroll_callback(
      window, ctrl_key, shift_key, (float)xpos, (float)ypos,
      rescale_mouse_scroll(yoffset, window->mouse_scroll_scale_factor));
  }
}

static void surface_update_framebuffer_size(window_t* window)
{
  if (window) {
    float yscale = 1.0;
    glfwGetFramebufferSize(window->handle, (int*)&(window->surface.width),
                           (int*)&window->surface.height);
    glfwGetWindowContentScale(window->handle, &window->surface.dpscale,
                              &yscale);
  }
}

static void glfw_window_size_callback(GLFWwindow* src_window, int width,
                                      int height)
{
  UNUSED_VAR(width);
  UNUSED_VAR(height);

  surface_update_framebuffer_size(
    (window_t*)glfwGetWindowUserPointer(src_window));
}

static keycode_t remap_glfw_key_code(int key)
{
  keycode_t key_code = KEY_UNKNOWN;
  switch (key) {
    case GLFW_KEY_ESCAPE:
      key_code = KEY_ESCAPE;
      break;
    case GLFW_KEY_TAB:
      key_code = KEY_TAB;
      break;
    case GLFW_KEY_LEFT_SHIFT:
      key_code = KEY_LEFT_SHIFT;
      break;
    case GLFW_KEY_RIGHT_SHIFT:
      key_code = KEY_RIGHT_SHIFT;
      break;
    case GLFW_KEY_LEFT_CONTROL:
      key_code = KEY_LEFT_CONTROL;
      break;
    case GLFW_KEY_RIGHT_CONTROL:
      key_code = KEY_RIGHT_CONTROL;
      break;
    case GLFW_KEY_LEFT_ALT:
      key_code = KEY_LEFT_ALT;
      break;
    case GLFW_KEY_RIGHT_ALT:
      key_code = KEY_RIGHT_ALT;
      break;
    case GLFW_KEY_LEFT_SUPER:
      key_code = KEY_LEFT_SUPER;
      break;
    case GLFW_KEY_RIGHT_SUPER:
      key_code = KEY_RIGHT_SUPER;
      break;
    case GLFW_KEY_MENU:
      key_code = KEY_MENU;
      break;
    case GLFW_KEY_NUM_LOCK:
      key_code = KEY_NUM_LOCK;
      break;
    case GLFW_KEY_CAPS_LOCK:
      key_code = KEY_CAPS_LOCK;
      break;
    case GLFW_KEY_PRINT_SCREEN:
      key_code = KEY_PRINT_SCREEN;
      break;
    case GLFW_KEY_SCROLL_LOCK:
      key_code = KEY_SCROLL_LOCK;
      break;
    case GLFW_KEY_PAUSE:
      key_code = KEY_PAUSE;
      break;
    case GLFW_KEY_DELETE:
      key_code = KEY_DELETE;
      break;
    case GLFW_KEY_BACKSPACE:
      key_code = KEY_BACKSPACE;
      break;
    case GLFW_KEY_ENTER:
      key_code = KEY_ENTER;
      break;
    case GLFW_KEY_HOME:
      key_code = KEY_HOME;
      break;
    case GLFW_KEY_END:
      key_code = KEY_END;
      break;
    case GLFW_KEY_PAGE_UP:
      key_code = KEY_PAGE_UP;
      break;
    case GLFW_KEY_PAGE_DOWN:
      key_code = KEY_PAGE_DOWN;
      break;
    case GLFW_KEY_INSERT:
      key_code = KEY_INSERT;
      break;
    case GLFW_KEY_LEFT:
      key_code = KEY_LEFT;
      break;
    case GLFW_KEY_RIGHT:
      key_code = KEY_RIGHT;
      break;
    case GLFW_KEY_DOWN:
      key_code = KEY_DOWN;
      break;
    case GLFW_KEY_UP:
      key_code = KEY_UP;
      break;
    case GLFW_KEY_F1:
      key_code = KEY_F1;
      break;
    case GLFW_KEY_F2:
      key_code = KEY_F2;
      break;
    case GLFW_KEY_F3:
      key_code = KEY_F3;
      break;
    case GLFW_KEY_F4:
      key_code = KEY_F4;
      break;
    case GLFW_KEY_F5:
      key_code = KEY_F5;
      break;
    case GLFW_KEY_F6:
      key_code = KEY_F6;
      break;
    case GLFW_KEY_F7:
      key_code = KEY_F7;
      break;
    case GLFW_KEY_F8:
      key_code = KEY_F8;
      break;
    case GLFW_KEY_F9:
      key_code = KEY_F9;
      break;
    case GLFW_KEY_F10:
      key_code = KEY_F10;
      break;
    case GLFW_KEY_F11:
      key_code = KEY_F11;
      break;
    case GLFW_KEY_F12:
      key_code = KEY_F12;
      break;
    case GLFW_KEY_F13:
      key_code = KEY_F13;
      break;
    case GLFW_KEY_F14:
      key_code = KEY_F14;
      break;
    case GLFW_KEY_F15:
      key_code = KEY_F15;
      break;
    case GLFW_KEY_F16:
      key_code = KEY_F16;
      break;
    case GLFW_KEY_F17:
      key_code = KEY_F17;
      break;
    case GLFW_KEY_F18:
      key_code = KEY_F18;
      break;
    case GLFW_KEY_F19:
      key_code = KEY_F19;
      break;
    case GLFW_KEY_F20:
      key_code = KEY_F20;
      break;
    case GLFW_KEY_F21:
      key_code = KEY_F21;
      break;
    case GLFW_KEY_F22:
      key_code = KEY_F22;
      break;
    case GLFW_KEY_F23:
      key_code = KEY_F23;
      break;
    case GLFW_KEY_F24:
      key_code = KEY_F24;
      break;
    case GLFW_KEY_F25:
      key_code = KEY_F25;
      break;

    /* Numeric keypad */
    case GLFW_KEY_KP_DIVIDE:
      key_code = KEY_KP_DIVIDE;
      break;
    case GLFW_KEY_KP_MULTIPLY:
      key_code = KEY_KP_MULTIPLY;
      break;
    case GLFW_KEY_KP_SUBTRACT:
      key_code = KEY_KP_SUBTRACT;
      break;
    case GLFW_KEY_KP_ADD:
      key_code = KEY_KP_ADD;
      break;

    /* These should have been detected in secondary keysym test above! */
    case GLFW_KEY_KP_0:
      key_code = KEY_KP_0;
      break;
    case GLFW_KEY_KP_1:
      key_code = KEY_KP_1;
      break;
    case GLFW_KEY_KP_2:
      key_code = KEY_KP_2;
      break;
    case GLFW_KEY_KP_3:
      key_code = KEY_KP_3;
      break;
    case GLFW_KEY_KP_4:
      key_code = KEY_KP_4;
      break;
    case GLFW_KEY_KP_6:
      key_code = KEY_KP_6;
      break;
    case GLFW_KEY_KP_7:
      key_code = KEY_KP_7;
      break;
    case GLFW_KEY_KP_8:
      key_code = KEY_KP_8;
      break;
    case GLFW_KEY_KP_9:
      key_code = KEY_KP_9;
      break;
    case GLFW_KEY_KP_DECIMAL:
      key_code = KEY_KP_DECIMAL;
      break;
    case GLFW_KEY_KP_EQUAL:
      key_code = KEY_KP_EQUAL;
      break;
    case GLFW_KEY_KP_ENTER:
      key_code = KEY_KP_ENTER;
      break;

    /*
     * Last resort: Check for printable keys (should not happen if the XKB
     * extension is available). This will give a layout dependent mapping
     * (which is wrong, and we may miss some keys, especially on non-US
     * keyboards), but it's better than nothing...
     */
    case GLFW_KEY_A:
      key_code = KEY_A;
      break;
    case GLFW_KEY_B:
      key_code = KEY_B;
      break;
    case GLFW_KEY_C:
      key_code = KEY_C;
      break;
    case GLFW_KEY_D:
      key_code = KEY_D;
      break;
    case GLFW_KEY_E:
      key_code = KEY_E;
      break;
    case GLFW_KEY_F:
      key_code = KEY_F;
      break;
    case GLFW_KEY_G:
      key_code = KEY_G;
      break;
    case GLFW_KEY_H:
      key_code = KEY_H;
      break;
    case GLFW_KEY_I:
      key_code = KEY_I;
      break;
    case GLFW_KEY_J:
      key_code = KEY_J;
      break;
    case GLFW_KEY_K:
      key_code = KEY_K;
      break;
    case GLFW_KEY_L:
      key_code = KEY_L;
      break;
    case GLFW_KEY_M:
      key_code = KEY_M;
      break;
    case GLFW_KEY_N:
      key_code = KEY_N;
      break;
    case GLFW_KEY_O:
      key_code = KEY_O;
      break;
    case GLFW_KEY_P:
      key_code = KEY_P;
      break;
    case GLFW_KEY_Q:
      key_code = KEY_Q;
      break;
    case GLFW_KEY_R:
      key_code = KEY_R;
      break;
    case GLFW_KEY_S:
      key_code = KEY_S;
      break;
    case GLFW_KEY_T:
      key_code = KEY_T;
      break;
    case GLFW_KEY_U:
      key_code = KEY_U;
      break;
    case GLFW_KEY_V:
      key_code = KEY_V;
      break;
    case GLFW_KEY_W:
      key_code = KEY_W;
      break;
    case GLFW_KEY_X:
      key_code = KEY_X;
      break;
    case GLFW_KEY_Y:
      key_code = KEY_Y;
      break;
    case GLFW_KEY_Z:
      key_code = KEY_Z;
      break;
    case GLFW_KEY_1:
      key_code = KEY_1;
      break;
    case GLFW_KEY_2:
      key_code = KEY_2;
      break;
    case GLFW_KEY_3:
      key_code = KEY_3;
      break;
    case GLFW_KEY_4:
      key_code = KEY_4;
      break;
    case GLFW_KEY_5:
      key_code = KEY_5;
      break;
    case GLFW_KEY_6:
      key_code = KEY_6;
      break;
    case GLFW_KEY_7:
      key_code = KEY_7;
      break;
    case GLFW_KEY_8:
      key_code = KEY_8;
      break;
    case GLFW_KEY_9:
      key_code = KEY_9;
      break;
    case GLFW_KEY_0:
      key_code = KEY_0;
      break;
    case GLFW_KEY_SPACE:
      key_code = KEY_SPACE;
      break;
    case GLFW_KEY_MINUS:
      key_code = KEY_MINUS;
      break;
    case GLFW_KEY_EQUAL:
      key_code = KEY_EQUAL;
      break;
    case GLFW_KEY_LEFT_BRACKET:
      key_code = KEY_LEFT_BRACKET;
      break;
    case GLFW_KEY_RIGHT_BRACKET:
      key_code = KEY_RIGHT_BRACKET;
      break;
    case GLFW_KEY_BACKSLASH:
      key_code = KEY_BACKSLASH;
      break;
    case GLFW_KEY_SEMICOLON:
      key_code = KEY_SEMICOLON;
      break;
    case GLFW_KEY_APOSTROPHE:
      key_code = KEY_APOSTROPHE;
      break;
    case GLFW_KEY_GRAVE_ACCENT:
      key_code = KEY_GRAVE_ACCENT;
      break;
    case GLFW_KEY_COMMA:
      key_code = KEY_COMMA;
      break;
    case GLFW_KEY_PERIOD:
      key_code = KEY_PERIOD;
      break;
    case GLFW_KEY_SLASH:
      key_code = KEY_SLASH;
      break;
    case GLFW_KEY_WORLD_1:
      key_code = KEY_WORLD_1; /* At least in some layouts... */
      break;
    default:
      key_code = KEY_UNKNOWN; /* No matching translation was found */
      break;
  }

  return key_code;
}
