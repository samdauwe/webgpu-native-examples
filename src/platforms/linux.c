#include "../core/log.h"
#include "../core/macro.h"
#include "../core/platform.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/time.h>

#include <X11/Xlib.h>
#include <X11/Xresource.h>
#include <X11/Xutil.h>

#include "../../lib/wgpu_native/wgpu_native.h"

#if __has_include("vulkan/vulkan.h")
#define DAWN_ENABLE_BACKEND_VULKAN
#endif

#ifdef DAWN_ENABLE_BACKEND_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xlib.h>
#endif

struct window {
  Window handle;
  XIM im; /* implementation model */
  XIC ic; /* input context */
  VkSurfaceKHR surface;
  /* common data */
  int should_close;
  char keys[KEY_NUM];
  char buttons[BUTTON_NUM];
  callbacks_t callbacks;
  void* userdata;
  uint32_t width;
  uint32_t height;
};

/* platform initialization */

static Display* g_display = NULL;
static XContext g_context;

static void open_display(void)
{
  XInitThreads();
  g_display = XOpenDisplay(NULL);
  assert(g_display != NULL);
  g_context = XUniqueContext();
}

static void close_display(void)
{
  XCloseDisplay(g_display);
  g_display = NULL;
}

static void initialize_path(void)
{
  char path[PATH_SIZE];
  ssize_t bytes;
  int error;

  bytes = readlink("/proc/self/exe", path, PATH_SIZE - 1);
  assert(bytes != -1);
  path[bytes]         = '\0';
  *strrchr(path, '/') = '\0';

  error = chdir(path);
  assert(error == 0);
  error = chdir("assets");
  assert(error == 0);
}

void platform_initialize(void)
{
  assert(g_display == NULL);
  open_display();
  initialize_path();
}

void platform_terminate(void)
{
  assert(g_display != NULL);
  close_display();
}

/* window related functions */

/**
 * @brief Helper to obtain a Vulkan surface from the supplied window.
 * @param device WebGPU device
 * @param dpy display on which the device will be bound
 * @param window window on which the device will be bound
 * @return window surface (or \c VK_NULL_HANDLE if creation failed)
 */
static VkSurfaceKHR
wgpu_create_vk_surface_from_xlib(WGPUDevice device, Display* dpy, Window window)
{
  VkResult err         = VK_SUCCESS;
  VkSurfaceKHR surface = VK_NULL_HANDLE;

#ifdef __linux__
  VkXlibSurfaceCreateInfoKHR info;
  memset(&info, 0, sizeof(info));
  info.sType  = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  info.pNext  = NULL;
  info.flags  = 0;
  info.dpy    = dpy;
  info.window = window;
  err = vkCreateXlibSurfaceKHR(wgpu_get_backend_instance(device), &info, NULL,
                               &surface);
#endif

  if (err != VK_SUCCESS) {
    log_fatal("Could not create surface!", err);
  }

  return surface;
}

window_t* window_create(const char* title, uint32_t width, uint32_t height,
                        int resizable)
{
  window_t* window;
  int screen;
  Atom delete_window;
  XSetWindowAttributes window_attrs;
  XSetWindowAttributes attr;
  XSizeHints* size_hints;
  XClassHint* class_hint;
  const char* wm_delete_window_name = "WM_DELETE_WINDOW";

  assert(g_display && width > 0 && height > 0);

  window = (window_t*)malloc(sizeof(window_t));
  memset(window, 0, sizeof(window_t));
  window->width   = width;
  window->height  = height;
  window->surface = VK_NULL_HANDLE;

  screen = DefaultScreen(g_display);

  /* initialize container window attributes */
  memset(&window_attrs, 0, sizeof(window_attrs));
  window_attrs.background_pixel  = 0;
  window_attrs.background_pixmap = 0;
  window_attrs.border_pixel      = 0;
  window_attrs.event_mask
    = 0                     /* */
      | ButtonPressMask     /* handle button press events */
      | ButtonReleaseMask   /* handle button release events */
      | ExposureMask        /* handle container redraw  */
      | KeyPressMask        /* handle key press events */
      | KeyReleaseMask      /* handle key release events */
      | PointerMotionMask   /* handle pointer motion events */
      | StructureNotifyMask /* handle container notifications */
    ;

  window->handle
    = XCreateWindow(g_display,                     /* display */
                    RootWindow(g_display, screen), /* parent */
                    0, 0,                          /* x, y*/
                    width, height, 0, /* width, height, border_width*/
                    DefaultDepth(g_display, screen),  /* depth */
                    InputOutput,                      /* class */
                    DefaultVisual(g_display, screen), /* visual */
                    CWBorderPixel | CWEventMask,      /* valuemask */
                    &window_attrs                     /* attributes */
    );

  if (!resizable) {
    /* not resizable */
    size_hints             = XAllocSizeHints();
    size_hints->flags      = PMinSize | PMaxSize;
    size_hints->min_width  = width;
    size_hints->max_width  = width;
    size_hints->min_height = height;
    size_hints->max_height = height;
    XSetWMNormalHints(g_display, window->handle, size_hints);
    XFree(size_hints);
  }

  /* clear window to black. */
  memset(&attr, 0, sizeof(attr));
  XChangeWindowAttributes(g_display, window->handle, CWBackPixel, &attr);

  /* application name */
  class_hint            = XAllocClassHint();
  class_hint->res_name  = (char*)title;
  class_hint->res_class = (char*)title;
  XSetClassHint(g_display, window->handle, class_hint);
  XFree(class_hint);

  /* close handler */
  XInternAtoms(g_display, (char**)&wm_delete_window_name, 1, False,
               &delete_window);
  XSetWMProtocols(g_display, window->handle, &delete_window, 1);

  XMapWindow(g_display, window->handle);
  XStoreName(g_display, window->handle, title);

  window->im = XOpenIM(g_display, NULL, NULL, NULL);
  window->ic = XCreateIC(window->im,                               /* */
                         XNInputStyle,                             /* */
                         0 | XIMPreeditNothing | XIMStatusNothing, /* */
                         XNClientWindow, window->handle,           /* */
                         NULL                                      /* */
  );

  XSaveContext(g_display, window->handle, g_context, (XPointer)window);
  XFlush(g_display);
  return window;
}

void window_destroy(window_t* window)
{
  XDestroyIC(window->ic);
  XCloseIM(window->im);

  XUnmapWindow(g_display, window->handle);
  XDeleteContext(g_display, window->handle, g_context);

  XDestroyWindow(g_display, window->handle);
  XFlush(g_display);

  free(window);
}

int window_should_close(window_t* window)
{
  return window->should_close;
}

void window_set_title(window_t* window, const char* title)
{
  XStoreName(g_display, window->handle, title);
}

void window_set_userdata(window_t* window, void* userdata)
{
  window->userdata = userdata;
}

void* window_get_userdata(window_t* window)
{
  return window->userdata;
}

void* window_get_surface(void* device, window_t* window)
{
  if (window->surface == VK_NULL_HANDLE) {
    window->surface
      = wgpu_create_vk_surface_from_xlib(device, g_display, window->handle);
  }
  return window->surface;
}

void window_get_size(window_t* window, uint32_t* width, uint32_t* height)
{
  *width  = window->width;
  *height = window->height;
}

void window_get_aspect_ratio(window_t* window, float* aspect_ratio)
{
  *aspect_ratio = (float)window->width / (float)window->height;
}

/* input related functions */

static void handle_key_event(window_t* window, int virtual_key, char pressed)
{
  KeySym* keysyms;
  KeySym keysym;
  keycode_t key;
  int dummy;

  keysyms = XGetKeyboardMapping(g_display, virtual_key, 1, &dummy);
  keysym  = keysyms[0];
  XFree(keysyms);

  switch (keysym) {
    case XK_Escape:
      key = KEY_ESCAPE;
      break;
    case XK_Tab:
      key = KEY_TAB;
      break;
    case XK_Shift_L:
      key = KEY_LEFT_SHIFT;
      break;
    case XK_Shift_R:
      key = KEY_RIGHT_SHIFT;
      break;
    case XK_Control_L:
      key = KEY_LEFT_CONTROL;
      break;
    case XK_Control_R:
      key = KEY_RIGHT_CONTROL;
      break;
    case XK_Meta_L:
    case XK_Alt_L:
      key = KEY_LEFT_ALT;
      break;
    case XK_Mode_switch:      // Mapped to Alt_R on many keyboards
    case XK_ISO_Level3_Shift: // AltGr on at least some machines
    case XK_Meta_R:
    case XK_Alt_R:
      key = KEY_RIGHT_ALT;
      break;
    case XK_Super_L:
      key = KEY_LEFT_SUPER;
      break;
    case XK_Super_R:
      key = KEY_RIGHT_SUPER;
      break;
    case XK_Menu:
      key = KEY_MENU;
      break;
    case XK_Num_Lock:
      key = KEY_NUM_LOCK;
      break;
    case XK_Caps_Lock:
      key = KEY_CAPS_LOCK;
      break;
    case XK_Print:
      key = KEY_PRINT_SCREEN;
      break;
    case XK_Scroll_Lock:
      key = KEY_SCROLL_LOCK;
      break;
    case XK_Pause:
      key = KEY_PAUSE;
      break;
    case XK_Delete:
      key = KEY_DELETE;
      break;
    case XK_BackSpace:
      key = KEY_BACKSPACE;
      break;
    case XK_Return:
      key = KEY_ENTER;
      break;
    case XK_Home:
      key = KEY_HOME;
      break;
    case XK_End:
      key = KEY_END;
      break;
    case XK_Page_Up:
      key = KEY_PAGE_UP;
      break;
    case XK_Page_Down:
      key = KEY_PAGE_DOWN;
      break;
    case XK_Insert:
      key = KEY_INSERT;
      break;
    case XK_Left:
      key = KEY_LEFT;
      break;
    case XK_Right:
      key = KEY_RIGHT;
      break;
    case XK_Down:
      key = KEY_DOWN;
      break;
    case XK_Up:
      key = KEY_UP;
      break;
    case XK_F1:
      key = KEY_F1;
      break;
    case XK_F2:
      key = KEY_F2;
      break;
    case XK_F3:
      key = KEY_F3;
      break;
    case XK_F4:
      key = KEY_F4;
      break;
    case XK_F5:
      key = KEY_F5;
      break;
    case XK_F6:
      key = KEY_F6;
      break;
    case XK_F7:
      key = KEY_F7;
      break;
    case XK_F8:
      key = KEY_F8;
      break;
    case XK_F9:
      key = KEY_F9;
      break;
    case XK_F10:
      key = KEY_F10;
      break;
    case XK_F11:
      key = KEY_F11;
      break;
    case XK_F12:
      key = KEY_F12;
      break;
    case XK_F13:
      key = KEY_F13;
      break;
    case XK_F14:
      key = KEY_F14;
      break;
    case XK_F15:
      key = KEY_F15;
      break;
    case XK_F16:
      key = KEY_F16;
      break;
    case XK_F17:
      key = KEY_F17;
      break;
    case XK_F18:
      key = KEY_F18;
      break;
    case XK_F19:
      key = KEY_F19;
      break;
    case XK_F20:
      key = KEY_F20;
      break;
    case XK_F21:
      key = KEY_F21;
      break;
    case XK_F22:
      key = KEY_F22;
      break;
    case XK_F23:
      key = KEY_F23;
      break;
    case XK_F24:
      key = KEY_F24;
      break;
    case XK_F25:
      key = KEY_F25;
      break;

    // Numeric keypad
    case XK_KP_Divide:
      key = KEY_KP_DIVIDE;
      break;
    case XK_KP_Multiply:
      key = KEY_KP_MULTIPLY;
      break;
    case XK_KP_Subtract:
      key = KEY_KP_SUBTRACT;
      break;
    case XK_KP_Add:
      key = KEY_KP_ADD;
      break;

    // These should have been detected in secondary keysym test above!
    case XK_KP_Insert:
      key = KEY_KP_0;
      break;
    case XK_KP_End:
      key = KEY_KP_1;
      break;
    case XK_KP_Down:
      key = KEY_KP_2;
      break;
    case XK_KP_Page_Down:
      key = KEY_KP_3;
      break;
    case XK_KP_Left:
      key = KEY_KP_4;
      break;
    case XK_KP_Right:
      key = KEY_KP_6;
      break;
    case XK_KP_Home:
      key = KEY_KP_7;
      break;
    case XK_KP_Up:
      key = KEY_KP_8;
      break;
    case XK_KP_Page_Up:
      key = KEY_KP_9;
      break;
    case XK_KP_Delete:
      key = KEY_KP_DECIMAL;
      break;
    case XK_KP_Equal:
      key = KEY_KP_EQUAL;
      break;
    case XK_KP_Enter:
      key = KEY_KP_ENTER;
      break;

    // Last resort: Check for printable keys (should not happen if the XKB
    // extension is available). This will give a layout dependent mapping
    // (which is wrong, and we may miss some keys, especially on non-US
    // keyboards), but it's better than nothing...
    case XK_a:
      key = KEY_A;
      break;
    case XK_b:
      key = KEY_B;
      break;
    case XK_c:
      key = KEY_C;
      break;
    case XK_d:
      key = KEY_D;
      break;
    case XK_e:
      key = KEY_E;
      break;
    case XK_f:
      key = KEY_F;
      break;
    case XK_g:
      key = KEY_G;
      break;
    case XK_h:
      key = KEY_H;
      break;
    case XK_i:
      key = KEY_I;
      break;
    case XK_j:
      key = KEY_J;
      break;
    case XK_k:
      key = KEY_K;
      break;
    case XK_l:
      key = KEY_L;
      break;
    case XK_m:
      key = KEY_M;
      break;
    case XK_n:
      key = KEY_N;
      break;
    case XK_o:
      key = KEY_O;
      break;
    case XK_p:
      key = KEY_P;
      break;
    case XK_q:
      key = KEY_Q;
      break;
    case XK_r:
      key = KEY_R;
      break;
    case XK_s:
      key = KEY_S;
      break;
    case XK_t:
      key = KEY_T;
      break;
    case XK_u:
      key = KEY_U;
      break;
    case XK_v:
      key = KEY_V;
      break;
    case XK_w:
      key = KEY_W;
      break;
    case XK_x:
      key = KEY_X;
      break;
    case XK_y:
      key = KEY_Y;
      break;
    case XK_z:
      key = KEY_Z;
      break;
    case XK_1:
      key = KEY_1;
      break;
    case XK_2:
      key = KEY_2;
      break;
    case XK_3:
      key = KEY_3;
      break;
    case XK_4:
      key = KEY_4;
      break;
    case XK_5:
      key = KEY_5;
      break;
    case XK_6:
      key = KEY_6;
      break;
    case XK_7:
      key = KEY_7;
      break;
    case XK_8:
      key = KEY_8;
      break;
    case XK_9:
      key = KEY_9;
      break;
    case XK_0:
      key = KEY_0;
      break;
    case XK_space:
      key = KEY_SPACE;
      break;
    case XK_minus:
      key = KEY_MINUS;
      break;
    case XK_equal:
      key = KEY_EQUAL;
      break;
    case XK_bracketleft:
      key = KEY_LEFT_BRACKET;
      break;
    case XK_bracketright:
      key = KEY_RIGHT_BRACKET;
      break;
    case XK_backslash:
      key = KEY_BACKSLASH;
      break;
    case XK_semicolon:
      key = KEY_SEMICOLON;
      break;
    case XK_apostrophe:
      key = KEY_APOSTROPHE;
      break;
    case XK_grave:
      key = KEY_GRAVE_ACCENT;
      break;
    case XK_comma:
      key = KEY_COMMA;
      break;
    case XK_period:
      key = KEY_PERIOD;
      break;
    case XK_slash:
      key = KEY_SLASH;
      break;
    case XK_less:
      key = KEY_WORLD_1; // At least in some layouts...
      break;
    default:
      key = KEY_UNKNOWN; // No matching translation was found
      break;
  }

  if ((key < KEY_NUM) && (key != KEY_UNKNOWN)) {
    window->keys[key] = pressed;
    if (window->callbacks.key_callback) {
      window->callbacks.key_callback(window, key, pressed);
    }
  }
}

static void handle_button_event(window_t* window, int xbutton, char pressed)
{
  if (xbutton == Button1 || xbutton == Button2
      || xbutton == Button3) { /* mouse button */
    button_t button         = (xbutton == Button1) ? BUTTON_L :
                              (xbutton == Button2) ? BUTTON_M :
                                                     BUTTON_R;
    window->buttons[button] = pressed;
    if (window->callbacks.button_callback) {
      window->callbacks.button_callback(window, button, pressed);
    }
  }
  else if (xbutton == Button4 || xbutton == Button5) { /* mouse wheel */
    if (window->callbacks.scroll_callback) {
      float offset = xbutton == Button4 ? 1 : -1;
      window->callbacks.scroll_callback(window, offset);
    }
  }
}

static void handle_client_event(window_t* window, XClientMessageEvent* event)
{
  static Atom protocols     = None;
  static Atom delete_window = None;
  if (protocols == None) {
    protocols     = XInternAtom(g_display, "WM_PROTOCOLS", True);
    delete_window = XInternAtom(g_display, "WM_DELETE_WINDOW", True);
    assert(protocols != None);
    assert(delete_window != None);
  }
  if (event->message_type == protocols) {
    Atom protocol = event->data.l[0];
    if (protocol == delete_window) {
      window->should_close = 1;
    }
  }
}

static void handle_configure_event(window_t* window, XConfigureEvent* event)
{
  if (window->width != (uint32_t)event->width
      || window->height != (uint32_t)event->height) {
    window->width  = event->width;
    window->height = event->height;
    if (window->callbacks.resize_callback) {
      window->callbacks.resize_callback(window, event->width, event->height);
    }
  }
}

static void process_event(XEvent* event)
{
  Window handle;
  window_t* window;
  int error;

  handle = event->xany.window;
  error  = XFindContext(g_display, handle, g_context, (XPointer*)&window);
  if (error != 0) {
    return;
  }

  if (event->type == ClientMessage) {
    handle_client_event(window, &event->xclient);
  }
  else if (event->type == ConfigureNotify) {
    handle_configure_event(window, &event->xconfigure);
  }
  else if (event->type == KeyPress) {
    handle_key_event(window, event->xkey.keycode, 1);
  }
  else if (event->type == KeyRelease) {
    handle_key_event(window, event->xkey.keycode, 0);
  }
  else if (event->type == ButtonPress) {
    handle_button_event(window, event->xbutton.button, 1);
  }
  else if (event->type == ButtonRelease) {
    handle_button_event(window, event->xbutton.button, 0);
  }
}

void input_poll_events(void)
{
  int count = XPending(g_display);
  while (count > 0) {
    XEvent event;
    XNextEvent(g_display, &event);
    process_event(&event);
    count -= 1;
  }
  XFlush(g_display);
}

int input_key_pressed(window_t* window, keycode_t key)
{
  assert(key >= 0 && key < KEY_NUM);
  return window->keys[key];
}

int input_button_pressed(window_t* window, button_t button)
{
  assert(button >= 0 && button < BUTTON_NUM);
  return window->buttons[button];
}

void input_query_cursor(window_t* window, float* xpos, float* ypos)
{
  Window root, child;
  int root_x, root_y, window_x, window_y;
  unsigned int mask;
  XQueryPointer(g_display, window->handle, &root, &child, &root_x, &root_y,
                &window_x, &window_y, &mask);
  *xpos = (float)window_x;
  *ypos = (float)window_y;
}

void input_set_callbacks(window_t* window, callbacks_t callbacks)
{
  window->callbacks = callbacks;
}

/* misc platform functions */

static double get_native_time(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void get_local_time(date_t* current_date)
{
  struct timeval te;
  gettimeofday(&te, NULL);
  time_t T               = time(NULL);
  struct tm tm           = *localtime(&T);
  long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
  current_date->msec     = (int)(milliseconds % (1000));
  current_date->sec      = tm.tm_sec;
  current_date->min      = tm.tm_min;
  current_date->hour     = tm.tm_hour;
  current_date->day      = tm.tm_mday;
  current_date->month    = tm.tm_mon + 1;
  current_date->year     = tm.tm_year + 1900;
  current_date->day_sec  = ((float)current_date->msec) / 1000.0
                          + current_date->sec + current_date->min * 60
                          + current_date->hour * 3600;
  return;
}

float platform_get_time(void)
{
  static double initial = -1;
  if (initial < 0) {
    initial = get_native_time();
  }
  return (float)(get_native_time() - initial);
}
