#include "wgpu_common.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- *
 * WebGPU Context
 * -------------------------------------------------------------------------- */

static wgpu_context_t wgpu_context;

/* Forward declarations */
static void wgpu_platform_start(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_init(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_discard(wgpu_context_t* wgpu_context);
static void wgpu_swapchain_resized(wgpu_context_t* wgpu_context);
static WGPUTextureView wgpu_swapchain_next(wgpu_context_t* wgpu_context);

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

  wgpu_platform_start(&wgpu_context);
}

static void glfw_key_cb(GLFWwindow* window, int key, int scancode, int action,
                        int mods)
{
  UNUSED_VAR(scancode);
  UNUSED_VAR(mods);
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  if ((action == GLFW_PRESS) && wgpu_context->key_down_cb) {
    wgpu_context->key_down_cb(key);
  }
  else if ((action == GLFW_RELEASE) && wgpu_context->key_up_cb) {
    wgpu_context->key_up_cb(key);
  }
}

static void glfw_char_cb(GLFWwindow* window, unsigned int chr)
{
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  if (wgpu_context->char_cb) {
    wgpu_context->char_cb(chr);
  }
}

static void glfw_mousebutton_cb(GLFWwindow* window, int button, int action,
                                int mods)
{
  UNUSED_VAR(mods);
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  if ((action == GLFW_PRESS) && wgpu_context->mouse_btn_down_cb) {
    wgpu_context->mouse_btn_down_cb(button);
  }
  else if ((action == GLFW_RELEASE) && wgpu_context->mouse_btn_up_cb) {
    wgpu_context->mouse_btn_up_cb(button);
  }
}

static void glfw_cursorpos_cb(GLFWwindow* window, double xpos, double ypos)
{
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  if (wgpu_context->mouse_pos_cb) {
    wgpu_context->mouse_pos_cb((float)xpos, (float)ypos);
  }
}

static void glfw_scroll_cb(GLFWwindow* window, double xoffset, double yoffset)
{
  UNUSED_VAR(xoffset);
  wgpu_context_t* wgpu_context
    = (wgpu_context_t*)glfwGetWindowUserPointer(window);
  if (wgpu_context->mouse_wheel_cb) {
    wgpu_context->mouse_wheel_cb((float)yoffset);
  }
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
  wgpu_context->async_setup_done = true;
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

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
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
    .mappedAtCreation = false,
  };

  const uint32_t initial_size
    = (desc->initial.size == 0) ? desc->size : desc->initial.size;

  if (desc->initial.data && initial_size > 0 && initial_size <= desc->size) {
    buffer_desc.mappedAtCreation = true;
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

WGPUBlendState wgpu_create_blend_state(bool enable_blend)
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
