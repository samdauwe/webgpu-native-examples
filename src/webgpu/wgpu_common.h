#ifndef WGPU_COMMON_H_
#define WGPU_COMMON_H_

#include <stdlib.h>

#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- *
 * Macros
 * -------------------------------------------------------------------------- */

/* Define bool, false, true if not defined */
#ifndef __bool_true_false_are_defined
#define bool int
#define false 0
#define true 1
#define size_t uint64_t
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Context
 * -------------------------------------------------------------------------- */

typedef struct wgpu_context_t wgpu_context_t;

typedef void (*wgpu_init_func)(struct wgpu_context_t* wgpu_context);
typedef void (*wgpu_frame_func)(struct wgpu_context_t* wgpu_context);
typedef void (*wgpu_shutdown_func)(struct wgpu_context_t* wgpu_context);
typedef void (*wgpu_key_func)(int key);
typedef void (*wgpu_char_func)(uint32_t c);
typedef void (*wgpu_mouse_btn_func)(int btn);
typedef void (*wgpu_mouse_pos_func)(float x, float y);
typedef void (*wgpu_mouse_wheel_func)(float v);

typedef struct {
  int width;
  int height;
  int sample_count;
  bool no_depth_buffer;
  const char* title;
  wgpu_init_func init_cb;
  wgpu_frame_func frame_cb;
  wgpu_shutdown_func shutdown_cb;
} wgpu_desc_t;

struct wgpu_context_t {
  wgpu_desc_t desc;
  bool async_setup_failed;
  bool async_setup_done;
  int width;
  int height;
  wgpu_key_func key_down_cb;
  wgpu_key_func key_up_cb;
  wgpu_char_func char_cb;
  wgpu_mouse_btn_func mouse_btn_down_cb;
  wgpu_mouse_btn_func mouse_btn_up_cb;
  wgpu_mouse_pos_func mouse_pos_cb;
  wgpu_mouse_wheel_func mouse_wheel_cb;
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  WGPUSurface surface;
  WGPUTextureFormat render_format;
  WGPUTexture msaa_tex;
  WGPUTexture depth_stencil_tex;
  WGPUTextureView swapchain_view;
  WGPUTextureView msaa_view;
  WGPUTextureView depth_stencil_view;
};

void wgpu_start(const wgpu_desc_t* desc);

/* -------------------------------------------------------------------------- *
 * GLFW WebGPU Extension
 * Ref: https://github.com/eliemichel/glfw3webgpu/
 * -------------------------------------------------------------------------- */

/**
 * @brief Creates a WebGPU surface for the specified window.
 */
WGPUSurface glfw_create_surface_for_window(WGPUInstance instance,
                                           GLFWwindow* window);

/* -------------------------------------------------------------------------- *
 * Time functions
 * -------------------------------------------------------------------------- */

uint64_t nano_time(void);

/* -------------------------------------------------------------------------- *
 * Macros
 * -------------------------------------------------------------------------- */

/* Define NULL if not defined */
#ifndef NULL
#define NULL ((void*)0)
#endif

#define UNUSED_VAR(x) ((void)(x))
#define UNUSED_FUNCTION(x) ((void)(x))

#ifndef CODE
#define CODE(...) #__VA_ARGS__
#endif

#ifndef STRVIEW
#define STRVIEW(X) (WGPUStringView){X, sizeof(X) - 1}
#endif

#define WGPU_RELEASE_RESOURCE(Type, Name)                                      \
  if (Name) {                                                                  \
    wgpu##Type##Release(Name);                                                 \
    Name = NULL;                                                               \
  }

#ifdef __cplusplus
}
#endif

#endif /* WGPU_COMMON_H_ */
