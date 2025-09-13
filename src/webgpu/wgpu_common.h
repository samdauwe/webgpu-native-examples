#ifndef WGPU_COMMON_H_
#define WGPU_COMMON_H_

#include <stdlib.h>

#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define DEFAULT_WINDOW_WIDTH (1280)
#define DEFAULT_WINDOW_HEIGHT (720)

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

typedef int (*wgpu_init_func)(struct wgpu_context_t* wgpu_context);
typedef int (*wgpu_frame_func)(struct wgpu_context_t* wgpu_context);
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
  WGPUTextureFormat depth_stencil_format;
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
 * WebGPU buffer helper functions
 * -------------------------------------------------------------------------- */

typedef struct wgpu_buffer_desc_t {
  const char* label;
  WGPUBufferUsage usage;
  uint32_t size;
  uint32_t count; /* Numer of elements in the buffer (optional) */
  struct {
    const void* data;
    uint32_t size;
  } initial;
  WGPUBool mapped_at_creation;
} wgpu_buffer_desc_t;

typedef struct wgpu_buffer_t {
  WGPUBuffer buffer;
  WGPUBufferUsage usage;
  uint32_t size;
  uint32_t count; /* Numer of elements in the buffer (optional) */
} wgpu_buffer_t;

/* WebGPU buffer create / destroy */
wgpu_buffer_t wgpu_create_buffer(struct wgpu_context_t* wgpu_context,
                                 const wgpu_buffer_desc_t* desc);
void wgpu_destroy_buffer(wgpu_buffer_t* buffer);

/* -------------------------------------------------------------------------- *
 * WebGPU texture helper functions
 * -------------------------------------------------------------------------- */

typedef struct wgpu_texture_desc_t {
  WGPUExtent3D extent;
  WGPUTextureFormat format;
  struct {
    const void* ptr;
    size_t size;
  } pixels;
  int8_t is_dirty; /* Pixel data has been update */
} wgpu_texture_desc_t;

typedef struct wgpu_texture_t {
  wgpu_texture_desc_t desc;
  WGPUTexture handle;
  WGPUTextureView view;
  WGPUSampler sampler;
  int8_t initialized; /* Texture is initialized */
} wgpu_texture_t;

/* WebGPU texture create / destroy */
wgpu_texture_t wgpu_create_texture(struct wgpu_context_t* wgpu_context,
                                   const wgpu_texture_desc_t* desc);
wgpu_texture_t
wgpu_create_color_bars_texture(struct wgpu_context_t* wgpu_context,
                               uint32_t width, uint32_t height);
void wgpu_recreate_texture(struct wgpu_context_t* wgpu_context,
                           wgpu_texture_t* texture);
void wgpu_image_to_texure(wgpu_context_t* wgpu_context, WGPUTexture texture,
                          void* pixels, WGPUExtent3D size, uint32_t channels);
void wgpu_destroy_texture(wgpu_texture_t* texture);

/* -------------------------------------------------------------------------- *
 * WebGPU shader helper functions
 * -------------------------------------------------------------------------- */

/* WebGPU shader */
typedef struct wgpu_shader_desc_t {
  const char* label;
  /* WGSL source code ( ref: https://www.w3.org/TR/WGSL ) */
  const char* wgsl_source_code;
  const char* entry;
  struct {
    uint32_t count;
    WGPUConstantEntry const* entries;
  } constants; /* Pipeline shader constant s*/
} wgpu_shader_desc_t;

WGPUShaderModule wgpu_create_shader_module(WGPUDevice device,
                                           const char* wgsl_source_code);

/* -------------------------------------------------------------------------- *
 * WebGPU pipeline helper functions
 * -------------------------------------------------------------------------- */

typedef struct create_depth_stencil_state_desc_t {
  WGPUTextureFormat format;
  bool depth_write_enabled;
} create_depth_stencil_state_desc_t;

WGPUBlendState wgpu_create_blend_state(bool enable_blend);
WGPUDepthStencilState
wgpu_create_depth_stencil_state(create_depth_stencil_state_desc_t* desc);

/* -------------------------------------------------------------------------- *
 * Math
 * -------------------------------------------------------------------------- */

/**
 * @brief Generates a random float number in range [min, max].
 * @param min minimum number
 * @param max maximum number
 * @return random float number in range [min, max]
 */
float random_float_min_max(float min, float max);

/**
 * @brief Generates a random float number in range [0.0f, 1.0f].
 * @return random float number in range [0.0f, 1.0f]
 */
float random_float(void);

/* -------------------------------------------------------------------------- *
 * Macros
 * -------------------------------------------------------------------------- */

/* Define NULL if not defined */
#ifndef NULL
#define NULL ((void*)0)
#endif

#define UNUSED_VAR(x) ((void)(x))
#define UNUSED_FUNCTION(x) ((void)(x))

#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define VALUE_OR(val, def) ((val == 0) ? def : val)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLAMP(x, lo, hi) (MIN(hi, MAX(lo, x)))

#ifndef CODE
#define CODE(...) #__VA_ARGS__
#endif

#ifndef NDEBUG
#define ASSERT(expression)                                                     \
  {                                                                            \
    if (!(expression)) {                                                       \
      printf("Assertion(%s) failed: file \"%s\", line %d\n", #expression,      \
             __FILE__, __LINE__);                                              \
    }                                                                          \
  }
#else
#define ASSERT(expression) NULL;
#endif

#ifndef STRVIEW
#define STRVIEW(X) ((WGPUStringView){X, sizeof(X) - 1})
#endif

#define WGPU_RELEASE_RESOURCE(Type, Name)                                      \
  if (Name) {                                                                  \
    wgpu##Type##Release(Name);                                                 \
    Name = NULL;                                                               \
  }

#define WGPU_VERTATTR_DESC(l, f, o)                                            \
  (WGPUVertexAttribute)                                                        \
  {                                                                            \
    .shaderLocation = l, .format = f, .offset = o,                             \
  }

#define WGPU_VERTBUFFERLAYOUT_DESC(s, a)                                       \
  {                                                                            \
    .arrayStride    = s,                                                       \
    .stepMode       = WGPUVertexStepMode_Vertex,                               \
    .attributeCount = sizeof(a) / sizeof(a[0]),                                \
    .attributes     = a,                                                       \
  };

#define WPU_VERTEXSTATE_DESC(b)                                                \
  {                                                                            \
    .vertexBufferCount = 1,                                                    \
    .vertexBuffers     = &b,                                                   \
  }

#define WGPU_VERTSTATE(name, bindSize, ...)                                    \
  WGPUVertexAttribute vertAttrDesc##name[] = {__VA_ARGS__};                    \
  WGPUVertexBufferLayout name##VertBuffLayoutDesc                              \
    = WGPU_VERTBUFFERLAYOUT_DESC(bindSize, vertAttrDesc##name);                \
  WGPUVertexStateDescriptor vert_state_##name                                  \
    = WPU_VERTEXSTATE_DESC(name##VertBuffLayoutDesc);

#define WGPU_VERTEX_BUFFER_LAYOUT(name, bind_size, ...)                        \
  WGPUVertexAttribute vert_attr_desc_##name[] = {__VA_ARGS__};                 \
  WGPUVertexBufferLayout name##_vertex_buffer_layout                           \
    = WGPU_VERTBUFFERLAYOUT_DESC(bind_size, vert_attr_desc_##name);

#define FREE_TEXTURE_PIXELS(tex)                                               \
  do {                                                                         \
    if ((tex).desc.pixels.ptr) {                                               \
      free((void*)(tex).desc.pixels.ptr);                                      \
      (tex).desc.pixels.ptr  = NULL;                                           \
      (tex).desc.pixels.size = 0;                                              \
    }                                                                          \
  } while (0)

/* Constants */
#define PI 3.14159265358979323846f   /* pi */
#define PI2 6.28318530717958647692f  /* pi * 2 */
#define PI_2 1.57079632679489661923f /* pi/2 */

#define TO_RADIANS(degrees) ((PI / 180.0f) * (degrees))
#define TO_DEGREES(radians) ((180.0f / PI) * (radians))

#ifdef __cplusplus
}
#endif

#endif /* WGPU_COMMON_H_ */
