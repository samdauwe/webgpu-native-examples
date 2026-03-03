#ifndef WGPU_COMMON_H_
#define WGPU_COMMON_H_

#include "core/input.h"

#include <stdbool.h>
#include <stdint.h>
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
 * WebGPU Context
 * -------------------------------------------------------------------------- */

typedef struct wgpu_context_t wgpu_context_t;

typedef int (*wgpu_init_func)(struct wgpu_context_t* wgpu_context);
typedef int (*wgpu_frame_func)(struct wgpu_context_t* wgpu_context);
typedef void (*wgpu_shutdown_func)(struct wgpu_context_t* wgpu_context);
typedef void (*wgpu_input_event_func)(struct wgpu_context_t* wgpu_context,
                                      const input_event_t* input_event);

typedef struct {
  int width;
  int height;
  int sample_count;
  WGPUBool no_depth_buffer;
  const char* title;
  wgpu_init_func init_cb;
  wgpu_frame_func frame_cb;
  wgpu_shutdown_func shutdown_cb;
  wgpu_input_event_func input_event_cb;
  const WGPUFeatureName* required_features;
  uint32_t required_feature_count;
} wgpu_desc_t;

/* Forward declarations */
typedef struct wgpu_mipmap_generator_t wgpu_mipmap_generator_t;
typedef struct wgpu_panorama_to_cubemap_converter_t
  wgpu_panorama_to_cubemap_converter_t;

struct wgpu_context_t {
  wgpu_desc_t desc;
  WGPUBool async_setup_failed;
  WGPUBool async_setup_done;
  int width;
  int height;
  wgpu_input_event_func input_event_cb;
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
  wgpu_mipmap_generator_t* mipmap_generator; /* Lazily created on demand */
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
WGPUBuffer wgpu_create_buffer_from_data(wgpu_context_t* wgpu_context,
                                        const void* data, size_t size,
                                        WGPUBufferUsage usage);
wgpu_buffer_t wgpu_create_buffer(struct wgpu_context_t* wgpu_context,
                                 const wgpu_buffer_desc_t* desc);
void wgpu_destroy_buffer(wgpu_buffer_t* buffer);

/* -------------------------------------------------------------------------- *
 * WebGPU texture helper functions
 * -------------------------------------------------------------------------- */

/**
 * @brief Texture view dimension hint for mipmap generation.
 * When set to _Undefined, the dimension is auto-detected from the texture.
 */
typedef enum wgpu_mipmap_view_dimension_t {
  WGPU_MIPMAP_VIEW_UNDEFINED = 0, /* Auto-detect from texture */
  WGPU_MIPMAP_VIEW_2D,            /* 2D texture */
  WGPU_MIPMAP_VIEW_2D_ARRAY,      /* 2D array texture */
  WGPU_MIPMAP_VIEW_CUBE,          /* Cube texture */
  WGPU_MIPMAP_VIEW_CUBE_ARRAY,    /* Cube array texture */
} wgpu_mipmap_view_dimension_t;

typedef struct wgpu_texture_desc_t {
  WGPUExtent3D extent;
  WGPUTextureFormat format;
  uint32_t mip_level_count;
  WGPUTextureUsage usage;
  struct {
    const void* ptr;
    size_t size;
  } pixels;
  int8_t is_dirty;         /* Pixel data has been updated */
  int8_t generate_mipmaps; /* Generate mipmaps after texture creation */
  wgpu_mipmap_view_dimension_t mipmap_view_dimension; /* View dimension hint */
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
                               const wgpu_texture_desc_t* desc);
void wgpu_recreate_texture(struct wgpu_context_t* wgpu_context,
                           wgpu_texture_t* texture);
void wgpu_image_to_texure(wgpu_context_t* wgpu_context, WGPUTexture texture,
                          void* pixels, WGPUExtent3D size, uint32_t channels);
void wgpu_destroy_texture(wgpu_texture_t* texture);

/* -------------------------------------------------------------------------- *
 * WebGPU mipmap generator
 *
 * Usage example:
 *
 *  // Simple: auto mip levels + generation
 *  wgpu_texture_t tex = wgpu_create_texture(ctx, &(wgpu_texture_desc_t){
 *    .extent = {512, 512, 1},
 *    .format = WGPUTextureFormat_RGBA8Unorm,
 *    .pixels = { .ptr = data, .size = data_size },
 *    .generate_mipmaps = 1, // <-- just set this
 *  });
 *
 *  // Cube texture with explicit view hint
 *  wgpu_texture_t cube = wgpu_create_texture(ctx, &(wgpu_texture_desc_t){
 *    .extent = {256, 256, 6},
 *    .format = WGPUTextureFormat_RGBA8Unorm,
 *    .pixels = { .ptr = cube_data, .size = cube_data_size },
 *    .generate_mipmaps = 1,
 *    .mipmap_view_dimension = WGPU_MIPMAP_VIEW_CUBE,
 *  });
 *
 *  // Or call directly on an existing texture
 *  gpu_generate_mipmaps(ctx, my_texture, WGPU_MIPMAP_VIEW_2D);
 * -------------------------------------------------------------------------- */

/**
 * @brief Computes the number of mip levels for a texture of the given size.
 */
uint32_t wgpu_texture_mip_level_count(uint32_t width, uint32_t height);

/**
 * @brief Generates mipmaps for the given texture.
 *
 * Creates a render pipeline (cached per format+view dimension) that samples
 * from mip level N-1 and renders into mip level N using a fullscreen triangle
 * with bilinear filtering. Supports 2D, 2D-array, cube, and cube-array
 * textures.
 *
 * The mipmap generator is lazily created and cached in wgpu_context_t.
 *
 * @param wgpu_context  The WebGPU context (generator is cached here)
 * @param texture       The texture to generate mipmaps for
 * @param view_dim      View dimension hint (0 = auto-detect)
 */
void wgpu_generate_mipmaps(wgpu_context_t* wgpu_context, WGPUTexture texture,
                           wgpu_mipmap_view_dimension_t view_dim);

/**
 * @brief Destroys the cached mipmap generator and frees all resources.
 * Called automatically during context shutdown.
 */
void wgpu_mipmap_generator_destroy(wgpu_mipmap_generator_t* generator);

/* -------------------------------------------------------------------------- *
 * WebGPU panorama-to-cubemap converter
 *
 * Converts an equirectangular panorama texture to a cubemap using a compute
 * shader. The panorama is uploaded as RGBA32Float and converted to an
 * RGBA16Float cubemap with manual bilinear filtering.
 *
 * Usage example:
 *
 *  // Create a reusable converter
 *  wgpu_panorama_to_cubemap_converter_t* converter
 *    = wgpu_panorama_to_cubemap_converter_create(device);
 *
 *  // Convert panorama data to an existing cubemap texture
 *  wgpu_panorama_to_cubemap_converter_convert(
 *    converter, panorama_rgba, width, height, cubemap_texture);
 *
 *  // Cleanup when done
 *  wgpu_panorama_to_cubemap_converter_destroy(converter);
 * -------------------------------------------------------------------------- */

/**
 * @brief Creates a panorama-to-cubemap converter.
 *        Initializes the compute pipeline, sampler, and per-face uniform
 *        buffers. The converter can be reused for multiple conversions.
 *
 * @param device  The WebGPU device
 * @return Pointer to the converter, or NULL on failure.
 */
wgpu_panorama_to_cubemap_converter_t*
wgpu_panorama_to_cubemap_converter_create(WGPUDevice device);

/**
 * @brief Uploads equirectangular panorama data and converts it to a cubemap.
 *        The panorama is uploaded as RGBA32Float, then a compute shader
 *        performs the equirectangular-to-cubemap projection with bilinear
 *        filtering, writing RGBA16Float output per face.
 *
 * @param converter        The converter instance
 * @param panorama_data    RGBA float pixel data (4 floats per pixel)
 * @param panorama_width   Width of the panorama (should be 2× height)
 * @param panorama_height  Height of the panorama
 * @param environment_cubemap  Output cubemap texture (must be created with
 *                         RGBA16Float format, StorageBinding usage, 6 layers)
 * @return true on success, false on failure.
 */
bool wgpu_panorama_to_cubemap_converter_convert(
  wgpu_panorama_to_cubemap_converter_t* converter, const float* panorama_data,
  uint32_t panorama_width, uint32_t panorama_height,
  WGPUTexture environment_cubemap);

/**
 * @brief Destroys the converter and frees all GPU resources.
 */
void wgpu_panorama_to_cubemap_converter_destroy(
  wgpu_panorama_to_cubemap_converter_t* converter);

/* -------------------------------------------------------------------------- *
 * WebGPU environment map (IBL) helper functions
 *
 * Complete Image-Based Lighting pipeline:
 *   HDR panorama (.hdr) → equirectangular cubemap → mipmapped cubemap
 *   → irradiance cubemap + prefiltered specular cubemap + BRDF LUT
 *
 * Usage example:
 *
 *  // Load HDR panorama from file
 *  wgpu_environment_t env = {0};
 *  wgpu_environment_load_from_file(&env, "assets/environments/pisa.hdr");
 *
 *  // Create all IBL textures from the loaded panorama
 *  wgpu_ibl_textures_t ibl = {0};
 *  wgpu_ibl_textures_desc_t ibl_desc = {
 *    .environment_size = 1024,     // cubemap face size
 *    .irradiance_size  = 64,       // irradiance face size
 *    .prefiltered_size = 512,      // specular prefiltered face size
 *    .brdf_lut_size    = 128,      // BRDF LUT size
 *    .num_samples      = 1024,     // Monte Carlo samples
 *  };
 *  wgpu_ibl_textures_from_environment(ctx, &env, &ibl_desc, &ibl);
 *
 *  // Use ibl.environment_cubemap, ibl.irradiance_cubemap,
 *  //     ibl.prefiltered_cubemap, ibl.brdf_lut in your bind groups
 *
 *  // Cleanup
 *  wgpu_ibl_textures_destroy(&ibl);
 *  wgpu_environment_release(&env);
 * -------------------------------------------------------------------------- */

/**
 * @brief CPU-side HDR environment data loaded from an equirectangular
 *        panorama (.hdr). No GPU resources are created at this stage.
 */
typedef struct wgpu_environment_t {
  float* data;     /* RGBA float pixel data (owned, allocated by stbi) */
  uint32_t width;  /* Panorama width (must be 2× height) */
  uint32_t height; /* Panorama height */
  float rotation;  /* Y-axis rotation angle in radians */
} wgpu_environment_t;

/**
 * @brief Configuration for IBL texture generation.
 *        All sizes are per-face dimensions (square). Zero = use defaults.
 */
typedef struct wgpu_ibl_textures_desc_t {
  uint32_t environment_size; /* Cubemap face size (default: 1024) */
  uint32_t irradiance_size;  /* Irradiance cubemap face size (default: 64) */
  uint32_t prefiltered_size; /* Prefiltered specular face size (default: 512)*/
  uint32_t brdf_lut_size;    /* BRDF integration LUT size (default: 128) */
  uint32_t num_samples;      /* Monte Carlo sample count (default: 1024) */
} wgpu_ibl_textures_desc_t;

/**
 * @brief GPU-side IBL textures produced from an environment map.
 *        All textures use RGBA16Float format (except noted).
 */
typedef struct wgpu_ibl_textures_t {
  WGPUTexture environment_cubemap;  /* Mipmapped env cubemap (6 faces) */
  WGPUTextureView environment_view; /* Cube view of env cubemap */
  WGPUTexture irradiance_cubemap;   /* Diffuse irradiance (6 faces) */
  WGPUTextureView irradiance_view;  /* Cube view of irradiance */
  WGPUTexture prefiltered_cubemap;  /* Specular prefiltered (6 faces) */
  WGPUTextureView prefiltered_view; /* Cube view of prefiltered */
  WGPUTexture brdf_lut;             /* 2D BRDF integration LUT */
  WGPUTextureView brdf_lut_view;    /* 2D view of BRDF LUT */
  WGPUSampler environment_sampler;  /* Trilinear, repeat */
  WGPUSampler brdf_lut_sampler;     /* Linear, clamp-to-edge */
  uint32_t prefiltered_mip_levels;  /* Mip count of prefiltered cubemap */
} wgpu_ibl_textures_t;

/* --- Environment loading (CPU-only, no GPU) --- */

/**
 * @brief Loads an equirectangular HDR panorama from file.
 *        The image must have a 2:1 aspect ratio. Images wider than 4096px
 *        are bilinearly downsampled to 4096×2048.
 * @return true on success, false on failure.
 */
bool wgpu_environment_load_from_file(wgpu_environment_t* env,
                                     const char* filepath);

/**
 * @brief Loads an equirectangular HDR panorama from memory.
 * @return true on success, false on failure.
 */
bool wgpu_environment_load_from_memory(wgpu_environment_t* env,
                                       const uint8_t* data, uint32_t size);

/**
 * @brief Releases CPU-side environment pixel data.
 */
void wgpu_environment_release(wgpu_environment_t* env);

/* --- IBL texture generation (GPU compute) --- */

/**
 * @brief Generates all IBL textures from a loaded environment panorama.
 *        Performs the full pipeline: upload panorama → cubemap conversion →
 *        mipmap generation → irradiance/specular/BRDF LUT computation.
 *        All operations are submitted to the GPU queue synchronously.
 *
 * @param wgpu_context  The WebGPU context
 * @param env           Loaded environment data (CPU-side HDR pixels)
 * @param desc          IBL generation parameters (NULL = use defaults)
 * @param ibl           Output IBL textures (caller must destroy with
 *                      wgpu_ibl_textures_destroy)
 * @return true on success, false on failure.
 */
bool wgpu_ibl_textures_from_environment(wgpu_context_t* wgpu_context,
                                        const wgpu_environment_t* env,
                                        const wgpu_ibl_textures_desc_t* desc,
                                        wgpu_ibl_textures_t* ibl);

/**
 * @brief Releases all GPU resources held by the IBL textures.
 */
void wgpu_ibl_textures_destroy(wgpu_ibl_textures_t* ibl);

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
  WGPUBool depth_write_enabled;
} create_depth_stencil_state_desc_t;

WGPUBlendState wgpu_create_blend_state(WGPUBool enable_blend);
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
#define VALUE_OR_DEFAULT(p, prop, def) ((p) ? (desc)->prop : (def))

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
