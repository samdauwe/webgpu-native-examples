#include "webgpu/imgui_overlay.h"
#include "webgpu/wgpu_common.h"

#include <cglm/cglm.h>
#include <math.h>
#include <string.h>

#define SOKOL_TIME_IMPL
#include <sokol_time.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#endif
#include <cimgui.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* -------------------------------------------------------------------------- *
 * WebGPU Example - Blending
 *
 * This example demonstrates the use of blending in WebGPU. It shows how to
 * configure different blend operations, source factors, and destination
 * factors for both color and alpha channels. The example displays two
 * overlapping images with various blend modes that can be selected through
 * a GUI.
 *
 * Ref:
 * https://github.com/webgpu/webgpu-samples/tree/main/src/sample/blending
 * -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

static const char* textured_quad_shader_wgsl;

/* -------------------------------------------------------------------------- *
 * Constants
 * -------------------------------------------------------------------------- */

#define TEXTURE_SIZE (300u)
#define NUM_CIRCLES (3u)

/* Blend operations */
typedef enum blend_operation_t {
  BLEND_OP_ADD              = 0,
  BLEND_OP_SUBTRACT         = 1,
  BLEND_OP_REVERSE_SUBTRACT = 2,
  BLEND_OP_MIN              = 3,
  BLEND_OP_MAX              = 4,
  BLEND_OP_COUNT            = 5,
} blend_operation_t;

/* Blend factors */
typedef enum blend_factor_t {
  BLEND_FACTOR_ZERO                = 0,
  BLEND_FACTOR_ONE                 = 1,
  BLEND_FACTOR_SRC                 = 2,
  BLEND_FACTOR_ONE_MINUS_SRC       = 3,
  BLEND_FACTOR_SRC_ALPHA           = 4,
  BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 5,
  BLEND_FACTOR_DST                 = 6,
  BLEND_FACTOR_ONE_MINUS_DST       = 7,
  BLEND_FACTOR_DST_ALPHA           = 8,
  BLEND_FACTOR_ONE_MINUS_DST_ALPHA = 9,
  BLEND_FACTOR_SRC_ALPHA_SATURATED = 10,
  BLEND_FACTOR_CONSTANT            = 11,
  BLEND_FACTOR_ONE_MINUS_CONSTANT  = 12,
  BLEND_FACTOR_COUNT               = 13,
} blend_factor_t;

/* Blend presets */
typedef enum blend_preset_t {
  PRESET_DEFAULT               = 0,
  PRESET_PREMULTIPLIED_BLEND   = 1,
  PRESET_UNPREMULTIPLIED_BLEND = 2,
  PRESET_DESTINATION_OVER      = 3,
  PRESET_SOURCE_IN             = 4,
  PRESET_DESTINATION_IN        = 5,
  PRESET_SOURCE_OUT            = 6,
  PRESET_DESTINATION_OUT       = 7,
  PRESET_SOURCE_ATOP           = 8,
  PRESET_DESTINATION_ATOP      = 9,
  PRESET_ADDITIVE              = 10,
  PRESET_COUNT                 = 11,
} blend_preset_t;

/* Alpha mode */
typedef enum alpha_mode_t {
  ALPHA_MODE_OPAQUE        = 0,
  ALPHA_MODE_PREMULTIPLIED = 1,
  ALPHA_MODE_COUNT         = 2,
} alpha_mode_t;

/* Texture set type */
typedef enum texture_set_t {
  TEXTURE_SET_PREMULTIPLIED   = 0,
  TEXTURE_SET_UNPREMULTIPLIED = 1,
  TEXTURE_SET_COUNT           = 2,
} texture_set_t;

/* Blend component settings */
typedef struct blend_component_t {
  blend_operation_t operation;
  blend_factor_t src_factor;
  blend_factor_t dst_factor;
} blend_component_t;

/* Preset definition */
typedef struct preset_def_t {
  blend_component_t color;
  blend_component_t alpha;
} preset_def_t;

/* Uniform data */
typedef struct uniforms_t {
  mat4 matrix;
} uniforms_t;

/* -------------------------------------------------------------------------- *
 * State
 * -------------------------------------------------------------------------- */

static struct {
  /* Textures */
  struct {
    wgpu_texture_t src_unpremultiplied;
    wgpu_texture_t dst_unpremultiplied;
    wgpu_texture_t src_premultiplied;
    wgpu_texture_t dst_premultiplied;
    WGPUSampler sampler;
  } textures;
  /* Pipelines */
  WGPURenderPipeline dst_pipeline;
  WGPURenderPipeline src_pipeline;
  WGPUPipelineLayout pipeline_layout;
  WGPUBindGroupLayout bind_group_layout;
  /* Shader module */
  WGPUShaderModule shader_module;
  /* Uniform buffers */
  struct {
    wgpu_buffer_t src;
    wgpu_buffer_t dst;
  } uniform_buffers;
  /* Bind groups for premultiplied textures */
  struct {
    WGPUBindGroup src;
    WGPUBindGroup dst;
  } bind_groups_premultiplied;
  /* Bind groups for unpremultiplied textures */
  struct {
    WGPUBindGroup src;
    WGPUBindGroup dst;
  } bind_groups_unpremultiplied;
  /* Render pass */
  WGPURenderPassColorAttachment color_attachment;
  WGPURenderPassDescriptor render_pass_descriptor;
  /* Settings */
  struct {
    alpha_mode_t alpha_mode;
    texture_set_t texture_set;
    blend_preset_t preset;
    blend_component_t color;
    blend_component_t alpha;
    float constant_color[3];
    float constant_alpha;
    float clear_color[3];
    float clear_alpha;
    bool clear_premultiply;
  } settings;
  /* UI strings */
  const char* operation_names[BLEND_OP_COUNT];
  const char* factor_names[BLEND_FACTOR_COUNT];
  const char* preset_names[PRESET_COUNT];
  const char* alpha_mode_names[ALPHA_MODE_COUNT];
  const char* texture_set_names[TEXTURE_SET_COUNT];
  /* Preset definitions */
  preset_def_t preset_defs[PRESET_COUNT];
  /* State flags */
  WGPUBool pipeline_needs_rebuild;
  WGPUBool initialized;
  uint64_t last_frame_time;
} state = {
  .color_attachment = {
    .loadOp     = WGPULoadOp_Clear,
    .storeOp    = WGPUStoreOp_Store,
    .clearValue = {0.0, 0.0, 0.0, 0.0},
    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
  },
  .render_pass_descriptor = {
    .colorAttachmentCount   = 1,
    .colorAttachments       = &state.color_attachment,
    .depthStencilAttachment = NULL,
  },
  .settings = {
    .alpha_mode        = ALPHA_MODE_PREMULTIPLIED,
    .texture_set       = TEXTURE_SET_PREMULTIPLIED,
    .preset            = PRESET_PREMULTIPLIED_BLEND,
    .color = {
      .operation  = BLEND_OP_ADD,
      .src_factor = BLEND_FACTOR_ONE,
      .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    },
    .alpha = {
      .operation  = BLEND_OP_ADD,
      .src_factor = BLEND_FACTOR_ONE,
      .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    },
    .constant_color = {1.0f, 0.5f, 0.25f},
    .constant_alpha = 1.0f,
    .clear_color = {0.0f, 0.0f, 0.0f},
    .clear_alpha = 0.0f,
    .clear_premultiply = true,
  },
  .operation_names = {
    "add",              /* */
    "subtract",         /* */
    "reverse-subtract", /* */
    "min",              /* */
    "max",              /* */
  },
  .factor_names = {
    "zero",               /* */
    "one",                /* */
    "src",                /* */
    "one-minus-src",      /* */
    "src-alpha",          /* */
    "one-minus-src-alpha",/* */
    "dst",                /* */
    "one-minus-dst",      /* */
    "dst-alpha",          /* */
    "one-minus-dst-alpha",/* */
    "src-alpha-saturated",/* */
    "constant",           /* */
    "one-minus-constant", /* */
  },
  .preset_names = {
    "default (copy)",                   /* */
    "premultiplied blend (source-over)",/* */
    "un-premultiplied blend",           /* */
    "destination-over",                 /* */
    "source-in",                        /* */
    "destination-in",                   /* */
    "source-out",                       /* */
    "destination-out",                  /* */
    "source-atop",                      /* */
    "destination-atop",                 /* */
    "additive (lighten)",               /* */
  },
  .alpha_mode_names = {
    "opaque",       /* */
    "premultiplied",/* */
  },
  .texture_set_names = {
    "premultiplied alpha",   /* */
    "un-premultiplied alpha",/* */
  },
  .preset_defs = {
    /* PRESET_DEFAULT - default (copy) */
    [PRESET_DEFAULT] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ZERO },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ZERO },
    },
    /* PRESET_PREMULTIPLIED_BLEND - premultiplied blend (source-over) */
    [PRESET_PREMULTIPLIED_BLEND] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
    },
    /* PRESET_UNPREMULTIPLIED_BLEND - un-premultiplied blend */
    [PRESET_UNPREMULTIPLIED_BLEND] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_SRC_ALPHA, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_SRC_ALPHA, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
    },
    /* PRESET_DESTINATION_OVER - destination-over */
    [PRESET_DESTINATION_OVER] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_ONE },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_ONE },
    },
    /* PRESET_SOURCE_IN - source-in */
    [PRESET_SOURCE_IN] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_DST_ALPHA, .dst_factor = BLEND_FACTOR_ZERO },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_DST_ALPHA, .dst_factor = BLEND_FACTOR_ZERO },
    },
    /* PRESET_DESTINATION_IN - destination-in */
    [PRESET_DESTINATION_IN] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ZERO, .dst_factor = BLEND_FACTOR_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ZERO, .dst_factor = BLEND_FACTOR_SRC_ALPHA },
    },
    /* PRESET_SOURCE_OUT - source-out */
    [PRESET_SOURCE_OUT] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_ZERO },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_ZERO },
    },
    /* PRESET_DESTINATION_OUT - destination-out */
    [PRESET_DESTINATION_OUT] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ZERO, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ZERO, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
    },
    /* PRESET_SOURCE_ATOP - source-atop */
    [PRESET_SOURCE_ATOP] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_DST_ALPHA, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_DST_ALPHA, .dst_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA },
    },
    /* PRESET_DESTINATION_ATOP - destination-atop */
    [PRESET_DESTINATION_ATOP] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_SRC_ALPHA },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE_MINUS_DST_ALPHA, .dst_factor = BLEND_FACTOR_SRC_ALPHA },
    },
    /* PRESET_ADDITIVE - additive (lighten) */
    [PRESET_ADDITIVE] = {
      .color = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ONE },
      .alpha = { .operation = BLEND_OP_ADD, .src_factor = BLEND_FACTOR_ONE, .dst_factor = BLEND_FACTOR_ONE },
    },
  },
  .pipeline_needs_rebuild = true,
};

/* -------------------------------------------------------------------------- *
 * Conversion helpers
 * -------------------------------------------------------------------------- */

static WGPUBlendOperation get_wgpu_blend_operation(blend_operation_t op)
{
  switch (op) {
    case BLEND_OP_ADD:
      return WGPUBlendOperation_Add;
    case BLEND_OP_SUBTRACT:
      return WGPUBlendOperation_Subtract;
    case BLEND_OP_REVERSE_SUBTRACT:
      return WGPUBlendOperation_ReverseSubtract;
    case BLEND_OP_MIN:
      return WGPUBlendOperation_Min;
    case BLEND_OP_MAX:
      return WGPUBlendOperation_Max;
    default:
      return WGPUBlendOperation_Add;
  }
}

static WGPUBlendFactor get_wgpu_blend_factor(blend_factor_t factor)
{
  switch (factor) {
    case BLEND_FACTOR_ZERO:
      return WGPUBlendFactor_Zero;
    case BLEND_FACTOR_ONE:
      return WGPUBlendFactor_One;
    case BLEND_FACTOR_SRC:
      return WGPUBlendFactor_Src;
    case BLEND_FACTOR_ONE_MINUS_SRC:
      return WGPUBlendFactor_OneMinusSrc;
    case BLEND_FACTOR_SRC_ALPHA:
      return WGPUBlendFactor_SrcAlpha;
    case BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return WGPUBlendFactor_OneMinusSrcAlpha;
    case BLEND_FACTOR_DST:
      return WGPUBlendFactor_Dst;
    case BLEND_FACTOR_ONE_MINUS_DST:
      return WGPUBlendFactor_OneMinusDst;
    case BLEND_FACTOR_DST_ALPHA:
      return WGPUBlendFactor_DstAlpha;
    case BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return WGPUBlendFactor_OneMinusDstAlpha;
    case BLEND_FACTOR_SRC_ALPHA_SATURATED:
      return WGPUBlendFactor_SrcAlphaSaturated;
    case BLEND_FACTOR_CONSTANT:
      return WGPUBlendFactor_Constant;
    case BLEND_FACTOR_ONE_MINUS_CONSTANT:
      return WGPUBlendFactor_OneMinusConstant;
    default:
      return WGPUBlendFactor_One;
  }
}

/* -------------------------------------------------------------------------- *
 * Image generation functions (C99 equivalent of Canvas 2D)
 * -------------------------------------------------------------------------- */

/* Helper for HSL to RGB conversion */
static float hue_to_rgb(float p, float q, float t)
{
  if (t < 0.0f)
    t += 1.0f;
  if (t > 1.0f)
    t -= 1.0f;
  if (t < 1.0f / 6.0f)
    return p + (q - p) * 6.0f * t;
  if (t < 1.0f / 2.0f)
    return q;
  if (t < 2.0f / 3.0f)
    return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  return p;
}

/* HSL to RGB conversion */
static void hsl_to_rgb(float h, float s, float l, float* r, float* g, float* b)
{
  if (s == 0.0f) {
    *r = *g = *b = l;
    return;
  }

  float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
  float p = 2.0f * l - q;

  *r = hue_to_rgb(p, q, h + 1.0f / 3.0f);
  *g = hue_to_rgb(p, q, h);
  *b = hue_to_rgb(p, q, h - 1.0f / 3.0f);
}

/**
 * @brief Creates source image with 3 overlapping circles using screen blend
 *
 * This mimics the TypeScript createSourceImage() function which uses canvas 2D
 * with radial gradients and screen composite operation.
 */
static void create_source_image(uint8_t* pixels, uint32_t size,
                                bool premultiply)
{
  /* Clear to transparent */
  memset(pixels, 0, size * size * 4);

  /* For each circle */
  const float center = (float)size / 2.0f;
  const float radius = (float)size / 3.0f;
  const float offset = (float)size / 6.0f;

  /* Temporary float buffer for screen blending */
  float* temp = (float*)calloc(size * size * 4, sizeof(float));

  for (uint32_t c = 0; c < NUM_CIRCLES; ++c) {
    /* Calculate circle center based on rotation */
    /* In TypeScript, canvas rotates BEFORE each draw, accumulating rotation:
     * - Circle 0: rotate 120° first -> position at 120° -> Red (h=0/3)
     * - Circle 1: rotate +120° (total 240°) -> position at 240° -> Green
     * (h=1/3)
     * - Circle 2: rotate +120° (total 360°/0°) -> position at 0° -> Blue
     * (h=2/3) So we need to add 120° to the base angle (c+1) instead of just c
     */
    float angle = (float)(c + 1) * (2.0f * PI / (float)NUM_CIRCLES);
    float cx    = center + cosf(angle) * offset;
    float cy    = center + sinf(angle) * offset;

    /* Get color for this circle */
    float h = (float)c / (float)NUM_CIRCLES;
    float r_col, g_col, b_col;
    hsl_to_rgb(h, 1.0f, 0.5f, &r_col, &g_col, &b_col);

    /* Draw circle with radial gradient */
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        float dx   = (float)x - cx;
        float dy   = (float)y - cy;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < radius) {
          /* Calculate gradient: full color at radius/2, fade to transparent at
           * radius */
          float t = dist / radius;
          float alpha;
          if (t < 0.5f) {
            alpha = 1.0f;
          }
          else {
            alpha = 1.0f - (t - 0.5f) * 2.0f;
          }

          /* Screen blend: result = 1 - (1 - dst) * (1 - src) */
          uint32_t idx = (y * size + x) * 4;
          float src_r  = r_col * alpha;
          float src_g  = g_col * alpha;
          float src_b  = b_col * alpha;
          float src_a  = alpha;

          /* Screen blend for RGB, normal blend for alpha */
          temp[idx + 0] = 1.0f - (1.0f - temp[idx + 0]) * (1.0f - src_r);
          temp[idx + 1] = 1.0f - (1.0f - temp[idx + 1]) * (1.0f - src_g);
          temp[idx + 2] = 1.0f - (1.0f - temp[idx + 2]) * (1.0f - src_b);
          temp[idx + 3] = temp[idx + 3] + src_a * (1.0f - temp[idx + 3]);
        }
      }
    }
  }

  /* Convert to uint8 with optional premultiplication */
  for (uint32_t i = 0; i < size * size; ++i) {
    float r = temp[i * 4 + 0];
    float g = temp[i * 4 + 1];
    float b = temp[i * 4 + 2];
    float a = temp[i * 4 + 3];

    if (premultiply) {
      r *= a;
      g *= a;
      b *= a;
    }

    pixels[i * 4 + 0] = (uint8_t)(CLAMP(r, 0.0f, 1.0f) * 255.0f);
    pixels[i * 4 + 1] = (uint8_t)(CLAMP(g, 0.0f, 1.0f) * 255.0f);
    pixels[i * 4 + 2] = (uint8_t)(CLAMP(b, 0.0f, 1.0f) * 255.0f);
    pixels[i * 4 + 3] = (uint8_t)(CLAMP(a, 0.0f, 1.0f) * 255.0f);
  }

  free(temp);
}

/**
 * @brief Creates destination image with diagonal rainbow stripes
 *
 * This mimics the TypeScript createDestinationImage() function which creates
 * a linear gradient with alternating transparent stripes.
 */
static void create_destination_image(uint8_t* pixels, uint32_t size,
                                     bool premultiply)
{
  /* Clear to transparent */
  memset(pixels, 0, size * size * 4);

  for (uint32_t y = 0; y < size; ++y) {
    for (uint32_t x = 0; x < size; ++x) {
      /* Linear gradient from top-left to bottom-right */
      float t = ((float)x + (float)y) / (2.0f * (float)size);

      /* Rainbow gradient: hue from 0 to 1 (inverted as in TypeScript) */
      float h = 1.0f - t;
      float r, g, b;
      hsl_to_rgb(h, 1.0f, 0.5f, &r, &g, &b);

      /* Create diagonal stripes (every 32 pixels, 16 visible, 16 transparent)
       */
      /* Rotated -45 degrees as in TypeScript: stripes go from bottom-left to
       * upper-right. In TypeScript, canvas is rotated -45° then horizontal
       * stripes drawn. This is equivalent to using (x + y) as the stripe
       * coordinate. */
      float stripe_coord = (float)x + (float)y;
      int stripe_idx     = (int)floorf(stripe_coord / 32.0f);
      bool visible       = (stripe_idx % 2) == 0;

      /* Also check stripe position within the cycle */
      float in_stripe = fmodf(stripe_coord, 32.0f);
      if (in_stripe < 0)
        in_stripe += 32.0f;
      visible = in_stripe >= 16.0f;

      uint32_t idx = (y * size + x) * 4;
      if (visible) {
        float alpha = 1.0f;
        if (premultiply) {
          r *= alpha;
          g *= alpha;
          b *= alpha;
        }
        pixels[idx + 0] = (uint8_t)(r * 255.0f);
        pixels[idx + 1] = (uint8_t)(g * 255.0f);
        pixels[idx + 2] = (uint8_t)(b * 255.0f);
        pixels[idx + 3] = 255;
      }
      else {
        pixels[idx + 0] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
        pixels[idx + 3] = 0;
      }
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Texture initialization
 * -------------------------------------------------------------------------- */

static void init_textures(wgpu_context_t* wgpu_context)
{
  const uint32_t tex_size    = TEXTURE_SIZE;
  const uint32_t pixel_count = tex_size * tex_size * 4;

  /* Allocate pixel buffers */
  uint8_t* src_pixels = (uint8_t*)malloc(pixel_count);
  uint8_t* dst_pixels = (uint8_t*)malloc(pixel_count);

  /* Create unpremultiplied textures */
  create_source_image(src_pixels, tex_size, false);
  state.textures.src_unpremultiplied = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent
      = {.width = tex_size, .height = tex_size, .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {.ptr = src_pixels, .size = pixel_count},
    });

  create_destination_image(dst_pixels, tex_size, false);
  state.textures.dst_unpremultiplied = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent
      = {.width = tex_size, .height = tex_size, .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {.ptr = dst_pixels, .size = pixel_count},
    });

  /* Create premultiplied textures */
  create_source_image(src_pixels, tex_size, true);
  state.textures.src_premultiplied = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent
      = {.width = tex_size, .height = tex_size, .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {.ptr = src_pixels, .size = pixel_count},
    });

  create_destination_image(dst_pixels, tex_size, true);
  state.textures.dst_premultiplied = wgpu_create_texture(
    wgpu_context,
    &(wgpu_texture_desc_t){
      .extent
      = {.width = tex_size, .height = tex_size, .depthOrArrayLayers = 1},
      .format = WGPUTextureFormat_RGBA8Unorm,
      .usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
      .pixels = {.ptr = dst_pixels, .size = pixel_count},
    });

  /* Create sampler */
  state.textures.sampler = wgpuDeviceCreateSampler(
    wgpu_context->device, &(WGPUSamplerDescriptor){
                            .label         = STRVIEW("Texture sampler"),
                            .addressModeU  = WGPUAddressMode_ClampToEdge,
                            .addressModeV  = WGPUAddressMode_ClampToEdge,
                            .addressModeW  = WGPUAddressMode_ClampToEdge,
                            .magFilter     = WGPUFilterMode_Linear,
                            .minFilter     = WGPUFilterMode_Linear,
                            .mipmapFilter  = WGPUMipmapFilterMode_Linear,
                            .maxAnisotropy = 1,
                          });
  ASSERT(state.textures.sampler != NULL);

  free(src_pixels);
  free(dst_pixels);
}

/* -------------------------------------------------------------------------- *
 * Uniform buffer initialization
 * -------------------------------------------------------------------------- */

static void init_uniform_buffers(wgpu_context_t* wgpu_context)
{
  state.uniform_buffers.src = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Source uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(uniforms_t),
                  });

  state.uniform_buffers.dst = wgpu_create_buffer(
    wgpu_context, &(wgpu_buffer_desc_t){
                    .label = "Destination uniform buffer",
                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                    .size  = sizeof(uniforms_t),
                  });
}

/* -------------------------------------------------------------------------- *
 * Bind group layout and pipeline layout
 * -------------------------------------------------------------------------- */

static void init_bind_group_layout(wgpu_context_t* wgpu_context)
{
  WGPUBindGroupLayoutEntry entries[3] = {
    [0] = {
      .binding    = 0,
      .visibility = WGPUShaderStage_Fragment,
      .sampler    = {.type = WGPUSamplerBindingType_Filtering},
    },
    [1] = {
      .binding    = 1,
      .visibility = WGPUShaderStage_Fragment,
      .texture    = {
        .sampleType    = WGPUTextureSampleType_Float,
        .viewDimension = WGPUTextureViewDimension_2D,
        .multisampled  = false,
      },
    },
    [2] = {
      .binding    = 2,
      .visibility = WGPUShaderStage_Vertex,
      .buffer     = {
        .type           = WGPUBufferBindingType_Uniform,
        .minBindingSize = sizeof(uniforms_t),
      },
    },
  };

  state.bind_group_layout = wgpuDeviceCreateBindGroupLayout(
    wgpu_context->device, &(WGPUBindGroupLayoutDescriptor){
                            .label      = STRVIEW("Bind group layout"),
                            .entryCount = ARRAY_SIZE(entries),
                            .entries    = entries,
                          });

  state.pipeline_layout = wgpuDeviceCreatePipelineLayout(
    wgpu_context->device, &(WGPUPipelineLayoutDescriptor){
                            .label                = STRVIEW("Pipeline layout"),
                            .bindGroupLayoutCount = 1,
                            .bindGroupLayouts     = &state.bind_group_layout,
                          });
}

/* -------------------------------------------------------------------------- *
 * Bind groups initialization
 * -------------------------------------------------------------------------- */

static void init_bind_groups(wgpu_context_t* wgpu_context)
{
  /* Premultiplied source bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0, .sampler = state.textures.sampler},
      [1]
      = {.binding = 1, .textureView = state.textures.src_premultiplied.view},
      [2] = {.binding = 2,
             .buffer  = state.uniform_buffers.src.buffer,
             .size    = sizeof(uniforms_t)},
    };
    state.bind_groups_premultiplied.src = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Premultiplied source bind group"),
        .layout     = state.bind_group_layout,
        .entryCount = ARRAY_SIZE(entries),
        .entries    = entries,
      });
  }

  /* Premultiplied destination bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0, .sampler = state.textures.sampler},
      [1]
      = {.binding = 1, .textureView = state.textures.dst_premultiplied.view},
      [2] = {.binding = 2,
             .buffer  = state.uniform_buffers.dst.buffer,
             .size    = sizeof(uniforms_t)},
    };
    state.bind_groups_premultiplied.dst = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Premultiplied destination bind group"),
        .layout     = state.bind_group_layout,
        .entryCount = ARRAY_SIZE(entries),
        .entries    = entries,
      });
  }

  /* Unpremultiplied source bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0, .sampler = state.textures.sampler},
      [1]
      = {.binding = 1, .textureView = state.textures.src_unpremultiplied.view},
      [2] = {.binding = 2,
             .buffer  = state.uniform_buffers.src.buffer,
             .size    = sizeof(uniforms_t)},
    };
    state.bind_groups_unpremultiplied.src = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Unpremultiplied source bind group"),
        .layout     = state.bind_group_layout,
        .entryCount = ARRAY_SIZE(entries),
        .entries    = entries,
      });
  }

  /* Unpremultiplied destination bind group */
  {
    WGPUBindGroupEntry entries[3] = {
      [0] = {.binding = 0, .sampler = state.textures.sampler},
      [1]
      = {.binding = 1, .textureView = state.textures.dst_unpremultiplied.view},
      [2] = {.binding = 2,
             .buffer  = state.uniform_buffers.dst.buffer,
             .size    = sizeof(uniforms_t)},
    };
    state.bind_groups_unpremultiplied.dst = wgpuDeviceCreateBindGroup(
      wgpu_context->device,
      &(WGPUBindGroupDescriptor){
        .label      = STRVIEW("Unpremultiplied destination bind group"),
        .layout     = state.bind_group_layout,
        .entryCount = ARRAY_SIZE(entries),
        .entries    = entries,
      });
  }
}

/* -------------------------------------------------------------------------- *
 * Pipeline initialization
 * -------------------------------------------------------------------------- */

static void init_dst_pipeline(wgpu_context_t* wgpu_context)
{
  /* Color blend state (no blending for destination) */
  WGPUBlendState blend_state = wgpu_create_blend_state(false);

  WGPURenderPipelineDescriptor pipeline_desc = {
    .label  = STRVIEW("Destination pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module     = state.shader_module,
      .entryPoint = STRVIEW("vs"),
    },
    .fragment = &(WGPUFragmentState){
      .module      = state.shader_module,
      .entryPoint  = STRVIEW("fs"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.dst_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
  ASSERT(state.dst_pipeline != NULL);
}

static void init_src_pipeline(wgpu_context_t* wgpu_context)
{
  /* Release previous pipeline if exists */
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.src_pipeline);

  /* Make blend component valid (min/max operations require one/one factors) */
  blend_component_t color = state.settings.color;
  blend_component_t alpha = state.settings.alpha;

  if (color.operation == BLEND_OP_MIN || color.operation == BLEND_OP_MAX) {
    color.src_factor = BLEND_FACTOR_ONE;
    color.dst_factor = BLEND_FACTOR_ONE;
  }
  if (alpha.operation == BLEND_OP_MIN || alpha.operation == BLEND_OP_MAX) {
    alpha.src_factor = BLEND_FACTOR_ONE;
    alpha.dst_factor = BLEND_FACTOR_ONE;
  }

  WGPUBlendState blend_state = {
    .color = {
      .operation = get_wgpu_blend_operation(color.operation),
      .srcFactor = get_wgpu_blend_factor(color.src_factor),
      .dstFactor = get_wgpu_blend_factor(color.dst_factor),
    },
    .alpha = {
      .operation = get_wgpu_blend_operation(alpha.operation),
      .srcFactor = get_wgpu_blend_factor(alpha.src_factor),
      .dstFactor = get_wgpu_blend_factor(alpha.dst_factor),
    },
  };

  WGPURenderPipelineDescriptor pipeline_desc = {
    .label  = STRVIEW("Source pipeline"),
    .layout = state.pipeline_layout,
    .vertex = {
      .module     = state.shader_module,
      .entryPoint = STRVIEW("vs"),
    },
    .fragment = &(WGPUFragmentState){
      .module      = state.shader_module,
      .entryPoint  = STRVIEW("fs"),
      .targetCount = 1,
      .targets     = &(WGPUColorTargetState){
        .format    = wgpu_context->render_format,
        .blend     = &blend_state,
        .writeMask = WGPUColorWriteMask_All,
      },
    },
    .primitive = {
      .topology = WGPUPrimitiveTopology_TriangleList,
    },
    .multisample = {
      .count = 1,
      .mask  = 0xFFFFFFFF,
    },
  };

  state.src_pipeline
    = wgpuDeviceCreateRenderPipeline(wgpu_context->device, &pipeline_desc);
  ASSERT(state.src_pipeline != NULL);

  state.pipeline_needs_rebuild = false;
}

/* -------------------------------------------------------------------------- *
 * Uniform update
 * -------------------------------------------------------------------------- */

static void update_uniforms(wgpu_context_t* wgpu_context)
{
  const float canvas_width  = (float)wgpu_context->width;
  const float canvas_height = (float)wgpu_context->height;

  /* Orthographic projection matrix */
  mat4 projection;
  glm_ortho(0.0f, canvas_width, canvas_height, 0.0f, -1.0f, 1.0f, projection);

  /* Source uniform - scale to texture size */
  {
    mat4 scale_matrix;
    glm_mat4_identity(scale_matrix);
    glm_scale(scale_matrix,
              (vec3){(float)TEXTURE_SIZE, (float)TEXTURE_SIZE, 1.0f});

    uniforms_t uniforms;
    glm_mat4_mul(projection, scale_matrix, uniforms.matrix);
    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.src.buffer,
                         0, &uniforms, sizeof(uniforms));
  }

  /* Destination uniform - scale to texture size */
  {
    mat4 scale_matrix;
    glm_mat4_identity(scale_matrix);
    glm_scale(scale_matrix,
              (vec3){(float)TEXTURE_SIZE, (float)TEXTURE_SIZE, 1.0f});

    uniforms_t uniforms;
    glm_mat4_mul(projection, scale_matrix, uniforms.matrix);
    wgpuQueueWriteBuffer(wgpu_context->queue, state.uniform_buffers.dst.buffer,
                         0, &uniforms, sizeof(uniforms));
  }
}

/* -------------------------------------------------------------------------- *
 * Apply preset
 * -------------------------------------------------------------------------- */

static void apply_preset(void)
{
  preset_def_t* preset         = &state.preset_defs[state.settings.preset];
  state.settings.color         = preset->color;
  state.settings.alpha         = preset->alpha;
  state.pipeline_needs_rebuild = true;
}

/* -------------------------------------------------------------------------- *
 * GUI rendering
 * -------------------------------------------------------------------------- */

static void render_gui(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  igSetNextWindowPos((ImVec2){(float)wgpu_context->width - 360.0f, 10.0f},
                     ImGuiCond_FirstUseEver, (ImVec2){0.0f, 0.0f});
  igSetNextWindowSize((ImVec2){350.0f, 0.0f}, ImGuiCond_FirstUseEver);

  igBegin("Blending Settings", NULL, ImGuiWindowFlags_AlwaysAutoResize);

  /* Canvas alpha mode */
  {
    int alpha_mode = (int)state.settings.alpha_mode;
    if (imgui_overlay_combo_box("Canvas alphaMode", &alpha_mode,
                                state.alpha_mode_names, ALPHA_MODE_COUNT)) {
      state.settings.alpha_mode = (alpha_mode_t)alpha_mode;
    }
  }

  /* Texture data */
  {
    int texture_set = (int)state.settings.texture_set;
    if (imgui_overlay_combo_box("Texture data", &texture_set,
                                state.texture_set_names, TEXTURE_SET_COUNT)) {
      state.settings.texture_set = (texture_set_t)texture_set;
    }
  }

  /* Preset */
  {
    int preset = (int)state.settings.preset;
    if (imgui_overlay_combo_box("Preset", &preset, state.preset_names,
                                PRESET_COUNT)) {
      state.settings.preset = (blend_preset_t)preset;
      apply_preset();
    }
  }

  igSeparator();

  /* Color blend settings */
  if (igCollapsingHeaderBoolPtr("Color", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    {
      int op = (int)state.settings.color.operation;
      if (imgui_overlay_combo_box("operation##color", &op,
                                  state.operation_names, BLEND_OP_COUNT)) {
        state.settings.color.operation = (blend_operation_t)op;
        state.pipeline_needs_rebuild   = true;
      }
    }
    {
      int factor = (int)state.settings.color.src_factor;
      if (imgui_overlay_combo_box("srcFactor##color", &factor,
                                  state.factor_names, BLEND_FACTOR_COUNT)) {
        state.settings.color.src_factor = (blend_factor_t)factor;
        state.pipeline_needs_rebuild    = true;
      }
    }
    {
      int factor = (int)state.settings.color.dst_factor;
      if (imgui_overlay_combo_box("dstFactor##color", &factor,
                                  state.factor_names, BLEND_FACTOR_COUNT)) {
        state.settings.color.dst_factor = (blend_factor_t)factor;
        state.pipeline_needs_rebuild    = true;
      }
    }
  }

  /* Alpha blend settings */
  if (igCollapsingHeaderBoolPtr("Alpha", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    {
      int op = (int)state.settings.alpha.operation;
      if (imgui_overlay_combo_box("operation##alpha", &op,
                                  state.operation_names, BLEND_OP_COUNT)) {
        state.settings.alpha.operation = (blend_operation_t)op;
        state.pipeline_needs_rebuild   = true;
      }
    }
    {
      int factor = (int)state.settings.alpha.src_factor;
      if (imgui_overlay_combo_box("srcFactor##alpha", &factor,
                                  state.factor_names, BLEND_FACTOR_COUNT)) {
        state.settings.alpha.src_factor = (blend_factor_t)factor;
        state.pipeline_needs_rebuild    = true;
      }
    }
    {
      int factor = (int)state.settings.alpha.dst_factor;
      if (imgui_overlay_combo_box("dstFactor##alpha", &factor,
                                  state.factor_names, BLEND_FACTOR_COUNT)) {
        state.settings.alpha.dst_factor = (blend_factor_t)factor;
        state.pipeline_needs_rebuild    = true;
      }
    }
  }

  /* Constant settings */
  if (igCollapsingHeaderBoolPtr("Constant", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    float color[4]
      = {state.settings.constant_color[0], state.settings.constant_color[1],
         state.settings.constant_color[2], state.settings.constant_alpha};
    if (igColorEdit4("color##constant", color, ImGuiColorEditFlags_None)) {
      state.settings.constant_color[0] = color[0];
      state.settings.constant_color[1] = color[1];
      state.settings.constant_color[2] = color[2];
      state.settings.constant_alpha    = color[3];
    }
  }

  /* Clear color settings */
  if (igCollapsingHeaderBoolPtr("Clear Color", NULL,
                                ImGuiTreeNodeFlags_DefaultOpen)) {
    igCheckbox("premultiply", &state.settings.clear_premultiply);
    igDragFloat("alpha##clear", &state.settings.clear_alpha, 0.01f, 0.0f, 1.0f,
                "%.2f", 0);
    float color[4]
      = {state.settings.clear_color[0], state.settings.clear_color[1],
         state.settings.clear_color[2], 1.0f};
    if (igColorEdit3("color##clear", color, ImGuiColorEditFlags_None)) {
      state.settings.clear_color[0] = color[0];
      state.settings.clear_color[1] = color[1];
      state.settings.clear_color[2] = color[2];
    }
  }

  igEnd();
}

/* -------------------------------------------------------------------------- *
 * Input handling
 * -------------------------------------------------------------------------- */

static void input_event_cb(wgpu_context_t* wgpu_context,
                           const input_event_t* input_event)
{
  imgui_overlay_handle_input(wgpu_context, input_event);
}

/* -------------------------------------------------------------------------- *
 * Init / Frame / Shutdown
 * -------------------------------------------------------------------------- */

static int init(wgpu_context_t* wgpu_context)
{
  if (!wgpu_context) {
    return EXIT_FAILURE;
  }

  stm_setup();

  /* Create shader module */
  state.shader_module = wgpu_create_shader_module(wgpu_context->device,
                                                  textured_quad_shader_wgsl);
  ASSERT(state.shader_module != NULL);

  /* Initialize resources */
  init_textures(wgpu_context);
  init_uniform_buffers(wgpu_context);
  init_bind_group_layout(wgpu_context);
  init_bind_groups(wgpu_context);
  init_dst_pipeline(wgpu_context);
  init_src_pipeline(wgpu_context);

  /* Initialize ImGui */
  imgui_overlay_init(wgpu_context);

  state.initialized = true;
  return EXIT_SUCCESS;
}

static int frame(wgpu_context_t* wgpu_context)
{
  if (!state.initialized) {
    return EXIT_FAILURE;
  }

  /* Rebuild pipeline if needed */
  if (state.pipeline_needs_rebuild) {
    init_src_pipeline(wgpu_context);
  }

  /* Update uniforms */
  update_uniforms(wgpu_context);

  /* Calculate clear color */
  float mult
    = state.settings.clear_premultiply ? state.settings.clear_alpha : 1.0f;
  state.color_attachment.clearValue = (WGPUColor){
    .r = state.settings.clear_color[0] * mult,
    .g = state.settings.clear_color[1] * mult,
    .b = state.settings.clear_color[2] * mult,
    .a = state.settings.clear_alpha,
  };

  /* Select bind groups based on texture set */
  WGPUBindGroup src_bind_group, dst_bind_group;
  if (state.settings.texture_set == TEXTURE_SET_PREMULTIPLIED) {
    src_bind_group = state.bind_groups_premultiplied.src;
    dst_bind_group = state.bind_groups_premultiplied.dst;
  }
  else {
    src_bind_group = state.bind_groups_unpremultiplied.src;
    dst_bind_group = state.bind_groups_unpremultiplied.dst;
  }

  /* Set render target */
  state.color_attachment.view = wgpu_context->swapchain_view;

  /* Create command encoder and render pass */
  WGPUCommandEncoder cmd_encoder
    = wgpuDeviceCreateCommandEncoder(wgpu_context->device, NULL);
  WGPURenderPassEncoder rpass_enc = wgpuCommandEncoderBeginRenderPass(
    cmd_encoder, &state.render_pass_descriptor);

  /* Draw destination texture without blending */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.dst_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, dst_bind_group, 0, NULL);
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);

  /* Draw source texture with blending */
  wgpuRenderPassEncoderSetPipeline(rpass_enc, state.src_pipeline);
  wgpuRenderPassEncoderSetBindGroup(rpass_enc, 0, src_bind_group, 0, NULL);
  wgpuRenderPassEncoderSetBlendConstant(rpass_enc,
                                        &(WGPUColor){
                                          .r = state.settings.constant_color[0],
                                          .g = state.settings.constant_color[1],
                                          .b = state.settings.constant_color[2],
                                          .a = state.settings.constant_alpha,
                                        });
  wgpuRenderPassEncoderDraw(rpass_enc, 6, 1, 0, 0);

  wgpuRenderPassEncoderEnd(rpass_enc);

  /* Submit command buffer */
  WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd_encoder, NULL);
  wgpuQueueSubmit(wgpu_context->queue, 1, &cmd_buffer);

  /* Render GUI */
  imgui_overlay_new_frame(wgpu_context,
                          (float)stm_sec(stm_laptime(&state.last_frame_time)));
  render_gui(wgpu_context);
  imgui_overlay_render(wgpu_context);

  /* Cleanup */
  wgpuRenderPassEncoderRelease(rpass_enc);
  wgpuCommandBufferRelease(cmd_buffer);
  wgpuCommandEncoderRelease(cmd_encoder);

  return EXIT_SUCCESS;
}

static void shutdown(wgpu_context_t* wgpu_context)
{
  UNUSED_VAR(wgpu_context);

  imgui_overlay_shutdown();

  wgpu_destroy_texture(&state.textures.src_unpremultiplied);
  wgpu_destroy_texture(&state.textures.dst_unpremultiplied);
  wgpu_destroy_texture(&state.textures.src_premultiplied);
  wgpu_destroy_texture(&state.textures.dst_premultiplied);
  WGPU_RELEASE_RESOURCE(Sampler, state.textures.sampler);

  wgpu_destroy_buffer(&state.uniform_buffers.src);
  wgpu_destroy_buffer(&state.uniform_buffers.dst);

  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups_premultiplied.src);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups_premultiplied.dst);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups_unpremultiplied.src);
  WGPU_RELEASE_RESOURCE(BindGroup, state.bind_groups_unpremultiplied.dst);
  WGPU_RELEASE_RESOURCE(BindGroupLayout, state.bind_group_layout);
  WGPU_RELEASE_RESOURCE(PipelineLayout, state.pipeline_layout);

  WGPU_RELEASE_RESOURCE(RenderPipeline, state.dst_pipeline);
  WGPU_RELEASE_RESOURCE(RenderPipeline, state.src_pipeline);
  WGPU_RELEASE_RESOURCE(ShaderModule, state.shader_module);
}

int main(void)
{
  wgpu_start(&(wgpu_desc_t){
    .title          = "Blending",
    .init_cb        = init,
    .frame_cb       = frame,
    .shutdown_cb    = shutdown,
    .input_event_cb = input_event_cb,
  });

  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- *
 * WGSL Shader
 * -------------------------------------------------------------------------- */

// clang-format off
static const char* textured_quad_shader_wgsl = CODE(
  struct OurVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
  };

  struct Uniforms {
    matrix: mat4x4f,
  };

  @group(0) @binding(2) var<uniform> uni: Uniforms;

  @vertex fn vs(
    @builtin(vertex_index) vertexIndex : u32
  ) -> OurVertexShaderOutput {
    let pos = array(
      vec2f( 0.0,  0.0),  // center
      vec2f( 1.0,  0.0),  // right, center
      vec2f( 0.0,  1.0),  // center, top

      // 2nd triangle
      vec2f( 0.0,  1.0),  // center, top
      vec2f( 1.0,  0.0),  // right, center
      vec2f( 1.0,  1.0),  // right, top
    );

    var vsOutput: OurVertexShaderOutput;
    let xy = pos[vertexIndex];
    vsOutput.position = uni.matrix * vec4f(xy, 0.0, 1.0);
    vsOutput.texcoord = xy;
    return vsOutput;
  }

  @group(0) @binding(0) var ourSampler: sampler;
  @group(0) @binding(1) var ourTexture: texture_2d<f32>;

  @fragment fn fs(fsInput: OurVertexShaderOutput) -> @location(0) vec4f {
    return textureSample(ourTexture, ourSampler, fsInput.texcoord);
  }
);
// clang-format on
